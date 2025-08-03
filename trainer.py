"""
训练器模块 - 实现训练、验证和测试功能
严格按照技术方案实现训练策略
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional
import json

from models import HandGestureGCNWithBatch, count_parameters, count_flops
from dataset import HandGestureDataLoader
from config import *

class HandGestureTrainer:
    """手语识别训练器"""

    def __init__(self, model: nn.Module, device: str = DEVICE):
        self.model = model.to(device)
        self.device = device
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

        # 早停
        self.patience_counter = 0

        # 创建结果目录
        Path(RESULTS_DIR).mkdir(exist_ok=True)
        Path(MODEL_SAVE_DIR).mkdir(exist_ok=True)

        # 打印模型设备确认
        print(f"模型已移到设备: {next(self.model.parameters()).device}")

    def setup_training(self, learning_rate: float = LEARNING_RATE,
                      weight_decay: float = WEIGHT_DECAY,
                      label_smoothing: float = LABEL_SMOOTHING,
                      num_epochs: int = NUM_EPOCHS):
        """设置训练参数"""
        # 损失函数：交叉熵 + 标签平滑
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # 优化器：AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器：CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )

        print(f"训练设置完成:")
        print(f"  学习率: {learning_rate}")
        print(f"  权重衰减: {weight_decay}")
        print(f"  标签平滑: {label_smoothing}")
        print(f"  设备: {self.device}")

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="训练")

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = batch.to(self.device)
            if self.device == "cuda":
                torch.cuda.synchronize()  # 同步GPU以准确计算时间（可选，调试用）

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch.y)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate_epoch(self, val_loader) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                # 移动数据到设备
                batch = batch.to(self.device)
                if self.device == "cuda":
                    torch.cuda.synchronize()

                # 前向传播
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.y)

                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs: int = NUM_EPOCHS,
              patience: int = PATIENCE, save_best: bool = True):
        """完整训练流程"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"早停耐心值: {patience}")

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader)

            # 学习率调度
            self.scheduler.step()

            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # 打印结果
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0

                if save_best:
                    self.save_model(f"best_model.pth")
                    print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
            else:
                self.patience_counter += 1

            # 早停检查
            if self.patience_counter >= patience:
                print(f"早停触发，最佳验证准确率: {self.best_val_accuracy:.2f}% (Epoch {self.best_epoch + 1})")
                break

        training_time = time.time() - start_time
        print(f"\n训练完成，总用时: {training_time:.2f} 秒")
        print(f"最佳验证准确率: {self.best_val_accuracy:.2f}% (Epoch {self.best_epoch + 1})")

    def test(self, test_loader, classes: List[str]) -> Dict:
        """测试模型"""
        print("开始测试...")

        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="测试"):
                # 移动数据到设备
                batch = batch.to(self.device)
                if self.device == "cuda":
                    torch.cuda.synchronize()

                # 前向传播
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.y)

                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()

                # 收集预测结果
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        # 计算指标
        test_accuracy = 100 * correct / total
        test_loss = total_loss / len(test_loader)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)

        # 获取实际出现的类别
        unique_labels = sorted(list(set(all_labels)))
        actual_classes = [classes[i] for i in unique_labels if i < len(classes)]

        # 分类报告
        try:
            report = classification_report(all_labels, all_predictions,
                                         target_names=actual_classes, output_dict=True)
        except ValueError as e:
            print(f"分类报告生成失败: {e}")
            print(f"实际类别数量: {len(unique_labels)}, 类别名称数量: {len(classes)}")
            # 使用数字标签生成报告
            report = classification_report(all_labels, all_predictions, output_dict=True)

        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels
        }

        print(f"测试结果:")
        print(f"  准确率: {test_accuracy:.2f}%")
        print(f"  损失: {test_loss:.4f}")
        print(f"  F1-Macro: {f1_macro:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")

        return results

    def save_model(self, filename: str):
        """保存模型"""
        save_path = os.path.join(MODEL_SAVE_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, save_path)
        print(f"模型已保存到: {save_path}")

    def load_model(self, filename: str):
        """加载模型"""
        load_path = os.path.join(MODEL_SAVE_DIR, filename)
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.best_epoch = checkpoint['best_epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']

        print(f"模型已从 {load_path} 加载")

    def plot_training_history(self, save_path: str = None):
        """绘制训练历史"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', color='blue')
        ax1.plot(self.val_losses, label='验证损失', color='red')
        ax1.set_title('损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率', color='blue')
        ax2.plot(self.val_accuracies, label='验证准确率', color='red')
        ax2.axhline(y=self.best_val_accuracy, color='green', linestyle='--',
                   label=f'最佳验证准确率: {self.best_val_accuracy:.2f}%')
        ax2.set_title('准确率曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        # 学习率曲线
        lr_history = []
        for param_group in self.optimizer.param_groups:
            lr_history.append(param_group['lr'])
        ax3.plot(lr_history, color='purple')
        ax3.set_title('学习率变化')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)

        # 损失差值
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        ax4.plot(loss_diff, color='orange')
        ax4.set_title('训练验证损失差值')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('|Train Loss - Val Loss|')
        ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")

        plt.show()

    def plot_confusion_matrix(self, cm: np.ndarray, classes: List[str],
                            save_path: str = None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 确保类别数量匹配
        if len(classes) != cm.shape[0]:
            print(f"警告: 混淆矩阵维度 {cm.shape} 与类别数量 {len(classes)} 不匹配")
            # 使用数字标签
            class_labels = [f'类别{i}' for i in range(cm.shape[0])]
        else:
            class_labels = classes

        # 绘制热力图
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('混淆矩阵 (百分比)')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵图已保存到: {save_path}")

        plt.show()

    def save_results(self, test_results: Dict, save_path: str = None):
        """保存测试结果"""
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, "test_results.json")

        # 转换numpy数组为列表以便JSON序列化
        results_to_save = {
            'test_accuracy': test_results['test_accuracy'],
            'test_loss': test_results['test_loss'],
            'f1_macro': test_results['f1_macro'],
            'f1_weighted': test_results['f1_weighted'],
            'confusion_matrix': test_results['confusion_matrix'].tolist(),
            'classification_report': test_results['classification_report'],
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'model_parameters': count_parameters(self.model)
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)

        print(f"测试结果已保存到: {save_path}")

def main():
    """主训练流程"""
    print("=== 手语识别模型训练 ===")

    # 加载数据
    data_loader = HandGestureDataLoader()
    if not data_loader.load_processed_data():
        return

    # 分割数据集
    if not data_loader.split_dataset():
        return

    # 获取数据加载器
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=BATCH_SIZE
    )

    if not all([train_loader, val_loader, test_loader]):
        return

    # 创建模型
    num_classes = len(data_loader.classes)
    model = HandGestureGCNWithBatch(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        dropout=DROPOUT_RATE
    )

    print(f"模型参数数量: {count_parameters(model):,}")

    # 创建训练器
    trainer = HandGestureTrainer(model)
    trainer.setup_training()

    # 训练模型
    trainer.train(train_loader, val_loader)

    # 绘制训练历史
    trainer.plot_training_history(
        save_path=os.path.join(RESULTS_DIR, "training_history.png")
    )

    # 测试模型
    test_results = trainer.test(test_loader, data_loader.classes)

    # 绘制混淆矩阵
    trainer.plot_confusion_matrix(
        test_results['confusion_matrix'],
        data_loader.classes,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png")
    )

    # 保存结果
    trainer.save_results(test_results)

if __name__ == "__main__":
    main()
