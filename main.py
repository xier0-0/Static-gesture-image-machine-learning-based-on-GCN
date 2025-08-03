"""
主程序文件 - 手语识别系统
整合所有模块并执行完整的训练流程
"""
import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_preprocessing import DatasetProcessor
from dataset import HandGestureDataLoader, analyze_dataset
from models import HandGestureGCNWithBatch, count_parameters, count_flops
from trainer import HandGestureTrainer
import torch

def preprocess_data(data_root=DATA_ROOT):
    """数据预处理"""
    print("=== 数据预处理阶段 ===")

    # 检查数据目录
    if not os.path.exists(data_root):
        print(f"错误: 数据目录 {data_root} 不存在")
        return False

    # 检查是否已经预处理过
    if os.path.exists("processed_data.pkl"):
        print("发现已预处理的数据文件，跳过预处理阶段")
        return True

    # 开始预处理
    processor = DatasetProcessor(data_root)
    processed_data = processor.process_dataset()

    if processed_data and len(processed_data['data']) > 0:
        print(f"数据预处理完成，共处理 {len(processed_data['data'])} 个样本")
        return True
    else:
        print("数据预处理失败")
        return False

def analyze_data():
    """数据分析"""
    print("\n=== 数据分析阶段 ===")

    data_loader = HandGestureDataLoader()
    if data_loader.load_processed_data():
        analyze_dataset(data_loader)
        return True
    return False

def train_model(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS):
    """模型训练"""
    print("\n=== 模型训练阶段 ===")

    # 加载数据
    data_loader = HandGestureDataLoader()
    if not data_loader.load_processed_data():
        print("无法加载预处理数据")
        return False

    # 分割数据集
    if not data_loader.split_dataset(
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    ):
        print("数据集分割失败")
        return False

    # 获取数据加载器
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=batch_size
    )

    if not all([train_loader, val_loader, test_loader]):
        print("数据加载器创建失败")
        return False

    # 创建模型
    num_classes = len(data_loader.classes)
    model = HandGestureGCNWithBatch(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        dropout=DROPOUT_RATE
    )

    # 打印模型信息
    print(f"模型配置:")
    print(f"  输入维度: {INPUT_DIM}")
    print(f"  隐藏维度: {HIDDEN_DIM}")
    print(f"  类别数量: {num_classes}")
    print(f"  参数数量: {count_parameters(model):,}")

    # 估算FLOPs
    if len(data_loader.data_list) > 0:
        sample_data = data_loader.data_list[0]
        estimated_flops = count_flops(model, sample_data)
        print(f"  估算FLOPs: {estimated_flops:,}")

    # 创建训练器
    trainer = HandGestureTrainer(model, device=DEVICE)
    trainer.setup_training(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        label_smoothing=LABEL_SMOOTHING,
        num_epochs=num_epochs
    )

    # 训练模型
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        patience=PATIENCE
    )

    # 绘制训练历史
    trainer.plot_training_history(
        save_path=os.path.join(RESULTS_DIR, "training_history.png")
    )

    # 测试模型
    print("\n=== 模型测试阶段 ===")
    test_results = trainer.test(test_loader, data_loader.classes)

    # 绘制混淆矩阵
    trainer.plot_confusion_matrix(
        test_results['confusion_matrix'],
        data_loader.classes,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png")
    )

    # 保存结果
    trainer.save_results(test_results)

    # 打印最终结果
    print(f"\n=== 训练完成 ===")
    print(f"最佳验证准确率: {trainer.best_val_accuracy:.2f}%")
    print(f"测试准确率: {test_results['test_accuracy']:.2f}%")
    print(f"F1-Macro: {test_results['f1_macro']:.4f}")
    print(f"F1-Weighted: {test_results['f1_weighted']:.4f}")

    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="手语识别系统")
    parser.add_argument("--mode", choices=["preprocess", "analyze", "train", "all"],
                       default="all", help="运行模式")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT,
                       help="数据根目录")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help="批大小")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                       help="训练轮数")

    args = parser.parse_args()

    # 更新配置 - 使用局部变量而不是全局变量
    data_root = args.data_root
    batch_size = args.batch_size
    num_epochs = args.epochs

    print("=== 手语识别系统 ===")
    print(f"数据目录: {data_root}")
    print(f"批大小: {batch_size}")
    print(f"训练轮数: {num_epochs}")
    print(f"设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"可用GPU: {torch.cuda.get_device_name(0)}")

    success = True

    if args.mode in ["preprocess", "all"]:
        success &= preprocess_data(data_root)

    if args.mode in ["analyze", "all"] and success:
        success &= analyze_data()

    if args.mode in ["train", "all"] and success:
        success &= train_model(batch_size, num_epochs)

    if success:
        print("\n=== 所有任务完成 ===")
        print("结果文件保存在以下目录:")
        print(f"  模型文件: {MODEL_SAVE_DIR}")
        print(f"  训练图表: {RESULTS_DIR}")
        print(f"  日志文件: {LOG_DIR}")
    else:
        print("\n=== 任务执行失败 ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
