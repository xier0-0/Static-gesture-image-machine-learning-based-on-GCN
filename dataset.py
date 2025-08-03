"""
数据集模块 - 实现数据加载和批处理
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pickle
import numpy as np
from typing import List, Tuple, Optional
import random
from sklearn.model_selection import train_test_split
from config import *

class HandGestureDataset(Dataset):
    """手语数据集类"""

    def __init__(self, data_list: List[Data], transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        if self.transform:
            data = self.transform(data)

        return data

class HandGestureDataLoader:
    """数据加载器管理类"""

    def __init__(self, processed_data_path: str = "processed_data.pkl"):
        self.processed_data_path = processed_data_path
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.classes = None
        self.class_to_idx = None

    def load_processed_data(self):
        """加载预处理后的数据"""
        try:
            with open(self.processed_data_path, 'rb') as f:
                data_dict = pickle.load(f)

            self.data_list = data_dict['data']
            self.classes = data_dict['classes']
            self.class_to_idx = data_dict['class_to_idx']

            print(f"加载数据成功: {len(self.data_list)} 个样本, {len(self.classes)} 个类别")
            print(f"类别: {self.classes}")

            return True

        except FileNotFoundError:
            print(f"未找到预处理数据文件: {self.processed_data_path}")
            print("请先运行 data_preprocessing.py 进行数据预处理")
            return False

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1,
                     test_ratio: float = 0.1, random_seed: int = 42):
        """分割数据集"""
        if not hasattr(self, 'data_list'):
            print("请先加载数据")
            return False

        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 检查每个类别的样本数量
        class_counts = {}
        for data in self.data_list:
            class_name = data.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("类别样本分布:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} 个样本")

        # 过滤掉样本太少的类别（少于2个样本）
        valid_classes = {class_name for class_name, count in class_counts.items() if count >= 2}
        if len(valid_classes) < len(class_counts):
            print(f"警告: 过滤掉 {len(class_counts) - len(valid_classes)} 个样本太少的类别")

        # 过滤数据
        filtered_data = [data for data in self.data_list if data.class_name in valid_classes]

        if len(filtered_data) == 0:
            print("错误: 没有足够的有效数据")
            return False

        print(f"过滤后剩余 {len(filtered_data)} 个样本")

        # 按类别分层采样
        try:
            train_data, temp_data = train_test_split(
                filtered_data,
                test_size=(val_ratio + test_ratio),
                random_state=random_seed,
                stratify=[data.y.item() for data in filtered_data]
            )

            # 分割验证集和测试集
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio_adjusted),
                random_state=random_seed,
                stratify=[data.y.item() for data in temp_data]
            )
        except ValueError as e:
            print(f"分层采样失败: {e}")
            print("使用随机分割...")
            # 如果分层采样失败，使用随机分割
            train_data, temp_data = train_test_split(
                filtered_data,
                test_size=(val_ratio + test_ratio),
                random_state=random_seed
            )

            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio_adjusted),
                random_state=random_seed
            )

        # 创建数据集对象
        self.train_dataset = HandGestureDataset(train_data)
        self.val_dataset = HandGestureDataset(val_data)
        self.test_dataset = HandGestureDataset(test_data)

        print(f"数据集分割完成:")
        print(f"  训练集: {len(train_data)} 个样本")
        print(f"  验证集: {len(val_data)} 个样本")
        print(f"  测试集: {len(test_data)} 个样本")

        return True

    def get_dataloaders(self, batch_size: int = 64, shuffle: bool = True,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取数据加载器"""
        if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
            print("请先分割数据集")
            return None, None, None

        def collate_fn(batch):
            """自定义批处理函数"""
            return Batch.from_data_list(batch)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if DEVICE == "cuda" else False  # 加速CPU到GPU传输
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if DEVICE == "cuda" else False
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if DEVICE == "cuda" else False
        )

        return train_loader, val_loader, test_loader

    def get_class_distribution(self):
        """获取类别分布"""
        if not hasattr(self, 'data_list'):
            return None

        class_counts = {}
        for data in self.data_list:
            class_name = data.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return class_counts

def create_data_transforms():
    """创建数据变换"""
    def add_noise(data: Data, noise_std: float = 0.01) -> Data:
        """添加噪声增强"""
        if random.random() < 0.5:  # 50%概率添加噪声
            noise = torch.randn_like(data.x) * noise_std
            data.x = data.x + noise
        return data

    def random_rotation(data: Data, max_angle: float = 0.1) -> Data:
        """随机旋转增强"""
        if random.random() < 0.3:  # 30%概率旋转
            angle = (random.random() - 0.5) * 2 * max_angle
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=torch.float32)

            # 只对坐标部分进行旋转
            coords = data.x[:, :3]
            rotated_coords = torch.matmul(coords, rotation_matrix.T)
            data.x[:, :3] = rotated_coords

        return data

    def combined_transform(data: Data) -> Data:
        """组合变换"""
        data = add_noise(data)
        data = random_rotation(data)
        return data

    return combined_transform

def analyze_dataset(data_loader: HandGestureDataLoader):
    """分析数据集"""
    print("=== 数据集分析 ===")

    # 类别分布
    class_dist = data_loader.get_class_distribution()
    if class_dist:
        print("类别分布:")
        for class_name, count in sorted(class_dist.items()):
            print(f"  {class_name}: {count} 个样本")

    # 特征统计
    if hasattr(data_loader, 'data_list') and data_loader.data_list:
        features = torch.stack([data.x for data in data_loader.data_list])
        print(f"\n特征统计:")
        print(f"  特征维度: {features.shape}")
        print(f"  特征均值: {features.mean(dim=(0, 1))}")
        print(f"  特征标准差: {features.std(dim=(0, 1))}")
        print(f"  特征范围: [{features.min()}, {features.max()}]")

    # 图结构统计
    if hasattr(data_loader, 'data_list') and data_loader.data_list:
        edge_counts = [data.edge_index.size(1) for data in data_loader.data_list]
        print(f"\n图结构统计:")
        print(f"  平均边数: {np.mean(edge_counts):.1f}")
        print(f"  边数范围: [{min(edge_counts)}, {max(edge_counts)}]")

if __name__ == "__main__":
    # 测试数据加载
    data_loader = HandGestureDataLoader()

    if data_loader.load_processed_data():
        # 分析数据集
        analyze_dataset(data_loader)

        # 分割数据集
        if data_loader.split_dataset():
            # 获取数据加载器
            train_loader, val_loader, test_loader = data_loader.get_dataloaders()

            if train_loader:
                print(f"\n数据加载器测试:")
                for batch_idx, batch in enumerate(train_loader):
                    print(f"  批次 {batch_idx + 1}: {batch.num_graphs} 个图")
                    print(f"    节点特征形状: {batch.x.shape}")
                    print(f"    边索引形状: {batch.edge_index.shape}")
                    print(f"    标签形状: {batch.y.shape}")
                    break
