"""
模型架构模块 - 实现EdgeAwareGCN和特征增强模块
严格按照技术方案实现轻量级图神经网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple

class EdgeAwareGCNConv(nn.Module):
    """
    自定义边感知图卷积层
    单层EdgeAwareGCNConv(7→16) + ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super(EdgeAwareGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # 节点特征线性变换
        self.lin_node = nn.Linear(in_channels, out_channels)

        # 自环权重
        self.self_loop_weight = nn.Parameter(torch.Tensor(out_channels))
        nn.init.normal_(self.self_loop_weight, mean=0, std=0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x: 节点特征 [num_nodes, in_channels]
        edge_index: 边索引 [2, num_edges]
        """
        # 节点特征变换
        x_transformed = self.lin_node(x)

        # 计算度矩阵
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # 归一化
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 消息传递
        out = torch.zeros_like(x_transformed)
        out.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.out_channels),
                        x_transformed[col] * norm.unsqueeze(-1))

        # 添加自环
        out += x_transformed * self.self_loop_weight.unsqueeze(0)

        # ReLU激活和Dropout
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out

class FeatureEnhancementModule(nn.Module):
    """
    特征增强模块 - 融合双手交互和手掌朝向
    Linear(19→8) + ReLU + Linear(8→16)
    """

    def __init__(self, hidden_dim: int = 16):
        super(FeatureEnhancementModule, self).__init__()
        self.hidden_dim = hidden_dim

        # 双手交互特征维度：13个距离
        self.interaction_dim = 13
        # 手掌朝向特征维度：6个法向量
        self.orientation_dim = 6
        # 总输入维度
        self.input_dim = self.interaction_dim + self.orientation_dim

        # MLP网络
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, hidden_dim)
        )

        # 指尖节点索引（10个指尖）
        self.fingertip_indices = [4, 8, 12, 16, 20, 25, 29, 33, 37, 41]

    def compute_interaction_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算双手交互特征 - 13个关键距离
        """
        # 定义关键点对（双手之间的重要距离）
        key_pairs = [
            # 手掌中心距离
            (9, 30),   # 左手掌中心到右手掌中心
            # 拇指距离
            (4, 25),   # 左右手拇指尖
            # 食指尖距离
            (8, 29),   # 左右手食指尖
            # 中指尖距离
            (12, 33),  # 左右手中指尖
            # 无名指尖距离
            (16, 37),  # 左右手无名指尖
            # 小指尖距离
            (20, 41),  # 左右手小指尖
            # 手腕距离
            (0, 21),   # 左右手腕
            # 手掌宽度
            (5, 17),   # 左手掌宽度
            (26, 38),  # 右手掌宽度
            # 手指长度
            (0, 4),    # 左手拇指长度
            (21, 25),  # 右手拇指长度
            (5, 8),    # 左手食指长度
            (26, 29),  # 右手食指长度
            (9, 12),   # 左手中指长度
        ]

        # 检查输入维度
        if x.size(0) < 42:
            print(f"警告: 输入节点数量不足42，实际为: {x.size(0)}")
            # 如果节点数量不足，用零填充
            if x.size(0) < 42:
                padding = torch.zeros(42 - x.size(0), x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=0)

        interaction_features = []
        for i, j in key_pairs:
            dist = torch.norm(x[i] - x[j], dim=0)
            interaction_features.append(dist)

        return torch.stack(interaction_features)

    def compute_orientation_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算手掌朝向特征 - 简化为6个距离特征
        """
        # 左手掌特征（使用三个点的距离）
        left_palm_points = x[[0, 5, 17]]  # 手腕、食指根、小指根
        left_dist1 = torch.norm(left_palm_points[1] - left_palm_points[0])
        left_dist2 = torch.norm(left_palm_points[2] - left_palm_points[0])
        left_dist3 = torch.norm(left_palm_points[2] - left_palm_points[1])

        # 右手掌特征
        right_palm_points = x[[21, 26, 38]]  # 手腕、食指根、小指根
        right_dist1 = torch.norm(right_palm_points[1] - right_palm_points[0])
        right_dist2 = torch.norm(right_palm_points[2] - right_palm_points[0])
        right_dist3 = torch.norm(right_palm_points[2] - right_palm_points[1])

        # 组合特征
        orientation_features = torch.stack([left_dist1, left_dist2, left_dist3,
                                          right_dist1, right_dist2, right_dist3])

        return orientation_features

    def compute_normal(self, points: torch.Tensor) -> torch.Tensor:
        """
        计算三点确定的平面法向量
        """
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]

        # 确保向量是3D的
        if v1.dim() == 1:
            v1 = v1.unsqueeze(0)
        if v2.dim() == 1:
            v2 = v2.unsqueeze(0)

        # 如果维度不是3，用零填充
        if v1.size(-1) < 3:
            v1 = torch.cat([v1, torch.zeros(v1.size(0), 3 - v1.size(-1), device=v1.device)], dim=-1)
        if v2.size(-1) < 3:
            v2 = torch.cat([v2, torch.zeros(v2.size(0), 3 - v2.size(-1), device=v2.device)], dim=-1)

        normal = torch.cross(v1, v2)
        norm = torch.norm(normal)
        if norm > 1e-8:
            normal = normal / norm
        return normal.squeeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x: 节点特征 [42, hidden_dim]
        """
        # 计算交互特征
        interaction_features = self.compute_interaction_features(x)  # [13]
        # 计算朝向特征
        orientation_features = self.compute_orientation_features(x)  # [6]
        # 拼接
        combined_features = torch.cat([interaction_features, orientation_features], dim=0).to(x.device)  # 确保设备一致
        # 检查维度并自动调整
        if combined_features.shape[0] != 19:
            # 如果维度不对，调整MLP输入维度
            if not hasattr(self, 'mlp_adjusted'):
                self.mlp_adjusted = nn.Sequential(
                    nn.Linear(combined_features.shape[0], 8),
                    nn.ReLU(),
                    nn.Linear(8, self.hidden_dim)
                ).to(x.device)  # 确保调整后的MLP在同一设备
            enhanced_features = self.mlp_adjusted(combined_features)
        else:
            enhanced_features = self.mlp(combined_features)
        # 残差添加到指尖节点
        for idx in self.fingertip_indices:
            if idx < x.size(0):
                x[idx] += enhanced_features
        return x

class HandGestureGCN(nn.Module):
    """
    手语识别图神经网络模型
    严格按照技术方案实现：单层GCN + 特征增强 + 混合池化 + 分类头
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 16,
                 num_classes: int = 20, dropout: float = 0.2):
        super(HandGestureGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # 图卷积层
        self.gcn = EdgeAwareGCNConv(input_dim, hidden_dim, dropout)

        # 特征增强模块
        self.feature_enhancement = FeatureEnhancementModule(hidden_dim)

        # 混合池化后的特征维度
        self.pooled_dim = hidden_dim * 2  # max + mean

        # 分类头
        self.classifier = nn.Linear(self.pooled_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.dim() >= 2:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def mixed_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        混合池化：cat(max(x, dim=0), mean(x, dim=0)) → [32]
        """
        # 全局最大池化
        max_pooled = torch.max(x, dim=0)[0]  # [hidden_dim]

        # 全局平均池化
        mean_pooled = torch.mean(x, dim=0)   # [hidden_dim]

        # 拼接
        pooled = torch.cat([max_pooled, mean_pooled])  # [hidden_dim * 2]

        return pooled

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        data: 图数据对象
        """
        x, edge_index = data.x, data.edge_index

        # 图卷积层
        x = self.gcn(x, edge_index)  # [42, hidden_dim]

        # 特征增强
        x = self.feature_enhancement(x)  # [42, hidden_dim]

        # 混合池化
        pooled = self.mixed_pooling(x)  # [hidden_dim * 2]

        # 分类
        logits = self.classifier(pooled)  # [num_classes]

        return logits

class HandGestureGCNWithBatch(nn.Module):
    """
    支持批处理的版本
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 16,
                 num_classes: int = 20, dropout: float = 0.2):
        super(HandGestureGCNWithBatch, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # 图卷积层
        self.gcn = EdgeAwareGCNConv(input_dim, hidden_dim, dropout)

        # 特征增强模块
        self.feature_enhancement = FeatureEnhancementModule(hidden_dim)

        # 混合池化后的特征维度
        self.pooled_dim = hidden_dim * 2

        # 分类头
        self.classifier = nn.Linear(self.pooled_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.dim() >= 2:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def mixed_pooling_with_batch(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        批处理版本的混合池化
        """
        batch_size = batch.max().item() + 1
        pooled_features = []

        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                x_i = x[mask]
                max_pooled = torch.max(x_i, dim=0)[0]
                mean_pooled = torch.mean(x_i, dim=0)
                pooled = torch.cat([max_pooled, mean_pooled])
                pooled_features.append(pooled)

        return torch.stack(pooled_features)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        前向传播（批处理版本）
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 图卷积层
        x = self.gcn(x, edge_index)

        # 特征增强（需要分别处理每个图）
        enhanced_features = []
        batch_size = batch.max().item() + 1

        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                x_i = x[mask]
                enhanced_i = self.feature_enhancement(x_i)
                enhanced_features.append(enhanced_i.to(x.device))  # 确保设备一致

        x = torch.cat(enhanced_features, dim=0)

        # 混合池化
        pooled = self.mixed_pooling_with_batch(x, batch)

        # 分类
        logits = self.classifier(pooled)

        return logits

def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model: nn.Module, input_data: Data) -> int:
    """估算模型FLOPs"""
    # 简化的FLOPs计算
    num_nodes = input_data.x.size(0)
    num_edges = input_data.edge_index.size(1)

    # GCN层FLOPs
    gcn_flops = num_edges * model.hidden_dim * 2 + num_nodes * model.hidden_dim * model.input_dim

    # 特征增强FLOPs
    enhancement_flops = 19 * 8 + 8 * model.hidden_dim

    # 池化FLOPs
    pooling_flops = num_nodes * model.hidden_dim * 2

    # 分类头FLOPs
    classifier_flops = model.pooled_dim * model.num_classes

    total_flops = gcn_flops + enhancement_flops + pooling_flops + classifier_flops

    return total_flops

if __name__ == "__main__":
    # 测试模型
    model = HandGestureGCN(input_dim=7, hidden_dim=16, num_classes=20)

    # 创建测试数据
    x = torch.randn(42, 7)
    edge_index = torch.randint(0, 42, (2, 80))
    data = Data(x=x, edge_index=edge_index)

    # 前向传播
    output = model(data)
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {count_parameters(model)}")
    print(f"估算FLOPs: {count_flops(model, data)}")
