"""
数据预处理模块 - MediaPipe关键点提取和特征工程
严格按照技术方案实现节点特征提取和边特征提取
"""
import cv2
import mediapipe as mp
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Dict
import os
from pathlib import Path
from tqdm import tqdm
import pickle

class HandKeypointExtractor:
    """手部关键点提取器"""
    
    def __init__(self, max_hands=2, confidence_threshold=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_hands,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
    def extract_keypoints(self, image_path: str) -> Optional[np.ndarray]:
        """
        从图像中提取手部关键点
        返回: 42个关键点的3D坐标 (42, 3) 或 None
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测手部
            results = self.hands.process(image_rgb)
            
            if not results.multi_hand_landmarks:
                return None
                
            # 初始化关键点数组 (42个点，左右手各21个)
            keypoints = np.zeros((42, 3))
            
            # 处理检测到的手
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= 2:  # 最多处理两只手
                    break
                    
                # 确定是左手还是右手
                hand_type = results.multi_handedness[hand_idx].classification[0].label
                start_idx = 0 if hand_type == "Left" else 21
                
                # 提取21个关键点
                for i, landmark in enumerate(hand_landmarks.landmark):
                    keypoints[start_idx + i] = [landmark.x, landmark.y, landmark.z]
            
            return keypoints
            
        except Exception as e:
            print(f"提取关键点失败 {image_path}: {e}")
            return None
    
    def __del__(self):
        self.hands.close()

class FeatureEngineer:
    """特征工程模块 - 严格按照技术方案实现"""
    
    def __init__(self):
        # 定义手部骨骼连接（MediaPipe 21点连接）
        self.hand_connections = [
            # 拇指
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 食指
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 中指
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 无名指
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 小指
            (0, 17), (17, 18), (18, 19), (19, 20),
            # 手掌连接
            (5, 9), (9, 13), (13, 17)
        ]
        
        # 构建完整的42点连接（左右手）
        self.full_connections = []
        # 左手连接
        for i, j in self.hand_connections:
            self.full_connections.append((i, j))
        # 右手连接
        for i, j in self.hand_connections:
            self.full_connections.append((i + 21, j + 21))
    
    def compute_baseline_lengths(self, keypoints: np.ndarray) -> Tuple[float, float]:
        """
        计算基准长度
        左手基准：||P0 - P5||（手腕到食指根）
        右手基准：||P21 - P26||
        """
        # 左手基准
        left_baseline = np.linalg.norm(keypoints[0] - keypoints[5])
        
        # 右手基准
        right_baseline = np.linalg.norm(keypoints[21] - keypoints[26])
        
        # 动态调整：如果单手检测，使用默认值
        if left_baseline < 1e-6:
            left_baseline = 0.1
        if right_baseline < 1e-6:
            right_baseline = 0.1
            
        return left_baseline, right_baseline
    
    def compute_node_features(self, keypoints: np.ndarray, 
                            left_baseline: float, 
                            right_baseline: float) -> np.ndarray:
        """
        计算节点特征 - 严格按照技术方案实现
        返回: (42, 7) 特征矩阵
        """
        features = np.zeros((42, 7))
        baselines = [left_baseline] * 21 + [right_baseline] * 21
        
        # 计算手掌中心
        palm_left = np.mean(keypoints[[0, 5, 17]], axis=0)
        palm_right = np.mean(keypoints[[21, 26, 38]], axis=0)
        palms = [palm_left] * 21 + [palm_right] * 21
        
        # 构建邻接关系
        neighbors = [[] for _ in range(42)]
        for i, j in self.full_connections:
            neighbors[i].append(j)
            neighbors[j].append(i)
        
        for i in range(42):
            # 1. 相对坐标 (3维)
            rel_pos = (keypoints[i] - palms[i]) / baselines[i]
            features[i, :3] = rel_pos
            
            # 2. 相邻边长度方差 (1维)
            if len(neighbors[i]) > 0:
                dists = []
                for j in neighbors[i]:
                    dist = np.linalg.norm(keypoints[j] - keypoints[i]) / baselines[i]
                    dists.append(dist)
                features[i, 3] = np.var(dists) if len(dists) > 1 else 0
            else:
                features[i, 3] = 0
            
            # 3. 手标记 (1维)
            features[i, 4] = 0 if i < 21 else 1
            
            # 4. 最大夹角 (1维)
            if len(neighbors[i]) >= 2:
                vecs = []
                for j in neighbors[i]:
                    vec = keypoints[j] - keypoints[i]
                    vecs.append(vec)
                
                angles = []
                for j in range(len(vecs)):
                    for k in range(j + 1, len(vecs)):
                        cos_angle = np.dot(vecs[j], vecs[k]) / (np.linalg.norm(vecs[j]) * np.linalg.norm(vecs[k]) + 1e-8)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        angles.append(angle)
                
                features[i, 5] = max(angles) / np.pi if angles else 0
            else:
                features[i, 5] = 0
            
            # 5. 速度预备 (1维，静态图像设为0)
            features[i, 6] = 0
        
        return features
    
    def compute_edge_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        计算边特征 - 方向向量
        返回: (num_edges, 3) 方向向量
        """
        edge_features = []
        
        for i, j in self.full_connections:
            # 计算方向向量
            direction = keypoints[j] - keypoints[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            else:
                direction = np.array([0, 0, 0])
            
            edge_features.append(direction)
        
        return np.array(edge_features)
    
    def embed_edge_features_to_nodes(self, node_features: np.ndarray, 
                                   edge_features: np.ndarray) -> np.ndarray:
        """
        将边特征嵌入到节点特征中
        """
        # 计算每个节点的入边方向平均值
        edge_to_node = [[] for _ in range(42)]
        
        for edge_idx, (i, j) in enumerate(self.full_connections):
            edge_to_node[j].append(edge_features[edge_idx])
        
        # 将平均入边方向加到节点特征
        for i in range(42):
            if edge_to_node[i]:
                avg_direction = np.mean(edge_to_node[i], axis=0)
                node_features[i, :3] += avg_direction * 0.1  # 小权重融合
        
        return node_features
    
    def build_graph_data(self, keypoints: np.ndarray) -> Data:
        """
        构建图数据对象
        """
        # 计算基准长度
        left_baseline, right_baseline = self.compute_baseline_lengths(keypoints)
        
        # 计算节点特征
        node_features = self.compute_node_features(keypoints, left_baseline, right_baseline)
        
        # 计算边特征
        edge_features = self.compute_edge_features(keypoints)
        
        # 嵌入边特征到节点
        node_features = self.embed_edge_features_to_nodes(node_features, edge_features)
        
        # 构建边索引（无向图）
        edge_index = []
        for i, j in self.full_connections:
            edge_index.append([i, j])
            edge_index.append([j, i])  # 无向图
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 转换为PyTorch张量
        x = torch.tensor(node_features, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index)

class DatasetProcessor:
    """数据集处理器"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.extractor = HandKeypointExtractor()
        self.feature_engineer = FeatureEngineer()
        
    def process_dataset(self, save_path: str = "processed_data.pkl"):
        """
        处理整个数据集
        """
        print("开始处理数据集...")
        
        # 获取所有类别
        classes = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        print(f"发现 {len(classes)} 个类别: {classes}")
        
        # 处理每个类别的图像
        all_data = []
        failed_count = 0
        
        for class_name in tqdm(classes, desc="处理类别"):
            class_dir = self.data_root / class_name
            class_idx = class_to_idx[class_name]
            
            # 获取该类别下的所有图像
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
            
            for image_path in image_files:
                # 提取关键点
                keypoints = self.extractor.extract_keypoints(str(image_path))
                
                if keypoints is not None:
                    # 构建图数据
                    try:
                        graph_data = self.feature_engineer.build_graph_data(keypoints)
                        graph_data.y = torch.tensor([class_idx], dtype=torch.long)
                        graph_data.image_path = str(image_path)
                        graph_data.class_name = class_name
                        
                        all_data.append(graph_data)
                    except Exception as e:
                        print(f"构建图数据失败 {image_path}: {e}")
                        failed_count += 1
                else:
                    failed_count += 1
        
        print(f"处理完成: 成功 {len(all_data)} 个样本, 失败 {failed_count} 个样本")
        
        # 保存处理后的数据
        processed_data = {
            'data': all_data,
            'classes': classes,
            'class_to_idx': class_to_idx
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"数据已保存到 {save_path}")
        return processed_data

if __name__ == "__main__":
    # 测试数据预处理
    processor = DatasetProcessor("data1")
    processed_data = processor.process_dataset()
