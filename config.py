"""
配置文件 - 手语识别系统参数设置
"""
import os
from pathlib import Path
import torch

# 数据集配置
DATA_ROOT = "data0"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# 模型配置
INPUT_DIM = 7  # 节点特征维度
HIDDEN_DIM = 16  # 隐藏层维度
OUTPUT_DIM = 20  # 类别数量（根据数据集自动调整）
NUM_LAYERS = 1  # GCN层数

# 训练配置
BATCH_SIZE = 64  # 增大批大小以适应更大的数据集
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.2
LABEL_SMOOTHING = 0.1
NUM_EPOCHS = 50  # 减少训练轮数（数据更多，收敛更快）
PATIENCE = 10  # 减少早停耐心值

# 数据预处理配置
IMAGE_SIZE = (224, 224)  # 输入图像尺寸
MAX_HANDS = 2  # 最大检测手数
CONFIDENCE_THRESHOLD = 0.5  # MediaPipe置信度阈值

# 路径配置
MODEL_SAVE_DIR = "models"
LOG_DIR = "logs"
RESULTS_DIR = "results"

# 创建必要的目录
for dir_path in [MODEL_SAVE_DIR, LOG_DIR, RESULTS_DIR]:
    Path(dir_path).mkdir(exist_ok=True)

# 设备配置
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True  # 启用CUDNN自动优化，提升性能
