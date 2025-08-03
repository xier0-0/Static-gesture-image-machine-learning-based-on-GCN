# 轻量级双手势识别系统

基于图神经网络的手语识别系统，严格按照技术方案实现，支持静态RGB图像中的双手势识别。

## 项目特点

- **轻量级设计**: 参数 < 3K，推理FLOPs < 10K/样本
- **高精度**: 目标精度 > 90%
- **模块化架构**: 便于维护和扩展
- **科研标准**: 完整的训练、验证、测试流程

## 技术方案

### 核心架构
- **输入**: MediaPipe提取42个手部关键点
- **特征工程**: 7维节点特征（相对坐标、长度方差、手标记、最大夹角、速度预备）
- **图结构**: 生物连接边，无向图
- **模型**: 单层EdgeAwareGCN + 特征增强 + 混合池化 + 分类头

### 特征流
```
原始图像 → MediaPipe检测 → 42节点 → 节点特征(7维) → 边特征嵌入 → 
GCN层 → 特征增强 → 混合池化 → 分类输出
```

## 文件结构

```
gesture_GCN/
├── config.py              # 配置文件
├── data_preprocessing.py  # 数据预处理模块
├── models.py             # 模型架构
├── dataset.py            # 数据集处理
├── trainer.py            # 训练器
├── main.py               # 主程序
├── requirements.txt      # 依赖包
├── README.md            # 说明文档
├── data1/               # 数据集
├── models/              # 模型保存目录
├── results/             # 结果保存目录
└── logs/                # 日志目录
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 完整训练流程

```bash
python main.py --mode all
```

### 2. 分阶段执行

```bash
# 仅数据预处理
python main.py --mode preprocess

# 仅数据分析
python main.py --mode analyze

# 仅模型训练
python main.py --mode train
```

### 3. 自定义参数

```bash
python main.py --data_root data1 --batch_size 128 --epochs 100
```

## 数据集格式

数据集应按照以下结构组织：
```
data1/
├── 类别1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 类别2/
│   ├── image1.jpg
│   └── ...
└── ...
```

## 输出结果

训练完成后，系统会生成：

1. **模型文件**: `models/best_model.pth`
2. **训练图表**: `results/training_history.png`
3. **混淆矩阵**: `results/confusion_matrix.png`
4. **测试结果**: `results/test_results.json`

## 模型性能

- **参数数量**: ~2.5K
- **推理FLOPs**: < 10K/样本
- **目标精度**: > 90%
- **训练时间**: < 50 epochs

## 技术细节

### 节点特征 (7维)
1. 相对坐标 (3维): 相对于手掌中心的归一化坐标
2. 长度方差 (1维): 相邻边长度的方差
3. 手标记 (1维): 左手=0，右手=1
4. 最大夹角 (1维): 关节最大弯曲角度
5. 速度预备 (1维): 静态图像设为0

### 训练策略
- **损失函数**: 交叉熵 + 标签平滑(ε=0.1)
- **优化器**: AdamW(lr=1e-3)
- **学习率调度**: CosineAnnealingLR
- **正则化**: Dropout(0.2) + 权重衰减(1e-5)
- **早停**: 基于验证精度，patience=10

## 扩展性

- 支持视频序列扩展（预留速度维度）
- 可扩展更多GCN层
- 支持不同数据集格式
- 支持模型量化部署

## 故障排除

### 常见问题

1. **MediaPipe安装失败**
   ```bash
   pip install mediapipe --upgrade
   ```

2. **CUDA内存不足**
   - 减小batch_size
   - 使用CPU训练

3. **数据预处理失败**
   - 检查图像格式
   - 确保MediaPipe能检测到手部

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系开发者。 