# 导入路径处理库，用于跨平台路径操作
from pathlib import Path

# ========== 路径配置 ==========
# 项目根目录：当前文件的父目录的父目录
ROOT_DIR = Path(__file__).parent.parent

# 原始数据目录：存放未处理的原始文本数据
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

# 处理后数据目录：存放预处理后的训练数据（如JSONL格式）
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# 日志目录：存放TensorBoard训练日志
LOGS_DIR = ROOT_DIR / "logs"

# 模型目录：存放训练好的模型文件
MODELS_DIR = ROOT_DIR / "models"

# ========== 模型超参数 ==========
# 序列长度：用前5个字符预测第6个字符
SEQ_LEN = 8

# 批次大小：每次训练处理的样本数量
BATCH_SIZE = 64

# 词嵌入维度：将字符ID转换为128维向量
EMBEDDING_DIM = 256

# RNN隐藏层大小：循环神经网络的隐藏状态维度
HIDDEN_SIZE = 512

# ========== 训练超参数 ==========
# 学习率：控制参数更新的步长大小
LEARNING_RATE = 5e-4  # 0.0005

# 训练轮数：完整遍历训练集的次数
EPOCHS = 20
