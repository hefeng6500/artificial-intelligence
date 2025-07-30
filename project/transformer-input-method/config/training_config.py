"""训练配置文件"""

import os
from datetime import datetime

class TrainingConfig:
    """训练配置类"""
    
    # 基础训练参数
    batch_size = 32           # 批次大小
    learning_rate = 1e-4      # 学习率
    num_epochs = 10           # 训练轮数（减少用于测试）
    warmup_steps = 4000       # 预热步数
    max_seq_length = 256      # 最大序列长度
    
    # 语言配置
    src_lang = 'zh'           # 源语言
    tgt_lang = 'en'           # 目标语言
    
    # 词汇表配置
    vocab_size = 30000        # 词汇表大小
    min_freq = 2              # 最小词频
    
    # 模型配置
    src_vocab_size = 30000    # 源语言词汇表大小
    tgt_vocab_size = 30000    # 目标语言词汇表大小
    
    # 数据分割
    val_split = 0.1           # 验证集比例
    test_split = 0.1          # 测试集比例
    
    # 损失函数配置
    loss_function = 'cross_entropy'  # 损失函数类型
    loss_params = {}          # 损失函数参数
    
    # 优化器参数字典
    optimizer_params = {}     # 优化器额外参数
    scheduler_params = {}     # 调度器额外参数
    
    # 优化器参数
    optimizer = 'adam'        # 优化器类型
    beta1 = 0.9              # Adam beta1
    beta2 = 0.98             # Adam beta2
    epsilon = 1e-9           # Adam epsilon
    weight_decay = 0.0       # 权重衰减
    
    # 学习率调度
    scheduler = 'transformer'  # 学习率调度器类型
    lr_factor = 1.0           # 学习率因子
    lr_patience = 10          # 学习率衰减耐心值
    lr_min = 1e-6            # 最小学习率
    
    # 梯度裁剪
    grad_clip_norm = 1.0     # 梯度裁剪范数
    
    # 验证和保存
    eval_steps = 1000        # 验证步数间隔
    save_steps = 2000        # 保存步数间隔
    logging_steps = 100      # 日志记录步数间隔
    
    # 早停
    early_stopping = True    # 是否启用早停
    patience = 10            # 早停耐心值
    min_delta = 1e-4         # 最小改进阈值
    
    # 数据加载
    num_workers = 4          # 数据加载器工作进程数
    pin_memory = True        # 是否固定内存
    
    # 混合精度训练
    use_amp = True           # 是否使用自动混合精度
    
    # 数据路径
    data_dir = "./data/processed"      # 数据目录
    train_data_path = "./data/processed/train.json"  # 训练文件路径
    val_data_path = "./data/processed/val.json"      # 验证文件路径
    test_data_path = "./data/processed/test.json"    # 测试文件路径
    
    # 模型保存路径
    model_save_dir = "./data/models"  # 模型保存目录
    output_dir = "./outputs"  # 输出目录
    checkpoint_dir = "./checkpoints"  # 检查点目录
    log_dir = "./logs"       # 日志目录
    
    # TensorBoard
    use_tensorboard = True   # 是否使用 TensorBoard
    tensorboard_dir = "./runs"  # TensorBoard 日志目录
    
    # Weights & Biases
    use_wandb = False        # 是否使用 W&B
    wandb_project = "transformer-translation"  # W&B 项目名
    wandb_entity = None      # W&B 实体名
    
    # 实验配置
    experiment_name = f"transformer_translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    seed = 42                # 随机种子
    
    # 数据预处理
    max_train_samples = None  # 最大训练样本数（None 表示使用全部）
    max_valid_samples = None  # 最大验证样本数
    
    # 模型初始化
    init_method = 'xavier_uniform'  # 参数初始化方法
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        dirs = [
            cls.output_dir,
            cls.checkpoint_dir,
            cls.log_dir,
            cls.tensorboard_dir
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"✅ 创建目录完成: {dirs}")
    
    @classmethod
    def get_config_dict(cls):
        """获取配置字典"""
        return {
            'batch_size': cls.batch_size,
            'learning_rate': cls.learning_rate,
            'num_epochs': cls.num_epochs,
            'warmup_steps': cls.warmup_steps,
            'optimizer': cls.optimizer,
            'scheduler': cls.scheduler,
            'grad_clip_norm': cls.grad_clip_norm,
            'early_stopping': cls.early_stopping,
            'patience': cls.patience,
            'use_amp': cls.use_amp,
            'seed': cls.seed,
        }
    
    @classmethod
    def update_config(cls, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
    
    @classmethod
    def validate_config(cls):
        """验证配置参数"""
        assert cls.batch_size > 0, "batch_size must be positive"
        assert cls.learning_rate > 0, "learning_rate must be positive"
        assert cls.num_epochs > 0, "num_epochs must be positive"
        assert cls.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert cls.grad_clip_norm > 0, "grad_clip_norm must be positive"
        assert cls.patience > 0, "patience must be positive"
        assert cls.eval_steps > 0, "eval_steps must be positive"
        assert cls.save_steps > 0, "save_steps must be positive"
        assert cls.logging_steps > 0, "logging_steps must be positive"
        
        print("✅ 训练配置验证通过")
        return True


class DataConfig:
    """数据配置类"""
    
    # 数据集配置
    dataset_name = "custom"   # 数据集名称
    
    # 数据预处理
    min_length = 1           # 最小句子长度
    max_length = 256         # 最大句子长度
    
    # 数据增强
    use_data_augmentation = False  # 是否使用数据增强
    augmentation_prob = 0.1        # 数据增强概率
    
    # 词汇表配置
    vocab_min_freq = 2       # 词汇最小频率
    vocab_max_size = 30000   # 词汇表最大大小
    
    # 数据分割
    train_ratio = 0.8        # 训练集比例
    valid_ratio = 0.1        # 验证集比例
    test_ratio = 0.1         # 测试集比例
    
    # 缓存配置
    use_cache = True         # 是否使用缓存
    cache_dir = "./cache"    # 缓存目录
    
    @classmethod
    def validate_config(cls):
        """验证数据配置"""
        assert cls.min_length > 0, "min_length must be positive"
        assert cls.max_length > cls.min_length, "max_length must be greater than min_length"
        assert cls.vocab_min_freq > 0, "vocab_min_freq must be positive"
        assert cls.vocab_max_size > 0, "vocab_max_size must be positive"
        assert abs(cls.train_ratio + cls.valid_ratio + cls.test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        print("✅ 数据配置验证通过")
        return True


if __name__ == "__main__":
    # 验证配置
    TrainingConfig.validate_config()
    DataConfig.validate_config()
    
    # 创建目录
    TrainingConfig.create_dirs()
    
    # 打印配置信息
    print("\n训练配置:")
    for key, value in TrainingConfig.get_config_dict().items():
        print(f"  {key}: {value}")