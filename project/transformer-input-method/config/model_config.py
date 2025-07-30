"""模型配置文件"""

class ModelConfig:
    """Transformer 模型配置类"""
    
    # 模型架构参数
    d_model = 512          # 模型维度
    n_heads = 8            # 多头注意力头数
    n_layers = 6           # 编码器/解码器层数
    d_ff = 2048           # 前馈网络维度
    dropout = 0.1         # Dropout 概率
    
    # 序列参数
    max_seq_len = 512     # 最大序列长度
    pad_token_id = 0      # 填充标记 ID
    sos_token_id = 1      # 开始标记 ID
    eos_token_id = 2      # 结束标记 ID
    unk_token_id = 3      # 未知标记 ID
    
    # 词汇表参数
    vocab_size_zh = 30000  # 中文词汇表大小
    vocab_size_en = 30000  # 英文词汇表大小
    
    # 位置编码参数
    max_position_embeddings = 1024
    
    # 标签平滑
    label_smoothing = 0.1
    
    # 模型保存路径
    model_save_dir = "./checkpoints"
    best_model_path = "./checkpoints/best_model.pth"
    
    # 预训练模型路径（如果使用）
    pretrained_model_path = None
    
    # 设备配置
    device = "cuda"  # 或 "cpu"
    
    # 推理配置
    beam_size = 4         # 束搜索大小
    length_penalty = 0.6  # 长度惩罚
    max_decode_length = 256  # 最大解码长度
    
    @classmethod
    def get_config_dict(cls):
        """获取配置字典"""
        return {
            'd_model': cls.d_model,
            'n_heads': cls.n_heads,
            'n_layers': cls.n_layers,
            'd_ff': cls.d_ff,
            'dropout': cls.dropout,
            'max_seq_len': cls.max_seq_len,
            'vocab_size_zh': cls.vocab_size_zh,
            'vocab_size_en': cls.vocab_size_en,
            'pad_token_id': cls.pad_token_id,
            'sos_token_id': cls.sos_token_id,
            'eos_token_id': cls.eos_token_id,
            'unk_token_id': cls.unk_token_id,
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
        assert cls.d_model % cls.n_heads == 0, "d_model must be divisible by n_heads"
        assert cls.d_model > 0, "d_model must be positive"
        assert cls.n_heads > 0, "n_heads must be positive"
        assert cls.n_layers > 0, "n_layers must be positive"
        assert cls.d_ff > 0, "d_ff must be positive"
        assert 0 <= cls.dropout <= 1, "dropout must be between 0 and 1"
        assert cls.max_seq_len > 0, "max_seq_len must be positive"
        assert cls.vocab_size_zh > 0, "vocab_size_zh must be positive"
        assert cls.vocab_size_en > 0, "vocab_size_en must be positive"
        
        print("✅ 模型配置验证通过")
        return True


# 语言方向配置
class LanguageConfig:
    """语言配置类"""
    
    # 支持的语言方向
    SUPPORTED_DIRECTIONS = ['zh2en', 'en2zh']
    
    # 语言代码
    CHINESE = 'zh'
    ENGLISH = 'en'
    
    # 特殊标记
    SPECIAL_TOKENS = {
        'PAD': '<pad>',
        'SOS': '<sos>',
        'EOS': '<eos>',
        'UNK': '<unk>',
    }
    
    # 中文分词配置
    CHINESE_TOKENIZER = {
        'type': 'jieba',
        'cut_all': False,
        'HMM': True
    }
    
    # 英文分词配置
    ENGLISH_TOKENIZER = {
        'type': 'nltk',
        'lowercase': True,
        'remove_punctuation': False
    }


if __name__ == "__main__":
    # 验证配置
    ModelConfig.validate_config()
    
    # 打印配置信息
    print("模型配置:")
    for key, value in ModelConfig.get_config_dict().items():
        print(f"  {key}: {value}")