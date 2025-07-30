#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型脚本
用于快速验证模型是否能正常工作
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.tokenizer import BilingualTokenizer
from models.transformer import TransformerModel
from config.model_config import ModelConfig

def create_simple_test_data():
    """创建简单的测试数据"""
    test_data = [
        ("你好", "hello"),
        ("谢谢", "thank you"),
        ("再见", "goodbye"),
        ("早上好", "good morning"),
        ("晚安", "good night")
    ]
    return test_data

def test_model():
    """测试模型功能"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("开始测试模型...")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建测试数据
    test_data = create_simple_test_data()
    zh_texts = [item[0] for item in test_data]
    en_texts = [item[1] for item in test_data]
    
    # 创建分词器
    logger.info("创建分词器...")
    tokenizer = BilingualTokenizer()
    tokenizer.build_vocab(zh_texts, en_texts, min_freq=1, max_vocab_size=1000)
    
    logger.info(f"中文词汇表大小: {tokenizer.zh_vocab_size}")
    logger.info(f"英文词汇表大小: {tokenizer.en_vocab_size}")
    
    # 创建配置
    ModelConfig.update_config(
        vocab_size_zh=tokenizer.zh_vocab_size,
        vocab_size_en=tokenizer.en_vocab_size,
        d_model=128,  # 减小模型大小
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.1
    )
    config = ModelConfig
    
    # 创建模型
    logger.info("创建模型...")
    model = TransformerModel(
        src_vocab_size=tokenizer.zh_vocab_size,
        tgt_vocab_size=tokenizer.en_vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    logger.info(f"模型参数数: {model.get_trainable_parameters():,}")
    
    # 测试前向传播
    logger.info("测试前向传播...")
    batch_size = 2
    seq_len = 10
    
    # 创建虚拟输入
    src_input = torch.randint(1, tokenizer.zh_vocab_size, (batch_size, seq_len))
    tgt_input = torch.randint(1, tokenizer.en_vocab_size, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        output = model(src_input, tgt_input)
        logger.info(f"输出logits形状: {output['logits'].shape}")
    
    # 保存模型和分词器
    models_dir = project_root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("保存模型和分词器...")
    
    # 保存模型
    model_path = models_dir / "test_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.get_config_dict(),
        'zh_vocab_size': tokenizer.zh_vocab_size,
        'en_vocab_size': tokenizer.en_vocab_size
    }, model_path)
    
    # 保存分词器
    tokenizer_dir = models_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer.save(
        zh_path=str(tokenizer_dir / "zh_tokenizer.json"),
        en_path=str(tokenizer_dir / "en_tokenizer.json")
    )
    
    logger.info(f"模型已保存到: {model_path}")
    logger.info(f"分词器已保存到: {tokenizer_dir}")
    
    # 测试加载模型
    logger.info("测试加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 重新创建模型
    # 更新配置
    for key, value in checkpoint['config'].items():
        if hasattr(ModelConfig, key):
            setattr(ModelConfig, key, value)
    
    loaded_model = TransformerModel(
        src_vocab_size=checkpoint['zh_vocab_size'],
        tgt_vocab_size=checkpoint['en_vocab_size'],
        d_model=ModelConfig.d_model,
        n_heads=ModelConfig.n_heads,
        n_layers=ModelConfig.n_layers,
        d_ff=ModelConfig.d_ff,
        max_seq_len=ModelConfig.max_seq_len,
        dropout=ModelConfig.dropout
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("模型加载成功！")
    
    # 测试推理
    logger.info("测试推理...")
    loaded_model.eval()
    with torch.no_grad():
        test_output = loaded_model(src_input, tgt_input)
        logger.info(f"推理输出logits形状: {test_output['logits'].shape}")
    
    logger.info("模型测试完成！")
    return True

if __name__ == "__main__":
    try:
        test_model()
        print("\n✅ 模型测试成功！")
        print("📁 模型文件已保存到 data/models/ 目录")
        print("🔧 可以使用保存的模型进行测试和推理")
    except Exception as e:
        print(f"\n❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)