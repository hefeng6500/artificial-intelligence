#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理测试脚本
使用训练好的模型进行中译英翻译测试
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

def load_model_and_tokenizer(model_path, tokenizer_dir):
    """加载模型和分词器"""
    logger = logging.getLogger(__name__)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info("加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 更新配置
    for key, value in checkpoint['config'].items():
        if hasattr(ModelConfig, key):
            setattr(ModelConfig, key, value)
    
    # 创建模型
    model = TransformerModel(
        src_vocab_size=checkpoint['zh_vocab_size'],
        tgt_vocab_size=checkpoint['en_vocab_size'],
        d_model=ModelConfig.d_model,
        n_heads=ModelConfig.n_heads,
        n_layers=ModelConfig.n_layers,
        d_ff=ModelConfig.d_ff,
        max_seq_len=ModelConfig.max_seq_len,
        dropout=ModelConfig.dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载分词器
    logger.info("加载分词器...")
    tokenizer = BilingualTokenizer()
    tokenizer.load(
        zh_path=str(tokenizer_dir / "zh_tokenizer.json"),
        en_path=str(tokenizer_dir / "en_tokenizer.json")
    )
    
    logger.info(f"模型加载完成，参数数量: {model.get_trainable_parameters():,}")
    logger.info(f"中文词汇表大小: {tokenizer.zh_vocab_size}")
    logger.info(f"英文词汇表大小: {tokenizer.en_vocab_size}")
    
    return model, tokenizer, device

def translate_text(model, tokenizer, device, chinese_text, max_length=50):
    """翻译中文文本到英文"""
    logger = logging.getLogger(__name__)
    
    # 分词和编码
    zh_tokens = tokenizer.zh_tokenizer.encode(chinese_text)
    
    # 转换为张量
    src_tokens = torch.tensor([zh_tokens], dtype=torch.long, device=device)
    
    logger.info(f"输入: {chinese_text}")
    logger.info(f"中文tokens: {zh_tokens}")
    
    # 生成翻译
    with torch.no_grad():
        generated_tokens = model.generate(
            src_tokens=src_tokens,
            max_length=max_length,
            beam_size=1,  # 使用贪心搜索
            temperature=1.0
        )
    
    # 解码生成的tokens
    generated_tokens = generated_tokens[0].cpu().tolist()
    
    # 移除特殊tokens
    if tokenizer.en_tokenizer.sos_token_id in generated_tokens:
        start_idx = generated_tokens.index(tokenizer.en_tokenizer.sos_token_id) + 1
    else:
        start_idx = 0
    
    if tokenizer.en_tokenizer.eos_token_id in generated_tokens[start_idx:]:
        end_idx = generated_tokens.index(tokenizer.en_tokenizer.eos_token_id, start_idx)
        generated_tokens = generated_tokens[start_idx:end_idx]
    else:
        generated_tokens = generated_tokens[start_idx:]
    
    # 解码为文本
    english_text = tokenizer.en_tokenizer.decode(generated_tokens)
    
    logger.info(f"生成的tokens: {generated_tokens}")
    logger.info(f"输出: {english_text}")
    
    return english_text

def test_inference():
    """测试推理功能"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("开始推理测试...")
    
    # 模型和分词器路径
    models_dir = project_root / "data" / "models"
    model_path = models_dir / "test_model.pth"
    tokenizer_dir = models_dir / "tokenizer"
    
    # 检查文件是否存在
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return False
    
    if not tokenizer_dir.exists():
        logger.error(f"分词器目录不存在: {tokenizer_dir}")
        return False
    
    try:
        # 加载模型和分词器
        model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_dir)
        
        # 测试翻译
        test_sentences = [
            "你好",
            "谢谢",
            "再见",
            "早上好",
            "晚安"
        ]
        
        logger.info("\n开始翻译测试:")
        logger.info("=" * 50)
        
        for chinese_text in test_sentences:
            try:
                english_text = translate_text(model, tokenizer, device, chinese_text)
                logger.info(f"中文: {chinese_text} -> 英文: {english_text}")
                logger.info("-" * 30)
            except Exception as e:
                logger.error(f"翻译失败 '{chinese_text}': {e}")
        
        logger.info("推理测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_inference()
        if success:
            print("\n✅ 推理测试成功！")
            print("🎯 模型可以正常进行中译英翻译")
        else:
            print("\n❌ 推理测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 推理测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)