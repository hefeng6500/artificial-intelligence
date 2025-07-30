#!/usr/bin/env python3
"""使用CSV数据集训练模型的简化脚本"""

import os
import sys
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from config.training_config import TrainingConfig
from models.transformer import create_transformer_model
from utils.tokenizer import BilingualTokenizer
from utils.data_loader import TranslationDataLoader, create_data_loaders
from utils.metrics import TranslationMetrics
from training.trainer import Trainer
from training.optimizer import create_optimizer, create_scheduler
from training.loss import create_loss_function

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log', encoding='utf-8')
        ]
    )

def create_tokenizer_from_data(config: TrainingConfig) -> BilingualTokenizer:
    """从训练数据创建分词器"""
    logging.info("创建分词器...")
    
    # 读取训练数据来构建词汇表
    with open(config.train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 提取中英文文本
    zh_texts = [item['zh'] for item in train_data]
    en_texts = [item['en'] for item in train_data]
    
    # 创建分词器
    tokenizer = BilingualTokenizer()
    
    # 从数据构建词汇表
    logging.info("从训练数据构建词汇表...")
    tokenizer.build_vocab(
        zh_texts=zh_texts,
        en_texts=en_texts,
        min_freq=config.min_freq,
        max_vocab_size=config.vocab_size
    )
    
    logging.info(f"分词器创建完成:")
    logging.info(f"  中文词汇表大小: {tokenizer.zh_vocab_size}")
    logging.info(f"  英文词汇表大小: {tokenizer.en_vocab_size}")
    
    return tokenizer

def main():
    """主训练函数"""
    # 设置日志
    setup_logging()
    logging.info("开始训练 Transformer 翻译模型")
    
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用设备: {device}")
        
        # 创建配置
        config = TrainingConfig()
        
        # 创建必要的目录
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 检查数据文件是否存在
        if not os.path.exists(config.train_data_path):
            logging.error(f"训练数据文件不存在: {config.train_data_path}")
            logging.info("请先运行 prepare_data.py 来准备数据")
            return
        
        # 创建分词器
        tokenizer = create_tokenizer_from_data(config)
        
        # 更新配置中的词汇表大小
        config.src_vocab_size = tokenizer.zh_vocab_size
        config.tgt_vocab_size = tokenizer.en_vocab_size
        
        # 创建模型
        logging.info("创建模型...")
        model = create_transformer_model(config)
        model.to(device)
        
        # 打印模型信息
        total_params = model.get_trainable_parameters()
        logging.info(f"模型参数数: {total_params:,}")
        
        # 创建数据加载器
        logging.info("创建数据加载器...")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data_path=config.train_data_path,
            valid_data_path=config.val_data_path,
            test_data_path=config.test_data_path,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_length=config.max_seq_length,
            direction='zh2en',
            num_workers=0  # Windows上设为0避免多进程问题
        )
        
        logging.info(f"数据加载器创建完成:")
        logging.info(f"  训练集大小: {len(train_loader.dataset)}")
        logging.info(f"  验证集大小: {len(val_loader.dataset)}")
        logging.info(f"  测试集大小: {len(test_loader.dataset)}")
        
        # 创建优化器
        optimizer = create_optimizer(
            model,
            optimizer_type=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 创建学习率调度器
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=config.scheduler
        )
        
        # 创建损失函数
        pad_token_id = tokenizer.en_tokenizer.pad_token_id or 0
        criterion = create_loss_function(
            loss_type=config.loss_function,
            pad_token_id=pad_token_id
        )
        
        # 创建评估指标
        metrics = TranslationMetrics()
        
        # 创建训练器
        logging.info("创建训练器...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=criterion,  # 注意这里是loss_fn而不是criterion
            scheduler=scheduler,
            config=config.get_config_dict(),  # 传递配置字典
            device=device
        )
        
        # 开始训练
        logging.info("开始训练...")
        trainer.train()
        
        # 评估最终模型
        if test_loader:
            logging.info("评估最终模型...")
            test_metrics = trainer.validate_epoch(test_loader)
            logging.info(f"测试集结果: {test_metrics}")
        
        # 保存最终模型
        final_model_path = os.path.join(config.model_save_dir, "final_model.pt")
        trainer.save_checkpoint(final_model_path)
        
        # 保存分词器
        tokenizer_save_path = os.path.join(config.model_save_dir, "tokenizer")
        os.makedirs(tokenizer_save_path, exist_ok=True)
        zh_tokenizer_path = os.path.join(tokenizer_save_path, "zh_tokenizer.json")
        en_tokenizer_path = os.path.join(tokenizer_save_path, "en_tokenizer.json")
        tokenizer.save(zh_tokenizer_path, en_tokenizer_path)
        
        # 保存配置
        config_save_path = os.path.join(config.model_save_dir, "config.json")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'src_vocab_size': config.src_vocab_size,
                'tgt_vocab_size': config.tgt_vocab_size,
                'max_seq_length': config.max_seq_length,
                'src_lang': config.src_lang,
                'tgt_lang': config.tgt_lang
            }, f, indent=2, ensure_ascii=False)
        
        logging.info("训练完成！")
        logging.info(f"模型保存路径: {final_model_path}")
        logging.info(f"分词器保存路径: {tokenizer_save_path}")
        logging.info(f"配置保存路径: {config_save_path}")
        
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()