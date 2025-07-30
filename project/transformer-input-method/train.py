#!/usr/bin/env python3
"""主训练脚本"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    设置日志
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 配置日志
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    # 设置第三方库日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def load_config(config_path: str) -> TrainingConfig:
    """
    加载训练配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        训练配置对象
    """
    if not os.path.exists(config_path):
        logging.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return TrainingConfig()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 创建配置对象
        config = TrainingConfig()
        
        # 更新配置
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logging.warning(f"未知配置项: {key}")
        
        logging.info(f"成功加载配置文件: {config_path}")
        return config
        
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        logging.info("使用默认配置")
        return TrainingConfig()


def create_tokenizer(config: TrainingConfig) -> BilingualTokenizer:
    """
    创建分词器
    
    Args:
        config: 训练配置
        
    Returns:
        双语分词器
    """
    logging.info("创建分词器...")
    
    tokenizer = BilingualTokenizer(
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        vocab_size=config.vocab_size,
        min_freq=config.min_freq,
        max_length=config.max_seq_length
    )
    
    # 检查是否存在预训练的分词器
    tokenizer_path = os.path.join(config.model_save_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        logging.info(f"加载预训练分词器: {tokenizer_path}")
        tokenizer.load(tokenizer_path)
    else:
        logging.info("创建新的分词器")
        # 如果有训练数据，可以在这里构建词汇表
        if os.path.exists(config.train_data_path):
            logging.info("从训练数据构建词汇表...")
            # 这里可以添加从训练数据构建词汇表的逻辑
    
    logging.info(f"分词器创建完成，词汇表大小: {len(tokenizer.src_vocab)}")
    return tokenizer


def create_model(config: TrainingConfig, tokenizer: BilingualTokenizer) -> nn.Module:
    """
    创建模型
    
    Args:
        config: 训练配置
        tokenizer: 分词器
        
    Returns:
        Transformer 模型
    """
    logging.info("创建模型...")
    
    # 更新配置中的词汇表大小
    config.src_vocab_size = len(tokenizer.src_vocab)
    config.tgt_vocab_size = len(tokenizer.tgt_vocab)
    
    # 创建模型
    model = create_transformer_model(config)
    
    # 打印模型信息
    total_params = model.get_total_params()
    trainable_params = model.get_trainable_params()
    
    logging.info(f"模型创建完成:")
    logging.info(f"  总参数数: {total_params:,}")
    logging.info(f"  可训练参数数: {trainable_params:,}")
    logging.info(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model


def create_data_loaders_from_config(
    config: TrainingConfig,
    tokenizer: BilingualTokenizer
) -> tuple:
    """
    根据配置创建数据加载器
    
    Args:
        config: 训练配置
        tokenizer: 分词器
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    logging.info("创建数据加载器...")
    
    # 检查数据文件是否存在
    if not os.path.exists(config.train_data_path):
        logging.error(f"训练数据文件不存在: {config.train_data_path}")
        raise FileNotFoundError(f"训练数据文件不存在: {config.train_data_path}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data_path=config.train_data_path,
        valid_data_path=config.val_data_path,
        test_data_path=config.test_data_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_length,
        direction='zh2en',
        num_workers=config.num_workers
    )
    
    logging.info(f"数据加载器创建完成:")
    logging.info(f"  训练集大小: {len(train_loader.dataset) if train_loader else 0}")
    logging.info(f"  验证集大小: {len(val_loader.dataset) if val_loader else 0}")
    logging.info(f"  测试集大小: {len(test_loader.dataset) if test_loader else 0}")
    
    return train_loader, val_loader, test_loader


def create_training_components(
    model: nn.Module,
    config: TrainingConfig
) -> tuple:
    """
    创建训练组件
    
    Args:
        model: 模型
        config: 训练配置
        
    Returns:
        (optimizer, scheduler, criterion)
    """
    logging.info("创建训练组件...")
    
    # 创建优化器
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_type=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        **config.optimizer_params
    )
    
    # 创建学习率调度器
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.scheduler,
        **config.scheduler_params
    )
    
    # 创建损失函数
    criterion = create_loss_function(
        loss_type=config.loss_function,
        vocab_size=config.tgt_vocab_size,
        pad_idx=0,  # 假设 PAD token 的 ID 是 0
        **config.loss_params
    )
    
    logging.info(f"训练组件创建完成:")
    logging.info(f"  优化器: {config.optimizer}")
    logging.info(f"  学习率调度器: {config.scheduler}")
    logging.info(f"  损失函数: {config.loss_function}")
    
    return optimizer, scheduler, criterion


def main():
    """
    主训练函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Transformer 翻译模型训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.json",
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志文件路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    
    logging.info("开始训练 Transformer 翻译模型")
    logging.info(f"命令行参数: {vars(args)}")
    
    try:
        # 设置随机种子
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        
        # 设置设备
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"使用设备: {device}")
        
        # 加载配置
        config = load_config(args.config)
        config.device = device
        
        # 创建输出目录
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 保存配置
        config_save_path = os.path.join(config.model_save_dir, "training_config.json")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(vars(config), f, indent=2, ensure_ascii=False)
        
        # 创建分词器
        tokenizer = create_tokenizer(config)
        
        # 创建模型
        model = create_model(config, tokenizer)
        model.to(device)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders_from_config(
            config, tokenizer
        )
        
        # 创建训练组件
        optimizer, scheduler, criterion = create_training_components(model, config)
        
        # 创建评估指标
        metrics = TranslationMetrics()
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            metrics=metrics,
            config=config,
            device=device
        )
        
        # 恢复训练（如果指定）
        if args.resume:
            logging.info(f"恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        logging.info("开始训练...")
        trainer.train()
        
        # 评估最终模型
        if test_loader:
            logging.info("评估最终模型...")
            test_metrics = trainer.evaluate(test_loader)
            logging.info(f"测试集结果: {test_metrics}")
        
        # 保存最终模型
        final_model_path = os.path.join(config.model_save_dir, "final_model.pt")
        trainer.save_checkpoint(final_model_path, is_best=True)
        
        # 保存分词器
        tokenizer_save_path = os.path.join(config.model_save_dir, "tokenizer")
        tokenizer.save(tokenizer_save_path)
        
        logging.info("训练完成！")
        logging.info(f"模型保存路径: {final_model_path}")
        logging.info(f"分词器保存路径: {tokenizer_save_path}")
        
    except KeyboardInterrupt:
        logging.info("训练被用户中断")
        sys.exit(1)
    
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()