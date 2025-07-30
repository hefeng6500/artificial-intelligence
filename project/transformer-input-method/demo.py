#!/usr/bin/env python3
"""Transformer 翻译模型演示脚本"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from config.training_config import TrainingConfig
from models.transformer import TransformerModel, create_transformer_model
from utils.tokenizer import BilingualTokenizer
from utils.data_loader import create_data_loaders
from utils.metrics import TranslationMetrics
from utils.visualization import AttentionVisualizer, TrainingVisualizer
from inference.translator import Translator, TranslationConfig
from inference.beam_search import BeamSearch, BeamSearchConfig
from training.trainer import Trainer
from training.optimizer import create_optimizer, create_scheduler
from training.loss import create_loss_function

# 设置中文字体
rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def setup_logging():
    """
    设置日志
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_demo_data() -> Dict[str, List[Dict[str, str]]]:
    """
    创建演示数据

    Returns:
        演示数据字典
    """
    # 中英文翻译数据
    translation_data = [
        {"zh": "你好，世界！", "en": "Hello, world!"},
        {
            "zh": "我爱学习人工智能。",
            "en": "I love learning artificial intelligence.",
        },
        {"zh": "今天天气很好。", "en": "The weather is very nice today."},
        {
            "zh": "机器学习是一门有趣的学科。",
            "en": "Machine learning is an interesting subject.",
        },
        {
            "zh": "深度学习改变了世界。",
            "en": "Deep learning has changed the world.",
        },
        {
            "zh": "Transformer 是一个强大的模型。",
            "en": "Transformer is a powerful model.",
        },
        {
            "zh": "自然语言处理很有挑战性。",
            "en": "Natural language processing is challenging.",
        },
        {
            "zh": "我们正在构建一个翻译系统。",
            "en": "We are building a translation system.",
        },
        {
            "zh": "注意力机制是关键技术。",
            "en": "Attention mechanism is a key technology.",
        },
        {
            "zh": "这个项目非常有意义。",
            "en": "This project is very meaningful.",
        },
    ]

    return {
        "train": translation_data[:8],
        "val": translation_data[8:9],
        "test": translation_data[9:10],
    }


def save_demo_data(data: Dict[str, List[Dict[str, str]]], data_dir: str):
    """
    保存演示数据

    Args:
        data: 演示数据
        data_dir: 数据目录
    """
    os.makedirs(data_dir, exist_ok=True)

    for split, examples in data.items():
        file_path = os.path.join(data_dir, f"{split}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

        logging.info(f"保存 {split} 数据到: {file_path} ({len(examples)} 条)")


def create_demo_config() -> TrainingConfig:
    """
    创建演示配置

    Returns:
        训练配置
    """
    config = TrainingConfig()

    # 模型配置（小模型用于演示）
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 2
    config.d_ff = 256
    config.max_seq_length = 20  # 减小序列长度以匹配演示数据
    config.dropout = 0.1

    # 训练配置
    config.batch_size = 2
    config.learning_rate = 1e-3
    config.num_epochs = 10
    config.warmup_steps = 100
    config.max_grad_norm = 1.0

    # 数据配置
    config.vocab_size = 1000
    config.min_freq = 1
    config.src_lang = "zh"
    config.tgt_lang = "en"

    # 路径配置
    config.train_data_path = "data/demo/train.json"
    config.val_data_path = "data/demo/val.json"
    config.test_data_path = "data/demo/test.json"
    config.model_save_dir = "checkpoints/demo"
    config.log_dir = "logs/demo"

    # 验证和保存
    config.eval_steps = 5
    config.save_steps = 10
    config.logging_steps = 2

    return config


def demo_tokenizer():
    """
    演示分词器功能
    """
    print("\n" + "=" * 50)
    print("分词器演示")
    print("=" * 50)

    # 创建分词器
    tokenizer = BilingualTokenizer()

    # 示例文本
    zh_texts = ["你好世界", "我爱学习", "今天天气很好"]
    en_texts = ["Hello world", "I love learning", "The weather is nice today"]

    # 构建词汇表
    print("构建词汇表...")
    tokenizer.build_vocab(zh_texts, en_texts)

    print(f"中文词汇表大小: {tokenizer.zh_vocab_size}")
    print(f"英文词汇表大小: {tokenizer.en_vocab_size}")

    # 编码解码示例
    test_zh = "你好，世界！"
    test_en = "Hello, world!"

    print(f"\n原始中文: {test_zh}")
    zh_tokens = tokenizer.zh_tokenizer.encode(test_zh)
    print(f"编码结果: {zh_tokens}")
    zh_decoded = tokenizer.zh_tokenizer.decode(zh_tokens)
    print(f"解码结果: {zh_decoded}")

    print(f"\n原始英文: {test_en}")
    en_tokens = tokenizer.en_tokenizer.encode(test_en)
    print(f"编码结果: {en_tokens}")
    en_decoded = tokenizer.en_tokenizer.decode(en_tokens)
    print(f"解码结果: {en_decoded}")

    return tokenizer


def demo_model(config: TrainingConfig, tokenizer: BilingualTokenizer):
    """
    演示模型功能

    Args:
        config: 训练配置
        tokenizer: 分词器
    """
    print("\n" + "=" * 50)
    print("模型演示")
    print("=" * 50)

    # 更新配置
    config.src_vocab_size = max(tokenizer.zh_vocab_size, 100)  # 确保词汇表大小足够
    config.tgt_vocab_size = max(tokenizer.en_vocab_size, 100)
    # 调整模型维度以适应小词汇表
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 2
    config.d_ff = 512

    # 创建模型
    model = create_transformer_model(config)

    print(f"模型参数:")
    print(f"  总参数数: {model.get_model_size():,}")
    print(f"  可训练参数数: {model.get_trainable_parameters():,}")
    print(f"  模型大小: {model.get_model_size() * 4 / 1024 / 1024:.2f} MB")

    # 测试前向传播
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 创建测试输入
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    src_input = torch.randint(0, config.src_vocab_size, (batch_size, src_seq_len)).to(
        device
    )
    tgt_input = torch.randint(0, config.tgt_vocab_size, (batch_size, tgt_seq_len)).to(
        device
    )

    print(f"\n测试输入:")
    print(f"  源序列形状: {src_input.shape}")
    print(f"  目标序列形状: {tgt_input.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(src_input, tgt_input)

    print(f"\n输出形状: {output['logits'].shape}")
    print(
        f"输出范围: [{output['logits'].min().item():.4f}, {output['logits'].max().item():.4f}]"
    )

    return model


def demo_training(config: TrainingConfig, tokenizer: BilingualTokenizer):
    """
    演示训练过程

    Args:
        config: 训练配置
        tokenizer: 分词器
    """
    print("\n" + "=" * 50)
    print("训练演示")
    print("=" * 50)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    config.src_vocab_size = max(tokenizer.zh_vocab_size, 100)
    config.tgt_vocab_size = max(tokenizer.en_vocab_size, 100)
    # 调整模型维度以适应小词汇表
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 2
    config.d_ff = 512
    model = create_transformer_model(config)
    model.to(device)

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data_path=config.train_data_path,
        valid_data_path=config.val_data_path,
        test_data_path=config.test_data_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_length,
        direction="zh2en",
        num_workers=0,
    )

    print(f"数据加载器:")
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(val_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")

    # 创建训练组件
    optimizer = create_optimizer(
        model,
        optimizer_type=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.scheduler,
        warmup_steps=config.warmup_steps,
        total_steps=len(train_loader) * config.num_epochs,
    )

    criterion = create_loss_function(
        loss_type="cross_entropy", ignore_index=tokenizer.en_tokenizer.pad_token_id
    )

    metrics = TranslationMetrics()

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        scheduler=scheduler,
        config=config.get_config_dict(),
        device=device,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 评估模型
    if test_loader:
        print("\n评估测试集...")
        # 临时设置测试数据加载器为验证数据加载器
        original_val_loader = trainer.val_loader
        trainer.val_loader = test_loader
        test_metrics = trainer.validate_epoch()
        trainer.val_loader = original_val_loader
        print(f"测试集结果: {test_metrics}")

    return model, trainer


def demo_translation(model: nn.Module, tokenizer: BilingualTokenizer):
    """
    演示翻译功能

    Args:
        model: 训练好的模型
        tokenizer: 分词器
    """
    print("\n" + "=" * 50)
    print("翻译演示")
    print("=" * 50)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 创建翻译器
    translator = Translator(model, tokenizer, tokenizer, device)

    # 测试文本
    test_texts = [
        "你好，世界！",
        "我爱学习人工智能。",
        "今天天气很好。",
        "机器学习很有趣。",
    ]

    # 不同的翻译配置
    configs = {
        "贪心搜索": TranslationConfig(search_strategy="greedy", max_length=50),
        "束搜索": TranslationConfig(
            search_strategy="beam_search",
            beam_size=3,
            max_length=50,
            num_return_sequences=2,
            return_scores=True,
        ),
        "采样": TranslationConfig(
            search_strategy="sampling",
            temperature=0.8,
            top_k=10,
            max_length=50,
            num_return_sequences=2,
        ),
    }

    for text in test_texts:
        print(f"\n原文: {text}")

        for method_name, config in configs.items():
            print(f"\n{method_name}:")
            try:
                result = translator.translate(text, config)

                if isinstance(result, str):
                    print(f"  译文: {result}")
                elif isinstance(result, dict):
                    print(f"  译文: {result.get('text', '')}")
                    if "score" in result:
                        print(f"  分数: {result['score']:.4f}")
                elif isinstance(result, list):
                    for i, trans in enumerate(result, 1):
                        if isinstance(trans, str):
                            print(f"  候选{i}: {trans}")
                        elif isinstance(trans, dict):
                            print(
                                f"  候选{i}: {trans.get('text', '')} (分数: {trans.get('score', 0):.4f})"
                            )

            except Exception as e:
                print(f"  错误: {e}")

    # 显示翻译统计
    stats = translator.get_stats()
    print(f"\n翻译统计: {stats}")


def demo_attention_visualization(model: nn.Module, tokenizer: BilingualTokenizer):
    """
    演示注意力可视化

    Args:
        model: 模型
        tokenizer: 分词器
    """
    print("\n" + "=" * 50)
    print("注意力可视化演示")
    print("=" * 50)

    try:
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # 创建可视化器
        visualizer = AttentionVisualizer()

        # 测试文本
        src_text = "你好，世界！"
        tgt_text = "Hello, world!"

        # 编码文本
        src_tokens = tokenizer.zh_tokenizer.encode(src_text)
        tgt_tokens = tokenizer.en_tokenizer.encode(tgt_text)

        # 转换为张量
        src_input = torch.tensor([src_tokens], device=device)
        tgt_input = torch.tensor([tgt_tokens[:-1]], device=device)  # 移除最后一个token

        print(f"源文本: {src_text}")
        print(f"目标文本: {tgt_text}")
        print(f"源tokens: {src_tokens}")
        print(f"目标tokens: {tgt_tokens}")

        # 获取注意力权重
        with torch.no_grad():
            output = model(src_input, tgt_input, output_attentions=True)

        if hasattr(output, "attentions") and output.attentions:
            # 获取编码器注意力（最后一层）
            encoder_attention = output.attentions[-1][0]  # [n_heads, seq_len, seq_len]

            # 创建token标签
            src_labels = [
                tokenizer.zh_tokenizer.decode([token]) for token in src_tokens
            ]

            print(f"\n注意力权重形状: {encoder_attention.shape}")

            # 绘制注意力热力图
            output_dir = "outputs/demo"
            os.makedirs(output_dir, exist_ok=True)

            attention_file = os.path.join(output_dir, "attention_heatmap.png")
            visualizer.plot_attention_heatmap(
                encoder_attention.cpu().numpy(),
                src_labels,
                src_labels,
                save_path=attention_file,
            )

            print(f"注意力热力图保存到: {attention_file}")

            # 绘制多头注意力
            multihead_file = os.path.join(output_dir, "multihead_attention.png")
            visualizer.plot_multihead_attention(
                encoder_attention.cpu().numpy(),
                src_labels,
                src_labels,
                save_path=multihead_file,
            )

            print(f"多头注意力图保存到: {multihead_file}")

        else:
            print("模型没有返回注意力权重")

    except Exception as e:
        print(f"注意力可视化失败: {e}")
        import traceback

        traceback.print_exc()


def demo_metrics():
    """
    演示评估指标
    """
    print("\n" + "=" * 50)
    print("评估指标演示")
    print("=" * 50)

    # 创建评估指标实例
    metrics = TranslationMetrics()

    # 测试数据
    predictions = [
        "Hello, world!",
        "I love learning artificial intelligence.",
        "The weather is very nice today.",
    ]

    references = [
        "Hello, world!",
        "I love studying artificial intelligence.",
        "Today's weather is very good.",
    ]

    print("测试数据:")
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        print(f"  样本{i+1}:")
        print(f"    预测: {pred}")
        print(f"    参考: {ref}")

    # 计算BLEU分数
    bleu_scores = []
    corpus_bleu_scores = []

    for pred, ref in zip(predictions, references):
        bleu_result = metrics.calculate_bleu([pred], [[ref]])
        bleu_scores.append(bleu_result)
        corpus_bleu_scores.append(bleu_result["corpus_bleu"])

    avg_corpus_bleu = sum(corpus_bleu_scores) / len(corpus_bleu_scores)

    print(f"\nBLEU分数:")
    for i, bleu_result in enumerate(bleu_scores):
        print(f"  样本{i+1}: {bleu_result['corpus_bleu']:.4f}")
    print(f"  平均: {avg_corpus_bleu:.4f}")

    # 计算其他指标
    print(f"\n其他指标:")
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        meteor_result = metrics.calculate_meteor([pred], [ref])
        rouge_result = metrics.calculate_rouge([pred], [ref])

        print(f"  样本{i+1}:")
        print(f"    METEOR: {meteor_result['meteor_mean']:.4f}")
        print(f"    ROUGE-L: {rouge_result['rougeL_mean']:.4f}")


def main():
    """
    主演示函数
    """
    print("Transformer 翻译模型演示")
    print("=" * 60)

    # 设置日志
    setup_logging()

    try:
        # 创建演示数据
        print("创建演示数据...")
        demo_data = create_demo_data()
        save_demo_data(demo_data, "data/demo")

        # 创建演示配置
        config = create_demo_config()

        # 创建输出目录
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs("outputs/demo", exist_ok=True)

        # 演示分词器
        tokenizer = demo_tokenizer()

        # 演示模型
        model = demo_model(config, tokenizer)

        # 演示评估指标
        demo_metrics()

        trained_model, trainer = demo_training(config, tokenizer)

        # 演示翻译
        demo_translation(trained_model, tokenizer)

        # 演示注意力可视化
        demo_attention_visualization(trained_model, tokenizer)

        # 演示训练（可选，因为需要较长时间）
        # print("\n是否进行训练演示？(y/n): ", end="")
        # if input().lower().startswith('y'):
        #     trained_model, trainer = demo_training(config, tokenizer)

        #     # 演示翻译
        #     demo_translation(trained_model, tokenizer)

        #     # 演示注意力可视化
        #     demo_attention_visualization(trained_model, tokenizer)
        # else:
        #     print("跳过训练演示")

        #     # 使用未训练的模型进行演示
        #     demo_translation(model, tokenizer)
        #     demo_attention_visualization(model, tokenizer)

        print("\n" + "=" * 60)
        print("演示完成！")
        print("生成的文件:")
        print("  - data/demo/: 演示数据")
        print("  - outputs/demo/: 可视化结果")
        print("  - checkpoints/demo/: 模型检查点（如果进行了训练）")
        print("  - logs/demo/: 训练日志（如果进行了训练）")

    except KeyboardInterrupt:
        print("\n演示被用户中断")

    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
