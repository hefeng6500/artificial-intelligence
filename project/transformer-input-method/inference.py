#!/usr/bin/env python3
"""推理脚本"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from config.training_config import TrainingConfig
from models.transformer import TransformerModel
from utils.tokenizer import BilingualTokenizer
from inference.translator import Translator, TranslationConfig
from inference.generator import TextGenerator, GenerationConfig
from utils.metrics import calculate_bleu, TranslationMetrics


def setup_logging(log_level: str = "INFO"):
    """
    设置日志
    
    Args:
        log_level: 日志级别
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    device: torch.device
) -> tuple:
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
        tokenizer_path: 分词器路径
        device: 设备
        
    Returns:
        (model, tokenizer)
    """
    logging.info(f"加载模型: {model_path}")
    logging.info(f"加载分词器: {tokenizer_path}")
    
    # 加载分词器
    tokenizer = BilingualTokenizer()
    tokenizer.load(tokenizer_path)
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型配置
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 如果没有配置，尝试从模型状态推断
        logging.warning("检查点中没有找到配置，使用默认配置")
        config = TrainingConfig()
        config.src_vocab_size = len(tokenizer.src_vocab)
        config.tgt_vocab_size = len(tokenizer.tgt_vocab)
    
    # 创建模型
    model = TransformerModel(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    )
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logging.info("模型和分词器加载完成")
    return model, tokenizer


def translate_text(
    translator: Translator,
    text: str,
    config: TranslationConfig
) -> Dict[str, Any]:
    """
    翻译单个文本
    
    Args:
        translator: 翻译器
        text: 输入文本
        config: 翻译配置
        
    Returns:
        翻译结果
    """
    logging.info(f"翻译文本: {text}")
    
    # 执行翻译
    result = translator.translate(text, config)
    
    # 格式化结果
    if isinstance(result, str):
        return {
            'source': text,
            'translation': result,
            'method': config.search_strategy
        }
    elif isinstance(result, dict):
        return {
            'source': text,
            'translation': result.get('text', ''),
            'score': result.get('score', 0.0),
            'method': config.search_strategy
        }
    elif isinstance(result, list):
        return {
            'source': text,
            'translations': result,
            'method': config.search_strategy
        }
    else:
        return {
            'source': text,
            'translation': str(result),
            'method': config.search_strategy
        }


def translate_file(
    translator: Translator,
    input_file: str,
    output_file: str,
    config: TranslationConfig
):
    """
    翻译文件
    
    Args:
        translator: 翻译器
        input_file: 输入文件路径
        output_file: 输出文件路径
        config: 翻译配置
    """
    logging.info(f"翻译文件: {input_file} -> {output_file}")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    logging.info(f"共 {len(lines)} 行待翻译")
    
    # 批量翻译
    batch_size = config.batch_size
    all_results = []
    
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        logging.info(f"翻译批次 {i//batch_size + 1}/{(len(lines)-1)//batch_size + 1}")
        
        batch_results = translator.translate_batch(batch_lines, config)
        all_results.extend(batch_results)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            if isinstance(result, str):
                f.write(result + '\n')
            elif isinstance(result, dict) and 'text' in result:
                f.write(result['text'] + '\n')
            elif isinstance(result, list) and result:
                # 取第一个候选
                if isinstance(result[0], str):
                    f.write(result[0] + '\n')
                elif isinstance(result[0], dict):
                    f.write(result[0].get('text', '') + '\n')
                else:
                    f.write(str(result[0]) + '\n')
            else:
                f.write(str(result) + '\n')
    
    logging.info(f"翻译完成，结果保存到: {output_file}")


def evaluate_translation(
    translator: Translator,
    test_file: str,
    config: TranslationConfig
) -> Dict[str, float]:
    """
    评估翻译质量
    
    Args:
        translator: 翻译器
        test_file: 测试文件路径（包含源文本和参考翻译）
        config: 翻译配置
        
    Returns:
        评估指标
    """
    logging.info(f"评估翻译质量: {test_file}")
    
    # 读取测试数据
    src_texts = []
    ref_texts = []
    
    if test_file.endswith('.json'):
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            src_texts.append(item['source'])
            ref_texts.append(item['target'])
    
    elif test_file.endswith('.txt'):
        # 假设每行包含源文本和目标文本，用制表符分隔
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    src, tgt = line.strip().split('\t', 1)
                    src_texts.append(src)
                    ref_texts.append(tgt)
    
    else:
        raise ValueError(f"不支持的文件格式: {test_file}")
    
    logging.info(f"共 {len(src_texts)} 个测试样本")
    
    # 执行翻译
    translations = translator.translate_batch(src_texts, config)
    
    # 提取翻译文本
    pred_texts = []
    for result in translations:
        if isinstance(result, str):
            pred_texts.append(result)
        elif isinstance(result, dict) and 'text' in result:
            pred_texts.append(result['text'])
        elif isinstance(result, list) and result:
            if isinstance(result[0], str):
                pred_texts.append(result[0])
            elif isinstance(result[0], dict):
                pred_texts.append(result[0].get('text', ''))
            else:
                pred_texts.append(str(result[0]))
        else:
            pred_texts.append(str(result))
    
    # 计算评估指标
    metrics = TranslationMetrics()
    
    # 计算 BLEU 分数
    bleu_scores = []
    for pred, ref in zip(pred_texts, ref_texts):
        bleu = calculate_bleu([pred], [[ref]])
        bleu_scores.append(bleu)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    # 计算其他指标
    meteor_scores = []
    rouge_scores = []
    
    for pred, ref in zip(pred_texts, ref_texts):
        meteor = metrics.calculate_meteor(pred, ref)
        rouge = metrics.calculate_rouge(pred, ref)
        
        meteor_scores.append(meteor)
        rouge_scores.append(rouge['rouge-l']['f'])
    
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    
    results = {
        'bleu': avg_bleu,
        'meteor': avg_meteor,
        'rouge-l': avg_rouge,
        'num_samples': len(src_texts)
    }
    
    logging.info(f"评估结果: {results}")
    return results


def interactive_translation(translator: Translator, config: TranslationConfig):
    """
    交互式翻译
    
    Args:
        translator: 翻译器
        config: 翻译配置
    """
    logging.info("进入交互式翻译模式")
    logging.info("输入 'quit' 或 'exit' 退出")
    
    while True:
        try:
            # 获取用户输入
            text = input("\n请输入要翻译的文本: ").strip()
            
            if text.lower() in ['quit', 'exit', '退出']:
                break
            
            if not text:
                continue
            
            # 执行翻译
            result = translate_text(translator, text, config)
            
            # 显示结果
            print(f"\n原文: {result['source']}")
            
            if 'translation' in result:
                print(f"译文: {result['translation']}")
                if 'score' in result:
                    print(f"分数: {result['score']:.4f}")
            
            elif 'translations' in result:
                print("候选译文:")
                for i, trans in enumerate(result['translations'], 1):
                    if isinstance(trans, str):
                        print(f"  {i}. {trans}")
                    elif isinstance(trans, dict):
                        print(f"  {i}. {trans.get('text', '')} (分数: {trans.get('score', 0):.4f})")
            
            print(f"方法: {result['method']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"翻译过程中发生错误: {e}")
    
    logging.info("退出交互式翻译模式")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="Transformer 翻译模型推理")
    
    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型文件路径"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="分词器路径"
    )
    
    # 推理模式
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "file", "interactive", "evaluate"],
        default="interactive",
        help="推理模式"
    )
    
    # 输入输出
    parser.add_argument(
        "--input",
        type=str,
        help="输入文本或文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="测试文件路径（用于评估）"
    )
    
    # 翻译配置
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["greedy", "beam_search", "sampling"],
        default="beam_search",
        help="搜索策略"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="束搜索大小"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="最大生成长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k 采样"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p 采样"
    )
    parser.add_argument(
        "--num-return",
        type=int,
        default=1,
        help="返回候选数量"
    )
    
    # 其他参数
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    logging.info("开始 Transformer 翻译模型推理")
    logging.info(f"命令行参数: {vars(args)}")
    
    try:
        # 设置设备
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"使用设备: {device}")
        
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(
            args.model, args.tokenizer, device
        )
        
        # 创建翻译配置
        translation_config = TranslationConfig(
            max_length=args.max_length,
            search_strategy=args.strategy,
            beam_size=args.beam_size,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_return,
            batch_size=args.batch_size,
            return_scores=True
        )
        
        # 创建翻译器
        translator = Translator(model, tokenizer, tokenizer, device)
        
        # 根据模式执行不同操作
        if args.mode == "text":
            if not args.input:
                logging.error("文本模式需要指定 --input 参数")
                sys.exit(1)
            
            result = translate_text(translator, args.input, translation_config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        elif args.mode == "file":
            if not args.input or not args.output:
                logging.error("文件模式需要指定 --input 和 --output 参数")
                sys.exit(1)
            
            translate_file(translator, args.input, args.output, translation_config)
        
        elif args.mode == "evaluate":
            if not args.test_file:
                logging.error("评估模式需要指定 --test-file 参数")
                sys.exit(1)
            
            results = evaluate_translation(translator, args.test_file, translation_config)
            print(json.dumps(results, ensure_ascii=False, indent=2))
        
        elif args.mode == "interactive":
            interactive_translation(translator, translation_config)
        
        else:
            logging.error(f"未知模式: {args.mode}")
            sys.exit(1)
        
        # 显示统计信息
        stats = translator.get_stats()
        if stats['total_translations'] > 0:
            logging.info(f"翻译统计: {stats}")
        
        logging.info("推理完成")
        
    except KeyboardInterrupt:
        logging.info("推理被用户中断")
        sys.exit(1)
    
    except Exception as e:
        logging.error(f"推理过程中发生错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()