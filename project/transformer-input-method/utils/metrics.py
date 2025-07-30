"""评估指标模块"""

import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
from sacrebleu import BLEU
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# 下载必要的 NLTK 数据
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TranslationMetrics:
    """翻译评估指标类"""
    
    def __init__(self):
        self.bleu_scorer = BLEU()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
    
    def calculate_bleu(
        self, 
        predictions: List[str], 
        references: List[str],
        max_order: int = 4
    ) -> Dict[str, float]:
        """
        计算 BLEU 分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            max_order: 最大 n-gram 阶数
            
        Returns:
            BLEU 分数字典
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        # 使用 sacrebleu 计算整体 BLEU
        # sacrebleu 期望的格式: corpus_score(hypotheses, list_of_list_of_references)
        # 检查 references 的格式
        if len(references) > 0 and isinstance(references[0], list):
            # 如果 references 已经是列表的列表，直接使用
            references_list = references
        else:
            # 如果 references 是字符串列表，转换为列表的列表
            references_list = [[ref] for ref in references]
        
        corpus_bleu = self.bleu_scorer.corpus_score(predictions, references_list)
        
        # 计算句子级别的 BLEU
        sentence_bleus = []
        bleu_scores = {f'bleu_{i}': [] for i in range(1, max_order + 1)}
        
        # 处理句子级别的 BLEU 计算
        # 需要确保 references 的格式正确
        if len(references) > 0 and isinstance(references[0], list):
            # 如果 references 是列表的列表，提取第一个参考文本
            ref_strings = [ref[0] if len(ref) > 0 else "" for ref in references]
        else:
            # 如果 references 是字符串列表，直接使用
            ref_strings = references
            
        for pred, ref in zip(predictions, ref_strings):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            
            # 计算不同阶数的 BLEU
            for n in range(1, max_order + 1):
                try:
                    bleu_n = sentence_bleu(
                        ref_tokens, pred_tokens, 
                        weights=tuple([1/n] * n + [0] * (4-n)),
                        smoothing_function=self.smoothing_function
                    )
                    bleu_scores[f'bleu_{n}'].append(bleu_n)
                except:
                    bleu_scores[f'bleu_{n}'].append(0.0)
            
            # 计算整体句子 BLEU
            try:
                sent_bleu = sentence_bleu(
                    ref_tokens, pred_tokens,
                    smoothing_function=self.smoothing_function
                )
                sentence_bleus.append(sent_bleu)
            except:
                sentence_bleus.append(0.0)
        
        # 计算平均分数
        results = {
            'corpus_bleu': corpus_bleu.score,
            'sentence_bleu_mean': np.mean(sentence_bleus),
            'sentence_bleu_std': np.std(sentence_bleus)
        }
        
        for n in range(1, max_order + 1):
            results[f'bleu_{n}_mean'] = np.mean(bleu_scores[f'bleu_{n}'])
            results[f'bleu_{n}_std'] = np.std(bleu_scores[f'bleu_{n}'])
        
        return results
    
    def calculate_meteor(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算 METEOR 分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            METEOR 分数字典
        """
        meteor_scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                score = meteor_score([ref.split()], pred.split())
                meteor_scores.append(score)
            except:
                meteor_scores.append(0.0)
        
        return {
            'meteor_mean': np.mean(meteor_scores),
            'meteor_std': np.std(meteor_scores)
        }
    
    def calculate_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算 ROUGE 分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            ROUGE 分数字典
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            
            for rouge_type in rouge_scores.keys():
                rouge_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        results = {}
        for rouge_type, scores in rouge_scores.items():
            results[f'{rouge_type}_mean'] = np.mean(scores)
            results[f'{rouge_type}_std'] = np.std(scores)
        
        return results
    
    def calculate_all_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            所有指标的字典
        """
        results = {}
        
        # BLEU 分数
        try:
            bleu_results = self.calculate_bleu(predictions, references)
            results.update(bleu_results)
        except Exception as e:
            print(f"BLEU 计算错误: {e}")
        
        # METEOR 分数
        try:
            meteor_results = self.calculate_meteor(predictions, references)
            results.update(meteor_results)
        except Exception as e:
            print(f"METEOR 计算错误: {e}")
        
        # ROUGE 分数
        try:
            rouge_results = self.calculate_rouge(predictions, references)
            results.update(rouge_results)
        except Exception as e:
            print(f"ROUGE 计算错误: {e}")
        
        return results


def calculate_perplexity(loss: float) -> float:
    """
    计算困惑度
    
    Args:
        loss: 交叉熵损失
        
    Returns:
        困惑度值
    """
    return math.exp(loss)


def calculate_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    ignore_index: int = -100
) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测 logits [batch_size, seq_len, vocab_size]
        targets: 目标标签 [batch_size, seq_len]
        ignore_index: 忽略的索引
        
    Returns:
        准确率
    """
    pred_tokens = torch.argmax(predictions, dim=-1)
    
    # 创建掩码，忽略特定索引
    mask = (targets != ignore_index)
    
    # 计算正确预测的数量
    correct = (pred_tokens == targets) & mask
    total = mask.sum()
    
    if total == 0:
        return 0.0
    
    return (correct.sum().float() / total.float()).item()


def calculate_token_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int = 0
) -> Dict[str, float]:
    """
    计算 token 级别的准确率
    
    Args:
        predictions: 预测 logits
        targets: 目标标签
        pad_token_id: 填充 token ID
        
    Returns:
        准确率指标字典
    """
    pred_tokens = torch.argmax(predictions, dim=-1)
    
    # 创建非填充掩码
    non_pad_mask = (targets != pad_token_id)
    
    # 计算总体准确率
    correct = (pred_tokens == targets) & non_pad_mask
    total = non_pad_mask.sum()
    
    overall_accuracy = (correct.sum().float() / total.float()).item() if total > 0 else 0.0
    
    # 计算序列级别的准确率
    seq_correct = (correct.sum(dim=1) == non_pad_mask.sum(dim=1))
    seq_accuracy = seq_correct.float().mean().item()
    
    return {
        'token_accuracy': overall_accuracy,
        'sequence_accuracy': seq_accuracy
    }


def calculate_edit_distance(str1: str, str2: str) -> int:
    """
    计算编辑距离（Levenshtein 距离）
    
    Args:
        str1: 第一个字符串
        str2: 第二个字符串
        
    Returns:
        编辑距离
    """
    m, n = len(str1), len(str2)
    
    # 创建 DP 表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # 删除
                    dp[i][j-1],    # 插入
                    dp[i-1][j-1]   # 替换
                )
    
    return dp[m][n]


def calculate_character_error_rate(
    predictions: List[str], 
    references: List[str]
) -> float:
    """
    计算字符错误率 (CER)
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        
    Returns:
        字符错误率
    """
    total_chars = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        edit_dist = calculate_edit_distance(pred, ref)
        total_errors += edit_dist
        total_chars += len(ref)
    
    return total_errors / total_chars if total_chars > 0 else 0.0


def calculate_word_error_rate(
    predictions: List[str], 
    references: List[str]
) -> float:
    """
    计算词错误率 (WER)
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        
    Returns:
        词错误率
    """
    total_words = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        
        edit_dist = calculate_edit_distance(' '.join(pred_words), ' '.join(ref_words))
        total_errors += edit_dist
        total_words += len(ref_words)
    
    return total_errors / total_words if total_words > 0 else 0.0


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.translation_metrics = TranslationMetrics()
    
    def update(
        self, 
        predictions: List[str], 
        references: List[str], 
        loss: Optional[float] = None
    ):
        """
        更新指标
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            loss: 损失值
        """
        # 计算翻译指标
        translation_scores = self.translation_metrics.calculate_all_metrics(
            predictions, references
        )
        
        for metric, score in translation_scores.items():
            self.metrics[metric].append(score)
        
        # 计算错误率
        cer = calculate_character_error_rate(predictions, references)
        wer = calculate_word_error_rate(predictions, references)
        
        self.metrics['cer'].append(cer)
        self.metrics['wer'].append(wer)
        
        # 添加损失
        if loss is not None:
            self.metrics['loss'].append(loss)
            self.metrics['perplexity'].append(calculate_perplexity(loss))
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        获取平均指标
        
        Returns:
            平均指标字典
        """
        avg_metrics = {}
        
        for metric, values in self.metrics.items():
            if values:
                avg_metrics[f'avg_{metric}'] = np.mean(values)
                avg_metrics[f'std_{metric}'] = np.std(values)
        
        return avg_metrics
    
    def reset(self):
        """重置指标"""
        self.metrics.clear()
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """
        获取最新的指标
        
        Returns:
            最新指标字典
        """
        latest_metrics = {}
        
        for metric, values in self.metrics.items():
            if values:
                latest_metrics[metric] = values[-1]
        
        return latest_metrics


if __name__ == "__main__":
    # 测试评估指标
    predictions = [
        "hello world",
        "this is a test",
        "machine translation"
    ]
    
    references = [
        "hello world",
        "this is a test sentence",
        "machine learning"
    ]
    
    # 测试翻译指标
    metrics = TranslationMetrics()
    
    print("测试 BLEU 分数:")
    bleu_scores = metrics.calculate_bleu(predictions, references)
    for metric, score in bleu_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n测试所有指标:")
    all_scores = metrics.calculate_all_metrics(predictions, references)
    for metric, score in all_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # 测试错误率
    print("\n测试错误率:")
    cer = calculate_character_error_rate(predictions, references)
    wer = calculate_word_error_rate(predictions, references)
    print(f"  CER: {cer:.4f}")
    print(f"  WER: {wer:.4f}")
    
    # 测试指标跟踪器
    print("\n测试指标跟踪器:")
    tracker = MetricsTracker()
    tracker.update(predictions, references, loss=2.5)
    
    avg_metrics = tracker.get_average_metrics()
    for metric, score in avg_metrics.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n✅ 评估指标测试通过")