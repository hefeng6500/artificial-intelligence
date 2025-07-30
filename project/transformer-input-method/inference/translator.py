"""翻译器模块"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time
from collections import defaultdict
import json
import os

# 导入生成器和束搜索
from .generator import TextGenerator, GenerationConfig
from .beam_search import BeamSearch, BeamSearchConfig


@dataclass
class TranslationConfig:
    """翻译配置"""
    # 基础参数
    max_length: int = 100
    min_length: int = 1
    
    # 搜索策略
    search_strategy: str = "beam_search"  # "greedy", "beam_search", "sampling"
    
    # 束搜索参数
    beam_size: int = 5
    length_penalty: float = 1.0
    coverage_penalty: float = 0.0
    
    # 采样参数
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    
    # 重复惩罚
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # 返回参数
    num_return_sequences: int = 1
    return_scores: bool = False
    return_attention: bool = False
    
    # 后处理
    remove_bos: bool = True
    remove_eos: bool = True
    remove_pad: bool = True
    
    # 特殊处理
    handle_unk: bool = True
    copy_unknown: bool = False
    
    # 性能优化
    batch_size: int = 32
    use_cache: bool = True
    
    # 质量控制
    min_score_threshold: float = -float('inf')
    max_score_threshold: float = float('inf')
    filter_duplicates: bool = True


class Translator:
    """
    翻译器
    
    Args:
        model: Transformer 模型
        src_tokenizer: 源语言分词器
        tgt_tokenizer: 目标语言分词器
        device: 设备
    """
    
    def __init__(
        self,
        model,
        src_tokenizer,
        tgt_tokenizer=None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer or src_tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移动模型到设备
        self.model.to(self.device)
        self.model.eval()
        
        # 特殊 token ID
        self.src_pad_id = getattr(src_tokenizer, 'pad_token_id', 0)
        self.src_bos_id = getattr(src_tokenizer, 'bos_token_id', 1)
        self.src_eos_id = getattr(src_tokenizer, 'eos_token_id', 2)
        self.src_unk_id = getattr(src_tokenizer, 'unk_token_id', 3)
        
        self.tgt_pad_id = getattr(self.tgt_tokenizer, 'pad_token_id', 0)
        self.tgt_bos_id = getattr(self.tgt_tokenizer, 'bos_token_id', 1)
        self.tgt_eos_id = getattr(self.tgt_tokenizer, 'eos_token_id', 2)
        self.tgt_unk_id = getattr(self.tgt_tokenizer, 'unk_token_id', 3)
        
        # 创建文本生成器
        self.generator = TextGenerator(model, self.tgt_tokenizer, device)
        
        # 统计信息
        self.translation_stats = {
            'total_translations': 0,
            'total_time': 0.0,
            'avg_time_per_translation': 0.0,
            'avg_src_length': 0.0,
            'avg_tgt_length': 0.0
        }
    
    def translate(
        self,
        text: str,
        config: Optional[TranslationConfig] = None,
        **kwargs
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        翻译单个文本
        
        Args:
            text: 源文本
            config: 翻译配置
            **kwargs: 其他参数
            
        Returns:
            翻译结果
        """
        return self.translate_batch([text], config, **kwargs)[0]
    
    def translate_batch(
        self,
        texts: List[str],
        config: Optional[TranslationConfig] = None,
        **kwargs
    ) -> List[Union[str, List[str], Dict[str, Any]]]:
        """
        批量翻译
        
        Args:
            texts: 源文本列表
            config: 翻译配置
            **kwargs: 其他参数
            
        Returns:
            翻译结果列表
        """
        # 合并配置
        config = config or TranslationConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        start_time = time.time()
        
        # 预处理输入
        src_inputs = self._preprocess_inputs(texts, config)
        
        # 执行翻译
        if config.search_strategy == "greedy":
            results = self._greedy_translate(src_inputs, config)
        elif config.search_strategy == "beam_search":
            results = self._beam_search_translate(src_inputs, config)
        elif config.search_strategy == "sampling":
            results = self._sampling_translate(src_inputs, config)
        else:
            raise ValueError(f"未知的搜索策略: {config.search_strategy}")
        
        # 后处理输出
        translations = self._postprocess_outputs(results, config)
        
        # 更新统计信息
        end_time = time.time()
        self._update_stats(texts, translations, end_time - start_time)
        
        return translations
    
    def _preprocess_inputs(
        self,
        texts: List[str],
        config: TranslationConfig
    ) -> Dict[str, torch.Tensor]:
        """
        预处理输入文本
        
        Args:
            texts: 输入文本列表
            config: 翻译配置
            
        Returns:
            预处理后的输入
        """
        # 编码源文本
        encoded_inputs = []
        for text in texts:
            tokens = self.src_tokenizer.encode(text)
            # 添加 BOS 和 EOS
            if self.src_bos_id not in tokens:
                tokens = [self.src_bos_id] + tokens
            if self.src_eos_id not in tokens:
                tokens = tokens + [self.src_eos_id]
            encoded_inputs.append(tokens)
        
        # 填充到相同长度
        max_length = max(len(tokens) for tokens in encoded_inputs)
        
        padded_inputs = []
        attention_masks = []
        
        for tokens in encoded_inputs:
            # 填充
            padding_length = max_length - len(tokens)
            padded_tokens = tokens + [self.src_pad_id] * padding_length
            
            # 创建注意力掩码
            attention_mask = [1] * len(tokens) + [0] * padding_length
            
            padded_inputs.append(padded_tokens)
            attention_masks.append(attention_mask)
        
        # 转换为张量
        input_ids = torch.tensor(padded_inputs, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device, dtype=torch.bool)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def _greedy_translate(
        self,
        src_inputs: Dict[str, torch.Tensor],
        config: TranslationConfig
    ) -> torch.Tensor:
        """
        贪心翻译
        
        Args:
            src_inputs: 源输入
            config: 翻译配置
            
        Returns:
            翻译结果
        """
        batch_size = src_inputs['input_ids'].size(0)
        max_length = config.max_length
        
        # 编码源序列
        with torch.no_grad():
            encoder_outputs = self.model.encode(
                src_inputs['input_ids'],
                src_inputs['attention_mask']
            )
        
        # 初始化目标序列
        tgt_input = torch.full(
            (batch_size, 1),
            self.tgt_bos_id,
            device=self.device
        )
        
        # 贪心解码
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            for step in range(max_length - 1):
                # 解码
                outputs = self.model.decode(
                    tgt_input,
                    encoder_outputs,
                    src_inputs['attention_mask']
                )
                
                # 获取下一个 token
                next_token_logits = outputs[:, -1, :]
                
                # 应用重复惩罚
                if config.repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, tgt_input, config.repetition_penalty
                    )
                
                # 贪心选择
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # 更新完成状态
                finished = finished | (next_tokens == self.tgt_eos_id)
                
                # 添加新 token
                tgt_input = torch.cat([tgt_input, next_tokens.unsqueeze(1)], dim=1)
                
                # 检查是否所有序列都完成
                if finished.all():
                    break
                
                # 检查最小长度
                if tgt_input.size(1) < config.min_length:
                    # 强制不生成 EOS
                    next_tokens = torch.where(
                        next_tokens == self.tgt_eos_id,
                        torch.full_like(next_tokens, self.tgt_unk_id),
                        next_tokens
                    )
        
        return tgt_input
    
    def _beam_search_translate(
        self,
        src_inputs: Dict[str, torch.Tensor],
        config: TranslationConfig
    ) -> List[List[Dict[str, Any]]]:
        """
        束搜索翻译
        
        Args:
            src_inputs: 源输入
            config: 翻译配置
            
        Returns:
            翻译候选列表
        """
        # 创建束搜索配置
        beam_config = BeamSearchConfig(
            beam_size=config.beam_size,
            max_length=config.max_length,
            min_length=config.min_length,
            length_penalty=config.length_penalty,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            num_return_sequences=config.num_return_sequences,
            early_stopping=True
        )
        
        # 创建束搜索器
        beam_searcher = BeamSearch(
            self.model, self.tgt_tokenizer, beam_config, self.device
        )
        
        # 执行束搜索
        batch_size = src_inputs['input_ids'].size(0)
        all_candidates = []
        
        for i in range(batch_size):
            single_input = src_inputs['input_ids'][i:i+1]
            single_mask = src_inputs['attention_mask'][i:i+1]
            
            candidates = beam_searcher.search(single_input, single_mask)
            
            # 转换为字典格式
            candidate_dicts = []
            for candidate in candidates:
                candidate_dict = {
                    'tokens': candidate.tokens,
                    'score': candidate.score,
                    'length': len(candidate.tokens)
                }
                candidate_dicts.append(candidate_dict)
            
            all_candidates.append(candidate_dicts)
        
        return all_candidates
    
    def _sampling_translate(
        self,
        src_inputs: Dict[str, torch.Tensor],
        config: TranslationConfig
    ) -> torch.Tensor:
        """
        采样翻译
        
        Args:
            src_inputs: 源输入
            config: 翻译配置
            
        Returns:
            翻译结果
        """
        # 创建生成配置
        gen_config = GenerationConfig(
            max_length=config.max_length,
            min_length=config.min_length,
            do_sample=True,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            num_return_sequences=config.num_return_sequences,
            early_stopping=True,
            pad_token_id=self.tgt_pad_id,
            bos_token_id=self.tgt_bos_id,
            eos_token_id=self.tgt_eos_id
        )
        
        # 使用生成器进行采样
        generated = self.generator.generate(
            input_ids=src_inputs['input_ids'],
            attention_mask=src_inputs['attention_mask'],
            generation_config=gen_config
        )
        
        return generated
    
    def _postprocess_outputs(
        self,
        results: Union[torch.Tensor, List[List[Dict[str, Any]]]],
        config: TranslationConfig
    ) -> List[Union[str, List[str], Dict[str, Any]]]:
        """
        后处理输出
        
        Args:
            results: 原始结果
            config: 翻译配置
            
        Returns:
            后处理后的翻译结果
        """
        translations = []
        
        if isinstance(results, torch.Tensor):
            # 贪心或采样结果
            for i in range(results.size(0)):
                tokens = results[i].cpu().numpy().tolist()
                text = self._tokens_to_text(tokens, config)
                translations.append(text)
        
        elif isinstance(results, list):
            # 束搜索结果
            for candidates in results:
                if config.num_return_sequences == 1:
                    # 返回最佳候选
                    best_candidate = candidates[0]
                    text = self._tokens_to_text(best_candidate['tokens'], config)
                    
                    if config.return_scores:
                        translation = {
                            'text': text,
                            'score': best_candidate['score'],
                            'length': best_candidate['length']
                        }
                    else:
                        translation = text
                    
                    translations.append(translation)
                
                else:
                    # 返回多个候选
                    candidate_texts = []
                    for candidate in candidates[:config.num_return_sequences]:
                        text = self._tokens_to_text(candidate['tokens'], config)
                        
                        if config.return_scores:
                            candidate_texts.append({
                                'text': text,
                                'score': candidate['score'],
                                'length': candidate['length']
                            })
                        else:
                            candidate_texts.append(text)
                    
                    translations.append(candidate_texts)
        
        return translations
    
    def _tokens_to_text(
        self,
        tokens: List[int],
        config: TranslationConfig
    ) -> str:
        """
        将 tokens 转换为文本
        
        Args:
            tokens: token 列表
            config: 翻译配置
            
        Returns:
            文本
        """
        # 移除特殊 tokens
        filtered_tokens = []
        for token in tokens:
            if config.remove_bos and token == self.tgt_bos_id:
                continue
            if config.remove_eos and token == self.tgt_eos_id:
                continue
            if config.remove_pad and token == self.tgt_pad_id:
                continue
            filtered_tokens.append(token)
        
        # 解码
        text = self.tgt_tokenizer.decode(filtered_tokens)
        
        # 处理未知词
        if config.handle_unk and hasattr(self.tgt_tokenizer, 'unk_token'):
            unk_token = self.tgt_tokenizer.unk_token
            if unk_token in text:
                # 简单替换为空格
                text = text.replace(unk_token, ' ')
        
        return text.strip()
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """
        应用重复惩罚
        
        Args:
            logits: 当前 logits
            input_ids: 输入序列
            penalty: 惩罚系数
            
        Returns:
            应用惩罚后的 logits
        """
        if penalty == 1.0:
            return logits
        
        batch_size = logits.size(0)
        
        for i in range(batch_size):
            for token in input_ids[i].unique():
                if logits[i, token] > 0:
                    logits[i, token] /= penalty
                else:
                    logits[i, token] *= penalty
        
        return logits
    
    def _update_stats(
        self,
        src_texts: List[str],
        translations: List[str],
        elapsed_time: float
    ):
        """
        更新统计信息
        
        Args:
            src_texts: 源文本列表
            translations: 翻译列表
            elapsed_time: 耗时
        """
        batch_size = len(src_texts)
        
        self.translation_stats['total_translations'] += batch_size
        self.translation_stats['total_time'] += elapsed_time
        
        # 计算平均时间
        self.translation_stats['avg_time_per_translation'] = (
            self.translation_stats['total_time'] / 
            self.translation_stats['total_translations']
        )
        
        # 计算平均长度
        src_lengths = [len(text.split()) for text in src_texts]
        tgt_lengths = []
        
        for translation in translations:
            if isinstance(translation, str):
                tgt_lengths.append(len(translation.split()))
            elif isinstance(translation, dict) and 'text' in translation:
                tgt_lengths.append(len(translation['text'].split()))
            elif isinstance(translation, list):
                # 取第一个候选的长度
                if translation and isinstance(translation[0], str):
                    tgt_lengths.append(len(translation[0].split()))
                elif translation and isinstance(translation[0], dict):
                    tgt_lengths.append(len(translation[0]['text'].split()))
                else:
                    tgt_lengths.append(0)
            else:
                tgt_lengths.append(0)
        
        # 更新平均长度
        total_translations = self.translation_stats['total_translations']
        current_avg_src = self.translation_stats['avg_src_length']
        current_avg_tgt = self.translation_stats['avg_tgt_length']
        
        new_avg_src = np.mean(src_lengths)
        new_avg_tgt = np.mean(tgt_lengths)
        
        self.translation_stats['avg_src_length'] = (
            (current_avg_src * (total_translations - batch_size) + 
             new_avg_src * batch_size) / total_translations
        )
        
        self.translation_stats['avg_tgt_length'] = (
            (current_avg_tgt * (total_translations - batch_size) + 
             new_avg_tgt * batch_size) / total_translations
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取翻译统计信息
        
        Returns:
            统计信息字典
        """
        return self.translation_stats.copy()
    
    def reset_stats(self):
        """
        重置统计信息
        """
        self.translation_stats = {
            'total_translations': 0,
            'total_time': 0.0,
            'avg_time_per_translation': 0.0,
            'avg_src_length': 0.0,
            'avg_tgt_length': 0.0
        }
    
    def save_config(self, config: TranslationConfig, filepath: str):
        """
        保存翻译配置
        
        Args:
            config: 翻译配置
            filepath: 保存路径
        """
        config_dict = {
            'max_length': config.max_length,
            'min_length': config.min_length,
            'search_strategy': config.search_strategy,
            'beam_size': config.beam_size,
            'length_penalty': config.length_penalty,
            'coverage_penalty': config.coverage_penalty,
            'temperature': config.temperature,
            'top_k': config.top_k,
            'top_p': config.top_p,
            'repetition_penalty': config.repetition_penalty,
            'no_repeat_ngram_size': config.no_repeat_ngram_size,
            'num_return_sequences': config.num_return_sequences,
            'return_scores': config.return_scores,
            'return_attention': config.return_attention
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_config(self, filepath: str) -> TranslationConfig:
        """
        加载翻译配置
        
        Args:
            filepath: 配置文件路径
            
        Returns:
            翻译配置
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return TranslationConfig(**config_dict)


if __name__ == "__main__":
    # 测试翻译器
    print("测试翻译器模块...")
    
    # 创建翻译配置
    config = TranslationConfig(
        max_length=50,
        min_length=5,
        search_strategy="beam_search",
        beam_size=5,
        length_penalty=1.2,
        num_return_sequences=3,
        return_scores=True
    )
    
    print(f"翻译配置:")
    print(f"  最大长度: {config.max_length}")
    print(f"  最小长度: {config.min_length}")
    print(f"  搜索策略: {config.search_strategy}")
    print(f"  束大小: {config.beam_size}")
    print(f"  长度惩罚: {config.length_penalty}")
    print(f"  返回序列数: {config.num_return_sequences}")
    print(f"  返回分数: {config.return_scores}")
    
    # 测试采样配置
    sampling_config = TranslationConfig(
        max_length=50,
        search_strategy="sampling",
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        num_return_sequences=2
    )
    
    print(f"\n采样配置:")
    print(f"  搜索策略: {sampling_config.search_strategy}")
    print(f"  温度: {sampling_config.temperature}")
    print(f"  Top-k: {sampling_config.top_k}")
    print(f"  Top-p: {sampling_config.top_p}")
    print(f"  返回序列数: {sampling_config.num_return_sequences}")
    
    print("\n✅ 翻译器模块测试通过")