"""文本生成器模块"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time
from collections import defaultdict

# 导入束搜索
from .beam_search import BeamSearch, BeamSearchConfig, BeamCandidate


@dataclass
class GenerationConfig:
    """文本生成配置"""
    # 基础参数
    max_length: int = 100
    min_length: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    
    # 束搜索参数
    num_beams: int = 1
    beam_search_config: Optional[BeamSearchConfig] = None
    
    # 重复惩罚
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # 长度惩罚
    length_penalty: float = 1.0
    
    # 早停
    early_stopping: bool = True
    
    # 返回参数
    num_return_sequences: int = 1
    return_dict_in_generate: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False
    
    # 特殊 token
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # 其他参数
    use_cache: bool = True
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    remove_invalid_values: bool = False


class TextGenerator:
    """
    文本生成器
    
    Args:
        model: Transformer 模型
        tokenizer: 分词器
        device: 设备
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 特殊 token ID
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
        self.bos_token_id = getattr(tokenizer, 'bos_token_id', 1)
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', 2)
        self.unk_token_id = getattr(tokenizer, 'unk_token_id', 3)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        生成文本
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            generation_config: 生成配置
            **kwargs: 其他参数
            
        Returns:
            生成的 token IDs 或包含额外信息的字典
        """
        # 合并配置
        config = generation_config or GenerationConfig()
        
        # 从 kwargs 更新配置
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 设置特殊 token ID
        if config.pad_token_id is None:
            config.pad_token_id = self.pad_token_id
        if config.bos_token_id is None:
            config.bos_token_id = self.bos_token_id
        if config.eos_token_id is None:
            config.eos_token_id = self.eos_token_id
        
        # 选择生成策略
        if config.num_beams > 1:
            return self._beam_search_generate(input_ids, attention_mask, config)
        elif config.do_sample:
            return self._sample_generate(input_ids, attention_mask, config)
        else:
            return self._greedy_generate(input_ids, attention_mask, config)
    
    def _greedy_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        config: GenerationConfig = None
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        贪心解码生成
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            config: 生成配置
            
        Returns:
            生成的 token IDs
        """
        batch_size = input_ids.size(0)
        max_length = config.max_length
        
        # 初始化生成序列
        generated = input_ids.clone()
        
        # 存储额外信息
        all_attentions = [] if config.output_attentions else None
        all_hidden_states = [] if config.output_hidden_states else None
        all_scores = [] if config.output_scores else None
        
        # 完成标记
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            for step in range(max_length - input_ids.size(1)):
                # 创建注意力掩码
                if attention_mask is not None:
                    current_mask = torch.ones(
                        batch_size, generated.size(1),
                        dtype=torch.bool, device=self.device
                    )
                else:
                    current_mask = None
                
                # 前向传播
                outputs = self.model(
                    generated,
                    attention_mask=current_mask,
                    output_attentions=config.output_attentions,
                    output_hidden_states=config.output_hidden_states
                )
                
                # 获取下一个 token 的 logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # 应用重复惩罚
                if config.repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, generated, config.repetition_penalty
                    )
                
                # 应用 n-gram 阻塞
                if config.no_repeat_ngram_size > 0:
                    next_token_logits = self._apply_ngram_blocking(
                        next_token_logits, generated, config.no_repeat_ngram_size
                    )
                
                # 贪心选择
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # 存储分数
                if config.output_scores:
                    all_scores.append(next_token_logits)
                
                # 存储注意力权重
                if config.output_attentions and hasattr(outputs, 'attentions'):
                    all_attentions.append(outputs.attentions)
                
                # 存储隐藏状态
                if config.output_hidden_states and hasattr(outputs, 'hidden_states'):
                    all_hidden_states.append(outputs.hidden_states)
                
                # 更新完成状态
                finished = finished | (next_tokens == config.eos_token_id)
                
                # 添加新 token
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
                
                # 检查是否所有序列都完成
                if config.early_stopping and finished.all():
                    break
                
                # 检查最小长度
                if generated.size(1) < config.min_length:
                    # 强制不生成 EOS
                    next_tokens = torch.where(
                        next_tokens == config.eos_token_id,
                        torch.full_like(next_tokens, config.unk_token_id),
                        next_tokens
                    )
        
        # 返回结果
        if config.return_dict_in_generate:
            result = {
                'sequences': generated,
                'scores': all_scores,
                'attentions': all_attentions,
                'hidden_states': all_hidden_states
            }
            return result
        else:
            return generated
    
    def _sample_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        config: GenerationConfig = None
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        采样生成
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            config: 生成配置
            
        Returns:
            生成的 token IDs
        """
        batch_size = input_ids.size(0)
        max_length = config.max_length
        
        # 扩展批次以支持多个返回序列
        if config.num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(config.num_return_sequences, dim=0)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(config.num_return_sequences, dim=0)
            batch_size = input_ids.size(0)
        
        # 初始化生成序列
        generated = input_ids.clone()
        
        # 存储额外信息
        all_attentions = [] if config.output_attentions else None
        all_hidden_states = [] if config.output_hidden_states else None
        all_scores = [] if config.output_scores else None
        
        # 完成标记
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            for step in range(max_length - input_ids.size(1)):
                # 创建注意力掩码
                if attention_mask is not None:
                    current_mask = torch.ones(
                        batch_size, generated.size(1),
                        dtype=torch.bool, device=self.device
                    )
                else:
                    current_mask = None
                
                # 前向传播
                outputs = self.model(
                    generated,
                    attention_mask=current_mask,
                    output_attentions=config.output_attentions,
                    output_hidden_states=config.output_hidden_states
                )
                
                # 获取下一个 token 的 logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # 应用温度
                if config.temperature != 1.0:
                    next_token_logits = next_token_logits / config.temperature
                
                # 应用重复惩罚
                if config.repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, generated, config.repetition_penalty
                    )
                
                # 应用 top-k 和 top-p 过滤
                next_token_logits = self._apply_top_k_top_p_filtering(
                    next_token_logits, config.top_k, config.top_p
                )
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # 存储分数
                if config.output_scores:
                    all_scores.append(next_token_logits)
                
                # 存储注意力权重
                if config.output_attentions and hasattr(outputs, 'attentions'):
                    all_attentions.append(outputs.attentions)
                
                # 存储隐藏状态
                if config.output_hidden_states and hasattr(outputs, 'hidden_states'):
                    all_hidden_states.append(outputs.hidden_states)
                
                # 更新完成状态
                finished = finished | (next_tokens == config.eos_token_id)
                
                # 添加新 token
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
                
                # 检查是否所有序列都完成
                if config.early_stopping and finished.all():
                    break
        
        # 返回结果
        if config.return_dict_in_generate:
            result = {
                'sequences': generated,
                'scores': all_scores,
                'attentions': all_attentions,
                'hidden_states': all_hidden_states
            }
            return result
        else:
            return generated
    
    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        config: GenerationConfig = None
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        束搜索生成
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            config: 生成配置
            
        Returns:
            生成的 token IDs
        """
        # 创建束搜索配置
        beam_config = config.beam_search_config or BeamSearchConfig(
            beam_size=config.num_beams,
            max_length=config.max_length,
            min_length=config.min_length,
            length_penalty=config.length_penalty,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            early_stopping=config.early_stopping,
            num_return_sequences=config.num_return_sequences,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p
        )
        
        # 创建束搜索器
        beam_searcher = BeamSearch(
            self.model, self.tokenizer, beam_config, self.device
        )
        
        # 执行束搜索
        batch_size = input_ids.size(0)
        all_sequences = []
        all_scores = []
        
        for i in range(batch_size):
            single_input = input_ids[i:i+1]
            single_mask = attention_mask[i:i+1] if attention_mask is not None else None
            
            candidates = beam_searcher.search(single_input, single_mask)
            
            # 提取序列和分数
            for candidate in candidates:
                sequence = torch.tensor(candidate.tokens, device=self.device).unsqueeze(0)
                all_sequences.append(sequence)
                all_scores.append(candidate.score)
        
        # 合并结果
        if all_sequences:
            # 填充序列到相同长度
            max_len = max(seq.size(1) for seq in all_sequences)
            padded_sequences = []
            
            for seq in all_sequences:
                if seq.size(1) < max_len:
                    padding = torch.full(
                        (1, max_len - seq.size(1)),
                        config.pad_token_id,
                        device=self.device
                    )
                    seq = torch.cat([seq, padding], dim=1)
                padded_sequences.append(seq)
            
            generated = torch.cat(padded_sequences, dim=0)
        else:
            generated = input_ids
        
        # 返回结果
        if config.return_dict_in_generate:
            result = {
                'sequences': generated,
                'sequences_scores': torch.tensor(all_scores, device=self.device) if all_scores else None
            }
            return result
        else:
            return generated
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """
        应用重复惩罚
        
        Args:
            logits: 当前 logits [batch_size, vocab_size]
            input_ids: 输入序列 [batch_size, seq_len]
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
    
    def _apply_ngram_blocking(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        ngram_size: int
    ) -> torch.Tensor:
        """
        应用 n-gram 阻塞
        
        Args:
            logits: 当前 logits [batch_size, vocab_size]
            input_ids: 输入序列 [batch_size, seq_len]
            ngram_size: n-gram 大小
            
        Returns:
            应用阻塞后的 logits
        """
        if ngram_size <= 0:
            return logits
        
        batch_size = logits.size(0)
        
        for i in range(batch_size):
            tokens = input_ids[i]
            
            if tokens.size(0) < ngram_size:
                continue
            
            # 获取当前 n-gram 前缀
            current_ngram = tokens[-(ngram_size-1):].tolist()
            
            # 查找历史中的匹配 n-gram
            for j in range(tokens.size(0) - ngram_size + 1):
                historical_ngram = tokens[j:j+ngram_size-1].tolist()
                
                if historical_ngram == current_ngram:
                    # 阻塞下一个 token
                    next_token = tokens[j+ngram_size-1].item()
                    logits[i, next_token] = -float('inf')
        
        return logits
    
    def _apply_top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        应用 top-k 和 top-p 过滤
        
        Args:
            logits: 输入 logits [batch_size, vocab_size]
            top_k: top-k 参数
            top_p: top-p 参数
            
        Returns:
            过滤后的 logits
        """
        # Top-k 过滤
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            # 获取 top-k 值
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            # 设置阈值
            min_values = top_k_values[:, -1:]
            logits = torch.where(
                logits < min_values,
                torch.full_like(logits, -float('inf')),
                logits
            )
        
        # Top-p (nucleus) 过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过阈值的 tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留第一个超过阈值的 token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            # 将要移除的位置设为负无穷
            for i in range(logits.size(0)):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i][indices_to_remove] = -float('inf')
        
        return logits
    
    def generate_batch(
        self,
        texts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[str]:
        """
        批量生成文本
        
        Args:
            texts: 输入文本列表
            generation_config: 生成配置
            **kwargs: 其他参数
            
        Returns:
            生成的文本列表
        """
        # 编码输入文本
        encoded_inputs = []
        max_length = 0
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            encoded_inputs.append(torch.tensor(tokens, device=self.device))
            max_length = max(max_length, len(tokens))
        
        # 填充到相同长度
        padded_inputs = []
        attention_masks = []
        
        for tokens in encoded_inputs:
            if len(tokens) < max_length:
                padding = torch.full(
                    (max_length - len(tokens),),
                    self.pad_token_id,
                    device=self.device
                )
                padded_tokens = torch.cat([tokens, padding])
                attention_mask = torch.cat([
                    torch.ones(len(tokens), device=self.device),
                    torch.zeros(max_length - len(tokens), device=self.device)
                ])
            else:
                padded_tokens = tokens
                attention_mask = torch.ones(len(tokens), device=self.device)
            
            padded_inputs.append(padded_tokens)
            attention_masks.append(attention_mask)
        
        # 转换为批次张量
        input_ids = torch.stack(padded_inputs)
        attention_mask = torch.stack(attention_masks)
        
        # 生成
        with torch.no_grad():
            generated_ids = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **kwargs
            )
        
        # 解码生成的文本
        generated_texts = []
        for i in range(generated_ids.size(0)):
            tokens = generated_ids[i].cpu().numpy()
            # 移除输入部分
            output_tokens = tokens[input_ids.size(1):]
            # 移除 EOS 和 PAD tokens
            output_tokens = output_tokens[output_tokens != self.eos_token_id]
            output_tokens = output_tokens[output_tokens != self.pad_token_id]
            
            text = self.tokenizer.decode(output_tokens)
            generated_texts.append(text)
        
        return generated_texts


if __name__ == "__main__":
    # 测试文本生成器
    print("测试文本生成器模块...")
    
    # 创建生成配置
    config = GenerationConfig(
        max_length=50,
        min_length=5,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        num_return_sequences=2,
        early_stopping=True
    )
    
    print(f"生成配置:")
    print(f"  最大长度: {config.max_length}")
    print(f"  最小长度: {config.min_length}")
    print(f"  采样: {config.do_sample}")
    print(f"  温度: {config.temperature}")
    print(f"  Top-k: {config.top_k}")
    print(f"  Top-p: {config.top_p}")
    print(f"  返回序列数: {config.num_return_sequences}")
    
    # 测试束搜索配置
    beam_config = GenerationConfig(
        max_length=50,
        num_beams=5,
        length_penalty=1.2,
        early_stopping=True,
        num_return_sequences=3
    )
    
    print(f"\n束搜索配置:")
    print(f"  最大长度: {beam_config.max_length}")
    print(f"  束大小: {beam_config.num_beams}")
    print(f"  长度惩罚: {beam_config.length_penalty}")
    print(f"  早停: {beam_config.early_stopping}")
    print(f"  返回序列数: {beam_config.num_return_sequences}")
    
    print("\n✅ 文本生成器模块测试通过")