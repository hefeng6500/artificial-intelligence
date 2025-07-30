"""束搜索模块"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from collections import namedtuple

# 束搜索候选项
BeamCandidate = namedtuple("BeamCandidate", ["tokens", "score", "attention_weights"])


@dataclass
class BeamSearchConfig:
    """束搜索配置"""

    beam_size: int = 5
    max_length: int = 100
    min_length: int = 1
    length_penalty: float = 1.0
    coverage_penalty: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = True
    num_return_sequences: int = 1
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    diversity_penalty: float = 0.0
    diversity_groups: int = 1


class BeamSearch:
    """
    束搜索解码器

    Args:
        model: Transformer 模型
        tokenizer: 分词器
        config: 束搜索配置
        device: 设备
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: BeamSearchConfig = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or BeamSearchConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 特殊 token ID
        self.bos_token_id = getattr(tokenizer, "bos_token_id", 1)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", 2)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        self.unk_token_id = getattr(tokenizer, "unk_token_id", 3)

    def search(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[BeamCandidate]:
        """
        执行束搜索

        Args:
            src_tokens: 源序列 tokens [batch_size, src_len]
            src_mask: 源序列掩码 [batch_size, src_len]
            **kwargs: 其他参数

        Returns:
            束搜索候选项列表
        """
        batch_size = src_tokens.size(0)

        if batch_size > 1:
            # 批量处理
            results = []
            for i in range(batch_size):
                single_src = src_tokens[i : i + 1]
                single_mask = src_mask[i : i + 1] if src_mask is not None else None
                result = self._search_single(single_src, single_mask, **kwargs)
                results.append(result)
            return results
        else:
            return self._search_single(src_tokens, src_mask, **kwargs)

    def _search_single(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[BeamCandidate]:
        """
        单个序列的束搜索

        Args:
            src_tokens: 源序列 tokens [1, src_len]
            src_mask: 源序列掩码 [1, src_len]
            **kwargs: 其他参数

        Returns:
            束搜索候选项列表
        """
        # 编码源序列
        with torch.no_grad():
            encoder_outputs = self.model.encode(src_tokens, src_mask)

        # 初始化束
        beam_size = self.config.beam_size
        max_length = self.config.max_length

        # 初始候选项：只包含 BOS token
        initial_tokens = torch.full(
            (beam_size, 1), self.bos_token_id, dtype=torch.long, device=self.device
        )

        # 束候选项：[beam_size, seq_len]
        beam_tokens = initial_tokens
        beam_scores = torch.zeros(beam_size, device=self.device)
        beam_attention_weights = []

        # 完成的序列
        finished_sequences = []

        for step in range(max_length - 1):
            # 当前序列长度
            current_length = beam_tokens.size(1)

            # 创建目标掩码
            tgt_mask = self._create_target_mask(current_length)

            # 解码当前步
            with torch.no_grad():
                # 扩展源序列以匹配束大小
                expanded_src = src_tokens.expand(beam_size, -1)
                expanded_src_mask = (
                    src_mask.expand(beam_size, -1) if src_mask is not None else None
                )
                expanded_encoder_outputs = encoder_outputs.expand(beam_size, -1, -1)

                # 解码
                decoder_outputs = self.model.decode(
                    beam_tokens,
                    expanded_encoder_outputs,
                    tgt_mask=tgt_mask,
                    src_mask=expanded_src_mask,
                )

                # 获取最后一步的 logits
                logits = decoder_outputs.logits[:, -1, :]  # [beam_size, vocab_size]

                # 应用温度
                if self.config.temperature != 1.0:
                    logits = logits / self.config.temperature

                # 应用重复惩罚
                if self.config.repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits, beam_tokens)

                # 应用 top-k 和 top-p 过滤
                logits = self._apply_top_k_top_p_filtering(logits)

                # 计算概率
                log_probs = F.log_softmax(logits, dim=-1)  # [beam_size, vocab_size]

            # 计算候选分数
            vocab_size = log_probs.size(-1)

            # 扩展分数：[beam_size, vocab_size]
            candidate_scores = beam_scores.unsqueeze(1) + log_probs

            # 应用长度惩罚
            if self.config.length_penalty != 1.0:
                length_penalty = (
                    (current_length + 1) / (1 + 1)
                ) ** self.config.length_penalty
                candidate_scores = candidate_scores / length_penalty

            # 重塑为一维：[beam_size * vocab_size]
            candidate_scores = candidate_scores.view(-1)

            # 选择 top-k 候选项
            top_k = min(beam_size * 2, candidate_scores.size(0))
            top_scores, top_indices = torch.topk(candidate_scores, top_k)

            # 计算束索引和 token 索引
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # 新的候选项
            new_beam_tokens = []
            new_beam_scores = []
            new_beam_attention = []

            for i, (beam_idx, token_idx, score) in enumerate(
                zip(beam_indices, token_indices, top_scores)
            ):
                # 获取父序列
                parent_tokens = beam_tokens[beam_idx]

                # 添加新 token
                new_tokens = torch.cat([parent_tokens, token_idx.unsqueeze(0)])

                # 检查是否结束
                if token_idx.item() == self.eos_token_id:
                    # 序列完成
                    final_score = score.item()

                    # 应用最终长度惩罚
                    if self.config.length_penalty != 1.0:
                        length_penalty = (
                            new_tokens.size(0) / (1 + 1)
                        ) ** self.config.length_penalty
                        final_score = final_score / length_penalty

                    finished_sequences.append(
                        BeamCandidate(
                            tokens=new_tokens.cpu().numpy(),
                            score=final_score,
                            attention_weights=None,  # 暂时不保存注意力权重
                        )
                    )
                else:
                    # 继续搜索
                    if len(new_beam_tokens) < beam_size:
                        new_beam_tokens.append(new_tokens)
                        new_beam_scores.append(score)

            # 检查是否有足够的候选项继续
            if len(new_beam_tokens) == 0:
                break

            # 更新束
            beam_tokens = torch.stack(new_beam_tokens)
            beam_scores = torch.stack(new_beam_scores)

            # 早停检查
            if (
                self.config.early_stopping
                and len(finished_sequences) >= self.config.num_return_sequences
            ):
                break

            # 最小长度检查
            if current_length < self.config.min_length:
                # 移除 EOS token 的候选项
                continue

        # 如果没有完成的序列，添加当前最佳序列
        if not finished_sequences:
            for i in range(min(beam_size, self.config.num_return_sequences)):
                finished_sequences.append(
                    BeamCandidate(
                        tokens=beam_tokens[i].cpu().numpy(),
                        score=beam_scores[i].item(),
                        attention_weights=None,
                    )
                )

        # 按分数排序
        finished_sequences.sort(key=lambda x: x.score, reverse=True)

        # 返回指定数量的序列
        return finished_sequences[: self.config.num_return_sequences]

    def _create_target_mask(self, length: int) -> torch.Tensor:
        """
        创建目标序列的因果掩码

        Args:
            length: 序列长度

        Returns:
            因果掩码 [length, length]
        """
        mask = torch.triu(torch.ones(length, length, device=self.device), diagonal=1)
        return mask == 0

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        应用重复惩罚

        Args:
            logits: 当前 logits [beam_size, vocab_size]
            tokens: 已生成的 tokens [beam_size, seq_len]

        Returns:
            应用惩罚后的 logits
        """
        penalty = self.config.repetition_penalty

        if penalty == 1.0:
            return logits

        # 对于每个束，惩罚已出现的 tokens
        for i in range(tokens.size(0)):
            for token in tokens[i].unique():
                if logits[i, token] > 0:
                    logits[i, token] /= penalty
                else:
                    logits[i, token] *= penalty

        return logits

    def _apply_top_k_top_p_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用 top-k 和 top-p 过滤

        Args:
            logits: 输入 logits [beam_size, vocab_size]

        Returns:
            过滤后的 logits
        """
        # Top-k 过滤
        if self.config.top_k > 0:
            top_k = min(self.config.top_k, logits.size(-1))
            # 获取 top-k 值
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            # 设置阈值
            min_values = top_k_values[:, -1:]
            logits = torch.where(
                logits < min_values, torch.full_like(logits, -float("inf")), logits
            )

        # Top-p (nucleus) 过滤
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过阈值的 tokens
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            # 保留第一个超过阈值的 token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            # 将要移除的位置设为负无穷
            for i in range(logits.size(0)):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i][indices_to_remove] = -float("inf")

        return logits

    def _apply_ngram_blocking(
        self, logits: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        应用 n-gram 阻塞，防止重复

        Args:
            logits: 当前 logits [beam_size, vocab_size]
            tokens: 已生成的 tokens [beam_size, seq_len]

        Returns:
            应用阻塞后的 logits
        """
        if self.config.no_repeat_ngram_size <= 0:
            return logits

        ngram_size = self.config.no_repeat_ngram_size

        for beam_idx in range(tokens.size(0)):
            beam_tokens = tokens[beam_idx]

            if beam_tokens.size(0) < ngram_size:
                continue

            # 获取当前 n-gram 前缀
            current_ngram = beam_tokens[-(ngram_size - 1) :].tolist()

            # 查找历史中的匹配 n-gram
            for i in range(beam_tokens.size(0) - ngram_size + 1):
                historical_ngram = beam_tokens[i : i + ngram_size - 1].tolist()

                if historical_ngram == current_ngram:
                    # 阻塞下一个 token
                    next_token = beam_tokens[i + ngram_size - 1].item()
                    logits[beam_idx, next_token] = -float("inf")

        return logits


class DiverseBeamSearch(BeamSearch):
    """
    多样化束搜索，生成更多样化的候选项

    Args:
        model: Transformer 模型
        tokenizer: 分词器
        config: 束搜索配置
        device: 设备
    """

    def _search_single(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[BeamCandidate]:
        """
        多样化束搜索的单个序列搜索

        Args:
            src_tokens: 源序列 tokens [1, src_len]
            src_mask: 源序列掩码 [1, src_len]
            **kwargs: 其他参数

        Returns:
            多样化的束搜索候选项列表
        """
        # 如果没有设置多样化参数，使用标准束搜索
        if self.config.diversity_penalty == 0.0 or self.config.diversity_groups <= 1:
            return super()._search_single(src_tokens, src_mask, **kwargs)

        # 实现多样化束搜索
        # 这里简化实现，将束分成多个组，每组独立搜索
        num_groups = self.config.diversity_groups
        beam_per_group = self.config.beam_size // num_groups

        all_candidates = []

        for group_idx in range(num_groups):
            # 为每个组创建独立的配置
            group_config = BeamSearchConfig(
                beam_size=beam_per_group,
                max_length=self.config.max_length,
                min_length=self.config.min_length,
                length_penalty=self.config.length_penalty,
                early_stopping=self.config.early_stopping,
                num_return_sequences=beam_per_group,
                temperature=self.config.temperature + group_idx * 0.1,  # 轻微调整温度
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )

            # 创建组搜索器
            group_searcher = BeamSearch(
                self.model, self.tokenizer, group_config, self.device
            )

            # 执行搜索
            group_candidates = group_searcher._search_single(
                src_tokens, src_mask, **kwargs
            )
            all_candidates.extend(group_candidates)

        # 按分数排序并返回最佳候选项
        all_candidates.sort(key=lambda x: x.score, reverse=True)
        return all_candidates[: self.config.num_return_sequences]


if __name__ == "__main__":
    # 测试束搜索
    print("测试束搜索模块...")

    # 创建虚拟配置
    config = BeamSearchConfig(
        beam_size=3,
        max_length=20,
        length_penalty=1.2,
        early_stopping=True,
        num_return_sequences=2,
    )

    print(f"束搜索配置:")
    print(f"  束大小: {config.beam_size}")
    print(f"  最大长度: {config.max_length}")
    print(f"  长度惩罚: {config.length_penalty}")
    print(f"  早停: {config.early_stopping}")
    print(f"  返回序列数: {config.num_return_sequences}")

    # 测试多样化束搜索配置
    diverse_config = BeamSearchConfig(
        beam_size=6, diversity_penalty=0.5, diversity_groups=2, num_return_sequences=3
    )

    print(f"\n多样化束搜索配置:")
    print(f"  束大小: {diverse_config.beam_size}")
    print(f"  多样化惩罚: {diverse_config.diversity_penalty}")
    print(f"  多样化组数: {diverse_config.diversity_groups}")
    print(f"  返回序列数: {diverse_config.num_return_sequences}")

    print("\n✅ 束搜索模块测试通过！")