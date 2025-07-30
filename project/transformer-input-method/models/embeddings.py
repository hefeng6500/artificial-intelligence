"""嵌入层模块"""

import math
import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """词汇嵌入层"""
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = 0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(
            vocab_size, 
            d_model, 
            padding_idx=padding_idx
        )
        
        # 初始化嵌入权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化嵌入权重"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入标记序列 [batch_size, seq_len]
            
        Returns:
            嵌入向量 [batch_size, seq_len, d_model]
        """
        # 缩放嵌入向量（Transformer 论文中的做法）
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """可学习的位置嵌入"""
    
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的位置嵌入
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # 初始化位置嵌入
        self._init_weights()
    
    def _init_weights(self):
        """初始化位置嵌入权重"""
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置嵌入后的张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # 创建位置索引
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # 添加位置嵌入
        pos_emb = self.position_embedding(positions)
        
        return self.dropout(x + pos_emb)


class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置嵌入（固定的，不可学习）"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建正弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        pos_encoding = self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
        
        return self.dropout(x + pos_encoding)


class SegmentEmbedding(nn.Module):
    """段嵌入（用于区分不同的句子或段落）"""
    
    def __init__(self, num_segments: int, d_model: int):
        super().__init__()
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        
        # 初始化段嵌入
        self._init_weights()
    
    def _init_weights(self):
        """初始化段嵌入权重"""
        nn.init.normal_(self.segment_embedding.weight, mean=0, std=0.02)
    
    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            segment_ids: 段 ID 序列 [batch_size, seq_len]
            
        Returns:
            段嵌入向量 [batch_size, seq_len, d_model]
        """
        return self.segment_embedding(segment_ids)


class TransformerEmbedding(nn.Module):
    """Transformer 完整嵌入层（词汇 + 位置 + 段）"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        num_segments: int = 2,
        padding_idx: Optional[int] = 0,
        dropout: float = 0.1,
        use_segment_embedding: bool = False,
        position_embedding_type: str = 'sinusoidal'  # 'sinusoidal' or 'learnable'
    ):
        super().__init__()
        
        # 词汇嵌入
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        
        # 位置嵌入
        if position_embedding_type == 'sinusoidal':
            self.position_embedding = SinusoidalPositionalEmbedding(d_model, max_len, dropout)
        elif position_embedding_type == 'learnable':
            self.position_embedding = PositionalEmbedding(max_len, d_model, dropout)
        else:
            raise ValueError(f"Unknown position embedding type: {position_embedding_type}")
        
        # 段嵌入（可选）
        self.use_segment_embedding = use_segment_embedding
        if use_segment_embedding:
            self.segment_embedding = SegmentEmbedding(num_segments, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入标记序列 [batch_size, seq_len]
            segment_ids: 段 ID 序列 [batch_size, seq_len]（可选）
            
        Returns:
            嵌入向量 [batch_size, seq_len, d_model]
        """
        # 词汇嵌入
        token_emb = self.token_embedding(input_ids)
        
        # 位置嵌入
        embeddings = self.position_embedding(token_emb)
        
        # 段嵌入（如果使用）
        if self.use_segment_embedding and segment_ids is not None:
            segment_emb = self.segment_embedding(segment_ids)
            embeddings = embeddings + segment_emb
        
        # 层归一化和 dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class AdaptiveEmbedding(nn.Module):
    """自适应嵌入（用于处理大词汇表）"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        cutoffs: list,
        div_val: float = 1.0,
        padding_idx: Optional[int] = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.cutoffs = cutoffs + [vocab_size]
        self.div_val = div_val
        
        self.embeddings = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        for i in range(len(self.cutoffs)):
            if i == 0:
                # 第一个簇使用完整维度
                emb_dim = d_model
                start_idx = 0
            else:
                # 后续簇使用递减维度
                emb_dim = d_model // (div_val ** i)
                start_idx = self.cutoffs[i-1]
            
            end_idx = self.cutoffs[i]
            cluster_size = end_idx - start_idx
            
            # 创建嵌入层
            embedding = nn.Embedding(cluster_size, emb_dim, padding_idx=padding_idx if start_idx == 0 else None)
            self.embeddings.append(embedding)
            
            # 创建投影层（如果需要）
            if emb_dim != d_model:
                projection = nn.Linear(emb_dim, d_model, bias=False)
                self.projections.append(projection)
            else:
                self.projections.append(None)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入标记序列 [batch_size, seq_len]
            
        Returns:
            嵌入向量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.size()
        output = torch.zeros(batch_size, seq_len, self.d_model, device=input_ids.device)
        
        for i, (embedding, projection) in enumerate(zip(self.embeddings, self.projections)):
            if i == 0:
                start_idx = 0
            else:
                start_idx = self.cutoffs[i-1]
            
            end_idx = self.cutoffs[i]
            
            # 找到属于当前簇的标记
            mask = (input_ids >= start_idx) & (input_ids < end_idx)
            
            if mask.any():
                # 调整索引
                cluster_ids = input_ids[mask] - start_idx
                
                # 获取嵌入
                cluster_emb = embedding(cluster_ids)
                
                # 投影到目标维度
                if projection is not None:
                    cluster_emb = projection(cluster_emb)
                
                # 填充到输出张量
                output[mask] = cluster_emb
        
        return output * math.sqrt(self.d_model)


if __name__ == "__main__":
    # 测试词汇嵌入
    vocab_size = 10000
    d_model = 512
    seq_len = 20
    batch_size = 2
    
    # 创建测试数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 测试基础嵌入
    token_emb = TokenEmbedding(vocab_size, d_model)
    output = token_emb(input_ids)
    print(f"词汇嵌入输出形状: {output.shape}")
    
    # 测试完整嵌入
    full_emb = TransformerEmbedding(vocab_size, d_model)
    output = full_emb(input_ids)
    print(f"完整嵌入输出形状: {output.shape}")
    
    # 测试自适应嵌入
    cutoffs = [1000, 5000]
    adaptive_emb = AdaptiveEmbedding(vocab_size, d_model, cutoffs)
    output = adaptive_emb(input_ids)
    print(f"自适应嵌入输出形状: {output.shape}")
    
    print("✅ 嵌入层测试通过")