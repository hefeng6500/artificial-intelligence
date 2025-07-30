"""Transformer 模型实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .attention import MultiHeadAttention, create_padding_mask, create_combined_mask
from .embeddings import TransformerEmbedding


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            output: 输出张量
            attention_weights: 注意力权重
        """
        # 自注意力 + 残差连接 + 层归一化
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))
        
        return output, attention_weights


class DecoderLayer(nn.Module):
    """解码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 解码器输入 [batch_size, target_seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, source_seq_len, d_model]
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码
            
        Returns:
            output: 输出张量
            self_attention_weights: 自注意力权重
            cross_attention_weights: 交叉注意力权重
        """
        # 自注意力 + 残差连接 + 层归一化
        self_attn_output, self_attention_weights = self.self_attention(
            x, x, x, self_attn_mask
        )
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 交叉注意力 + 残差连接 + 层归一化
        cross_attn_output, cross_attention_weights = self.cross_attention(
            x, encoder_output, encoder_output, cross_attn_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        output = self.norm3(x + self.dropout(ff_output))
        
        return output, self_attention_weights, cross_attention_weights


class TransformerEncoder(nn.Module):
    """Transformer 编码器"""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            output: 编码器输出
            attention_weights: 所有层的注意力权重
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return self.norm(x), attention_weights


class TransformerDecoder(nn.Module):
    """Transformer 解码器"""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list, list]:
        """
        前向传播
        
        Args:
            x: 解码器输入
            encoder_output: 编码器输出
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码
            
        Returns:
            output: 解码器输出
            self_attention_weights: 自注意力权重
            cross_attention_weights: 交叉注意力权重
        """
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, self_attn_mask, cross_attn_mask
            )
            self_attention_weights.append(self_attn_weights)
            cross_attention_weights.append(cross_attn_weights)
        
        return self.norm(x), self_attention_weights, cross_attention_weights


class TransformerModel(nn.Module):
    """完整的 Transformer 模型"""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        
        # 嵌入层
        self.src_embedding = TransformerEmbedding(
            src_vocab_size, d_model, max_seq_len, padding_idx=pad_token_id, dropout=dropout
        )
        self.tgt_embedding = TransformerEmbedding(
            tgt_vocab_size, d_model, max_seq_len, padding_idx=pad_token_id, dropout=dropout
        )
        
        # 编码器和解码器
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, n_layers, d_ff, dropout)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            src_tokens: 源序列 [batch_size, src_seq_len]
            tgt_tokens: 目标序列 [batch_size, tgt_seq_len]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            包含输出和注意力权重的字典
        """
        # 创建掩码
        if src_mask is None:
            src_mask = create_padding_mask(src_tokens, self.pad_token_id)
        if tgt_mask is None:
            tgt_mask = create_combined_mask(tgt_tokens, self.pad_token_id)
        
        # 嵌入
        src_embedded = self.src_embedding(src_tokens)
        tgt_embedded = self.tgt_embedding(tgt_tokens)
        
        # 编码
        encoder_output, encoder_attention_weights = self.encoder(src_embedded, src_mask)
        
        # 解码
        decoder_output, decoder_self_attention_weights, decoder_cross_attention_weights = self.decoder(
            tgt_embedded, encoder_output, tgt_mask, src_mask
        )
        
        # 输出投影
        logits = self.output_projection(decoder_output)
        
        return {
            'logits': logits,
            'encoder_output': encoder_output,
            'decoder_output': decoder_output,
            'encoder_attention_weights': encoder_attention_weights,
            'decoder_self_attention_weights': decoder_self_attention_weights,
            'decoder_cross_attention_weights': decoder_cross_attention_weights
        }
    
    def encode(self, src_tokens: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码源序列"""
        if src_mask is None:
            src_mask = create_padding_mask(src_tokens, self.pad_token_id)
        
        src_embedded = self.src_embedding(src_tokens)
        encoder_output, _ = self.encoder(src_embedded, src_mask)
        
        return encoder_output
    
    def decode_step(
        self,
        tgt_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """解码单步"""
        if tgt_mask is None:
            tgt_mask = create_combined_mask(tgt_tokens, self.pad_token_id)
        
        tgt_embedded = self.tgt_embedding(tgt_tokens)
        decoder_output, _, _ = self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(
        self,
        src_tokens: torch.Tensor,
        max_length: int = 100,
        beam_size: int = 1,
        length_penalty: float = 1.0,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        生成翻译序列
        
        Args:
            src_tokens: 源序列 [batch_size, src_seq_len]
            max_length: 最大生成长度
            beam_size: 束搜索大小
            length_penalty: 长度惩罚
            temperature: 温度参数
            
        Returns:
            生成的序列 [batch_size, generated_seq_len]
        """
        self.eval()
        
        with torch.no_grad():
            if beam_size == 1:
                return self._greedy_search(src_tokens, max_length, temperature)
            else:
                return self._beam_search(src_tokens, max_length, beam_size, length_penalty)
    
    def _greedy_search(
        self, 
        src_tokens: torch.Tensor, 
        max_length: int, 
        temperature: float
    ) -> torch.Tensor:
        """贪心搜索"""
        batch_size = src_tokens.size(0)
        device = src_tokens.device
        
        # 编码源序列
        encoder_output = self.encode(src_tokens)
        src_mask = create_padding_mask(src_tokens, self.pad_token_id)
        
        # 初始化目标序列
        tgt_tokens = torch.full(
            (batch_size, 1), self.sos_token_id, dtype=torch.long, device=device
        )
        
        for _ in range(max_length - 1):
            # 解码当前序列
            logits = self.decode_step(tgt_tokens, encoder_output, src_mask=src_mask)
            
            # 获取最后一个时间步的 logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # 选择下一个 token
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到序列
            tgt_tokens = torch.cat([tgt_tokens, next_tokens], dim=1)
            
            # 检查是否所有序列都已结束
            if (next_tokens == self.eos_token_id).all():
                break
        
        return tgt_tokens
    
    def _beam_search(
        self, 
        src_tokens: torch.Tensor, 
        max_length: int, 
        beam_size: int, 
        length_penalty: float
    ) -> torch.Tensor:
        """束搜索（简化版本）"""
        # 这里实现一个简化的束搜索
        # 实际应用中可能需要更复杂的实现
        return self._greedy_search(src_tokens, max_length, 1.0)
    
    def get_model_size(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer_model(config) -> TransformerModel:
    """根据配置创建 Transformer 模型"""
    # 支持字典和对象两种配置方式
    if hasattr(config, '__dict__'):
        # 配置对象
        return TransformerModel(
            src_vocab_size=getattr(config, 'src_vocab_size', 10000),
            tgt_vocab_size=getattr(config, 'tgt_vocab_size', 8000),
            d_model=getattr(config, 'd_model', 512),
            n_heads=getattr(config, 'n_heads', 8),
            n_layers=getattr(config, 'n_layers', 6),
            d_ff=getattr(config, 'd_ff', 2048),
            max_seq_len=getattr(config, 'max_seq_length', 512),
            dropout=getattr(config, 'dropout', 0.1),
            pad_token_id=getattr(config, 'pad_token_id', 0),
            sos_token_id=getattr(config, 'sos_token_id', 1),
            eos_token_id=getattr(config, 'eos_token_id', 2)
        )
    else:
        # 字典配置
        return TransformerModel(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            d_model=config.get('d_model', 512),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 6),
            d_ff=config.get('d_ff', 2048),
            max_seq_len=config.get('max_seq_len', 512),
            dropout=config.get('dropout', 0.1),
            pad_token_id=config.get('pad_token_id', 0),
            sos_token_id=config.get('sos_token_id', 1),
            eos_token_id=config.get('eos_token_id', 2)
        )


if __name__ == "__main__":
    # 测试模型
    config = {
        'src_vocab_size': 10000,
        'tgt_vocab_size': 8000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 512,
        'dropout': 0.1
    }
    
    model = create_transformer_model(config)
    
    # 创建测试数据
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src_tokens = torch.randint(0, config['src_vocab_size'], (batch_size, src_seq_len))
    tgt_tokens = torch.randint(0, config['tgt_vocab_size'], (batch_size, tgt_seq_len))
    
    # 前向传播
    output = model(src_tokens, tgt_tokens)
    
    print(f"源序列形状: {src_tokens.shape}")
    print(f"目标序列形状: {tgt_tokens.shape}")
    print(f"输出 logits 形状: {output['logits'].shape}")
    print(f"模型参数数量: {model.get_model_size():,}")
    print(f"可训练参数数量: {model.get_trainable_parameters():,}")
    
    # 测试生成
    generated = model.generate(src_tokens, max_length=20)
    print(f"生成序列形状: {generated.shape}")
    
    print("✅ Transformer 模型测试通过")