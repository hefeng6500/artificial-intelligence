import torch
import numpy as np
from transformer_model import (
    MultiHeadAttention, 
    PositionwiseFeedForward, 
    PositionalEncoding,
    TransformerBlock,
    TransformerEncoder,
    TransformerDecoder,
    Transformer
)

def test_multi_head_attention():
    """
    测试多头注意力机制
    """
    print("测试多头注意力机制...")
    
    d_model = 128
    num_heads = 8
    seq_len = 10
    batch_size = 2
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, attention = mha(x, x, x)
    
    # 检查输出形状
    assert output.shape == (batch_size, seq_len, d_model), f"输出形状错误: {output.shape}"
    assert attention.shape == (batch_size, num_heads, seq_len, seq_len), f"注意力形状错误: {attention.shape}"
    
    # 检查注意力权重是否归一化
    attention_sum = torch.sum(attention, dim=-1)
    assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-6), "注意力权重未正确归一化"
    
    print("✓ 多头注意力机制测试通过")

def test_positional_encoding():
    """
    测试位置编码
    """
    print("测试位置编码...")
    
    d_model = 128
    max_len = 100
    
    # 创建位置编码
    pe = PositionalEncoding(d_model, max_len)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 50
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 应用位置编码
    output = pe(x)
    
    # 检查输出形状
    assert output.shape == x.shape, f"位置编码后形状错误: {output.shape}"
    
    # 检查位置编码是否被正确添加
    assert not torch.equal(output, x), "位置编码未被添加"
    
    print("✓ 位置编码测试通过")

def test_feedforward():
    """
    测试前馈网络
    """
    print("测试前馈网络...")
    
    d_model = 128
    d_ff = 512
    batch_size = 2
    seq_len = 10
    
    # 创建前馈网络
    ff = PositionwiseFeedForward(d_model, d_ff)
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = ff(x)
    
    # 检查输出形状
    assert output.shape == x.shape, f"前馈网络输出形状错误: {output.shape}"
    
    print("✓ 前馈网络测试通过")

def test_transformer_block():
    """
    测试 Transformer 块
    """
    print("测试 Transformer 块...")
    
    d_model = 128
    num_heads = 8
    d_ff = 512
    batch_size = 2
    seq_len = 10
    
    # 创建 Transformer 块
    block = TransformerBlock(d_model, num_heads, d_ff)
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, attention = block(x)
    
    # 检查输出形状
    assert output.shape == x.shape, f"Transformer 块输出形状错误: {output.shape}"
    assert attention.shape == (batch_size, num_heads, seq_len, seq_len), f"注意力形状错误: {attention.shape}"
    
    print("✓ Transformer 块测试通过")

def test_transformer_encoder():
    """
    测试 Transformer 编码器
    """
    print("测试 Transformer 编码器...")
    
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_layers = 3
    d_ff = 512
    max_len = 100
    batch_size = 2
    seq_len = 10
    
    # 创建编码器
    encoder = TransformerEncoder(
        vocab_size, d_model, num_heads, num_layers, d_ff, max_len
    )
    
    # 创建测试输入
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    output, attention_weights = encoder(src)
    
    # 检查输出形状
    assert output.shape == (batch_size, seq_len, d_model), f"编码器输出形状错误: {output.shape}"
    assert len(attention_weights) == num_layers, f"注意力权重层数错误: {len(attention_weights)}"
    
    for i, attn in enumerate(attention_weights):
        assert attn.shape == (batch_size, num_heads, seq_len, seq_len), \
            f"第 {i} 层注意力形状错误: {attn.shape}"
    
    print("✓ Transformer 编码器测试通过")

def test_transformer_decoder():
    """
    测试 Transformer 解码器
    """
    print("测试 Transformer 解码器...")
    
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_layers = 3
    d_ff = 512
    max_len = 100
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # 创建解码器
    decoder = TransformerDecoder(
        vocab_size, d_model, num_heads, num_layers, d_ff, max_len
    )
    
    # 创建测试输入
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    encoder_output = torch.randn(batch_size, src_len, d_model)
    
    # 前向传播
    output, attention_weights = decoder(tgt, encoder_output)
    
    # 检查输出形状
    assert output.shape == (batch_size, tgt_len, vocab_size), f"解码器输出形状错误: {output.shape}"
    assert len(attention_weights) == num_layers, f"注意力权重层数错误: {len(attention_weights)}"
    
    print("✓ Transformer 解码器测试通过")

def test_full_transformer():
    """
    测试完整的 Transformer 模型
    """
    print("测试完整的 Transformer 模型...")
    
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 512
    max_len = 100
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # 创建完整模型
    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads,
        num_encoder_layers, num_decoder_layers, d_ff, max_len
    )
    
    # 创建测试输入
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # 前向传播
    output, encoder_attention, decoder_attention = model(src, tgt)
    
    # 检查输出形状
    assert output.shape == (batch_size, tgt_len, tgt_vocab_size), f"模型输出形状错误: {output.shape}"
    assert len(encoder_attention) == num_encoder_layers, f"编码器注意力层数错误: {len(encoder_attention)}"
    assert len(decoder_attention) == num_decoder_layers, f"解码器注意力层数错误: {len(decoder_attention)}"
    
    print("✓ 完整 Transformer 模型测试通过")

def test_model_parameters():
    """
    测试模型参数数量
    """
    print("测试模型参数数量...")
    
    # 创建小型模型
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        max_len=50
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    assert total_params > 0, "模型没有参数"
    assert trainable_params == total_params, "存在不可训练的参数"
    
    print("✓ 模型参数测试通过")

def test_gradient_flow():
    """
    测试梯度流动
    """
    print("测试梯度流动...")
    
    # 创建小型模型
    model = Transformer(
        src_vocab_size=50,
        tgt_vocab_size=50,
        d_model=32,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_len=20
    )
    
    # 创建测试数据
    src = torch.randint(0, 50, (1, 5))
    tgt = torch.randint(0, 50, (1, 5))
    target = torch.randint(0, 50, (1, 5))
    
    # 前向传播
    output, _, _ = model(src, tgt)
    
    # 计算损失
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output.view(-1, 50), target.view(-1))
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_gradient = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradient = True
            break
    
    assert has_gradient, "模型参数没有梯度"
    
    print("✓ 梯度流动测试通过")

def run_all_tests():
    """
    运行所有测试
    """
    print("=== Transformer 模型测试 ===")
    print()
    
    try:
        test_multi_head_attention()
        test_positional_encoding()
        test_feedforward()
        test_transformer_block()
        test_transformer_encoder()
        test_transformer_decoder()
        test_full_transformer()
        test_model_parameters()
        test_gradient_flow()
        
        print()
        print("🎉 所有测试通过！")
        print("Transformer 模型实现正确，可以正常使用。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行所有测试
    run_all_tests()