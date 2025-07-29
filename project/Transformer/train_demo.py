import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from transformer_model import Transformer, visualize_attention
import time
import os

class SimpleTranslationDataset(Dataset):
    """
    简单的翻译数据集
    
    为了演示目的，我们创建一个简单的数字序列翻译任务：
    输入：[1, 2, 3, 4, 5]
    输出：[2, 3, 4, 5, 6] (每个数字加1)
    """
    
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=100):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # 生成数据
        self.data = []
        for _ in range(num_samples):
            # 生成随机序列
            src = torch.randint(1, vocab_size-1, (seq_len,))
            # 目标序列是源序列每个元素加1（简单的变换）
            tgt = (src + 1) % vocab_size
            # 添加特殊标记：0为PAD，vocab_size-1为EOS
            tgt = torch.cat([torch.tensor([0]), tgt[:-1]])  # 添加起始标记
            
            self.data.append((src, tgt))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_padding_mask(seq, pad_token=0):
    """
    创建填充掩码
    
    Args:
        seq: 输入序列 [batch_size, seq_len]
        pad_token: 填充标记
    
    Returns:
        mask: 掩码矩阵 [batch_size, 1, 1, seq_len]
    """
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    """
    创建前瞻掩码（下三角矩阵）
    
    Args:
        size: 序列长度
    
    Returns:
        mask: 前瞻掩码
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

def train_model():
    """
    训练 Transformer 模型
    """
    print("开始训练 Transformer 模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数
    vocab_size = 100
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    max_len = 50
    dropout = 0.1
    
    # 创建模型
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    ).to(device)
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集和数据加载器
    dataset = SimpleTranslationDataset(num_samples=5000, seq_len=10, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # 训练循环
    num_epochs = 20
    train_losses = []
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            
            # 创建目标输入和标签
            tgt_input = tgt[:, :-1]  # 解码器输入（去掉最后一个标记）
            tgt_output = tgt[:, 1:]  # 解码器目标（去掉第一个标记）
            
            # 创建掩码
            src_mask = create_padding_mask(src)
            tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output, _, _ = model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} 完成, 平均损失: {avg_loss:.4f}')
        
        # 更新学习率
        scheduler.step()
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', linewidth=2)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.grid(True)
    plt.show()
    
    # 创建 model 文件夹（如果不存在）
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型到 model 文件夹
    model_path = os.path.join(model_dir, 'transformer_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")
    
    return model, device

def inference_demo(model, device, vocab_size=100):
    """
    推理演示
    
    Args:
        model: 训练好的模型
        device: 设备
        vocab_size: 词汇表大小
    """
    print("\n开始推理演示...")
    
    model.eval()
    
    # 创建测试输入
    test_src = torch.tensor([[1, 5, 10, 15, 20, 25, 30, 35, 40, 45]]).to(device)
    print(f"输入序列: {test_src.squeeze().cpu().numpy()}")
    
    # 期望输出（每个数字加1）
    expected = test_src.squeeze().cpu().numpy() + 1
    print(f"期望输出: {expected}")
    
    # 推理过程
    with torch.no_grad():
        # 编码
        encoder_output, encoder_attention = model.encoder(test_src)
        
        # 解码（逐步生成）
        max_len = test_src.size(1)
        tgt_input = torch.zeros(1, 1, dtype=torch.long).to(device)  # 起始标记
        
        generated_sequence = []
        
        for i in range(max_len):
            # 创建掩码
            tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)
            
            # 解码
            decoder_output, decoder_attention = model.decoder(
                tgt_input, encoder_output, tgt_mask=tgt_mask
            )
            
            # 预测下一个词
            output = model.output_projection(decoder_output)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            generated_sequence.append(next_token.item())
            
            # 更新目标输入
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
        
        print(f"生成序列: {generated_sequence}")
        
        # 计算准确率
        correct = sum(1 for i, j in zip(generated_sequence, expected) if i == j)
        accuracy = correct / len(expected) * 100
        print(f"准确率: {accuracy:.2f}%")
        
        # 可视化注意力权重
        if len(encoder_attention) > 0:
            print("\n可视化编码器注意力权重...")
            
            # 创建词汇标签
            input_tokens = [f"token_{i}" for i in test_src.squeeze().cpu().numpy()]
            
            # 可视化第一层第一个头的注意力
            attn_weights = encoder_attention[0][0, 0].cpu().numpy()
            
            plt.figure(figsize=(10, 8))
            plt.imshow(attn_weights, cmap='Blues', aspect='auto')
            plt.colorbar()
            plt.title('编码器注意力权重 - 层 0, 头 0')
            plt.xlabel('Key 位置')
            plt.ylabel('Query 位置')
            
            # 添加数值标注
            for i in range(len(input_tokens)):
                for j in range(len(input_tokens)):
                    plt.text(j, i, f'{attn_weights[i, j]:.2f}', 
                            ha='center', va='center', fontsize=8)
            
            plt.xticks(range(len(input_tokens)), input_tokens, rotation=45)
            plt.yticks(range(len(input_tokens)), input_tokens)
            plt.tight_layout()
            plt.show()

def analyze_model_components(model):
    """
    分析模型组件
    
    Args:
        model: Transformer 模型
    """
    print("\n=== 模型组件分析 ===")
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 分析各组件参数
    print("\n各组件参数分布:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params:,} 参数 ({params/total_params*100:.1f}%)")
    
    # 分析编码器层
    print("\n编码器层分析:")
    for i, layer in enumerate(model.encoder.transformer_blocks):
        params = sum(p.numel() for p in layer.parameters())
        print(f"编码器层 {i}: {params:,} 参数")
    
    # 分析解码器层
    print("\n解码器层分析:")
    for i, layer in enumerate(model.decoder.decoder_layers):
        params = sum(p.numel() for p in layer.parameters())
        print(f"解码器层 {i}: {params:,} 参数")

def main():
    """
    主函数：运行完整的训练和推理流程
    """
    print("=== Transformer 模型演示 ===")
    print("这个演示将展示如何训练和使用 Transformer 模型")
    print("任务：学习将输入序列的每个数字加1")
    print("例如：[1, 2, 3, 4, 5] -> [2, 3, 4, 5, 6]")
    
    # 训练模型
    model, device = train_model()
    
    # 分析模型
    analyze_model_components(model)
    
    # 推理演示
    inference_demo(model, device)
    
    print("\n演示完成！")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()