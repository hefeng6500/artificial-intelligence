import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_model import Transformer
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体以解决可视化中文显示问题
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 如果上述字体不可用，尝试其他方案
try:
    plt.rcParams['font.family'] = ['SimHei']
except:
    try:
        plt.rcParams['font.family'] = ['Microsoft YaHei']
    except:
        print("警告：无法设置中文字体，可能会出现中文显示问题")
        print("建议安装 SimHei 或 Microsoft YaHei 字体")

class AttentionVisualizer:
    """
    注意力权重可视化工具
    
    提供多种方式来可视化和分析 Transformer 模型的注意力权重，
    帮助理解模型如何关注输入序列的不同部分。
    """
    
    def __init__(self, model, device='cpu'):
        """
        初始化可视化工具
        
        Args:
            model: Transformer 模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_attention_weights(self, src, tgt=None, src_mask=None, tgt_mask=None):
        """
        提取模型的注意力权重
        
        Args:
            src: 源序列
            tgt: 目标序列（可选）
            src_mask: 源掩码
            tgt_mask: 目标掩码
        
        Returns:
            encoder_attention: 编码器注意力权重
            decoder_attention: 解码器注意力权重（如果提供目标序列）
        """
        with torch.no_grad():
            if tgt is not None:
                # 完整的编码-解码过程
                _, encoder_attention, decoder_attention = self.model(src, tgt, src_mask, tgt_mask)
                return encoder_attention, decoder_attention
            else:
                # 仅编码过程
                _, encoder_attention = self.model.encoder(src, src_mask)
                return encoder_attention, None
    
    def plot_attention_heatmap(self, attention_weights, tokens, layer=0, head=0, 
                              title="注意力权重热力图", figsize=(10, 8)):
        """
        绘制注意力权重热力图
        
        Args:
            attention_weights: 注意力权重张量
            tokens: 词汇列表
            layer: 层索引
            head: 注意力头索引
            title: 图表标题
            figsize: 图表大小
        """
        # 提取指定层和头的注意力权重
        if isinstance(attention_weights, list):
            attn = attention_weights[layer][0, head].cpu().numpy()
        else:
            attn = attention_weights[0, head].cpu().numpy()
        
        plt.figure(figsize=figsize)
        
        # 创建自定义颜色映射
        colors = ['white', 'lightblue', 'blue', 'darkblue']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # 绘制热力图
        im = plt.imshow(attn, cmap=cmap, aspect='auto')
        
        # 添加颜色条
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('注意力权重', rotation=270, labelpad=20)
        
        # 设置标签
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)
        
        # 添加数值标注
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                text = plt.text(j, i, f'{attn[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.title(f'{title} - 层 {layer}, 头 {head}')
        plt.xlabel('Key 位置')
        plt.ylabel('Query 位置')
        plt.tight_layout()
        plt.show()
    
    def plot_multi_head_attention(self, attention_weights, tokens, layer=0, figsize=(15, 10)):
        """
        绘制多头注意力权重
        
        Args:
            attention_weights: 注意力权重张量
            tokens: 词汇列表
            layer: 层索引
            figsize: 图表大小
        """
        if isinstance(attention_weights, list):
            attn_layer = attention_weights[layer][0]  # [num_heads, seq_len, seq_len]
        else:
            attn_layer = attention_weights[0]  # [num_heads, seq_len, seq_len]
        
        num_heads = attn_layer.size(0)
        
        # 计算子图布局
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for head in range(num_heads):
            row = head // cols
            col = head % cols
            
            attn = attn_layer[head].cpu().numpy()
            
            im = axes[row, col].imshow(attn, cmap='Blues', aspect='auto')
            axes[row, col].set_title(f'头 {head}')
            axes[row, col].set_xticks(range(len(tokens)))
            axes[row, col].set_yticks(range(len(tokens)))
            axes[row, col].set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            axes[row, col].set_yticklabels(tokens, fontsize=8)
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        # 隐藏多余的子图
        for head in range(num_heads, rows * cols):
            row = head // cols
            col = head % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'多头注意力权重 - 层 {layer}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_attention_across_layers(self, attention_weights, tokens, head=0, figsize=(15, 10)):
        """
        绘制跨层的注意力权重变化
        
        Args:
            attention_weights: 注意力权重列表
            tokens: 词汇列表
            head: 注意力头索引
            figsize: 图表大小
        """
        num_layers = len(attention_weights)
        
        # 计算子图布局
        cols = 3
        rows = (num_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for layer in range(num_layers):
            row = layer // cols
            col = layer % cols
            
            attn = attention_weights[layer][0, head].cpu().numpy()
            
            im = axes[row, col].imshow(attn, cmap='Blues', aspect='auto')
            axes[row, col].set_title(f'层 {layer}')
            axes[row, col].set_xticks(range(len(tokens)))
            axes[row, col].set_yticks(range(len(tokens)))
            axes[row, col].set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            axes[row, col].set_yticklabels(tokens, fontsize=8)
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        # 隐藏多余的子图
        for layer in range(num_layers, rows * cols):
            row = layer // cols
            col = layer % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'跨层注意力权重变化 - 头 {head}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_attention_statistics(self, attention_weights, tokens, figsize=(15, 5)):
        """
        绘制注意力权重统计信息
        
        Args:
            attention_weights: 注意力权重列表
            tokens: 词汇列表
            figsize: 图表大小
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 收集所有注意力权重
        all_weights = []
        layer_means = []
        layer_stds = []
        
        for layer_attn in attention_weights:
            weights = layer_attn[0].cpu().numpy()  # [num_heads, seq_len, seq_len]
            all_weights.extend(weights.flatten())
            layer_means.append(weights.mean())
            layer_stds.append(weights.std())
        
        # 1. 注意力权重分布直方图
        axes[0].hist(all_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('注意力权重分布')
        axes[0].set_xlabel('注意力权重值')
        axes[0].set_ylabel('频次')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 各层平均注意力权重
        layers = range(len(attention_weights))
        axes[1].plot(layers, layer_means, 'o-', color='red', linewidth=2, markersize=6)
        axes[1].fill_between(layers, 
                           [m - s for m, s in zip(layer_means, layer_stds)],
                           [m + s for m, s in zip(layer_means, layer_stds)],
                           alpha=0.3, color='red')
        axes[1].set_title('各层注意力权重统计')
        axes[1].set_xlabel('层索引')
        axes[1].set_ylabel('平均注意力权重')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 注意力集中度（熵）
        entropies = []
        for layer_attn in attention_weights:
            layer_entropy = []
            weights = layer_attn[0].cpu().numpy()  # [num_heads, seq_len, seq_len]
            for head in range(weights.shape[0]):
                for i in range(weights.shape[1]):
                    # 计算每行的熵
                    row = weights[head, i, :]
                    entropy = -np.sum(row * np.log(row + 1e-10))
                    layer_entropy.append(entropy)
            entropies.append(np.mean(layer_entropy))
        
        axes[2].plot(layers, entropies, 's-', color='green', linewidth=2, markersize=6)
        axes[2].set_title('注意力集中度（熵）')
        axes[2].set_xlabel('层索引')
        axes[2].set_ylabel('平均熵')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_attention_patterns(self, attention_weights, tokens):
        """
        分析注意力模式
        
        Args:
            attention_weights: 注意力权重列表
            tokens: 词汇列表
        
        Returns:
            analysis: 分析结果字典
        """
        analysis = {
            'num_layers': len(attention_weights),
            'num_heads': attention_weights[0].size(1),
            'seq_len': attention_weights[0].size(2),
            'patterns': []
        }
        
        print("=== 注意力模式分析 ===")
        print(f"层数: {analysis['num_layers']}")
        print(f"注意力头数: {analysis['num_heads']}")
        print(f"序列长度: {analysis['seq_len']}")
        print()
        
        for layer_idx, layer_attn in enumerate(attention_weights):
            layer_patterns = []
            weights = layer_attn[0].cpu().numpy()  # [num_heads, seq_len, seq_len]
            
            print(f"层 {layer_idx}:")
            
            for head_idx in range(weights.shape[0]):
                head_weights = weights[head_idx]
                
                # 分析对角线注意力（自注意力）
                diagonal_attention = np.mean(np.diag(head_weights))
                
                # 分析局部注意力（相邻位置）
                local_attention = 0
                count = 0
                for i in range(head_weights.shape[0]):
                    for j in range(max(0, i-1), min(head_weights.shape[1], i+2)):
                        if i != j:
                            local_attention += head_weights[i, j]
                            count += 1
                local_attention = local_attention / count if count > 0 else 0
                
                # 分析全局注意力（远距离位置）
                global_attention = 0
                count = 0
                for i in range(head_weights.shape[0]):
                    for j in range(head_weights.shape[1]):
                        if abs(i - j) > 2:
                            global_attention += head_weights[i, j]
                            count += 1
                global_attention = global_attention / count if count > 0 else 0
                
                pattern = {
                    'head': head_idx,
                    'diagonal_attention': diagonal_attention,
                    'local_attention': local_attention,
                    'global_attention': global_attention
                }
                layer_patterns.append(pattern)
                
                print(f"  头 {head_idx}: 对角线={diagonal_attention:.3f}, "
                      f"局部={local_attention:.3f}, 全局={global_attention:.3f}")
            
            analysis['patterns'].append(layer_patterns)
            print()
        
        return analysis
    
    def create_attention_flow_diagram(self, attention_weights, tokens, layer=0, head=0, 
                                    threshold=0.1, figsize=(12, 8)):
        """
        创建注意力流向图
        
        Args:
            attention_weights: 注意力权重
            tokens: 词汇列表
            layer: 层索引
            head: 头索引
            threshold: 显示连接的最小权重阈值
            figsize: 图表大小
        """
        if isinstance(attention_weights, list):
            attn = attention_weights[layer][0, head].cpu().numpy()
        else:
            attn = attention_weights[0, head].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置词汇位置
        n_tokens = len(tokens)
        positions = [(i, 0) for i in range(n_tokens)]  # 水平排列
        
        # 绘制词汇节点
        for i, (x, y) in enumerate(positions):
            circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, tokens[i], ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 绘制注意力连接
        for i in range(n_tokens):
            for j in range(n_tokens):
                if attn[i, j] > threshold:
                    # 计算箭头位置
                    start_x, start_y = positions[j]
                    end_x, end_y = positions[i]
                    
                    # 调整箭头位置以避免重叠
                    if i != j:
                        # 计算弧形路径
                        mid_y = 1.0 if i > j else -1.0
                        
                        # 绘制弧形箭头
                        ax.annotate('', xy=(end_x, end_y + 0.3), xytext=(start_x, start_y + 0.3),
                                  arrowprops=dict(arrowstyle='->', 
                                                connectionstyle=f"arc3,rad={mid_y * 0.3}",
                                                color='red', 
                                                alpha=attn[i, j],
                                                linewidth=attn[i, j] * 5))
                        
                        # 添加权重标签
                        mid_x = (start_x + end_x) / 2
                        ax.text(mid_x, mid_y * 0.5, f'{attn[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-1, n_tokens)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'注意力流向图 - 层 {layer}, 头 {head}\n(阈值: {threshold})', fontsize=14)
        
        plt.tight_layout()
        plt.show()

def demo_attention_visualization():
    """
    注意力可视化演示
    """
    print("=== 注意力可视化演示 ===")
    
    # 创建简单的模型和数据
    vocab_size = 50
    d_model = 128
    num_heads = 4
    num_layers = 3
    
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=512,
        max_len=50
    )
    
    # 创建示例输入
    src = torch.tensor([[1, 5, 10, 15, 20, 25]])
    tokens = [f'token_{i}' for i in src.squeeze().numpy()]
    
    print(f"输入序列: {src.squeeze().numpy()}")
    print(f"词汇标记: {tokens}")
    
    # 创建可视化工具
    visualizer = AttentionVisualizer(model)
    
    # 提取注意力权重
    encoder_attention, _ = visualizer.extract_attention_weights(src)
    
    print(f"\n提取到 {len(encoder_attention)} 层的注意力权重")
    
    # 1. 绘制单个注意力头的热力图
    print("\n1. 绘制注意力热力图...")
    visualizer.plot_attention_heatmap(encoder_attention, tokens, layer=0, head=0)
    
    # 2. 绘制多头注意力
    print("\n2. 绘制多头注意力...")
    visualizer.plot_multi_head_attention(encoder_attention, tokens, layer=0)
    
    # 3. 绘制跨层注意力变化
    print("\n3. 绘制跨层注意力变化...")
    visualizer.plot_attention_across_layers(encoder_attention, tokens, head=0)
    
    # 4. 绘制注意力统计信息
    print("\n4. 绘制注意力统计信息...")
    visualizer.plot_attention_statistics(encoder_attention, tokens)
    
    # 5. 分析注意力模式
    print("\n5. 分析注意力模式...")
    analysis = visualizer.analyze_attention_patterns(encoder_attention, tokens)
    
    # 6. 创建注意力流向图
    print("\n6. 创建注意力流向图...")
    visualizer.create_attention_flow_diagram(encoder_attention, tokens, 
                                           layer=0, head=0, threshold=0.1)
    
    print("\n注意力可视化演示完成！")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行演示
    demo_attention_visualization()