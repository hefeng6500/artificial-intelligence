"""可视化模块"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class AttentionVisualizer:
    """注意力权重可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        source_tokens: List[str],
        target_tokens: List[str],
        layer_idx: int = 0,
        head_idx: int = 0,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        绘制注意力权重热力图
        
        Args:
            attention_weights: 注意力权重 [num_layers, num_heads, seq_len, seq_len]
            source_tokens: 源语言 tokens
            target_tokens: 目标语言 tokens
            layer_idx: 层索引
            head_idx: 头索引
            save_path: 保存路径
            title: 图表标题
        """
        # 提取指定层和头的注意力权重
        if attention_weights.dim() == 4:
            weights = attention_weights[layer_idx, head_idx].cpu().numpy()
        elif attention_weights.dim() == 3:
            weights = attention_weights[head_idx].cpu().numpy()
        else:
            weights = attention_weights.cpu().numpy()
        
        # 截取有效长度
        weights = weights[:len(target_tokens), :len(source_tokens)]
        
        plt.figure(figsize=self.figsize)
        
        # 创建热力图
        sns.heatmap(
            weights,
            xticklabels=source_tokens,
            yticklabels=target_tokens,
            cmap='Blues',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Attention Weight'}
        )
        
        if title is None:
            title = f'Attention Weights (Layer {layer_idx}, Head {head_idx})'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Source Tokens', fontsize=12)
        plt.ylabel('Target Tokens', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力权重图已保存到: {save_path}")
        
        plt.show()
    
    def plot_multi_head_attention(
        self,
        attention_weights: torch.Tensor,
        source_tokens: List[str],
        target_tokens: List[str],
        layer_idx: int = 0,
        max_heads: int = 8,
        save_path: Optional[str] = None
    ):
        """
        绘制多头注意力权重
        
        Args:
            attention_weights: 注意力权重
            source_tokens: 源语言 tokens
            target_tokens: 目标语言 tokens
            layer_idx: 层索引
            max_heads: 最大显示头数
            save_path: 保存路径
        """
        if attention_weights.dim() == 4:
            layer_weights = attention_weights[layer_idx]
        else:
            layer_weights = attention_weights
        
        num_heads = min(layer_weights.size(0), max_heads)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for head_idx in range(num_heads):
            weights = layer_weights[head_idx].cpu().numpy()
            weights = weights[:len(target_tokens), :len(source_tokens)]
            
            sns.heatmap(
                weights,
                xticklabels=source_tokens,
                yticklabels=target_tokens,
                cmap='Blues',
                ax=axes[head_idx],
                cbar=True,
                annot=False
            )
            
            axes[head_idx].set_title(f'Head {head_idx}', fontweight='bold')
            axes[head_idx].set_xlabel('Source')
            axes[head_idx].set_ylabel('Target')
        
        # 隐藏多余的子图
        for idx in range(num_heads, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Multi-Head Attention (Layer {layer_idx})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"多头注意力图已保存到: {save_path}")
        
        plt.show()
    
    def plot_attention_interactive(
        self,
        attention_weights: torch.Tensor,
        source_tokens: List[str],
        target_tokens: List[str],
        layer_idx: int = 0,
        head_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        创建交互式注意力权重图
        
        Args:
            attention_weights: 注意力权重
            source_tokens: 源语言 tokens
            target_tokens: 目标语言 tokens
            layer_idx: 层索引
            head_idx: 头索引
            save_path: 保存路径
        """
        if attention_weights.dim() == 4:
            weights = attention_weights[layer_idx, head_idx].cpu().numpy()
        elif attention_weights.dim() == 3:
            weights = attention_weights[head_idx].cpu().numpy()
        else:
            weights = attention_weights.cpu().numpy()
        
        weights = weights[:len(target_tokens), :len(source_tokens)]
        
        fig = go.Figure(data=go.Heatmap(
            z=weights,
            x=source_tokens,
            y=target_tokens,
            colorscale='Blues',
            showscale=True,
            hoverongaps=False,
            hovertemplate='Source: %{x}<br>Target: %{y}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Interactive Attention Weights (Layer {layer_idx}, Head {head_idx})',
            xaxis_title='Source Tokens',
            yaxis_title='Target Tokens',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"交互式注意力图已保存到: {save_path}")
        
        fig.show()


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        train_metrics: Optional[Dict[str, List[float]]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制训练曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            train_metrics: 训练指标字典
            val_metrics: 验证指标字典
            save_path: 保存路径
        """
        # 计算子图数量
        num_plots = 1  # 损失图
        if train_metrics:
            num_plots += len(train_metrics)
        
        # 计算子图布局
        cols = min(3, num_plots)
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        if num_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # 绘制损失曲线
        epochs = range(1, len(train_losses) + 1)
        axes[plot_idx].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses:
            val_epochs = range(1, len(val_losses) + 1)
            axes[plot_idx].plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        axes[plot_idx].set_title('Training and Validation Loss', fontweight='bold')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
        
        # 绘制其他指标
        if train_metrics:
            for metric_name, train_values in train_metrics.items():
                if plot_idx >= len(axes):
                    break
                
                epochs = range(1, len(train_values) + 1)
                axes[plot_idx].plot(epochs, train_values, 'b-', 
                                  label=f'Training {metric_name}', linewidth=2)
                
                if val_metrics and metric_name in val_metrics:
                    val_values = val_metrics[metric_name]
                    val_epochs = range(1, len(val_values) + 1)
                    axes[plot_idx].plot(val_epochs, val_values, 'r-', 
                                      label=f'Validation {metric_name}', linewidth=2)
                
                axes[plot_idx].set_title(f'{metric_name.title()}', fontweight='bold')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel(metric_name.title())
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # 隐藏多余的子图
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线图已保存到: {save_path}")
        
        plt.show()
    
    def plot_learning_rate_schedule(
        self,
        learning_rates: List[float],
        save_path: Optional[str] = None
    ):
        """
        绘制学习率调度曲线
        
        Args:
            learning_rates: 学习率列表
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        steps = range(len(learning_rates))
        plt.plot(steps, learning_rates, 'g-', linewidth=2)
        
        plt.title('Learning Rate Schedule', fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"学习率调度图已保存到: {save_path}")
        
        plt.show()
    
    def plot_gradient_norms(
        self,
        gradient_norms: List[float],
        save_path: Optional[str] = None
    ):
        """
        绘制梯度范数曲线
        
        Args:
            gradient_norms: 梯度范数列表
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        steps = range(len(gradient_norms))
        plt.plot(steps, gradient_norms, 'purple', linewidth=1, alpha=0.7)
        
        # 添加移动平均线
        if len(gradient_norms) > 10:
            window_size = min(50, len(gradient_norms) // 10)
            moving_avg = pd.Series(gradient_norms).rolling(window=window_size).mean()
            plt.plot(steps, moving_avg, 'red', linewidth=2, label=f'Moving Average ({window_size})')
            plt.legend()
        
        plt.title('Gradient Norms During Training', fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"梯度范数图已保存到: {save_path}")
        
        plt.show()


class ModelVisualizer:
    """模型结构可视化器"""
    
    def __init__(self):
        pass
    
    def plot_model_architecture(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        save_path: Optional[str] = None
    ):
        """
        绘制模型架构图
        
        Args:
            model: PyTorch 模型
            input_shape: 输入形状
            save_path: 保存路径
        """
        try:
            from torchviz import make_dot
            
            # 创建虚拟输入
            dummy_input = torch.randn(input_shape)
            
            # 前向传播
            output = model(dummy_input)
            
            # 创建计算图
            dot = make_dot(output, params=dict(model.named_parameters()))
            
            if save_path:
                dot.render(save_path, format='png')
                print(f"模型架构图已保存到: {save_path}.png")
            
            return dot
            
        except ImportError:
            print("请安装 torchviz: pip install torchviz")
            return None
    
    def plot_parameter_distribution(
        self,
        model: torch.nn.Module,
        save_path: Optional[str] = None
    ):
        """
        绘制模型参数分布
        
        Args:
            model: PyTorch 模型
            save_path: 保存路径
        """
        # 收集所有参数
        all_params = []
        param_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                all_params.extend(param.data.cpu().numpy().flatten())
                param_names.append(name)
        
        plt.figure(figsize=(12, 6))
        
        # 绘制参数分布直方图
        plt.subplot(1, 2, 1)
        plt.hist(all_params, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Parameter Distribution', fontweight='bold')
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 绘制参数统计信息
        plt.subplot(1, 2, 2)
        param_stats = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu().numpy()
                param_stats.append([
                    np.mean(param_data),
                    np.std(param_data),
                    np.min(param_data),
                    np.max(param_data)
                ])
                layer_names.append(name.split('.')[-2] if '.' in name else name)
        
        param_stats = np.array(param_stats)
        
        x = np.arange(len(layer_names))
        width = 0.2
        
        plt.bar(x - 1.5*width, param_stats[:, 0], width, label='Mean', alpha=0.8)
        plt.bar(x - 0.5*width, param_stats[:, 1], width, label='Std', alpha=0.8)
        plt.bar(x + 0.5*width, param_stats[:, 2], width, label='Min', alpha=0.8)
        plt.bar(x + 1.5*width, param_stats[:, 3], width, label='Max', alpha=0.8)
        
        plt.title('Parameter Statistics by Layer', fontweight='bold')
        plt.xlabel('Layer')
        plt.ylabel('Value')
        plt.xticks(x, layer_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"参数分布图已保存到: {save_path}")
        
        plt.show()


def plot_translation_examples(
    source_texts: List[str],
    target_texts: List[str],
    predicted_texts: List[str],
    scores: Optional[List[float]] = None,
    max_examples: int = 5,
    save_path: Optional[str] = None
):
    """
    绘制翻译示例对比
    
    Args:
        source_texts: 源文本列表
        target_texts: 目标文本列表
        predicted_texts: 预测文本列表
        scores: 评分列表
        max_examples: 最大显示示例数
        save_path: 保存路径
    """
    num_examples = min(len(source_texts), max_examples)
    
    fig, ax = plt.subplots(figsize=(14, 2 * num_examples))
    
    # 创建表格数据
    table_data = []
    for i in range(num_examples):
        row = [
            f"Ex {i+1}",
            source_texts[i][:50] + "..." if len(source_texts[i]) > 50 else source_texts[i],
            target_texts[i][:50] + "..." if len(target_texts[i]) > 50 else target_texts[i],
            predicted_texts[i][:50] + "..." if len(predicted_texts[i]) > 50 else predicted_texts[i]
        ]
        
        if scores:
            row.append(f"{scores[i]:.3f}")
        
        table_data.append(row)
    
    # 设置列标题
    columns = ['Example', 'Source', 'Target', 'Predicted']
    if scores:
        columns.append('Score')
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='left',
        loc='center',
        colWidths=[0.1, 0.3, 0.3, 0.3] if not scores else [0.1, 0.25, 0.25, 0.25, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.axis('off')
    ax.set_title('Translation Examples Comparison', fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"翻译示例对比图已保存到: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # 测试可视化功能
    print("测试可视化模块...")
    
    # 创建虚拟数据
    train_losses = [3.5, 2.8, 2.3, 1.9, 1.6, 1.4, 1.2, 1.1, 1.0, 0.9]
    val_losses = [3.2, 2.6, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.0, 0.95]
    
    train_metrics = {
        'bleu': [0.1, 0.15, 0.22, 0.28, 0.35, 0.42, 0.48, 0.52, 0.56, 0.60],
        'accuracy': [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.78, 0.82, 0.85]
    }
    
    val_metrics = {
        'bleu': [0.08, 0.12, 0.18, 0.25, 0.32, 0.38, 0.44, 0.48, 0.52, 0.55],
        'accuracy': [0.25, 0.35, 0.45, 0.55, 0.6, 0.65, 0.7, 0.73, 0.77, 0.8]
    }
    
    # 测试训练曲线可视化
    print("\n测试训练曲线可视化:")
    training_viz = TrainingVisualizer()
    training_viz.plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_metrics=train_metrics,
        val_metrics=val_metrics
    )
    
    # 测试注意力可视化
    print("\n测试注意力可视化:")
    attention_viz = AttentionVisualizer()
    
    # 创建虚拟注意力权重
    attention_weights = torch.randn(1, 8, 10, 12)  # [layers, heads, target_len, source_len]
    source_tokens = ['hello', 'world', 'this', 'is', 'a', 'test', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
    target_tokens = ['你好', '世界', '这', '是', '一个', '测试', '<eos>', '<pad>', '<pad>', '<pad>']
    
    attention_viz.plot_attention_weights(
        attention_weights=attention_weights,
        source_tokens=source_tokens[:8],
        target_tokens=target_tokens[:6],
        layer_idx=0,
        head_idx=0
    )
    
    # 测试翻译示例对比
    print("\n测试翻译示例对比:")
    source_texts = [
        "Hello world",
        "This is a test",
        "Machine translation"
    ]
    
    target_texts = [
        "你好世界",
        "这是一个测试",
        "机器翻译"
    ]
    
    predicted_texts = [
        "你好世界",
        "这是测试",
        "机器学习"
    ]
    
    scores = [0.95, 0.82, 0.65]
    
    plot_translation_examples(
        source_texts=source_texts,
        target_texts=target_texts,
        predicted_texts=predicted_texts,
        scores=scores
    )
    
    print("\n✅ 可视化模块测试通过")