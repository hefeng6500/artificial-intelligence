"""损失函数模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    
    Args:
        smoothing: 平滑参数，通常在 0.1 左右
        ignore_index: 忽略的索引，通常是 padding token
        reduction: 损失聚合方式
    """
    
    def __init__(
        self, 
        smoothing: float = 0.1, 
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input: 预测 logits [batch_size, seq_len, vocab_size] 或 [batch_size * seq_len, vocab_size]
            target: 目标标签 [batch_size, seq_len] 或 [batch_size * seq_len]
            
        Returns:
            损失值
        """
        # 重塑输入
        if input.dim() == 3:
            batch_size, seq_len, vocab_size = input.shape
            input = input.view(-1, vocab_size)
            target = target.view(-1)
        else:
            vocab_size = input.size(-1)
        
        # 创建掩码
        mask = (target != self.ignore_index)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        
        # 计算 log softmax
        log_probs = F.log_softmax(input, dim=-1)
        
        # 获取有效的预测和目标
        valid_log_probs = log_probs[mask]
        valid_targets = target[mask]
        
        # 计算标签平滑损失
        nll_loss = -valid_log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -valid_log_probs.mean(dim=-1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss 用于处理类别不平衡问题
    
    Args:
        alpha: 类别权重
        gamma: 聚焦参数
        ignore_index: 忽略的索引
        reduction: 损失聚合方式
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input: 预测 logits
            target: 目标标签
            
        Returns:
            损失值
        """
        # 重塑输入
        if input.dim() == 3:
            input = input.view(-1, input.size(-1))
            target = target.view(-1)
        
        # 创建掩码
        mask = (target != self.ignore_index)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        
        # 获取有效的预测和目标
        valid_input = input[mask]
        valid_target = target[mask]
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(valid_input, valid_target, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 应用 alpha 权重
        if self.alpha is not None:
            if self.alpha.device != valid_target.device:
                self.alpha = self.alpha.to(valid_target.device)
            alpha_t = self.alpha[valid_target]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    对比损失，用于句子级别的表示学习
    
    Args:
        temperature: 温度参数
        margin: 边界参数
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self, 
        embeddings1: torch.Tensor, 
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            embeddings1: 第一组嵌入 [batch_size, hidden_size]
            embeddings2: 第二组嵌入 [batch_size, hidden_size]
            labels: 标签，1 表示相似，0 表示不相似
            
        Returns:
            损失值
        """
        # 计算余弦相似度
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        similarity = torch.sum(embeddings1 * embeddings2, dim=1)
        
        # 计算对比损失
        positive_loss = labels * torch.pow(1 - similarity, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(similarity - self.margin, min=0), 2)
        
        loss = positive_loss + negative_loss
        
        return loss.mean()


class KLDivergenceLoss(nn.Module):
    """
    KL 散度损失，用于知识蒸馏
    
    Args:
        temperature: 温度参数
        reduction: 损失聚合方式
    """
    
    def __init__(self, temperature: float = 4.0, reduction: str = 'batchmean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            student_logits: 学生模型的 logits
            teacher_logits: 教师模型的 logits
            
        Returns:
            KL 散度损失
        """
        # 应用温度缩放
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # 计算 KL 散度
        kl_loss = F.kl_div(
            student_probs, 
            teacher_probs, 
            reduction=self.reduction
        )
        
        # 缩放损失
        return kl_loss * (self.temperature ** 2)


class MultiTaskLoss(nn.Module):
    """
    多任务损失，结合多个损失函数
    
    Args:
        loss_weights: 各个损失的权重
        loss_functions: 损失函数列表
    """
    
    def __init__(
        self, 
        loss_weights: dict, 
        loss_functions: dict
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.loss_functions = nn.ModuleDict(loss_functions)
    
    def forward(self, predictions: dict, targets: dict) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            predictions: 预测结果字典
            targets: 目标字典
            
        Returns:
            总损失和各个损失的字典
        """
        total_loss = 0
        loss_dict = {}
        
        for loss_name, loss_fn in self.loss_functions.items():
            if loss_name in predictions and loss_name in targets:
                loss_value = loss_fn(predictions[loss_name], targets[loss_name])
                weight = self.loss_weights.get(loss_name, 1.0)
                
                weighted_loss = weight * loss_value
                total_loss += weighted_loss
                
                loss_dict[f'{loss_name}_loss'] = loss_value.item()
                loss_dict[f'{loss_name}_weighted_loss'] = weighted_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


def create_loss_function(
    loss_type: str = 'cross_entropy',
    **kwargs
) -> nn.Module:
    """
    创建损失函数
    
    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数
        
    Returns:
        损失函数
    """
    if loss_type == 'cross_entropy':
        ignore_index = kwargs.get('ignore_index', -100)
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        ignore_index = kwargs.get('ignore_index', -100)
        return LabelSmoothingCrossEntropy(
            smoothing=smoothing,
            ignore_index=ignore_index
        )
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', None)
        gamma = kwargs.get('gamma', 2.0)
        ignore_index = kwargs.get('ignore_index', -100)
        return FocalLoss(
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index
        )
    
    elif loss_type == 'contrastive':
        temperature = kwargs.get('temperature', 0.07)
        margin = kwargs.get('margin', 0.5)
        return ContrastiveLoss(
            temperature=temperature,
            margin=margin
        )
    
    elif loss_type == 'kl_divergence':
        temperature = kwargs.get('temperature', 4.0)
        return KLDivergenceLoss(temperature=temperature)
    
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")


class AdaptiveLoss(nn.Module):
    """
    自适应损失，根据训练进度调整损失权重
    
    Args:
        base_loss: 基础损失函数
        adaptive_weight: 是否使用自适应权重
    """
    
    def __init__(
        self, 
        base_loss: nn.Module,
        adaptive_weight: bool = True
    ):
        super().__init__()
        self.base_loss = base_loss
        self.adaptive_weight = adaptive_weight
        self.step_count = 0
        self.loss_history = []
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input: 预测结果
            target: 目标标签
            
        Returns:
            损失值
        """
        loss = self.base_loss(input, target)
        
        if self.adaptive_weight and self.training:
            self.step_count += 1
            self.loss_history.append(loss.item())
            
            # 保持历史记录在合理范围内
            if len(self.loss_history) > 1000:
                self.loss_history = self.loss_history[-1000:]
            
            # 计算自适应权重
            if len(self.loss_history) > 10:
                recent_avg = sum(self.loss_history[-10:]) / 10
                overall_avg = sum(self.loss_history) / len(self.loss_history)
                
                # 如果最近的损失比整体平均高，增加权重
                if recent_avg > overall_avg:
                    adaptive_factor = min(2.0, recent_avg / overall_avg)
                    loss = loss * adaptive_factor
        
        return loss


if __name__ == "__main__":
    # 测试损失函数
    print("测试损失函数...")
    
    # 创建虚拟数据
    batch_size, seq_len, vocab_size = 4, 10, 1000
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 添加一些 padding tokens
    targets[:, -2:] = -100  # 最后两个位置是 padding
    
    print(f"输入形状: {logits.shape}")
    print(f"目标形状: {targets.shape}")
    
    # 测试标签平滑交叉熵
    print("\n测试标签平滑交叉熵:")
    label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss1 = label_smooth_loss(logits, targets)
    print(f"标签平滑损失: {loss1.item():.4f}")
    
    # 测试普通交叉熵对比
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    loss2 = ce_loss(logits.view(-1, vocab_size), targets.view(-1))
    print(f"普通交叉熵损失: {loss2.item():.4f}")
    
    # 测试 Focal Loss
    print("\n测试 Focal Loss:")
    focal_loss = FocalLoss(gamma=2.0)
    loss3 = focal_loss(logits, targets)
    print(f"Focal 损失: {loss3.item():.4f}")
    
    # 测试对比损失
    print("\n测试对比损失:")
    embeddings1 = torch.randn(batch_size, 512)
    embeddings2 = torch.randn(batch_size, 512)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    contrastive_loss = ContrastiveLoss()
    loss4 = contrastive_loss(embeddings1, embeddings2, labels)
    print(f"对比损失: {loss4.item():.4f}")
    
    # 测试 KL 散度损失
    print("\n测试 KL 散度损失:")
    student_logits = torch.randn(batch_size, vocab_size)
    teacher_logits = torch.randn(batch_size, vocab_size)
    
    kl_loss = KLDivergenceLoss(temperature=4.0)
    loss5 = kl_loss(student_logits, teacher_logits)
    print(f"KL 散度损失: {loss5.item():.4f}")
    
    # 测试损失函数创建
    print("\n测试损失函数创建:")
    loss_fn = create_loss_function('label_smoothing', smoothing=0.1)
    loss6 = loss_fn(logits, targets)
    print(f"创建的标签平滑损失: {loss6.item():.4f}")
    
    print("\n✅ 损失函数测试通过")