"""优化器和学习率调度器模块"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, 
    ReduceLROnPlateau, CyclicLR, OneCycleLR
)
from typing import Dict, Any, Optional, Union
import math
import warnings


class WarmupScheduler:
    """
    预热学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_steps: 预热步数
        base_scheduler: 基础调度器
        warmup_factor: 预热因子
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        warmup_factor: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.warmup_factor = warmup_factor
        self.step_count = 0
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """执行一步学习率调度"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # 预热阶段
            warmup_factor = self.warmup_factor + (1.0 - self.warmup_factor) * self.step_count / self.warmup_steps
            
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # 正常调度阶段
            if self.base_scheduler is not None:
                self.base_scheduler.step()
    
    def get_last_lr(self):
        """获取最后的学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """获取调度器状态字典"""
        return {
            'first_cycle_steps': self.first_cycle_steps,
            'cycle_mult': self.cycle_mult,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'gamma': self.gamma,
            'cur_cycle_steps': self.cur_cycle_steps,
            'cycle': self.cycle,
            'step_in_cycle': self.step_in_cycle,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态字典"""
        self.first_cycle_steps = state_dict['first_cycle_steps']
        self.cycle_mult = state_dict['cycle_mult']
        self.max_lr = state_dict['max_lr']
        self.min_lr = state_dict['min_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.gamma = state_dict['gamma']
        self.cur_cycle_steps = state_dict['cur_cycle_steps']
        self.cycle = state_dict['cycle']
        self.step_in_cycle = state_dict['step_in_cycle']
        self.step_count = state_dict['step_count']
    
    def state_dict(self):
        """获取调度器状态字典"""
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor,
            'step_count': self.step_count,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态字典"""
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
        self.factor = state_dict['factor']
        self.step_count = state_dict['step_count']
        self.base_lrs = state_dict['base_lrs']


class TransformerScheduler:
    """
    Transformer 原论文中的学习率调度器
    
    Args:
        optimizer: 优化器
        d_model: 模型维度
        warmup_steps: 预热步数
        factor: 缩放因子
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_count = 0
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """执行一步学习率调度"""
        self.step_count += 1
        
        # 计算学习率
        lr = self.factor * (self.d_model ** -0.5) * min(
            self.step_count ** -0.5,
            self.step_count * (self.warmup_steps ** -1.5)
        )
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """获取最后的学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """获取调度器状态字典"""
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor,
            'step_count': self.step_count,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态字典"""
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
        self.factor = state_dict['factor']
        self.step_count = state_dict['step_count']
        self.base_lrs = state_dict['base_lrs']


class CosineAnnealingWarmupRestarts:
    """
    带预热的余弦退火重启调度器
    
    Args:
        optimizer: 优化器
        first_cycle_steps: 第一个周期的步数
        cycle_mult: 周期倍数
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_steps: 预热步数
        gamma: 衰减因子
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0
    ):
        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0
        self.step_count = 0
    
    def step(self):
        """执行一步学习率调度"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # 预热阶段
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            # 余弦退火阶段
            self.step_in_cycle += 1
            
            if self.step_in_cycle > self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 1
                self.cur_cycle_steps = int(self.first_cycle_steps * (self.cycle_mult ** self.cycle))
            
            base_lr = self.min_lr + (self.max_lr - self.min_lr) * \
                     (1 + math.cos(math.pi * self.step_in_cycle / self.cur_cycle_steps)) / 2
            
            lr = base_lr * (self.gamma ** self.cycle)
        
        # 更新学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """获取最后的学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """获取调度器状态字典"""
        return {
            'first_cycle_steps': self.first_cycle_steps,
            'cycle_mult': self.cycle_mult,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'gamma': self.gamma,
            'cur_cycle_steps': self.cur_cycle_steps,
            'cycle': self.cycle,
            'step_in_cycle': self.step_in_cycle,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态字典"""
        self.first_cycle_steps = state_dict['first_cycle_steps']
        self.cycle_mult = state_dict['cycle_mult']
        self.max_lr = state_dict['max_lr']
        self.min_lr = state_dict['min_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.gamma = state_dict['gamma']
        self.cur_cycle_steps = state_dict['cur_cycle_steps']
        self.cycle = state_dict['cycle']
        self.step_in_cycle = state_dict['step_in_cycle']
        self.step_count = state_dict['step_count']


class AdamWWithDecay(optim.AdamW):
    """
    带权重衰减的 AdamW 优化器
    
    Args:
        params: 参数
        lr: 学习率
        betas: Adam 的 beta 参数
        eps: 数值稳定性参数
        weight_decay: 权重衰减
        amsgrad: 是否使用 AMSGrad
        exclude_from_weight_decay: 排除权重衰减的参数名
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        exclude_from_weight_decay: Optional[list] = None
    ):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.exclude_from_weight_decay = exclude_from_weight_decay or []
    
    def step(self, closure=None):
        """执行优化步骤"""
        # 临时保存权重衰减设置
        original_weight_decays = []
        
        for group in self.param_groups:
            original_weight_decays.append(group['weight_decay'])
            
            # 检查是否需要排除权重衰减
            for param_name in self.exclude_from_weight_decay:
                if any(param_name in name for name, _ in group.get('params_names', [])):
                    group['weight_decay'] = 0.0
                    break
        
        # 执行优化步骤
        loss = super().step(closure)
        
        # 恢复权重衰减设置
        for group, original_wd in zip(self.param_groups, original_weight_decays):
            group['weight_decay'] = original_wd
        
        return loss


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    **kwargs
) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型
        learning_rate: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他参数
        
    Returns:
        优化器
    """
    # 获取模型参数
    params = model.parameters()
    
    # 分组参数（可选）
    if kwargs.get('use_param_groups', False):
        # 为不同类型的参数设置不同的学习率和权重衰减
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        param_groups = [
            {
                'params': [
                    p for n, p in model.named_parameters() 
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': weight_decay,
                'lr': learning_rate
            },
            {
                'params': [
                    p for n, p in model.named_parameters() 
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': 0.0,
                'lr': learning_rate
            }
        ]
        params = param_groups
    
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            params,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            params,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    
    elif optimizer_type.lower() == 'adamw_custom':
        return AdamWWithDecay(
            params,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay,
            exclude_from_weight_decay=kwargs.get('exclude_from_weight_decay', [])
        )
    
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(
            params,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get('nesterov', True)
        )
    
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(
            params,
            lr=learning_rate,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.0)
        )
    
    elif optimizer_type.lower() == 'adagrad':
        return optim.Adagrad(
            params,
            lr=learning_rate,
            lr_decay=kwargs.get('lr_decay', 0),
            weight_decay=weight_decay,
            eps=kwargs.get('eps', 1e-10)
        )
    
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    **kwargs
) -> Union[torch.optim.lr_scheduler._LRScheduler, object]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        **kwargs: 其他参数
        
    Returns:
        学习率调度器
    """
    if scheduler_type.lower() == 'step':
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type.lower() == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    
    elif scheduler_type.lower() == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    
    elif scheduler_type.lower() == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 0)
        )
    
    elif scheduler_type.lower() == 'cyclic':
        return CyclicLR(
            optimizer,
            base_lr=kwargs.get('base_lr', 1e-5),
            max_lr=kwargs.get('max_lr', 1e-3),
            step_size_up=kwargs.get('step_size_up', 2000),
            mode=kwargs.get('mode', 'triangular')
        )
    
    elif scheduler_type.lower() == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=kwargs.get('total_steps', 1000),
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos')
        )
    
    elif scheduler_type.lower() == 'warmup':
        base_scheduler = None
        if 'base_scheduler_type' in kwargs:
            base_scheduler = create_scheduler(
                optimizer,
                kwargs['base_scheduler_type'],
                **kwargs.get('base_scheduler_kwargs', {})
            )
        
        return WarmupScheduler(
            optimizer,
            warmup_steps=kwargs.get('warmup_steps', 1000),
            base_scheduler=base_scheduler,
            warmup_factor=kwargs.get('warmup_factor', 0.1)
        )
    
    elif scheduler_type.lower() == 'transformer':
        return TransformerScheduler(
            optimizer,
            d_model=kwargs.get('d_model', 512),
            warmup_steps=kwargs.get('warmup_steps', 4000),
            factor=kwargs.get('factor', 1.0)
        )
    
    elif scheduler_type.lower() == 'cosine_warmup_restarts':
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=kwargs.get('first_cycle_steps', 1000),
            cycle_mult=kwargs.get('cycle_mult', 1.0),
            max_lr=kwargs.get('max_lr', 1e-3),
            min_lr=kwargs.get('min_lr', 1e-5),
            warmup_steps=kwargs.get('warmup_steps', 100),
            gamma=kwargs.get('gamma', 1.0)
        )
    
    elif scheduler_type.lower() == 'none':
        return None
    
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")


class GradientClipper:
    """
    梯度裁剪器
    
    Args:
        max_norm: 最大梯度范数
        norm_type: 范数类型
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        裁剪梯度
        
        Args:
            model: 模型
            
        Returns:
            梯度范数
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # 计算梯度范数
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), self.norm_type) for p in parameters]),
            self.norm_type
        )
        
        # 裁剪梯度
        if total_norm > self.max_norm:
            clip_coef = self.max_norm / (total_norm + 1e-6)
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        
        return total_norm.item()


if __name__ == "__main__":
    # 测试优化器和调度器
    print("测试优化器和调度器...")
    
    # 创建虚拟模型
    model = nn.Linear(100, 10)
    
    # 测试优化器创建
    print("\n测试优化器创建:")
    optimizer = create_optimizer(
        model, 
        optimizer_type='adamw',
        learning_rate=1e-4,
        weight_decay=1e-2
    )
    print(f"优化器类型: {type(optimizer).__name__}")
    print(f"学习率: {optimizer.param_groups[0]['lr']}")
    print(f"权重衰减: {optimizer.param_groups[0]['weight_decay']}")
    
    # 测试调度器创建
    print("\n测试调度器创建:")
    scheduler = create_scheduler(
        optimizer,
        scheduler_type='cosine',
        T_max=100
    )
    print(f"调度器类型: {type(scheduler).__name__}")
    
    # 测试 Transformer 调度器
    print("\n测试 Transformer 调度器:")
    transformer_scheduler = create_scheduler(
        optimizer,
        scheduler_type='transformer',
        d_model=512,
        warmup_steps=4000
    )
    print(f"Transformer 调度器类型: {type(transformer_scheduler).__name__}")
    
    # 测试预热调度器
    print("\n测试预热调度器:")
    warmup_scheduler = create_scheduler(
        optimizer,
        scheduler_type='warmup',
        warmup_steps=1000,
        base_scheduler_type='cosine',
        base_scheduler_kwargs={'T_max': 100}
    )
    print(f"预热调度器类型: {type(warmup_scheduler).__name__}")
    
    # 测试梯度裁剪
    print("\n测试梯度裁剪:")
    clipper = GradientClipper(max_norm=1.0)
    
    # 创建虚拟梯度
    dummy_input = torch.randn(32, 100)
    dummy_target = torch.randn(32, 10)
    
    output = model(dummy_input)
    loss = nn.MSELoss()(output, dummy_target)
    loss.backward()
    
    grad_norm = clipper.clip_gradients(model)
    print(f"梯度范数: {grad_norm:.4f}")
    
    # 测试学习率变化
    print("\n测试学习率变化:")
    initial_lr = optimizer.param_groups[0]['lr']
    print(f"初始学习率: {initial_lr:.6f}")
    
    # 执行几步调度
    for step in range(5):
        if hasattr(scheduler, 'step'):
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"步骤 {step+1} 学习率: {current_lr:.6f}")
    
    print("\n✅ 优化器和调度器测试通过")