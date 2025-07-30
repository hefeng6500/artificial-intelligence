"""训练器模块"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict
import wandb

# 导入项目模块
from models.transformer import TransformerModel
from utils.metrics import MetricsTracker, calculate_perplexity
from utils.visualization import TrainingVisualizer
from training.optimizer import GradientClipper
from training.loss import create_loss_function


@dataclass
class TrainingState:
    """训练状态"""
    epoch: int = 0
    step: int = 0
    best_loss: float = float('inf')
    best_metric: float = 0.0
    patience_counter: int = 0
    learning_rates: List[float] = None
    train_losses: List[float] = None
    val_losses: List[float] = None
    train_metrics: Dict[str, List[float]] = None
    val_metrics: Dict[str, List[float]] = None
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = []
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []
        if self.train_metrics is None:
            self.train_metrics = defaultdict(list)
        if self.val_metrics is None:
            self.val_metrics = defaultdict(list)
    
    def save(self, filepath: str):
        """保存训练状态"""
        # 手动创建状态字典
        state_dict = {
            'epoch': self.epoch,
            'step': self.step,
            'best_loss': self.best_loss,
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'learning_rates': self.learning_rates,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str):
        """加载训练状态"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
        
        # 转换 defaultdict
        if 'train_metrics' in state_dict:
            state_dict['train_metrics'] = defaultdict(list, state_dict['train_metrics'])
        if 'val_metrics' in state_dict:
            state_dict['val_metrics'] = defaultdict(list, state_dict['val_metrics'])
        
        return cls(**state_dict)


class Trainer:
    """
    Transformer 模型训练器
    
    Args:
        model: Transformer 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        loss_fn: 损失函数
        config: 训练配置
        device: 设备
    """
    
    def __init__(
        self,
        model: TransformerModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 配置
        self.config = config or {}
        self.max_epochs = self.config.get('max_epochs', 100)
        self.save_dir = Path(self.config.get('save_dir', './checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 损失函数
        if loss_fn is None:
            self.loss_fn = create_loss_function(
                loss_type=self.config.get('loss_type', 'label_smoothing'),
                smoothing=self.config.get('label_smoothing', 0.1),
                ignore_index=self.config.get('pad_token_id', 0)
            )
        else:
            self.loss_fn = loss_fn
        
        # 梯度裁剪
        self.gradient_clipper = GradientClipper(
            max_norm=self.config.get('max_grad_norm', 1.0)
        )
        
        # 评估指标
        self.metrics_tracker = MetricsTracker()
        
        # 可视化
        self.visualizer = TrainingVisualizer()
        
        # 训练状态
        self.state = TrainingState()
        
        # 日志记录
        self.setup_logging()
        
        # TensorBoard
        if self.config.get('use_tensorboard', True):
            self.writer = SummaryWriter(
                log_dir=self.config.get('tensorboard_dir', './runs')
            )
        else:
            self.writer = None
        
        # Weights & Biases
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'transformer-translation'),
                config=self.config
            )
        
        # 早停
        self.early_stopping = self.config.get('early_stopping', True)
        self.patience = self.config.get('patience', 10)
        self.min_delta = self.config.get('min_delta', 1e-4)
        
        # 混合精度训练
        self.use_amp = self.config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 累积梯度
        self.accumulation_steps = self.config.get('accumulation_steps', 1)
        
        # 验证频率
        self.val_frequency = self.config.get('val_frequency', 1)
        self.save_frequency = self.config.get('save_frequency', 5)
        
        # 回调函数
        self.callbacks = self.config.get('callbacks', [])
    
    def setup_logging(self):
        """设置日志记录"""
        log_dir = self.save_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        
        # 进度条
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch {self.state.epoch + 1}/{self.max_epochs}',
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            src_tokens = batch['src_ids'].to(self.device)
            tgt_tokens = batch['tgt_ids'].to(self.device)
            src_mask = batch.get('src_mask', None)
            tgt_mask = batch.get('tgt_mask', None)
            
            if src_mask is not None:
                src_mask = src_mask.to(self.device)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(self.device)
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        src_tokens, tgt_tokens[:, :-1],
                        src_mask=src_mask,
                        tgt_mask=tgt_mask[:, :-1, :-1] if tgt_mask is not None else None
                    )
                    # 确保logits和target的序列长度匹配
                    logits = outputs['logits']
                    target = tgt_tokens[:, 1:]
                    if logits.size(1) != target.size(1):
                        # 截断或填充logits以匹配target长度
                        min_len = min(logits.size(1), target.size(1))
                        logits = logits[:, :min_len, :]
                        target = target[:, :min_len]
                    # 重塑张量以适应CrossEntropyLoss
                    logits = logits.reshape(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
                    target = target.reshape(-1)  # [batch_size * seq_len]
                    loss = self.loss_fn(logits, target)
                    loss = loss / self.accumulation_steps
            else:
                outputs = self.model(
                    src_tokens, tgt_tokens[:, :-1],
                    src_mask=src_mask,
                    tgt_mask=tgt_mask[:, :-1, :-1] if tgt_mask is not None else None
                )
                # 调试输出形状
                print(f"Debug: logits shape: {outputs['logits'].shape}, target shape: {tgt_tokens[:, 1:].shape}")
                # 确保logits和target的序列长度匹配
                logits = outputs['logits']
                target = tgt_tokens[:, 1:]
                if logits.size(1) != target.size(1):
                    # 截断或填充logits以匹配target长度
                    min_len = min(logits.size(1), target.size(1))
                    logits = logits[:, :min_len, :]
                    target = target[:, :min_len]
                # 重塑张量以适应CrossEntropyLoss
                logits = logits.reshape(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
                target = target.reshape(-1)  # [batch_size * seq_len]
                loss = self.loss_fn(logits, target)
                loss = loss / self.accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 累积梯度
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # 梯度裁剪
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = self.gradient_clipper.clip_gradients(self.model)
                
                # 优化器步骤
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # 学习率调度
                if self.scheduler is not None:
                    if hasattr(self.scheduler, 'step'):
                        self.scheduler.step()
                
                # 记录学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                self.state.learning_rates.append(current_lr)
                
                # 更新步数
                self.state.step += 1
                
                # 记录到 TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Train/Loss', loss.item() * self.accumulation_steps, self.state.step)
                    self.writer.add_scalar('Train/LearningRate', current_lr, self.state.step)
                    self.writer.add_scalar('Train/GradientNorm', grad_norm, self.state.step)
                
                # 记录到 Weights & Biases
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'train/loss': loss.item() * self.accumulation_steps,
                        'train/learning_rate': current_lr,
                        'train/gradient_norm': grad_norm,
                        'step': self.state.step
                    })
            
            # 记录损失
            epoch_losses.append(loss.item() * self.accumulation_steps)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 执行回调
            for callback in self.callbacks:
                if hasattr(callback, 'on_batch_end'):
                    callback.on_batch_end(self.state.step, {
                        'loss': loss.item() * self.accumulation_steps,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        
        return {
            'loss': avg_loss,
            'perplexity': calculate_perplexity(avg_loss)
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个 epoch"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for batch in pbar:
                # 移动数据到设备
                src_tokens = batch['src_ids'].to(self.device)
                tgt_tokens = batch['tgt_ids'].to(self.device)
                src_mask = batch.get('src_mask', None)
                tgt_mask = batch.get('tgt_mask', None)
                
                if src_mask is not None:
                    src_mask = src_mask.to(self.device)
                if tgt_mask is not None:
                    tgt_mask = tgt_mask.to(self.device)
                
                # 前向传播
                outputs = self.model(
                    src_tokens, tgt_tokens[:, :-1],
                    src_mask=src_mask,
                    tgt_mask=tgt_mask[:, :-1, :-1] if tgt_mask is not None else None
                )
                
                # 计算损失
                # 确保logits和target的序列长度匹配
                logits = outputs['logits']
                target = tgt_tokens[:, 1:]
                if logits.size(1) != target.size(1):
                    # 截断或填充logits以匹配target长度
                    min_len = min(logits.size(1), target.size(1))
                    logits = logits[:, :min_len, :]
                    target = target[:, :min_len]
                # 重塑张量以适应CrossEntropyLoss
                logits = logits.reshape(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
                target = target.reshape(-1)  # [batch_size * seq_len]
                loss = self.loss_fn(logits, target)
                val_losses.append(loss.item())
                
                # 收集预测和参考文本（用于计算 BLEU 等指标）
                if hasattr(batch, 'src_texts') and hasattr(batch, 'tgt_texts'):
                    # 生成翻译
                    generated = self.model.generate(
                        src_tokens,
                        max_length=tgt_tokens.size(1),
                        src_mask=src_mask
                    )
                    
                    # 这里需要将 token 转换回文本，暂时跳过
                    # all_predictions.extend(generated_texts)
                    # all_references.extend(batch['tgt_texts'])
                
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        # 计算平均损失
        avg_loss = np.mean(val_losses)
        
        # 计算其他指标
        metrics = {
            'loss': avg_loss,
            'perplexity': calculate_perplexity(avg_loss)
        }
        
        # 如果有文本数据，计算 BLEU 等指标
        if all_predictions and all_references:
            self.metrics_tracker.update(all_predictions, all_references, avg_loss)
            text_metrics = self.metrics_tracker.get_latest_metrics()
            metrics.update(text_metrics)
        
        return metrics
    
    def save_checkpoint(
        self, 
        filepath: Optional[str] = None, 
        is_best: bool = False
    ):
        """保存检查点"""
        if filepath is None:
            filepath = self.save_dir / f'checkpoint_epoch_{self.state.epoch}.pt'
        
        checkpoint = {
            'epoch': self.state.epoch,
            'step': self.state.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': self.state.best_loss,
            'config': self.config,
            'training_state': {
                'epoch': self.state.epoch,
                'step': self.state.step,
                'best_loss': self.state.best_loss,
                'best_metric': self.state.best_metric,
                'patience_counter': self.state.patience_counter,
                'learning_rates': self.state.learning_rates,
                'train_losses': self.state.train_losses,
                'val_losses': self.state.val_losses,
                'train_metrics': dict(self.state.train_metrics),
                'val_metrics': dict(self.state.val_metrics)
            }
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")
        
        self.logger.info(f"保存检查点到: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        if 'training_state' in checkpoint:
            state_dict = checkpoint['training_state']
            if 'train_metrics' in state_dict:
                state_dict['train_metrics'] = defaultdict(list, state_dict['train_metrics'])
            if 'val_metrics' in state_dict:
                state_dict['val_metrics'] = defaultdict(list, state_dict['val_metrics'])
            
            for key, value in state_dict.items():
                setattr(self.state, key, value)
        
        self.logger.info(f"从 {filepath} 加载检查点")
    
    def train(self) -> Dict[str, List[float]]:
        """开始训练"""
        self.logger.info("开始训练...")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"模型参数数量: {self.model.get_trainable_parameters():,}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.state.epoch, self.max_epochs):
                self.state.epoch = epoch
                
                # 执行回调
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_start'):
                        callback.on_epoch_start(epoch)
                
                # 训练
                train_metrics = self.train_epoch()
                self.state.train_losses.append(train_metrics['loss'])
                
                for metric_name, value in train_metrics.items():
                    if metric_name != 'loss':
                        self.state.train_metrics[metric_name].append(value)
                
                # 验证
                val_metrics = {}
                if self.val_loader and (epoch + 1) % self.val_frequency == 0:
                    val_metrics = self.validate_epoch()
                    self.state.val_losses.append(val_metrics['loss'])
                    
                    for metric_name, value in val_metrics.items():
                        if metric_name != 'loss':
                            self.state.val_metrics[metric_name].append(value)
                
                # 记录到 TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
                    if val_metrics:
                        self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
                
                # 记录到 Weights & Biases
                if self.config.get('use_wandb', False):
                    log_dict = {'epoch': epoch, 'train/epoch_loss': train_metrics['loss']}
                    if val_metrics:
                        log_dict['val/epoch_loss'] = val_metrics['loss']
                    wandb.log(log_dict)
                
                # 打印进度
                log_msg = f"Epoch {epoch + 1}/{self.max_epochs} - "
                log_msg += f"Train Loss: {train_metrics['loss']:.4f}"
                if val_metrics:
                    log_msg += f", Val Loss: {val_metrics['loss']:.4f}"
                
                self.logger.info(log_msg)
                
                # 检查是否是最佳模型
                is_best = False
                if val_metrics:
                    current_loss = val_metrics['loss']
                    if current_loss < self.state.best_loss - self.min_delta:
                        self.state.best_loss = current_loss
                        self.state.patience_counter = 0
                        is_best = True
                    else:
                        self.state.patience_counter += 1
                else:
                    current_loss = train_metrics['loss']
                    if current_loss < self.state.best_loss - self.min_delta:
                        self.state.best_loss = current_loss
                        self.state.patience_counter = 0
                        is_best = True
                    else:
                        self.state.patience_counter += 1
                
                # 保存检查点
                if (epoch + 1) % self.save_frequency == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
                
                # 早停检查
                if self.early_stopping and self.state.patience_counter >= self.patience:
                    self.logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
                
                # 执行回调
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, train_metrics, val_metrics)
        
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {e}")
            raise
        
        finally:
            # 保存最终状态
            self.save_checkpoint()
            
            # 保存训练状态
            self.state.save(self.save_dir / 'training_state.json')
            
            # 关闭 TensorBoard
            if self.writer is not None:
                self.writer.close()
            
            # 关闭 Weights & Biases
            if self.config.get('use_wandb', False):
                wandb.finish()
            
            # 计算训练时间
            training_time = time.time() - start_time
            self.logger.info(f"训练完成，总用时: {training_time:.2f} 秒")
            
            # 绘制训练曲线
            if len(self.state.train_losses) > 0:
                self.visualizer.plot_training_curves(
                    train_losses=self.state.train_losses,
                    val_losses=self.state.val_losses if self.state.val_losses else None,
                    train_metrics=dict(self.state.train_metrics),
                    val_metrics=dict(self.state.val_metrics) if self.state.val_metrics else None,
                    save_path=str(self.save_dir / 'training_curves.png')
                )
        
        return {
            'train_losses': self.state.train_losses,
            'val_losses': self.state.val_losses,
            'train_metrics': dict(self.state.train_metrics),
            'val_metrics': dict(self.state.val_metrics)
        }


if __name__ == "__main__":
    # 测试训练器
    print("测试训练器...")
    
    # 这里需要实际的模型和数据加载器来测试
    # 暂时只测试训练状态的保存和加载
    
    print("\n测试训练状态:")
    state = TrainingState()
    state.epoch = 5
    state.step = 1000
    state.train_losses = [3.5, 2.8, 2.3, 1.9, 1.6]
    state.val_losses = [3.2, 2.6, 2.1, 1.8, 1.5]
    
    # 保存状态
    test_dir = Path('./test_checkpoints')
    test_dir.mkdir(exist_ok=True)
    
    state_file = test_dir / 'test_state.json'
    state.save(str(state_file))
    print(f"训练状态已保存到: {state_file}")
    
    # 加载状态
    loaded_state = TrainingState.load(str(state_file))
    print(f"加载的训练状态 - Epoch: {loaded_state.epoch}, Step: {loaded_state.step}")
    print(f"训练损失: {loaded_state.train_losses}")
    
    # 清理测试文件
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    print("\n✅ 训练器测试通过")