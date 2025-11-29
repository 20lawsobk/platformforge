"""
Cloud GPU Training Script for Production-Scale Code Generation Model

This script is designed to run on cloud GPU infrastructure (AWS, GCP, Azure, Lambda Labs, etc.)
for training larger models with more data and compute power.

Requirements:
- NVIDIA GPU with CUDA support (A100, H100, V100, or RTX 4090 recommended)
- PyTorch with CUDA
- 16GB+ GPU memory for medium models, 40GB+ for large models

Usage:
    # Quick test on GPU
    python cloud_training.py --mode quick --gpu 0
    
    # Full training
    python cloud_training.py --mode full --gpu 0
    
    # Multi-GPU training
    torchrun --nproc_per_node=4 cloud_training.py --mode full --distributed
    
    # Custom configuration
    python cloud_training.py --model-size large --epochs 100 --batch-size 32 --lr 1e-4
"""

import os
import sys
import json
import time
import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from server.ai_model.tokenizer import CodeTokenizer
from server.ai_model.transformer import create_model, CodeTransformer
from server.ai_model.training_data import get_all_training_data, get_training_stats


class CloudConfig:
    """Configuration for cloud training."""
    
    # Model configurations for different GPU memory sizes
    CONFIGS = {
        'tiny': {
            'embed_dim': 128,
            'num_layers': 2,
            'num_heads': 2,
            'vocab_size': 8000,
            'max_seq_len': 512,
            'gpu_memory': '4GB',
            'params': '~1M',
        },
        'small': {
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 4,
            'vocab_size': 16000,
            'max_seq_len': 1024,
            'gpu_memory': '8GB',
            'params': '~10M',
        },
        'medium': {
            'embed_dim': 512,
            'num_layers': 8,
            'num_heads': 8,
            'vocab_size': 32000,
            'max_seq_len': 2048,
            'gpu_memory': '16GB',
            'params': '~50M',
        },
        'large': {
            'embed_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'vocab_size': 50000,
            'max_seq_len': 2048,
            'gpu_memory': '24GB',
            'params': '~125M',
        },
        'xlarge': {
            'embed_dim': 1024,
            'num_layers': 16,
            'num_heads': 16,
            'vocab_size': 64000,
            'max_seq_len': 4096,
            'gpu_memory': '40GB+',
            'params': '~350M',
        },
        'xxlarge': {
            'embed_dim': 1536,
            'num_layers': 24,
            'num_heads': 24,
            'vocab_size': 100000,
            'max_seq_len': 4096,
            'gpu_memory': '80GB+',
            'params': '~1B',
        },
    }
    
    @classmethod
    def get_config(cls, size: str) -> dict:
        return cls.CONFIGS.get(size, cls.CONFIGS['medium'])
    
    @classmethod
    def recommend_config(cls, gpu_memory_gb: int) -> str:
        if gpu_memory_gb >= 80:
            return 'xxlarge'
        elif gpu_memory_gb >= 40:
            return 'xlarge'
        elif gpu_memory_gb >= 24:
            return 'large'
        elif gpu_memory_gb >= 16:
            return 'medium'
        elif gpu_memory_gb >= 8:
            return 'small'
        else:
            return 'tiny'


class CodeDatasetCloud(Dataset):
    """Optimized dataset for GPU training."""
    
    def __init__(
        self,
        data: List[str],
        tokenizer: CodeTokenizer,
        max_length: int = 1024,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Create examples from each text sample
        for text in data:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) < 16:  # Skip very short samples
                continue
            
            # For short sequences, just pad to max_length
            if len(tokens) <= max_length:
                chunk = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
                self.examples.append(chunk)
            else:
                # For long sequences, use sliding window
                for i in range(0, len(tokens) - max_length + 1, stride):
                    chunk = tokens[i:i + max_length]
                    self.examples.append(chunk)
                # Also include the last chunk if not already covered
                if len(tokens) > max_length and (len(tokens) - max_length) % stride != 0:
                    chunk = tokens[-max_length:]
                    self.examples.append(chunk)
        
        print(f"Created {len(self.examples)} training examples from {len(data)} samples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, labels


class CloudTrainer:
    """Production-grade trainer for cloud GPU training."""
    
    def __init__(
        self,
        model: CodeTransformer,
        tokenizer: CodeTokenizer,
        output_dir: str,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        distributed: bool = False,
        local_rank: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and torch.cuda.is_available()
        self.distributed = distributed
        self.local_rank = local_rank
        
        # Setup device
        if torch.cuda.is_available():
            if distributed:
                self.device = torch.device(f'cuda:{local_rank}')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            self.use_amp = False
        
        self.model = self.model.to(self.device)
        
        # Distributed training setup
        if distributed:
            self.model = DDP(self.model, device_ids=[local_rank])
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
    
    def _create_optimizer(self, num_training_steps: int):
        """Create optimizer with weight decay and learning rate schedule."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Cosine learning rate schedule with warmup
        warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(
        self,
        train_data: List[str],
        val_data: List[str],
        num_epochs: int = 10,
        batch_size: int = 8,
        max_length: int = 1024,
        eval_every: int = 1,
        save_every: int = 5,
        log_every: int = 10,
    ) -> Dict:
        """Run training loop."""
        
        print("\n" + "="*70)
        print("CLOUD GPU TRAINING")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Distributed: {self.distributed}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training Samples: {len(train_data)}")
        print(f"Validation Samples: {len(val_data)}")
        print(f"Batch Size: {batch_size}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {batch_size * self.gradient_accumulation_steps}")
        print(f"Max Sequence Length: {max_length}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Epochs: {num_epochs}")
        print("="*70 + "\n")
        
        # Create datasets
        train_dataset = CodeDatasetCloud(train_data, self.tokenizer, max_length)
        val_dataset = CodeDatasetCloud(val_data, self.tokenizer, max_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        
        # Calculate training steps
        steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
        num_training_steps = steps_per_epoch * num_epochs
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {num_training_steps}")
        print()
        
        # Create optimizer and scheduler
        self._create_optimizer(num_training_steps)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_loss = self._train_epoch(train_loader, epoch, num_epochs, log_every)
            epoch_time = time.time() - epoch_start
            
            # Validation
            val_loss = None
            if epoch % eval_every == 0:
                val_loss = self._validate(val_loader)
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint('best_model')
            
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}')
            
            # Log epoch results
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_ppl': math.exp(min(train_loss, 20)),
                'val_loss': val_loss,
                'val_ppl': math.exp(min(val_loss, 20)) if val_loss else None,
                'epoch_time': epoch_time,
                'lr': self.scheduler.get_last_lr()[0],
            }
            self.training_history.append(epoch_stats)
            
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {epoch_stats['train_ppl']:.2f}")
            if val_loss:
                print(f"  Val Loss: {val_loss:.4f} | Val PPL: {epoch_stats['val_ppl']:.2f}")
            print(f"  Time: {epoch_time:.1f}s | LR: {epoch_stats['lr']:.2e}")
            print()
        
        # Save final model
        self._save_checkpoint('final_model')
        
        total_time = time.time() - start_time
        
        print("="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total Time: {total_time/60:.2f} minutes")
        print(f"Best Validation Loss: {self.best_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("="*70)
        
        # Save training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return {
            'best_loss': self.best_loss,
            'history': self.training_history,
            'total_time': total_time,
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        num_epochs: int,
        log_every: int,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(input_ids)
                # Handle both dict output (with 'logits' key) and direct tensor output
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                )
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % log_every == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch}/{num_epochs} | Step {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(input_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state (handle DDP wrapper)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Save model
        torch.save(model.state_dict(), checkpoint_dir / 'model.pt')
        
        # Save tokenizer
        self.tokenizer.save(str(checkpoint_dir / 'tokenizer.json'))
        
        # Save config
        config = model.get_config()
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save training state (for resuming)
        training_state = {
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        if self.scaler:
            training_state['scaler_state'] = self.scaler.state_dict()
        
        torch.save(training_state, checkpoint_dir / 'training_state.pt')
        
        if self.local_rank == 0:
            print(f"  Saved checkpoint: {checkpoint_dir}")


def setup_distributed():
    """Setup distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    parser = argparse.ArgumentParser(description='Cloud GPU Training for Code Generation Model')
    
    # Training mode
    parser.add_argument('--mode', choices=['quick', 'full', 'custom'], default='full',
                        help='Training mode: quick (test), full (production), custom')
    
    # Model configuration
    parser.add_argument('--model-size', choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge'],
                        default='medium', help='Model size configuration')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints/cloud_model',
                        help='Output directory for checkpoints')
    parser.add_argument('--save-name', type=str, default='cloud_model',
                        help='Name for saved model')
    
    args = parser.parse_args()
    
    # Setup distributed if needed
    local_rank = 0
    if args.distributed:
        local_rank = setup_distributed()
    elif torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Print GPU info
    if local_rank == 0:
        print("\n" + "="*70)
        print("CLOUD GPU TRAINING SETUP")
        print("="*70)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            print(f"Recommended Config: {CloudConfig.recommend_config(int(gpu_memory))}")
        else:
            print("WARNING: No GPU detected! Training will be slow on CPU.")
        
        print(f"Selected Config: {args.model_size}")
        print("="*70 + "\n")
    
    # Get model configuration
    config = CloudConfig.get_config(args.model_size)
    
    # Set parameters based on mode
    if args.mode == 'quick':
        args.epochs = 5
        args.batch_size = 8
        config['max_seq_len'] = 512
    elif args.mode == 'full':
        args.epochs = 50
        args.batch_size = 16
    
    # Load training data
    if local_rank == 0:
        print("Loading training data...")
    
    all_data = get_all_training_data(shuffle=True)
    stats = get_training_stats()
    
    if local_rank == 0:
        print(f"Total samples: {stats['total_samples']}")
        print(f"Total characters: {stats['total_characters']:,}")
        print()
    
    # Split data
    train_size = int(len(all_data) * 0.9)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]
    
    # Train tokenizer
    if local_rank == 0:
        print("Training tokenizer...")
    
    tokenizer = CodeTokenizer(vocab_size=config['vocab_size'])
    tokenizer.train(train_data, min_frequency=1, verbose=(local_rank == 0))
    actual_vocab_size = len(tokenizer)
    
    if local_rank == 0:
        print(f"Vocabulary size: {actual_vocab_size}")
        print()
    
    # Create model
    if local_rank == 0:
        print(f"Creating {args.model_size} model...")
    
    model = create_model(actual_vocab_size, args.model_size)
    
    if local_rank == 0:
        print(f"Model parameters: {model.count_parameters():,}")
        print()
    
    # Create trainer
    output_dir = Path(__file__).parent / args.output_dir / args.save_name
    
    trainer = CloudTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        use_amp=not args.no_amp,
        distributed=args.distributed,
        local_rank=local_rank,
    )
    
    # Train
    result = trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        eval_every=1,
        save_every=10,
        log_every=10,
    )
    
    # Cleanup distributed
    if args.distributed:
        dist.destroy_process_group()
    
    return result


if __name__ == '__main__':
    main()
