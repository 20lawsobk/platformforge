"""
Training Pipeline for Custom Code Generation Model
Includes data loading, training loop, checkpointing, and evaluation
"""

import os
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from .tokenizer import CodeTokenizer
from .transformer import CodeTransformer, create_model


class CodeDataset(Dataset):
    """
    Dataset for code generation training.
    Supports loading from text files or in-memory data.
    """
    
    def __init__(
        self,
        data: List[str],
        tokenizer: CodeTokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.samples = []
        
        for text in data:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) <= max_length:
                self.samples.append(tokens)
            else:
                for i in range(0, len(tokens) - max_length + 1, stride):
                    self.samples.append(tokens[i:i + max_length])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.samples[idx]
        
        if len(tokens) < self.max_length:
            padding = [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
            tokens = tokens + padding
            attention_mask = [1] * len(self.samples[idx]) + [0] * len(padding)
        else:
            attention_mask = [1] * self.max_length
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


class Trainer:
    """
    Training manager for code generation models.
    
    Features:
    - Gradient accumulation
    - Mixed precision training
    - Learning rate scheduling
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: CodeTransformer,
        tokenizer: CodeTokenizer,
        output_dir: str = './checkpoints',
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
    
    def _get_lr(self, step: int, total_steps: int) -> float:
        """Calculate learning rate with warmup and cosine decay."""
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        
        progress = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
        return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _update_lr(self, step: int, total_steps: int):
        """Update learning rate for all parameter groups."""
        lr = self._get_lr(step, total_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        total_steps = len(dataloader) * total_epochs
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss'] / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss'] / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                current_step = epoch * len(dataloader) + batch_idx
                lr = self._update_lr(current_step, total_steps)
                
                self.global_step += 1
            
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask, labels)
            total_loss += outputs['loss'].item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_data: List[str],
        val_data: Optional[List[str]] = None,
        num_epochs: int = 10,
        batch_size: int = 8,
        max_length: int = 512,
        save_every: int = 1,
        eval_every: int = 1
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_data: List of training text samples
            val_data: Optional validation data
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            max_length: Maximum sequence length
            save_every: Save checkpoint every N epochs
            eval_every: Evaluate every N epochs
            
        Returns:
            Training history dictionary
        """
        train_dataset = CodeDataset(train_data, self.tokenizer, max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = None
        if val_data:
            val_dataset = CodeDataset(val_data, self.tokenizer, max_length)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Model Parameters: {self.model.count_parameters():,}")
        print(f"  Training Samples: {len(train_dataset)}")
        print(f"  Validation Samples: {len(val_dataset) if val_data else 0}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Max Length: {max_length}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            train_loss = self.train_epoch(train_loader, epoch, num_epochs)
            
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_ppl': math.exp(min(train_loss, 20)),
            }
            
            if val_loader and (epoch + 1) % eval_every == 0:
                val_loss = self.evaluate(val_loader)
                epoch_stats['val_loss'] = val_loss
                epoch_stats['val_ppl'] = math.exp(min(val_loss, 20))
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best_model')
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}')
            
            self.training_history.append(epoch_stats)
            
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {epoch_stats['train_ppl']:.2f}")
            if 'val_loss' in epoch_stats:
                print(f"  Val Loss: {val_loss:.4f} | Val PPL: {epoch_stats['val_ppl']:.2f}")
        
        self.save_checkpoint('final_model')
        
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return {'history': self.training_history, 'best_loss': self.best_loss}
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), checkpoint_dir / 'model.pt')
        
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(self.model.get_config(), f, indent=2)
        
        self.tokenizer.save(str(checkpoint_dir / 'tokenizer.json'))
        
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(training_state, checkpoint_dir / 'training_state.pt')
        
        print(f"Saved checkpoint: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        with open(checkpoint_dir / 'config.json') as f:
            config = json.load(f)
        
        self.model.load_state_dict(torch.load(checkpoint_dir / 'model.pt', map_location=self.device))
        self.tokenizer.load(str(checkpoint_dir / 'tokenizer.json'))
        
        training_state_path = checkpoint_dir / 'training_state.pt'
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=self.device)
            self.global_step = training_state['global_step']
            self.epoch = training_state['epoch']
            self.best_loss = training_state['best_loss']
            self.optimizer.load_state_dict(training_state['optimizer_state'])
        
        print(f"Loaded checkpoint: {checkpoint_dir}")


def train_code_model(
    training_data: List[str],
    validation_data: Optional[List[str]] = None,
    model_size: str = 'small',
    vocab_size: int = 10000,
    output_dir: str = './checkpoints',
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    max_length: int = 512
) -> Tuple[CodeTransformer, CodeTokenizer]:
    """
    Convenience function to train a code generation model.
    
    Args:
        training_data: List of code samples for training
        validation_data: Optional validation samples
        model_size: Size of model ('tiny', 'small', 'medium', 'large')
        vocab_size: Size of tokenizer vocabulary
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (trained model, tokenizer)
    """
    print("Initializing tokenizer...")
    tokenizer = CodeTokenizer(vocab_size=vocab_size)
    tokenizer.train(training_data, min_frequency=2, verbose=True)
    
    print(f"\nCreating {model_size} model...")
    model = create_model(len(tokenizer), model_size)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        learning_rate=learning_rate
    )
    
    print("\nStarting training...")
    trainer.train(
        train_data=training_data,
        val_data=validation_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_length=max_length
    )
    
    return model, tokenizer


if __name__ == '__main__':
    sample_code = [
        """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
        
        """class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    
    def is_empty(self):
        return len(self.items) == 0""",
        
        """async function fetchUser(id) {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
        throw new Error('User not found');
    }
    return response.json();
}""",
        
        """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)""",
        
        """import React, { useState, useEffect } from 'react';

function Counter() {
    const [count, setCount] = useState(0);
    
    useEffect(() => {
        document.title = `Count: ${count}`;
    }, [count]);
    
    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={() => setCount(count + 1)}>
                Increment
            </button>
        </div>
    );
}""",
    ]
    
    model, tokenizer = train_code_model(
        training_data=sample_code,
        model_size='tiny',
        vocab_size=5000,
        num_epochs=3,
        batch_size=2,
        max_length=256
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
