"""
Training Script for Custom Code Generation Model
Runs the full training pipeline with comprehensive data
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch

from .tokenizer import CodeTokenizer
from .transformer import create_model
from .trainer import Trainer, CodeDataset
from .training_data import get_all_training_data, get_training_data_by_category

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def train_model(
    model_size: str = 'small',
    vocab_size: int = 8000,
    num_epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    max_length: int = 256,
    save_name: str = 'code_model'
):
    """
    Train the code generation model.
    
    Args:
        model_size: Size of model ('tiny', 'small', 'medium')
        vocab_size: Maximum vocabulary size
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        max_length: Maximum sequence length
        save_name: Name for the saved checkpoint
    """
    print("="*70)
    print("Custom AI Model Training")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    print("Loading training data...")
    all_data = get_all_training_data()
    data_by_category = get_training_data_by_category()
    
    print(f"Total samples: {len(all_data)}")
    for category, samples in data_by_category.items():
        print(f"  - {category}: {len(samples)} samples")
    print()
    
    train_size = int(len(all_data) * 0.9)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print()
    
    print("Training tokenizer...")
    tokenizer = CodeTokenizer(vocab_size=vocab_size)
    tokenizer.train(train_data, min_frequency=1, verbose=True)
    actual_vocab_size = len(tokenizer)
    print(f"Final vocabulary size: {actual_vocab_size}")
    print()
    
    print(f"Creating {model_size} model...")
    model = create_model(actual_vocab_size, model_size)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model config: {model.get_config()}")
    print()
    
    output_dir = CHECKPOINT_DIR / save_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        warmup_steps=50,
        gradient_accumulation_steps=2,
    )
    
    print("Starting training...")
    start_time = time.time()
    
    result = trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_length=max_length,
        save_every=5,
        eval_every=1
    )
    
    training_time = time.time() - start_time
    
    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Best validation loss: {result['best_loss']:.4f}")
    print(f"Checkpoint saved to: {output_dir}")
    print()
    
    print("Training history:")
    for epoch_stats in result['history']:
        epoch = epoch_stats['epoch']
        train_loss = epoch_stats['train_loss']
        train_ppl = epoch_stats['train_ppl']
        val_info = ""
        if 'val_loss' in epoch_stats:
            val_info = f" | Val Loss: {epoch_stats['val_loss']:.4f}, Val PPL: {epoch_stats['val_ppl']:.2f}"
        print(f"  Epoch {epoch}: Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}{val_info}")
    
    print()
    print("Testing generation...")
    model.eval()
    
    test_prompts = [
        "def fibonacci(",
        "class User",
        "async function fetch",
        "SELECT * FROM",
        "resource \"aws_",
    ]
    
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id
            )
        
        output = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output[:100]}...")
    
    training_info = {
        'model_size': model_size,
        'vocab_size': actual_vocab_size,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'training_time_minutes': training_time / 60,
        'best_loss': result['best_loss'],
        'final_epoch': len(result['history']),
        'training_date': datetime.now().isoformat(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'total_samples': len(all_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
    }
    
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return model, tokenizer, result


def quick_train():
    """Quick training for testing (fewer epochs, smaller model)."""
    return train_model(
        model_size='tiny',
        vocab_size=5000,
        num_epochs=5,
        batch_size=4,
        learning_rate=1e-3,
        max_length=128,
        save_name='quick_model'
    )


def full_train():
    """Full training with larger model and more epochs."""
    return train_model(
        model_size='small',
        vocab_size=10000,
        num_epochs=20,
        batch_size=4,
        learning_rate=3e-4,
        max_length=256,
        save_name='full_model'
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train custom code generation model')
    parser.add_argument('--mode', choices=['quick', 'full', 'custom'], default='quick',
                        help='Training mode: quick (fast test), full (production), custom')
    parser.add_argument('--model-size', choices=['tiny', 'small', 'medium', 'large'], default='small')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--vocab-size', type=int, default=8000)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--save-name', type=str, default='code_model')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_train()
    elif args.mode == 'full':
        full_train()
    else:
        train_model(
            model_size=args.model_size,
            vocab_size=args.vocab_size,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
            save_name=args.save_name
        )
