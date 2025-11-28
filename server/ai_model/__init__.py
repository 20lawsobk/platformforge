"""
Custom AI Model Training System

This package provides a complete framework for training and deploying
custom transformer-based AI models for code generation.

Components:
- tokenizer: BPE tokenizer for code and text
- transformer: Custom transformer architecture
- trainer: Training pipeline with checkpointing
- inference: Inference engine for generation

Usage:
    from server.ai_model import train_code_model, CodeGenerator
    
    # Train a model
    model, tokenizer = train_code_model(
        training_data=['def hello(): return "world"', ...],
        model_size='small',
        num_epochs=10
    )
    
    # Or load and use a trained model
    generator = CodeGenerator('path/to/checkpoint')
    result = generator.generate('def add(a, b):')
"""

from .tokenizer import BytePairTokenizer, CodeTokenizer
from .transformer import CodeTransformer, create_model
from .trainer import Trainer, CodeDataset, train_code_model
from .inference import CodeGenerator, AIAssistant, load_generator

__all__ = [
    'BytePairTokenizer',
    'CodeTokenizer',
    'CodeTransformer',
    'create_model',
    'Trainer',
    'CodeDataset',
    'train_code_model',
    'CodeGenerator',
    'AIAssistant',
    'load_generator',
]

__version__ = '0.1.0'
