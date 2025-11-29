"""
Custom AI Model Training System

This package provides a complete framework for training and deploying
custom transformer-based AI models for code generation.

Components:
- tokenizer: BPE tokenizer for code and text
- transformer: Custom transformer architecture
- trainer: Training pipeline with checkpointing
- inference: Inference engine for generation
- knowledge_base: Comprehensive catalog of languages, frameworks, and capabilities

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
    
    # Query the knowledge base
    from server.ai_model import get_language_info, search_knowledge_base
    python_info = get_language_info('python')
    results = search_knowledge_base('kubernetes')
"""

from .tokenizer import BytePairTokenizer, CodeTokenizer
from .transformer import CodeTransformer, create_model
from .trainer import Trainer, CodeDataset, train_code_model
from .inference import CodeGenerator, AIAssistant, load_generator
from .knowledge_base import (
    LANGUAGES_DATABASE,
    FRAMEWORKS_DATABASE,
    INFRASTRUCTURE_DATABASE,
    PATTERNS_DATABASE,
    SKILLS_DATABASE,
    get_language_info,
    get_frameworks_for_language,
    get_infrastructure_by_category,
    get_patterns_by_type,
    get_all_skills,
    get_skills_by_category,
    search_knowledge_base,
    get_all_languages,
    get_all_framework_names,
    get_all_infrastructure_categories,
    get_all_pattern_categories,
    get_all_skill_categories,
    get_statistics,
)

__all__ = [
    # Tokenizer
    'BytePairTokenizer',
    'CodeTokenizer',
    # Transformer
    'CodeTransformer',
    'create_model',
    # Trainer
    'Trainer',
    'CodeDataset',
    'train_code_model',
    # Inference
    'CodeGenerator',
    'AIAssistant',
    'load_generator',
    # Knowledge Base - Databases
    'LANGUAGES_DATABASE',
    'FRAMEWORKS_DATABASE',
    'INFRASTRUCTURE_DATABASE',
    'PATTERNS_DATABASE',
    'SKILLS_DATABASE',
    # Knowledge Base - Helper Functions
    'get_language_info',
    'get_frameworks_for_language',
    'get_infrastructure_by_category',
    'get_patterns_by_type',
    'get_all_skills',
    'get_skills_by_category',
    'search_knowledge_base',
    'get_all_languages',
    'get_all_framework_names',
    'get_all_infrastructure_categories',
    'get_all_pattern_categories',
    'get_all_skill_categories',
    'get_statistics',
]

__version__ = '0.1.0'
