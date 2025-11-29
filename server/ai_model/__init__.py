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
from .safety_guards import (
    Severity as SafetySeverity,
    SafetyLevel,
    ActionCategory,
    DetectedRisk,
    ConfirmationRequired,
    ValidationResult,
    RiskAssessmentResult,
    SafetyConfig,
    DestructiveActionDetector,
    ActionValidator,
    SafeAlternatives,
    assess_risk,
    is_destructive,
    requires_confirmation,
    get_safe_alternative,
    validate_against_instructions,
)
from .cost_estimator import (
    CostUnit,
    ActionType as CostActionType,
    Complexity,
    AlertLevel,
    CostBreakdown,
    CostEstimate,
    CostRecord,
    BudgetAlert,
    OptimizationSuggestion,
    PricingModel,
    TokenPricingModel,
    ComputePricingModel,
    StoragePricingModel,
    ActionPricingModel,
    CostEstimator,
    CostTracker,
    BudgetManager,
    CostOptimizer,
    format_cost,
    compare_costs,
    get_cost_breakdown,
    estimate_action_cost,
    create_budget_manager,
)
from .checkpoint_system import (
    CheckpointType,
    TriggerType,
    CompressionLevel,
    DiffType,
    FileState,
    DatabaseState,
    CheckpointMetadata,
    Checkpoint,
    RestoreResult,
    FileDiff,
    DatabaseDiff,
    DiffResult,
    RetentionPolicy,
    AutoCheckpointRule,
    FileSnapshot,
    AutoCheckpoint,
    DiffViewer,
    CheckpointManager,
    RecoveryOption,
    quick_checkpoint,
    rollback_to_last,
    get_recovery_options,
    create_checkpoint_before_action,
    set_default_manager,
)
from .context_manager import (
    InstructionCategory,
    ContextPriority,
    PatternConfidence,
    MessageRole,
    CodingStyle,
    UserPreference,
    ExplicitInstruction,
    ProjectContext,
    ConversationMessage,
    LearnedPattern,
    ContextItem,
    ConversationMemory,
    LearningMemory,
    InstructionTracker,
    ContextWindow,
    ContextManager,
    get_default_manager as get_default_context_manager,
    set_default_manager as set_default_context_manager,
    get_project_summary,
    search_context,
    remember,
    recall,
    load_project_context,
    add_instruction,
    check_conflicts,
    learn_from_correction,
)
from .error_handler import (
    ErrorSeverity,
    ErrorCategory,
    EffortLevel,
    Solution,
    FixResult,
    ErrorContext,
    UserFriendlyError,
    ErrorPattern,
    ErrorPatternMatcher,
    ErrorTranslator,
    SelfServiceResolver,
    ErrorLogEntry,
    ErrorReport,
    ErrorLogger,
    ERROR_PATTERNS,
    COMMON_ERRORS,
    get_default_translator,
    get_default_resolver,
    get_default_logger,
    get_error_context,
    handle_error,
    suggest_fixes,
    auto_fix,
    format_error_for_display,
)
from .deployment_engine import (
    DeploymentType,
    VMSize,
    DeploymentStatus,
    HealthStatus,
    DeploymentStrategy,
    Region,
    PortMapping,
    HealthCheckConfig,
    ScalingConfig,
    ScheduleConfig,
    DeploymentConfig,
    DeploymentSnapshot,
    DeploymentResult,
    RollbackResult,
    ScaleResult,
    DeploymentMetrics,
    LogEntry,
    BuildOptimizer,
    AutoScaler,
    HealthChecker,
    BlueGreenStrategy,
    CanaryStrategy,
    RollingStrategy,
    MultiRegionDeployer,
    EdgeDeployer,
    DeploymentEngine,
    estimate_deployment_time,
    estimate_deployment_cost,
    validate_config,
    generate_deployment_url,
    get_engine,
    deploy,
    rollback,
    get_status,
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
    # Safety Guards - Enums and Types
    'SafetySeverity',
    'SafetyLevel',
    'ActionCategory',
    # Safety Guards - Data Classes
    'DetectedRisk',
    'ConfirmationRequired',
    'ValidationResult',
    'RiskAssessmentResult',
    # Safety Guards - Classes
    'SafetyConfig',
    'DestructiveActionDetector',
    'ActionValidator',
    'SafeAlternatives',
    # Safety Guards - Functions
    'assess_risk',
    'is_destructive',
    'requires_confirmation',
    'get_safe_alternative',
    'validate_against_instructions',
    # Cost Estimator - Enums and Types
    'CostUnit',
    'CostActionType',
    'Complexity',
    'AlertLevel',
    # Cost Estimator - Data Classes
    'CostBreakdown',
    'CostEstimate',
    'CostRecord',
    'BudgetAlert',
    'OptimizationSuggestion',
    # Cost Estimator - Pricing Models
    'PricingModel',
    'TokenPricingModel',
    'ComputePricingModel',
    'StoragePricingModel',
    'ActionPricingModel',
    # Cost Estimator - Classes
    'CostEstimator',
    'CostTracker',
    'BudgetManager',
    'CostOptimizer',
    # Cost Estimator - Functions
    'format_cost',
    'compare_costs',
    'get_cost_breakdown',
    'estimate_action_cost',
    'create_budget_manager',
    # Checkpoint System - Enums and Types
    'CheckpointType',
    'TriggerType',
    'CompressionLevel',
    'DiffType',
    # Checkpoint System - Data Classes
    'FileState',
    'DatabaseState',
    'CheckpointMetadata',
    'Checkpoint',
    'RestoreResult',
    'FileDiff',
    'DatabaseDiff',
    'DiffResult',
    'RetentionPolicy',
    'AutoCheckpointRule',
    'RecoveryOption',
    # Checkpoint System - Classes
    'FileSnapshot',
    'AutoCheckpoint',
    'DiffViewer',
    'CheckpointManager',
    # Checkpoint System - Functions
    'quick_checkpoint',
    'rollback_to_last',
    'get_recovery_options',
    'create_checkpoint_before_action',
    'set_default_manager',
    # Context Manager - Enums and Types
    'InstructionCategory',
    'ContextPriority',
    'PatternConfidence',
    'MessageRole',
    # Context Manager - Data Classes
    'CodingStyle',
    'UserPreference',
    'ExplicitInstruction',
    'ProjectContext',
    'ConversationMessage',
    'LearnedPattern',
    'ContextItem',
    # Context Manager - Classes
    'ConversationMemory',
    'LearningMemory',
    'InstructionTracker',
    'ContextWindow',
    'ContextManager',
    # Context Manager - Functions
    'get_default_context_manager',
    'set_default_context_manager',
    'get_project_summary',
    'search_context',
    'remember',
    'recall',
    'load_project_context',
    'add_instruction',
    'check_conflicts',
    'learn_from_correction',
    # Error Handler - Enums and Types
    'ErrorSeverity',
    'ErrorCategory',
    'EffortLevel',
    # Error Handler - Data Classes
    'Solution',
    'FixResult',
    'ErrorContext',
    'UserFriendlyError',
    'ErrorPattern',
    'ErrorLogEntry',
    'ErrorReport',
    # Error Handler - Classes
    'ErrorPatternMatcher',
    'ErrorTranslator',
    'SelfServiceResolver',
    'ErrorLogger',
    # Error Handler - Databases
    'ERROR_PATTERNS',
    'COMMON_ERRORS',
    # Error Handler - Functions
    'get_default_translator',
    'get_default_resolver',
    'get_default_logger',
    'get_error_context',
    'handle_error',
    'suggest_fixes',
    'auto_fix',
    'format_error_for_display',
    # Deployment Engine - Enums and Types
    'DeploymentType',
    'VMSize',
    'DeploymentStatus',
    'HealthStatus',
    'DeploymentStrategy',
    'Region',
    # Deployment Engine - Data Classes
    'PortMapping',
    'HealthCheckConfig',
    'ScalingConfig',
    'ScheduleConfig',
    'DeploymentConfig',
    'DeploymentSnapshot',
    'DeploymentResult',
    'RollbackResult',
    'ScaleResult',
    'DeploymentMetrics',
    'LogEntry',
    # Deployment Engine - Classes
    'BuildOptimizer',
    'AutoScaler',
    'HealthChecker',
    'BlueGreenStrategy',
    'CanaryStrategy',
    'RollingStrategy',
    'MultiRegionDeployer',
    'EdgeDeployer',
    'DeploymentEngine',
    # Deployment Engine - Functions
    'estimate_deployment_time',
    'estimate_deployment_cost',
    'validate_config',
    'generate_deployment_url',
    'get_engine',
    'deploy',
    'rollback',
    'get_status',
]

__version__ = '0.1.0'
