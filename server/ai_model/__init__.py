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
- kv_store: Key-Value Store with persistence, TTL, and atomic operations
- object_storage: S3-compatible Object Storage with buckets and presigned URLs

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
    
    # Use the Key-Value Store
    from server.ai_model import KeyValueStore, kv_get, kv_set
    
    # Simple usage with helper functions
    kv_set("user:123", {"name": "Alice", "score": 100})
    user = kv_get("user:123")
    
    # Direct store usage with TTL
    store = KeyValueStore("my_namespace")
    store.set("session", {"token": "abc"}, ttl=3600)  # Expires in 1 hour
    
    # Atomic operations
    store.increment("counter", 1)
    store.append("log", "new entry")
    
    # Use the Object Storage
    from server.ai_model import StorageClient, upload_from_text, download_as_bytes
    
    # Using the high-level client
    client = StorageClient()
    client.upload_from_text("bucket/file.txt", "Hello, World!")
    content = client.download_as_text("bucket/file.txt")
    
    # Using helper functions
    upload_from_text("mybucket/doc.txt", "Document content")
    data = download_as_bytes("mybucket/doc.txt")
    
    # Generate presigned URLs
    url = client.get_presigned_url("bucket/file.txt", expires_in=3600)
    
    # Use the Secrets Manager
    from server.ai_model import SecretsManager, Secret, SecretScope
    
    # Initialize the manager with a master password
    manager = SecretsManager(master_password="secure_master_password")
    
    # Set a secret
    manager.set_secret("API_KEY", "sk-1234567890", scope=SecretScope.APP)
    
    # Get a secret
    value = manager.get_secret("API_KEY")
    
    # List secrets (names only, not values)
    names = manager.list_secrets()
    
    # Rotate a secret
    manager.rotate_secret("API_KEY", "sk-new-value")
    
    # Export to environment variables
    manager.export_to_env()
    
    # Use the environment injector
    from server.ai_model import EnvironmentInjector
    injector = EnvironmentInjector(manager)
    injector.inject_all()
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
from .kv_store import (
    StorageLimit,
    StoreError,
    KeyError as KVKeyError,
    KeyTooLargeError,
    ValueTooLargeError,
    StoreLimitExceededError,
    StoreNotFoundError,
    SerializationError,
    AtomicOperationError,
    StoredValue,
    StoreStats,
    BatchResult,
    Serializer,
    JSONSerializer,
    KeyValueStore,
    StoreManager,
    get_default_manager as get_default_kv_manager,
    get_default_store,
    set_default_store,
    set_default_manager as set_default_kv_manager,
    get as kv_get,
    set as kv_set,
    delete as kv_delete,
    exists as kv_exists,
    keys as kv_keys,
    clear as kv_clear,
    increment as kv_increment,
    append as kv_append,
    get_many as kv_get_many,
    set_many as kv_set_many,
    delete_many as kv_delete_many,
    stats as kv_stats,
    create_store,
    list_stores,
    delete_store,
    format_size,
)
from .object_storage import (
    StorageLimit as ObjectStorageLimit,
    StorageClass,
    ObjectStorageError,
    BucketNotFoundError,
    BucketAlreadyExistsError,
    BucketNotEmptyError,
    ObjectNotFoundError,
    ObjectTooLargeError,
    KeyTooLongError,
    QuotaExceededError,
    InvalidBucketNameError,
    PresignedUrlExpiredError,
    InvalidPresignedUrlError,
    MultipartUploadError,
    ObjectMetadata,
    StorageObject,
    ListObjectsResult,
    BucketInfo,
    MultipartUpload,
    PresignedUrl,
    ContentTypeDetector,
    Bucket,
    ObjectStorage,
    StorageClient,
    get_default_client as get_default_storage_client,
    set_default_client as set_default_storage_client,
    upload_from_text,
    upload_from_bytes,
    upload_from_filename,
    download_as_text,
    download_as_bytes,
    download_to_filename,
    list_objects,
    delete_object,
    object_exists,
    get_presigned_url,
    format_size as format_storage_size,
)
from .secrets_manager import (
    SecretScope,
    AccessLevel,
    DeploymentEnvironment as SecretDeploymentEnvironment,
    AuditAction,
    SecretStorageLimit,
    SecretsError,
    SecretNotFoundError,
    SecretAlreadyExistsError,
    SecretExpiredError,
    SecretAccessDeniedError,
    InvalidSecretNameError,
    SecretValueTooLargeError,
    SecretQuotaExceededError,
    EncryptionError,
    MasterPasswordRequiredError,
    SecretVersion,
    AuditLogEntry,
    SecretMetadata,
    Secret,
    Encryptor,
    FallbackEncryptor,
    AuditLogger,
    SecretStore,
    SecretsManager,
    EnvironmentInjector,
    get_default_manager as get_default_secrets_manager,
    set_default_manager as set_default_secrets_manager,
    init_secrets,
    set_secret,
    get_secret,
    delete_secret,
    list_secrets,
    rotate_secret,
    export_to_env,
    load_from_env,
    secure_zero_memory,
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
    # Key-Value Store - Enums and Limits
    'StorageLimit',
    # Key-Value Store - Exceptions
    'StoreError',
    'KVKeyError',
    'KeyTooLargeError',
    'ValueTooLargeError',
    'StoreLimitExceededError',
    'StoreNotFoundError',
    'SerializationError',
    'AtomicOperationError',
    # Key-Value Store - Data Classes
    'StoredValue',
    'StoreStats',
    'BatchResult',
    # Key-Value Store - Serializers
    'Serializer',
    'JSONSerializer',
    # Key-Value Store - Main Classes
    'KeyValueStore',
    'StoreManager',
    # Key-Value Store - Manager Functions
    'get_default_kv_manager',
    'get_default_store',
    'set_default_store',
    'set_default_kv_manager',
    # Key-Value Store - Helper Functions
    'kv_get',
    'kv_set',
    'kv_delete',
    'kv_exists',
    'kv_keys',
    'kv_clear',
    'kv_increment',
    'kv_append',
    'kv_get_many',
    'kv_set_many',
    'kv_delete_many',
    'kv_stats',
    # Key-Value Store - Store Management
    'create_store',
    'list_stores',
    'delete_store',
    'format_size',
    # Object Storage - Enums and Limits
    'ObjectStorageLimit',
    'StorageClass',
    # Object Storage - Exceptions
    'ObjectStorageError',
    'BucketNotFoundError',
    'BucketAlreadyExistsError',
    'BucketNotEmptyError',
    'ObjectNotFoundError',
    'ObjectTooLargeError',
    'KeyTooLongError',
    'QuotaExceededError',
    'InvalidBucketNameError',
    'PresignedUrlExpiredError',
    'InvalidPresignedUrlError',
    'MultipartUploadError',
    # Object Storage - Data Classes
    'ObjectMetadata',
    'StorageObject',
    'ListObjectsResult',
    'BucketInfo',
    'MultipartUpload',
    'PresignedUrl',
    # Object Storage - Classes
    'ContentTypeDetector',
    'Bucket',
    'ObjectStorage',
    'StorageClient',
    # Object Storage - Client Functions
    'get_default_storage_client',
    'set_default_storage_client',
    # Object Storage - Upload Functions
    'upload_from_text',
    'upload_from_bytes',
    'upload_from_filename',
    # Object Storage - Download Functions
    'download_as_text',
    'download_as_bytes',
    'download_to_filename',
    # Object Storage - Management Functions
    'list_objects',
    'delete_object',
    'object_exists',
    'get_presigned_url',
    'format_storage_size',
    # Secrets Manager - Enums and Types
    'SecretScope',
    'AccessLevel',
    'SecretDeploymentEnvironment',
    'AuditAction',
    'SecretStorageLimit',
    # Secrets Manager - Exceptions
    'SecretsError',
    'SecretNotFoundError',
    'SecretAlreadyExistsError',
    'SecretExpiredError',
    'SecretAccessDeniedError',
    'InvalidSecretNameError',
    'SecretValueTooLargeError',
    'SecretQuotaExceededError',
    'EncryptionError',
    'MasterPasswordRequiredError',
    # Secrets Manager - Data Classes
    'SecretVersion',
    'AuditLogEntry',
    'SecretMetadata',
    'Secret',
    # Secrets Manager - Encryption Classes
    'Encryptor',
    'FallbackEncryptor',
    # Secrets Manager - Main Classes
    'AuditLogger',
    'SecretStore',
    'SecretsManager',
    'EnvironmentInjector',
    # Secrets Manager - Manager Functions
    'get_default_secrets_manager',
    'set_default_secrets_manager',
    'init_secrets',
    # Secrets Manager - Helper Functions
    'set_secret',
    'get_secret',
    'delete_secret',
    'list_secrets',
    'rotate_secret',
    'export_to_env',
    'load_from_env',
    # Secrets Manager - Utilities
    'secure_zero_memory',
]

__version__ = '0.1.0'
