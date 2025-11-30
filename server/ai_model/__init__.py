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
- app_testing: Comprehensive browser automation testing system
- web_search: Web search integration with multiple backends, caching, and content extraction
- image_generation: AI image generation with multiple backends, style presets, and editing
- stripe_integration: Comprehensive Stripe payments integration with subscriptions and webhooks
- automations: Bot/automation framework for Slack, Telegram, Discord with scheduling and workflows

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
    
    # Use the App Testing System
    from server.ai_model import (
        TestSession, TestStep, TestRunner, TestReporter,
        IssueDetector, AutoFixer, VisualTester,
        TestActionType, run_single_test, generate_test_report
    )
    
    # Create a test session with fluent API
    session = TestSession(name="Login Flow Test", base_url="http://localhost:5000")
    session.navigate("/login")
    session.fill("#email", "test@example.com")
    session.fill("#password", "password123")
    session.click("#submit-btn")
    session.assert_visible(".dashboard")
    
    # Run the test
    runner = TestRunner()
    result = await runner.run_session(session)
    
    # Generate HTML report
    reporter = TestReporter()
    report = reporter.generate_report([result], "Login Tests")
    reporter.export(report, format=ReportFormat.HTML)
    
    # Detect issues automatically
    detector = IssueDetector()
    issues = await detector.detect_all(browser)
    
    # Get fix suggestions
    fixer = AutoFixer()
    fixes = fixer.suggest_fixes(issues)
    
    # Visual regression testing
    visual_tester = VisualTester(baseline_dir="./baselines")
    diff = await visual_tester.compare_against_baseline(browser, "homepage")
    if not diff.passed:
        print(f"Visual regression detected: {diff.diff_percentage:.2%} changed")
    
    # Use the Web Search System
    from server.ai_model import (
        WebSearcher, SearchQuery, SearchResult, SearchResponse,
        ContentExtractor, DocumentationFetcher, SearchCache,
        search, search_images, search_news,
        extract_content, fetch_documentation,
    )
    
    # Create a searcher and search
    searcher = WebSearcher()
    response = await searcher.search("python async programming")
    
    # Use advanced query options
    query = SearchQuery(
        query="machine learning",
        max_results=10,
        freshness=FreshnessFilter.WEEK,
    )
    response = await searcher.search(query)
    
    # Image and news search
    images = await searcher.search_images("cute cats", max_results=20)
    news = await searcher.search_news("technology", freshness="day")
    
    # Extract content from URLs
    extractor = ContentExtractor()
    content = await extractor.extract_content("https://example.com/article")
    
    # Fetch API documentation
    doc_fetcher = DocumentationFetcher()
    docs = await doc_fetcher.fetch_documentation("https://docs.python.org/3/")
    
    # Use the Image Generation System
    from server.ai_model import (
        ImageGenerator, ImagePrompt, StylePreset, AspectRatio,
        PromptEnhancer, ImageEditor, StylePresets,
        image_generate, image_enhance_prompt, image_estimate_cost,
    )
    
    # Quick generation with style
    result = await image_generate("A sunset over mountains", style=StylePreset.PHOTOREALISTIC)
    image = result.image
    image.save("sunset.png")
    
    # Full control with ImagePrompt
    prompt = ImagePrompt(
        text="A cyberpunk cityscape",
        negative_prompt="blurry, low quality",
        style=StylePreset.ILLUSTRATION,
        aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    )
    generator = ImageGenerator()
    result = await generator.generate(prompt, count=4)
    
    # Enhance prompts for better results
    enhancer = PromptEnhancer()
    enhanced = enhancer.enhance("cat on couch")
    # -> "A fluffy domestic cat sitting comfortably on a modern velvet couch..."
    
    # Apply style presets
    styled_prompt = StylePresets.apply(prompt, StylePreset.OIL_PAINTING)
    
    # Batch generation
    prompts = [ImagePrompt(text=t) for t in ["sunset", "sunrise", "night sky"]]
    results = await generator.generate_batch(prompts, max_concurrent=3)
    
    # Image editing
    editor = ImageEditor()
    variation = await editor.create_variation(image, strength=0.5)
    inpainted = await editor.inpaint(image, mask, "replace with flowers")
    upscaled = await editor.upscale(image, scale=2)
    
    # Cost estimation
    estimate = generator.estimate_cost(prompt)
    print(f"Expected cost: ${estimate.expected_cost:.4f}")
    
    # Usage statistics
    stats = generator.get_usage_stats()
    print(f"Total generations: {stats.total_generations}")
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
from .app_testing import (
    ActionType as TestActionType,
    TestStatus,
    IssueType,
    IssueSeverity,
    AssertionType,
    BrowserType,
    RecordingFormat,
    ReportFormat,
    WaitCondition,
    TIMEOUT_DEFAULTS,
    ACCESSIBILITY_RULES,
    ElementLocator,
    BrowserConfig,
    Screenshot,
    VideoRecording,
    ConsoleMessage,
    NetworkRequest,
    NetworkResponse,
    PerformanceMetrics,
    DetectedIssue,
    TestStep,
    TestResult,
    TestSession,
    TestSuite,
    VisualDiff,
    Fix as TestFix,
    TestReport,
    BrowserError,
    ElementNotFoundError,
    NavigationError,
    TimeoutError as BrowserTimeoutError,
    AssertionError as TestAssertionError,
    BrowserController,
    SimulatedBrowserController,
    TestRunner,
    IssueDetector,
    AutoFixer,
    VisualTester,
    TestReporter,
    get_runner as get_test_runner,
    get_issue_detector,
    get_auto_fixer,
    get_visual_tester,
    get_reporter as get_test_reporter,
    run_test_suite,
    run_single_test,
    detect_issues as detect_test_issues,
    suggest_fixes as suggest_test_fixes,
    generate_report as generate_test_report,
    record_video,
    create_test_session,
    navigate as test_navigate,
    click as test_click,
    type_text as test_type_text,
    fill as test_fill,
    wait_for as test_wait_for,
    screenshot as test_screenshot,
    assert_visible as test_assert_visible,
    assert_text as test_assert_text,
)
from .web_search import (
    SearchBackend,
    SearchType,
    FreshnessFilter,
    SafeSearchLevel,
    ContentType as WebContentType,
    CacheEvictionPolicy,
    SearchLimit,
    WebSearchError,
    SearchBackendError,
    RateLimitExceededError,
    QueryTooLongError,
    ContentExtractionError,
    CacheError,
    InvalidBackendError,
    TimeoutError as SearchTimeoutError,
    SearchResult,
    ImageResult,
    NewsResult,
    SearchQuery,
    SearchResponse,
    ExtractedContent,
    DocumentationPage,
    CachedResult,
    RateLimitState,
    HTMLContentParser,
    SearchBackendInterface,
    DuckDuckGoBackend,
    BraveSearchBackend,
    SerpAPIBackend,
    CustomSearchBackend,
    SearchCache,
    RateLimiter,
    ContentExtractor,
    DocumentationFetcher,
    WebSearcher,
    get_default_searcher,
    set_default_searcher,
    get_default_extractor,
    get_default_doc_fetcher,
    search as web_search,
    search_images as web_search_images,
    search_news as web_search_news,
    extract_content,
    fetch_documentation,
    get_cached_result as get_cached_search_result,
    clear_cache as clear_search_cache,
    format_results as format_search_results,
)
from .image_generation import (
    ImageBackend,
    ImageFormat,
    AspectRatio,
    ImageSize,
    ImageQuality,
    StylePreset,
    EditOperation,
    GenerationStatus,
    ImageGenerationError,
    InvalidPromptError,
    BackendUnavailableError,
    GenerationQuotaExceededError,
    ImageTooLargeError,
    UnsupportedOperationError,
    ContentFilterError,
    RateLimitError as ImageRateLimitError,
    GeneratedImage,
    ImagePrompt,
    GenerationResult,
    EditRequest,
    CostEstimate as ImageCostEstimate,
    UsageStats as ImageUsageStats,
    StylePresets,
    PromptEnhancer,
    ImageBackendInterface,
    DallE3Backend,
    StabilityAIBackend,
    MidjourneyBackend,
    LocalSDBackend,
    ImageCache,
    ImageEditor,
    ImageGenerator,
    get_default_generator,
    set_default_generator,
    generate as image_generate,
    generate_batch as image_generate_batch,
    enhance_prompt as image_enhance_prompt,
    apply_style as image_apply_style,
    create_variation as image_create_variation,
    edit_image as image_edit_image,
    estimate_cost as image_estimate_cost,
    get_usage_stats as image_get_usage_stats,
    list_styles as image_list_styles,
    list_backends as image_list_backends,
    format_cost as image_format_cost,
)
from .auth_system import (
    AuthProvider,
    UserStatus,
    SessionStatus,
    TokenType,
    HashAlgorithm,
    Role,
    Permission,
    AuditEvent as AuthAuditEvent,
    AuthLimit,
    AuthError,
    InvalidCredentialsError,
    UserNotFoundError,
    UserExistsError,
    SessionExpiredError,
    SessionNotFoundError,
    InvalidTokenError,
    TokenExpiredError,
    AccountLockedError,
    AccountSuspendedError,
    PermissionDeniedError,
    MFARequiredError,
    MFAInvalidCodeError,
    RateLimitExceededError as AuthRateLimitExceededError,
    WeakPasswordError,
    OAuthError,
    UserProfile,
    OAuthAccount,
    MFASettings,
    User,
    Session,
    AuditLogEntry as AuthAuditLogEntry,
    LoginResult,
    PasswordHasher,
    PasswordValidator,
    TokenManager,
    OAuthProvider,
    GoogleOAuthProvider,
    GitHubOAuthProvider,
    DiscordOAuthProvider,
    RoleManager,
    MFAManager,
    RateLimiter as AuthRateLimiter,
    AuthManager,
    get_default_manager as get_default_auth_manager,
    set_default_manager as set_default_auth_manager,
    register as auth_register,
    login as auth_login,
    logout as auth_logout,
    validate_session as auth_validate_session,
    get_user as auth_get_user,
    verify_email as auth_verify_email,
    reset_password as auth_reset_password,
    check_permission as auth_check_permission,
)
from .stripe_integration import (
    PaymentStatus,
    SubscriptionStatus,
    InvoiceStatus,
    RefundStatus,
    RefundReason,
    PaymentMethodType,
    WebhookEventType,
    BillingInterval,
    ProrationBehavior,
    CollectionMethod,
    CheckoutMode,
    Currency,
    StripeLimit,
    StripeError,
    CardError,
    RateLimitError as StripeRateLimitError,
    InvalidRequestError as StripeInvalidRequestError,
    AuthenticationError as StripeAuthenticationError,
    PermissionError as StripePermissionError,
    ResourceNotFoundError,
    IdempotencyError,
    WebhookSignatureError,
    PaymentFailedError,
    SubscriptionError,
    InvoiceError,
    RefundError,
    Address,
    CardDetails,
    PaymentMethod,
    Customer as StripeCustomer,
    Price,
    Product as StripeProduct,
    SubscriptionItem,
    Subscription as StripeSubscription,
    InvoiceLineItem,
    Invoice as StripeInvoice,
    PaymentIntent,
    Refund,
    CheckoutSession,
    PortalSession,
    WebhookEvent,
    UsageRecord,
    Coupon,
    TaxRate,
    CustomerManager,
    SubscriptionManager,
    PaymentManager,
    InvoiceManager,
    WebhookHandler,
    ProductManager,
    CheckoutManager,
    StripeClient,
    get_default_client as get_default_stripe_client,
    set_default_client as set_default_stripe_client,
    init_stripe,
    create_customer as stripe_create_customer,
    update_customer as stripe_update_customer,
    create_subscription as stripe_create_subscription,
    cancel_subscription as stripe_cancel_subscription,
    update_subscription as stripe_update_subscription,
    create_payment_intent as stripe_create_payment_intent,
    capture_payment as stripe_capture_payment,
    create_checkout_session as stripe_create_checkout_session,
    create_portal_session as stripe_create_portal_session,
    handle_webhook as stripe_handle_webhook,
    verify_webhook_signature as stripe_verify_webhook_signature,
    create_refund as stripe_create_refund,
    list_invoices as stripe_list_invoices,
    format_amount as stripe_format_amount,
)
from .automations import (
    BotPlatform,
    BotStatus,
    MessageType,
    TriggerType as AutomationTriggerType,
    ActionType as AutomationActionType,
    JobStatus,
    JobPriority,
    ConversationState,
    ScheduleType,
    AutomationLimit,
    AutomationError,
    BotError,
    BotNotStartedError,
    BotAlreadyRunningError,
    BotConnectionError,
    RateLimitError as BotRateLimitError,
    WorkflowError,
    WorkflowNotFoundError,
    WorkflowExecutionError,
    TriggerError,
    InvalidTriggerError,
    JobError,
    JobNotFoundError,
    JobTimeoutError,
    ConversationError,
    ConversationNotFoundError,
    SchedulerError,
    InvalidCronExpressionError,
    User as BotUser,
    Channel,
    Message as BotMessage,
    MessageContext,
    Handler,
    RateLimiter as BotRateLimiter,
    BotStats,
    Bot,
    SlackBot,
    TelegramBot,
    DiscordBot,
    Trigger,
    Action,
    WorkflowExecution,
    Workflow,
    ScheduledTask,
    Scheduler,
    Job,
    JobQueue,
    Conversation,
    ConversationManager,
    WebhookRequest,
    WebhookHandler as AutomationWebhookHandler,
    WebhookReceiver,
    get_default_scheduler,
    set_default_scheduler,
    get_default_job_queue,
    set_default_job_queue,
    get_default_conversation_manager,
    set_default_conversation_manager,
    get_default_webhook_receiver,
    set_default_webhook_receiver,
    create_bot,
    create_workflow,
    schedule_job,
    get_conversation,
    create_conversation,
    end_conversation,
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
    # App Testing - Enums and Types
    'TestActionType',
    'TestStatus',
    'IssueType',
    'IssueSeverity',
    'AssertionType',
    'BrowserType',
    'RecordingFormat',
    'ReportFormat',
    'WaitCondition',
    # App Testing - Constants
    'TIMEOUT_DEFAULTS',
    'ACCESSIBILITY_RULES',
    # App Testing - Data Classes
    'ElementLocator',
    'BrowserConfig',
    'Screenshot',
    'VideoRecording',
    'ConsoleMessage',
    'NetworkRequest',
    'NetworkResponse',
    'PerformanceMetrics',
    'DetectedIssue',
    'TestStep',
    'TestResult',
    'TestSession',
    'TestSuite',
    'VisualDiff',
    'TestFix',
    'TestReport',
    # App Testing - Exceptions
    'BrowserError',
    'ElementNotFoundError',
    'NavigationError',
    'BrowserTimeoutError',
    'TestAssertionError',
    # App Testing - Main Classes
    'BrowserController',
    'SimulatedBrowserController',
    'TestRunner',
    'IssueDetector',
    'AutoFixer',
    'VisualTester',
    'TestReporter',
    # App Testing - Instance Getters
    'get_test_runner',
    'get_issue_detector',
    'get_auto_fixer',
    'get_visual_tester',
    'get_test_reporter',
    # App Testing - High-Level Functions
    'run_test_suite',
    'run_single_test',
    'detect_test_issues',
    'suggest_test_fixes',
    'generate_test_report',
    'record_video',
    'create_test_session',
    # App Testing - Step Helper Functions
    'test_navigate',
    'test_click',
    'test_type_text',
    'test_fill',
    'test_wait_for',
    'test_screenshot',
    'test_assert_visible',
    'test_assert_text',
    # Web Search - Enums and Types
    'SearchBackend',
    'SearchType',
    'FreshnessFilter',
    'SafeSearchLevel',
    'WebContentType',
    'CacheEvictionPolicy',
    'SearchLimit',
    # Web Search - Exceptions
    'WebSearchError',
    'SearchBackendError',
    'RateLimitExceededError',
    'QueryTooLongError',
    'ContentExtractionError',
    'CacheError',
    'InvalidBackendError',
    'SearchTimeoutError',
    # Web Search - Data Classes
    'SearchResult',
    'ImageResult',
    'NewsResult',
    'SearchQuery',
    'SearchResponse',
    'ExtractedContent',
    'DocumentationPage',
    'CachedResult',
    'RateLimitState',
    # Web Search - Backend Classes
    'SearchBackendInterface',
    'DuckDuckGoBackend',
    'BraveSearchBackend',
    'SerpAPIBackend',
    'CustomSearchBackend',
    # Web Search - Main Classes
    'HTMLContentParser',
    'SearchCache',
    'RateLimiter',
    'ContentExtractor',
    'DocumentationFetcher',
    'WebSearcher',
    # Web Search - Instance Getters
    'get_default_searcher',
    'set_default_searcher',
    'get_default_extractor',
    'get_default_doc_fetcher',
    # Web Search - High-Level Functions
    'web_search',
    'web_search_images',
    'web_search_news',
    'extract_content',
    'fetch_documentation',
    'get_cached_search_result',
    'clear_search_cache',
    'format_search_results',
    # Image Generation - Enums and Types
    'ImageBackend',
    'ImageFormat',
    'AspectRatio',
    'ImageSize',
    'ImageQuality',
    'StylePreset',
    'EditOperation',
    'GenerationStatus',
    # Image Generation - Exceptions
    'ImageGenerationError',
    'InvalidPromptError',
    'BackendUnavailableError',
    'GenerationQuotaExceededError',
    'ImageTooLargeError',
    'UnsupportedOperationError',
    'ContentFilterError',
    'ImageRateLimitError',
    # Image Generation - Data Classes
    'GeneratedImage',
    'ImagePrompt',
    'GenerationResult',
    'EditRequest',
    'ImageCostEstimate',
    'ImageUsageStats',
    # Image Generation - Style Classes
    'StylePresets',
    'PromptEnhancer',
    # Image Generation - Backend Classes
    'ImageBackendInterface',
    'DallE3Backend',
    'StabilityAIBackend',
    'MidjourneyBackend',
    'LocalSDBackend',
    # Image Generation - Main Classes
    'ImageCache',
    'ImageEditor',
    'ImageGenerator',
    # Image Generation - Instance Getters
    'get_default_generator',
    'set_default_generator',
    # Image Generation - High-Level Functions
    'image_generate',
    'image_generate_batch',
    'image_enhance_prompt',
    'image_apply_style',
    'image_create_variation',
    'image_edit_image',
    'image_estimate_cost',
    'image_get_usage_stats',
    'image_list_styles',
    'image_list_backends',
    'image_format_cost',
    # Auth System - Enums and Types
    'AuthProvider',
    'UserStatus',
    'SessionStatus',
    'TokenType',
    'HashAlgorithm',
    'Role',
    'Permission',
    'AuthAuditEvent',
    'AuthLimit',
    # Auth System - Exceptions
    'AuthError',
    'InvalidCredentialsError',
    'UserNotFoundError',
    'UserExistsError',
    'SessionExpiredError',
    'SessionNotFoundError',
    'InvalidTokenError',
    'TokenExpiredError',
    'AccountLockedError',
    'AccountSuspendedError',
    'PermissionDeniedError',
    'MFARequiredError',
    'MFAInvalidCodeError',
    'AuthRateLimitExceededError',
    'WeakPasswordError',
    'OAuthError',
    # Auth System - Data Classes
    'UserProfile',
    'OAuthAccount',
    'MFASettings',
    'User',
    'Session',
    'AuthAuditLogEntry',
    'LoginResult',
    # Auth System - Security Classes
    'PasswordHasher',
    'PasswordValidator',
    'TokenManager',
    # Auth System - OAuth Provider Classes
    'OAuthProvider',
    'GoogleOAuthProvider',
    'GitHubOAuthProvider',
    'DiscordOAuthProvider',
    # Auth System - Manager Classes
    'RoleManager',
    'MFAManager',
    'AuthRateLimiter',
    'AuthManager',
    # Auth System - Instance Getters
    'get_default_auth_manager',
    'set_default_auth_manager',
    # Auth System - Helper Functions
    'auth_register',
    'auth_login',
    'auth_logout',
    'auth_validate_session',
    'auth_get_user',
    'auth_verify_email',
    'auth_reset_password',
    'auth_check_permission',
    # Stripe Integration - Enums and Types
    'PaymentStatus',
    'SubscriptionStatus',
    'InvoiceStatus',
    'RefundStatus',
    'RefundReason',
    'PaymentMethodType',
    'WebhookEventType',
    'BillingInterval',
    'ProrationBehavior',
    'CollectionMethod',
    'CheckoutMode',
    'Currency',
    'StripeLimit',
    # Stripe Integration - Exceptions
    'StripeError',
    'CardError',
    'StripeRateLimitError',
    'StripeInvalidRequestError',
    'StripeAuthenticationError',
    'StripePermissionError',
    'ResourceNotFoundError',
    'IdempotencyError',
    'WebhookSignatureError',
    'PaymentFailedError',
    'SubscriptionError',
    'InvoiceError',
    'RefundError',
    # Stripe Integration - Data Classes
    'Address',
    'CardDetails',
    'PaymentMethod',
    'StripeCustomer',
    'Price',
    'StripeProduct',
    'SubscriptionItem',
    'StripeSubscription',
    'InvoiceLineItem',
    'StripeInvoice',
    'PaymentIntent',
    'Refund',
    'CheckoutSession',
    'PortalSession',
    'WebhookEvent',
    'UsageRecord',
    'Coupon',
    'TaxRate',
    # Stripe Integration - Manager Classes
    'CustomerManager',
    'SubscriptionManager',
    'PaymentManager',
    'InvoiceManager',
    'WebhookHandler',
    'ProductManager',
    'CheckoutManager',
    'StripeClient',
    # Stripe Integration - Instance Getters
    'get_default_stripe_client',
    'set_default_stripe_client',
    'init_stripe',
    # Stripe Integration - Helper Functions
    'stripe_create_customer',
    'stripe_update_customer',
    'stripe_create_subscription',
    'stripe_cancel_subscription',
    'stripe_update_subscription',
    'stripe_create_payment_intent',
    'stripe_capture_payment',
    'stripe_create_checkout_session',
    'stripe_create_portal_session',
    'stripe_handle_webhook',
    'stripe_verify_webhook_signature',
    'stripe_create_refund',
    'stripe_list_invoices',
    'stripe_format_amount',
    # Automations - Enums and Types
    'BotPlatform',
    'BotStatus',
    'MessageType',
    'AutomationTriggerType',
    'AutomationActionType',
    'JobStatus',
    'JobPriority',
    'ConversationState',
    'ScheduleType',
    'AutomationLimit',
    # Automations - Exceptions
    'AutomationError',
    'BotError',
    'BotNotStartedError',
    'BotAlreadyRunningError',
    'BotConnectionError',
    'BotRateLimitError',
    'WorkflowError',
    'WorkflowNotFoundError',
    'WorkflowExecutionError',
    'TriggerError',
    'InvalidTriggerError',
    'JobError',
    'JobNotFoundError',
    'JobTimeoutError',
    'ConversationError',
    'ConversationNotFoundError',
    'SchedulerError',
    'InvalidCronExpressionError',
    # Automations - Data Classes
    'BotUser',
    'Channel',
    'BotMessage',
    'MessageContext',
    'Handler',
    'BotRateLimiter',
    'BotStats',
    'Trigger',
    'Action',
    'WorkflowExecution',
    'ScheduledTask',
    'Job',
    'Conversation',
    'WebhookRequest',
    'AutomationWebhookHandler',
    # Automations - Main Classes
    'Bot',
    'SlackBot',
    'TelegramBot',
    'DiscordBot',
    'Workflow',
    'Scheduler',
    'JobQueue',
    'ConversationManager',
    'WebhookReceiver',
    # Automations - Instance Getters
    'get_default_scheduler',
    'set_default_scheduler',
    'get_default_job_queue',
    'set_default_job_queue',
    'get_default_conversation_manager',
    'set_default_conversation_manager',
    'get_default_webhook_receiver',
    'set_default_webhook_receiver',
    # Automations - Factory Functions
    'create_bot',
    'create_workflow',
    'schedule_job',
    'get_conversation',
    'create_conversation',
    'end_conversation',
]

__version__ = '0.1.0'
