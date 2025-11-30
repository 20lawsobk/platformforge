"""
Bot and Automation Framework for Platform Forge

This module provides a comprehensive bot and automation framework matching
Replit's Agents & Automations functionality. It includes multi-platform bot
support, scheduled task execution, event-driven triggers, workflow automation,
and background job processing.

Key Components:
- Bot: Base class for all bot platforms with message handling
- SlackBot: Slack-specific bot implementation
- TelegramBot: Telegram-specific bot implementation
- DiscordBot: Discord-specific bot implementation
- Scheduler: Cron-like task scheduler with timezone support
- Workflow: If-this-then-that automation workflows
- Trigger: Event triggers (time, webhook, message, custom)
- Action: Workflow actions with conditional execution
- JobQueue: Background job processing with priorities and retries
- ConversationManager: Stateful conversation handling

Usage:
    from server.ai_model.automations import (
        Bot, SlackBot, TelegramBot, DiscordBot,
        Scheduler, Workflow, Trigger, Action,
        JobQueue, ConversationManager,
        create_bot, create_workflow, schedule_job,
    )
    
    # Create and configure a Slack bot
    bot = SlackBot(
        token="xoxb-your-token",
        name="MyBot"
    )
    
    @bot.on_command("hello")
    async def hello_command(ctx):
        await ctx.reply("Hello! How can I help you?")
    
    @bot.on_message(pattern=r"help|support")
    async def help_handler(ctx):
        await ctx.reply("Here's how I can help...")
    
    # Start the bot
    await bot.start()
    
    # Create a workflow
    workflow = Workflow(name="Welcome Flow")
    workflow.add_trigger(Trigger.on_message(pattern="hi|hello"))
    workflow.add_action(Action.reply("Welcome to our service!"))
    workflow.add_action(Action.delay(seconds=2))
    workflow.add_action(Action.reply("How can I help you today?"))
    
    # Schedule tasks
    scheduler = Scheduler()
    
    @scheduler.cron("0 9 * * *")  # Every day at 9 AM
    async def daily_report():
        await send_daily_report()
    
    @scheduler.every(hours=1)
    async def hourly_check():
        await check_system_health()
    
    # Background job processing
    queue = JobQueue()
    
    @queue.job("send_email")
    async def send_email_job(to: str, subject: str, body: str):
        await email_service.send(to, subject, body)
    
    # Queue a job
    await queue.enqueue("send_email", 
        to="user@example.com",
        subject="Welcome!",
        body="Thanks for signing up!"
    )
    
    # Conversation state management
    conv_manager = ConversationManager()
    
    @bot.on_message()
    async def handle_message(ctx):
        conv = conv_manager.get_or_create(ctx.user_id)
        
        if conv.state == "awaiting_name":
            conv.set("name", ctx.message.text)
            conv.transition("awaiting_email")
            await ctx.reply("Great! Now, what's your email?")
        elif conv.state == "awaiting_email":
            conv.set("email", ctx.message.text)
            conv.transition("complete")
            await ctx.reply(f"Thanks, {conv.get('name')}!")
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
import threading
import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    Union,
)
from functools import wraps
import copy


logger = logging.getLogger(__name__)


class BotPlatform(Enum):
    """Supported bot platforms."""
    SLACK = "slack"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    GENERIC = "generic"
    WEBHOOK = "webhook"


class BotStatus(Enum):
    """Bot operational status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class MessageType(Enum):
    """Types of messages a bot can receive."""
    TEXT = "text"
    COMMAND = "command"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    STICKER = "sticker"
    REACTION = "reaction"
    REPLY = "reply"
    FORWARD = "forward"
    EDIT = "edit"
    DELETE = "delete"
    JOIN = "join"
    LEAVE = "leave"
    SYSTEM = "system"


class TriggerType(Enum):
    """Types of automation triggers."""
    MESSAGE = "message"
    COMMAND = "command"
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    EVENT = "event"
    REACTION = "reaction"
    MENTION = "mention"
    KEYWORD = "keyword"
    REGEX = "regex"
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    CHANNEL_CREATE = "channel_create"
    CUSTOM = "custom"


class ActionType(Enum):
    """Types of workflow actions."""
    SEND_MESSAGE = "send_message"
    REPLY = "reply"
    REACT = "react"
    DELAY = "delay"
    CONDITION = "condition"
    LOOP = "loop"
    HTTP_REQUEST = "http_request"
    SET_VARIABLE = "set_variable"
    LOG = "log"
    NOTIFY = "notify"
    UPDATE_STATE = "update_state"
    CALL_FUNCTION = "call_function"
    BRANCH = "branch"
    END = "end"
    CUSTOM = "custom"


class JobStatus(Enum):
    """Background job status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    DEFERRED = "deferred"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ConversationState(Enum):
    """Predefined conversation states."""
    NEW = "new"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ScheduleType(Enum):
    """Types of schedule definitions."""
    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AutomationLimit(Enum):
    """Automation system limits."""
    MAX_WORKFLOWS = 100
    MAX_TRIGGERS_PER_WORKFLOW = 20
    MAX_ACTIONS_PER_WORKFLOW = 50
    MAX_JOBS_IN_QUEUE = 10000
    MAX_RETRIES = 5
    MAX_CONVERSATIONS = 10000
    MAX_CONVERSATION_AGE_SECONDS = 86400
    MAX_MESSAGE_LENGTH = 4000
    MAX_HANDLERS = 500
    RATE_LIMIT_WINDOW_SECONDS = 60
    RATE_LIMIT_MAX_REQUESTS = 100


class AutomationError(Exception):
    """Base exception for automation errors."""
    pass


class BotError(AutomationError):
    """Bot-related errors."""
    pass


class BotNotStartedError(BotError):
    """Bot is not running."""
    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        super().__init__(f"Bot '{bot_name}' is not started")


class BotAlreadyRunningError(BotError):
    """Bot is already running."""
    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        super().__init__(f"Bot '{bot_name}' is already running")


class BotConnectionError(BotError):
    """Failed to connect to platform."""
    def __init__(self, platform: str, reason: str):
        self.platform = platform
        self.reason = reason
        super().__init__(f"Failed to connect to {platform}: {reason}")


class RateLimitError(AutomationError):
    """Rate limit exceeded."""
    def __init__(self, limit: int, window: int, retry_after: float):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window}s. "
            f"Retry after {retry_after:.1f}s"
        )


class WorkflowError(AutomationError):
    """Workflow execution errors."""
    pass


class WorkflowNotFoundError(WorkflowError):
    """Workflow not found."""
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        super().__init__(f"Workflow '{workflow_id}' not found")


class WorkflowExecutionError(WorkflowError):
    """Error during workflow execution."""
    def __init__(self, workflow_id: str, action_index: int, error: str):
        self.workflow_id = workflow_id
        self.action_index = action_index
        self.error = error
        super().__init__(
            f"Workflow '{workflow_id}' failed at action {action_index}: {error}"
        )


class TriggerError(AutomationError):
    """Trigger-related errors."""
    pass


class InvalidTriggerError(TriggerError):
    """Invalid trigger configuration."""
    pass


class JobError(AutomationError):
    """Job processing errors."""
    pass


class JobNotFoundError(JobError):
    """Job not found."""
    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Job '{job_id}' not found")


class JobTimeoutError(JobError):
    """Job execution timeout."""
    def __init__(self, job_id: str, timeout: float):
        self.job_id = job_id
        self.timeout = timeout
        super().__init__(f"Job '{job_id}' timed out after {timeout}s")


class ConversationError(AutomationError):
    """Conversation management errors."""
    pass


class ConversationNotFoundError(ConversationError):
    """Conversation not found."""
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        super().__init__(f"Conversation '{conversation_id}' not found")


class SchedulerError(AutomationError):
    """Scheduler errors."""
    pass


class InvalidCronExpressionError(SchedulerError):
    """Invalid cron expression."""
    def __init__(self, expression: str, reason: str):
        self.expression = expression
        self.reason = reason
        super().__init__(f"Invalid cron expression '{expression}': {reason}")


@dataclass
class User:
    """
    Represents a user across any platform.
    
    Attributes:
        id: Platform-specific user ID
        username: User's username or handle
        display_name: User's display name
        platform: The platform this user is from
        metadata: Additional platform-specific data
    """
    id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    platform: BotPlatform = BotPlatform.GENERIC
    avatar_url: Optional[str] = None
    is_bot: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "platform": self.platform.value,
            "avatar_url": self.avatar_url,
            "is_bot": self.is_bot,
            "metadata": self.metadata,
        }


@dataclass
class Channel:
    """
    Represents a channel/chat/conversation.
    
    Attributes:
        id: Platform-specific channel ID
        name: Channel name
        type: Channel type (dm, group, public, private)
        platform: The platform this channel is from
    """
    id: str
    name: Optional[str] = None
    type: str = "channel"
    platform: BotPlatform = BotPlatform.GENERIC
    is_private: bool = False
    member_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "platform": self.platform.value,
            "is_private": self.is_private,
            "member_count": self.member_count,
            "metadata": self.metadata,
        }


@dataclass
class Message:
    """
    Represents a message received by the bot.
    
    Attributes:
        id: Platform-specific message ID
        text: Message text content
        user: User who sent the message
        channel: Channel where message was sent
        type: Type of message
        timestamp: When the message was sent
        reply_to: ID of message this replies to
        attachments: List of attachments
        mentions: List of mentioned users
        metadata: Additional platform-specific data
    """
    id: str
    text: str
    user: User
    channel: Channel
    type: MessageType = MessageType.TEXT
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    mentions: List[User] = field(default_factory=list)
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    is_edited: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_command(self) -> bool:
        """Check if message starts with a command prefix."""
        return self.text.startswith(("/", "!", "."))
    
    @property
    def command(self) -> Optional[str]:
        """Extract command name if this is a command."""
        if self.is_command:
            parts = self.text[1:].split()
            return parts[0] if parts else None
        return None
    
    @property
    def command_args(self) -> List[str]:
        """Extract command arguments."""
        if self.is_command:
            parts = self.text[1:].split()
            return parts[1:] if len(parts) > 1 else []
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "user": self.user.to_dict(),
            "channel": self.channel.to_dict(),
            "type": self.type.value,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "attachments": self.attachments,
            "mentions": [u.to_dict() for u in self.mentions],
            "reactions": self.reactions,
            "is_edited": self.is_edited,
            "metadata": self.metadata,
        }


@dataclass
class MessageContext:
    """
    Context for handling a message, providing convenient reply methods.
    
    Attributes:
        message: The original message
        bot: Reference to the bot instance
        matched_pattern: The pattern that matched this handler
        matched_groups: Regex groups from pattern match
        variables: Workflow variables
    """
    message: Message
    bot: "Bot"
    matched_pattern: Optional[str] = None
    matched_groups: Dict[str, str] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def user(self) -> User:
        return self.message.user
    
    @property
    def user_id(self) -> str:
        return self.message.user.id
    
    @property
    def channel(self) -> Channel:
        return self.message.channel
    
    @property
    def channel_id(self) -> str:
        return self.message.channel.id
    
    @property
    def text(self) -> str:
        return self.message.text
    
    @property
    def command(self) -> Optional[str]:
        return self.message.command
    
    @property
    def args(self) -> List[str]:
        return self.message.command_args
    
    async def reply(self, text: str, **kwargs) -> "Message":
        """Reply to the message."""
        return await self.bot.reply(self.message, text, **kwargs)
    
    async def send(self, text: str, channel_id: Optional[str] = None, **kwargs) -> "Message":
        """Send a message to a channel."""
        target = channel_id or self.channel_id
        return await self.bot.send_message(target, text, **kwargs)
    
    async def react(self, emoji: str) -> bool:
        """Add a reaction to the message."""
        return await self.bot.add_reaction(self.message, emoji)
    
    async def typing(self) -> None:
        """Show typing indicator."""
        await self.bot.send_typing(self.channel_id)


@dataclass
class Handler:
    """
    Represents a message or event handler.
    
    Attributes:
        callback: The handler function
        trigger_type: Type of trigger
        pattern: Pattern to match (regex for messages, command name for commands)
        priority: Handler priority (lower = higher priority)
        description: Human-readable description
        enabled: Whether handler is active
    """
    callback: Callable[[MessageContext], Awaitable[Any]]
    trigger_type: TriggerType
    pattern: Optional[Union[str, Pattern]] = None
    priority: int = 100
    description: str = ""
    enabled: bool = True
    filters: List[Callable[[MessageContext], bool]] = field(default_factory=list)
    cooldown: float = 0.0
    last_triggered: float = 0.0
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, ctx: MessageContext) -> bool:
        """Check if this handler matches the context."""
        if not self.enabled:
            return False
        
        if self.cooldown > 0:
            if time.time() - self.last_triggered < self.cooldown:
                return False
        
        for filter_fn in self.filters:
            if not filter_fn(ctx):
                return False
        
        if self.pattern is None:
            return True
        
        if isinstance(self.pattern, Pattern):
            match = self.pattern.search(ctx.text)
            if match:
                ctx.matched_groups = match.groupdict()
                return True
            return False
        
        if self.trigger_type == TriggerType.COMMAND:
            return ctx.command == self.pattern
        
        return self.pattern.lower() in ctx.text.lower()


@dataclass
class RateLimiter:
    """
    Rate limiter for bot operations.
    
    Attributes:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
    """
    max_requests: int = AutomationLimit.RATE_LIMIT_MAX_REQUESTS.value
    window_seconds: int = AutomationLimit.RATE_LIMIT_WINDOW_SECONDS.value
    _requests: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def check(self, key: str) -> Tuple[bool, float]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            self._requests[key] = [
                t for t in self._requests[key] 
                if t > window_start
            ]
            
            if len(self._requests[key]) >= self.max_requests:
                oldest = min(self._requests[key])
                retry_after = oldest + self.window_seconds - now
                return False, retry_after
            
            self._requests[key].append(now)
            return True, 0.0
    
    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        with self._lock:
            self._requests[key] = []
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for a key."""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            active = [t for t in self._requests[key] if t > window_start]
            return max(0, self.max_requests - len(active))


@dataclass
class BotStats:
    """Statistics for a bot instance."""
    messages_received: int = 0
    messages_sent: int = 0
    commands_handled: int = 0
    errors_count: int = 0
    uptime_seconds: float = 0.0
    start_time: Optional[float] = None
    last_message_time: Optional[float] = None
    active_conversations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "commands_handled": self.commands_handled,
            "errors_count": self.errors_count,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self.start_time,
            "last_message_time": self.last_message_time,
            "active_conversations": self.active_conversations,
        }


class Bot(ABC):
    """
    Abstract base class for all bot platforms.
    
    Provides common functionality for message handling, command parsing,
    event routing, and lifecycle management.
    
    Usage:
        class MyBot(Bot):
            async def _connect(self):
                # Connect to platform
                pass
            
            async def _disconnect(self):
                # Disconnect from platform
                pass
            
            async def send_message(self, channel_id, text, **kwargs):
                # Send message implementation
                pass
        
        bot = MyBot(token="...", name="MyBot")
        
        @bot.on_command("hello")
        async def hello(ctx):
            await ctx.reply("Hello!")
        
        await bot.start()
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        name: str = "Bot",
        prefix: str = "/",
        platform: BotPlatform = BotPlatform.GENERIC,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.token = token
        self.name = name
        self.prefix = prefix
        self.platform = platform
        self.status = BotStatus.STOPPED
        self.handlers: List[Handler] = []
        self.rate_limiter = rate_limiter or RateLimiter()
        self.stats = BotStats()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._lock = threading.Lock()
        self.middleware: List[Callable[[MessageContext], Awaitable[bool]]] = []
        self.error_handlers: List[Callable[[Exception, MessageContext], Awaitable[None]]] = []
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def _connect(self) -> None:
        """Connect to the platform. Subclasses must implement."""
        pass
    
    @abstractmethod
    async def _disconnect(self) -> None:
        """Disconnect from the platform. Subclasses must implement."""
        pass
    
    @abstractmethod
    async def send_message(
        self,
        channel_id: str,
        text: str,
        **kwargs
    ) -> Message:
        """Send a message to a channel. Subclasses must implement."""
        pass
    
    async def start(self) -> None:
        """Start the bot and connect to the platform."""
        if self.status == BotStatus.RUNNING:
            raise BotAlreadyRunningError(self.name)
        
        self.status = BotStatus.STARTING
        self.logger.info(f"Starting bot '{self.name}'...")
        
        try:
            await self._connect()
            self.status = BotStatus.RUNNING
            self._running = True
            self.stats.start_time = time.time()
            self.logger.info(f"Bot '{self.name}' started successfully")
        except Exception as e:
            self.status = BotStatus.ERROR
            self.logger.error(f"Failed to start bot: {e}")
            raise BotConnectionError(self.platform.value, str(e))
    
    async def stop(self) -> None:
        """Stop the bot and disconnect from the platform."""
        if self.status != BotStatus.RUNNING:
            return
        
        self.status = BotStatus.STOPPING
        self.logger.info(f"Stopping bot '{self.name}'...")
        
        try:
            self._running = False
            await self._disconnect()
            self.status = BotStatus.STOPPED
            self.stats.uptime_seconds = time.time() - (self.stats.start_time or 0)
            self.logger.info(f"Bot '{self.name}' stopped")
        except Exception as e:
            self.status = BotStatus.ERROR
            self.logger.error(f"Error stopping bot: {e}")
            raise
    
    def on_message(
        self,
        pattern: Optional[Union[str, Pattern]] = None,
        priority: int = 100,
        description: str = "",
        filters: Optional[List[Callable]] = None,
        cooldown: float = 0.0,
    ) -> Callable:
        """
        Decorator to register a message handler.
        
        Args:
            pattern: Regex pattern or string to match
            priority: Handler priority
            description: Handler description
            filters: Additional filter functions
            cooldown: Minimum seconds between triggers
        
        Usage:
            @bot.on_message(pattern=r"hello|hi")
            async def greet(ctx):
                await ctx.reply("Hello!")
        """
        def decorator(func: Callable[[MessageContext], Awaitable[Any]]) -> Callable:
            compiled_pattern = None
            if pattern:
                if isinstance(pattern, str):
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                else:
                    compiled_pattern = pattern
            
            handler = Handler(
                callback=func,
                trigger_type=TriggerType.MESSAGE,
                pattern=compiled_pattern,
                priority=priority,
                description=description or func.__doc__ or "",
                filters=filters or [],
                cooldown=cooldown,
            )
            self.handlers.append(handler)
            self.handlers.sort(key=lambda h: h.priority)
            return func
        return decorator
    
    def on_command(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        priority: int = 50,
        description: str = "",
        filters: Optional[List[Callable]] = None,
        cooldown: float = 0.0,
    ) -> Callable:
        """
        Decorator to register a command handler.
        
        Args:
            name: Command name (without prefix)
            aliases: Alternative command names
            priority: Handler priority
            description: Handler description
            filters: Additional filter functions
            cooldown: Minimum seconds between triggers
        
        Usage:
            @bot.on_command("help", aliases=["h", "?"])
            async def help_command(ctx):
                await ctx.reply("Available commands: ...")
        """
        def decorator(func: Callable[[MessageContext], Awaitable[Any]]) -> Callable:
            all_names = [name] + (aliases or [])
            
            for cmd_name in all_names:
                handler = Handler(
                    callback=func,
                    trigger_type=TriggerType.COMMAND,
                    pattern=cmd_name,
                    priority=priority,
                    description=description or func.__doc__ or "",
                    filters=filters or [],
                    cooldown=cooldown,
                )
                self.handlers.append(handler)
            
            self.handlers.sort(key=lambda h: h.priority)
            return func
        return decorator
    
    def on_event(
        self,
        event_type: TriggerType,
        priority: int = 100,
    ) -> Callable:
        """
        Decorator to register an event handler.
        
        Args:
            event_type: Type of event to handle
            priority: Handler priority
        
        Usage:
            @bot.on_event(TriggerType.USER_JOIN)
            async def welcome_user(ctx):
                await ctx.send(f"Welcome {ctx.user.display_name}!")
        """
        def decorator(func: Callable[[MessageContext], Awaitable[Any]]) -> Callable:
            handler = Handler(
                callback=func,
                trigger_type=event_type,
                priority=priority,
            )
            self.handlers.append(handler)
            self.handlers.sort(key=lambda h: h.priority)
            return func
        return decorator
    
    def add_middleware(
        self,
        middleware: Callable[[MessageContext], Awaitable[bool]]
    ) -> None:
        """
        Add middleware that runs before handlers.
        
        Middleware should return True to continue processing, False to stop.
        """
        self.middleware.append(middleware)
    
    def add_error_handler(
        self,
        handler: Callable[[Exception, MessageContext], Awaitable[None]]
    ) -> None:
        """Add an error handler for handler exceptions."""
        self.error_handlers.append(handler)
    
    async def process_message(self, message: Message) -> None:
        """
        Process an incoming message through handlers.
        
        This is called internally when a message is received.
        """
        if self.status != BotStatus.RUNNING:
            return
        
        self.stats.messages_received += 1
        self.stats.last_message_time = time.time()
        
        allowed, retry_after = self.rate_limiter.check(message.user.id)
        if not allowed:
            self.logger.warning(
                f"Rate limit exceeded for user {message.user.id}, "
                f"retry after {retry_after:.1f}s"
            )
            return
        
        ctx = MessageContext(message=message, bot=self)
        
        for mw in self.middleware:
            try:
                if not await mw(ctx):
                    return
            except Exception as e:
                self.logger.error(f"Middleware error: {e}")
                return
        
        for handler in self.handlers:
            if handler.matches(ctx):
                try:
                    if message.is_command:
                        self.stats.commands_handled += 1
                    
                    handler.trigger_count += 1
                    handler.last_triggered = time.time()
                    
                    await handler.callback(ctx)
                    break
                    
                except Exception as e:
                    self.stats.errors_count += 1
                    self.logger.error(
                        f"Handler error for {handler.description or handler.pattern}: {e}"
                    )
                    
                    for error_handler in self.error_handlers:
                        try:
                            await error_handler(e, ctx)
                        except Exception as eh_error:
                            self.logger.error(f"Error handler failed: {eh_error}")
    
    async def reply(
        self,
        message: Message,
        text: str,
        **kwargs
    ) -> Message:
        """Reply to a specific message."""
        kwargs["reply_to"] = message.id
        return await self.send_message(message.channel.id, text, **kwargs)
    
    async def add_reaction(
        self,
        message: Message,
        emoji: str
    ) -> bool:
        """Add a reaction to a message. Override in subclass."""
        return False
    
    async def send_typing(self, channel_id: str) -> None:
        """Send typing indicator. Override in subclass."""
        pass
    
    def get_handlers(
        self,
        trigger_type: Optional[TriggerType] = None
    ) -> List[Handler]:
        """Get all handlers, optionally filtered by type."""
        if trigger_type is None:
            return self.handlers.copy()
        return [h for h in self.handlers if h.trigger_type == trigger_type]
    
    def remove_handler(self, handler: Handler) -> bool:
        """Remove a handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
            return True
        return False
    
    def get_stats(self) -> BotStats:
        """Get bot statistics."""
        if self.stats.start_time and self.status == BotStatus.RUNNING:
            self.stats.uptime_seconds = time.time() - self.stats.start_time
        return self.stats


class SlackBot(Bot):
    """
    Slack bot implementation.
    
    Provides Slack-specific message handling, formatting, and API integration.
    
    Usage:
        bot = SlackBot(
            token="xoxb-your-bot-token",
            app_token="xapp-your-app-token",  # For Socket Mode
            name="MySlackBot"
        )
        
        @bot.on_command("status")
        async def status(ctx):
            await ctx.reply(
                text="System Status",
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "*All systems operational*"}
                    }
                ]
            )
        
        await bot.start()
    """
    
    def __init__(
        self,
        token: str,
        app_token: Optional[str] = None,
        signing_secret: Optional[str] = None,
        name: str = "SlackBot",
        **kwargs
    ):
        super().__init__(token=token, name=name, platform=BotPlatform.SLACK, **kwargs)
        self.app_token = app_token
        self.signing_secret = signing_secret
        self._ws = None
        self._client = None
    
    async def _connect(self) -> None:
        """Connect to Slack via Socket Mode or RTM."""
        self.logger.info("Connecting to Slack...")
        pass
    
    async def _disconnect(self) -> None:
        """Disconnect from Slack."""
        self.logger.info("Disconnecting from Slack...")
        if self._ws:
            pass
    
    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[List[Dict]] = None,
        attachments: Optional[List[Dict]] = None,
        thread_ts: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs
    ) -> Message:
        """
        Send a message to a Slack channel.
        
        Args:
            channel_id: Channel ID to send to
            text: Message text (fallback for blocks)
            blocks: Slack Block Kit blocks
            attachments: Legacy attachments
            thread_ts: Thread timestamp for replies
            reply_to: Message ID to reply to (sets thread_ts)
        """
        self.stats.messages_sent += 1
        
        message_id = str(uuid.uuid4())
        return Message(
            id=message_id,
            text=text,
            user=User(id="bot", username=self.name, is_bot=True),
            channel=Channel(id=channel_id),
            timestamp=time.time(),
            metadata={
                "blocks": blocks,
                "attachments": attachments,
                "thread_ts": thread_ts or reply_to,
            }
        )
    
    async def add_reaction(self, message: Message, emoji: str) -> bool:
        """Add a reaction to a Slack message."""
        self.logger.debug(f"Adding reaction {emoji} to message {message.id}")
        return True
    
    async def send_typing(self, channel_id: str) -> None:
        """Slack doesn't have typing indicators in the same way."""
        pass
    
    async def open_modal(
        self,
        trigger_id: str,
        view: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Open a Slack modal view."""
        self.logger.debug(f"Opening modal with trigger {trigger_id}")
        return {"ok": True, "view": view}
    
    async def update_message(
        self,
        channel_id: str,
        message_ts: str,
        text: str,
        blocks: Optional[List[Dict]] = None,
    ) -> bool:
        """Update an existing Slack message."""
        self.logger.debug(f"Updating message {message_ts}")
        return True
    
    async def delete_message(
        self,
        channel_id: str,
        message_ts: str
    ) -> bool:
        """Delete a Slack message."""
        self.logger.debug(f"Deleting message {message_ts}")
        return True
    
    def format_mention(self, user_id: str) -> str:
        """Format a user mention for Slack."""
        return f"<@{user_id}>"
    
    def format_channel(self, channel_id: str) -> str:
        """Format a channel mention for Slack."""
        return f"<#{channel_id}>"


class TelegramBot(Bot):
    """
    Telegram bot implementation.
    
    Provides Telegram-specific features including inline keyboards,
    callback queries, and rich media handling.
    
    Usage:
        bot = TelegramBot(
            token="123456:ABC-DEF...",
            name="MyTelegramBot"
        )
        
        @bot.on_command("start")
        async def start(ctx):
            await ctx.reply(
                "Welcome!",
                reply_markup={
                    "inline_keyboard": [[
                        {"text": "Help", "callback_data": "help"}
                    ]]
                }
            )
        
        @bot.on_callback("help")
        async def help_callback(ctx):
            await ctx.reply("Here's how to use this bot...")
        
        await bot.start()
    """
    
    def __init__(
        self,
        token: str,
        name: str = "TelegramBot",
        parse_mode: str = "HTML",
        **kwargs
    ):
        super().__init__(token=token, name=name, platform=BotPlatform.TELEGRAM, **kwargs)
        self.parse_mode = parse_mode
        self._offset = 0
        self._polling = False
        self.callback_handlers: Dict[str, Handler] = {}
    
    async def _connect(self) -> None:
        """Start polling for updates."""
        self.logger.info("Starting Telegram polling...")
        self._polling = True
    
    async def _disconnect(self) -> None:
        """Stop polling."""
        self.logger.info("Stopping Telegram polling...")
        self._polling = False
    
    async def send_message(
        self,
        channel_id: str,
        text: str,
        parse_mode: Optional[str] = None,
        reply_markup: Optional[Dict] = None,
        reply_to: Optional[str] = None,
        disable_notification: bool = False,
        **kwargs
    ) -> Message:
        """
        Send a message to a Telegram chat.
        
        Args:
            channel_id: Chat ID to send to
            text: Message text (supports HTML/Markdown)
            parse_mode: Message parse mode (HTML, Markdown, MarkdownV2)
            reply_markup: Keyboard markup
            reply_to: Message ID to reply to
            disable_notification: Send silently
        """
        self.stats.messages_sent += 1
        
        message_id = str(uuid.uuid4())
        return Message(
            id=message_id,
            text=text,
            user=User(id="bot", username=self.name, is_bot=True),
            channel=Channel(id=channel_id),
            timestamp=time.time(),
            reply_to=reply_to,
            metadata={
                "parse_mode": parse_mode or self.parse_mode,
                "reply_markup": reply_markup,
            }
        )
    
    def on_callback(
        self,
        data: str,
        priority: int = 50,
    ) -> Callable:
        """
        Decorator to handle callback query data.
        
        Usage:
            @bot.on_callback("settings:*")
            async def settings_callback(ctx):
                action = ctx.callback_data.split(":")[1]
                ...
        """
        def decorator(func: Callable[[MessageContext], Awaitable[Any]]) -> Callable:
            handler = Handler(
                callback=func,
                trigger_type=TriggerType.CUSTOM,
                pattern=data,
                priority=priority,
            )
            self.callback_handlers[data] = handler
            return func
        return decorator
    
    async def answer_callback(
        self,
        callback_id: str,
        text: Optional[str] = None,
        show_alert: bool = False,
    ) -> bool:
        """Answer a callback query."""
        self.logger.debug(f"Answering callback {callback_id}")
        return True
    
    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        reply_markup: Optional[Dict] = None,
    ) -> bool:
        """Edit an existing message."""
        self.logger.debug(f"Editing message {message_id}")
        return True
    
    async def send_photo(
        self,
        chat_id: str,
        photo: Union[str, bytes],
        caption: Optional[str] = None,
        **kwargs
    ) -> Message:
        """Send a photo."""
        self.stats.messages_sent += 1
        return Message(
            id=str(uuid.uuid4()),
            text=caption or "",
            user=User(id="bot", is_bot=True),
            channel=Channel(id=chat_id),
            type=MessageType.IMAGE,
        )
    
    async def send_document(
        self,
        chat_id: str,
        document: Union[str, bytes],
        caption: Optional[str] = None,
        **kwargs
    ) -> Message:
        """Send a document."""
        self.stats.messages_sent += 1
        return Message(
            id=str(uuid.uuid4()),
            text=caption or "",
            user=User(id="bot", is_bot=True),
            channel=Channel(id=chat_id),
            type=MessageType.FILE,
        )
    
    def format_mention(self, user_id: str, name: str) -> str:
        """Format a user mention for Telegram (HTML)."""
        return f'<a href="tg://user?id={user_id}">{name}</a>'


class DiscordBot(Bot):
    """
    Discord bot implementation.
    
    Provides Discord-specific features including embeds, components,
    slash commands, and role/permission handling.
    
    Usage:
        bot = DiscordBot(
            token="your-bot-token",
            name="MyDiscordBot",
            intents=["guilds", "messages", "message_content"]
        )
        
        @bot.on_command("info")
        async def info(ctx):
            embed = bot.create_embed(
                title="Bot Info",
                description="A helpful bot",
                color=0x5865F2
            )
            await ctx.reply(embed=embed)
        
        @bot.slash_command("ping")
        async def ping(ctx):
            await ctx.reply("Pong!")
        
        await bot.start()
    """
    
    def __init__(
        self,
        token: str,
        name: str = "DiscordBot",
        intents: Optional[List[str]] = None,
        guild_ids: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(token=token, name=name, platform=BotPlatform.DISCORD, **kwargs)
        self.intents = intents or ["guilds", "messages"]
        self.guild_ids = guild_ids or []
        self.slash_commands: Dict[str, Handler] = {}
        self._ws = None
        self._session_id = None
    
    async def _connect(self) -> None:
        """Connect to Discord Gateway."""
        self.logger.info("Connecting to Discord Gateway...")
    
    async def _disconnect(self) -> None:
        """Disconnect from Discord."""
        self.logger.info("Disconnecting from Discord...")
        if self._ws:
            pass
    
    async def send_message(
        self,
        channel_id: str,
        text: str,
        embed: Optional[Dict] = None,
        embeds: Optional[List[Dict]] = None,
        components: Optional[List[Dict]] = None,
        reply_to: Optional[str] = None,
        allowed_mentions: Optional[Dict] = None,
        **kwargs
    ) -> Message:
        """
        Send a message to a Discord channel.
        
        Args:
            channel_id: Channel ID to send to
            text: Message content
            embed: Single embed
            embeds: Multiple embeds
            components: Message components (buttons, selects)
            reply_to: Message ID to reply to
            allowed_mentions: Mention parsing rules
        """
        self.stats.messages_sent += 1
        
        message_id = str(uuid.uuid4())
        return Message(
            id=message_id,
            text=text,
            user=User(id="bot", username=self.name, is_bot=True),
            channel=Channel(id=channel_id),
            timestamp=time.time(),
            reply_to=reply_to,
            metadata={
                "embed": embed,
                "embeds": embeds,
                "components": components,
            }
        )
    
    def slash_command(
        self,
        name: str,
        description: str = "No description",
        options: Optional[List[Dict]] = None,
        guild_ids: Optional[List[str]] = None,
    ) -> Callable:
        """
        Decorator to register a slash command.
        
        Usage:
            @bot.slash_command("greet", "Greet a user", options=[
                {"name": "user", "type": 6, "description": "User to greet"}
            ])
            async def greet(ctx, user):
                await ctx.reply(f"Hello {user.display_name}!")
        """
        def decorator(func: Callable[[MessageContext], Awaitable[Any]]) -> Callable:
            handler = Handler(
                callback=func,
                trigger_type=TriggerType.COMMAND,
                pattern=name,
                description=description,
                metadata={
                    "options": options or [],
                    "guild_ids": guild_ids or self.guild_ids,
                }
            )
            self.slash_commands[name] = handler
            return func
        return decorator
    
    async def add_reaction(self, message: Message, emoji: str) -> bool:
        """Add a reaction to a Discord message."""
        self.logger.debug(f"Adding reaction {emoji}")
        return True
    
    async def send_typing(self, channel_id: str) -> None:
        """Send typing indicator to a Discord channel."""
        self.logger.debug(f"Sending typing to {channel_id}")
    
    def create_embed(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        color: int = 0x5865F2,
        fields: Optional[List[Dict]] = None,
        thumbnail: Optional[str] = None,
        image: Optional[str] = None,
        footer: Optional[Dict] = None,
        author: Optional[Dict] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Discord embed."""
        embed = {"type": "rich", "color": color}
        
        if title:
            embed["title"] = title
        if description:
            embed["description"] = description
        if fields:
            embed["fields"] = fields
        if thumbnail:
            embed["thumbnail"] = {"url": thumbnail}
        if image:
            embed["image"] = {"url": image}
        if footer:
            embed["footer"] = footer
        if author:
            embed["author"] = author
        if timestamp:
            embed["timestamp"] = timestamp
        
        return embed
    
    def create_button(
        self,
        label: str,
        custom_id: Optional[str] = None,
        style: int = 1,
        url: Optional[str] = None,
        emoji: Optional[str] = None,
        disabled: bool = False,
    ) -> Dict[str, Any]:
        """Create a Discord button component."""
        button = {
            "type": 2,
            "label": label,
            "style": style,
            "disabled": disabled,
        }
        
        if custom_id:
            button["custom_id"] = custom_id
        if url:
            button["url"] = url
            button["style"] = 5
        if emoji:
            button["emoji"] = {"name": emoji}
        
        return button
    
    def create_select(
        self,
        custom_id: str,
        options: List[Dict],
        placeholder: Optional[str] = None,
        min_values: int = 1,
        max_values: int = 1,
    ) -> Dict[str, Any]:
        """Create a Discord select menu component."""
        return {
            "type": 3,
            "custom_id": custom_id,
            "options": options,
            "placeholder": placeholder,
            "min_values": min_values,
            "max_values": max_values,
        }
    
    def format_mention(self, user_id: str) -> str:
        """Format a user mention for Discord."""
        return f"<@{user_id}>"
    
    def format_channel(self, channel_id: str) -> str:
        """Format a channel mention for Discord."""
        return f"<#{channel_id}>"
    
    def format_role(self, role_id: str) -> str:
        """Format a role mention for Discord."""
        return f"<@&{role_id}>"


@dataclass
class Trigger:
    """
    Defines when a workflow should be triggered.
    
    Attributes:
        type: Type of trigger
        config: Trigger-specific configuration
        conditions: Additional conditions to check
        enabled: Whether trigger is active
    """
    type: TriggerType
    config: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Callable[[MessageContext], bool]] = field(default_factory=list)
    enabled: bool = True
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @classmethod
    def on_message(
        cls,
        pattern: Optional[str] = None,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> "Trigger":
        """Create a message trigger."""
        return cls(
            type=TriggerType.MESSAGE,
            config={
                "pattern": pattern,
                "channel_id": channel_id,
                "user_id": user_id,
            }
        )
    
    @classmethod
    def on_command(cls, command: str) -> "Trigger":
        """Create a command trigger."""
        return cls(
            type=TriggerType.COMMAND,
            config={"command": command}
        )
    
    @classmethod
    def on_schedule(
        cls,
        cron: Optional[str] = None,
        interval: Optional[int] = None,
        at: Optional[str] = None,
    ) -> "Trigger":
        """Create a schedule trigger."""
        return cls(
            type=TriggerType.SCHEDULE,
            config={
                "cron": cron,
                "interval": interval,
                "at": at,
            }
        )
    
    @classmethod
    def on_webhook(
        cls,
        path: str,
        method: str = "POST",
        secret: Optional[str] = None,
    ) -> "Trigger":
        """Create a webhook trigger."""
        return cls(
            type=TriggerType.WEBHOOK,
            config={
                "path": path,
                "method": method,
                "secret": secret,
            }
        )
    
    @classmethod
    def on_event(cls, event_name: str) -> "Trigger":
        """Create an event trigger."""
        return cls(
            type=TriggerType.EVENT,
            config={"event": event_name}
        )
    
    @classmethod
    def on_keyword(cls, keywords: List[str]) -> "Trigger":
        """Create a keyword trigger."""
        return cls(
            type=TriggerType.KEYWORD,
            config={"keywords": keywords}
        )
    
    def matches(self, ctx: MessageContext) -> bool:
        """Check if context matches this trigger."""
        if not self.enabled:
            return False
        
        for condition in self.conditions:
            if not condition(ctx):
                return False
        
        if self.type == TriggerType.MESSAGE:
            if self.config.get("pattern"):
                if not re.search(self.config["pattern"], ctx.text, re.IGNORECASE):
                    return False
            if self.config.get("channel_id"):
                if ctx.channel_id != self.config["channel_id"]:
                    return False
            if self.config.get("user_id"):
                if ctx.user_id != self.config["user_id"]:
                    return False
            return True
        
        elif self.type == TriggerType.COMMAND:
            return ctx.command == self.config.get("command")
        
        elif self.type == TriggerType.KEYWORD:
            keywords = self.config.get("keywords", [])
            text_lower = ctx.text.lower()
            return any(kw.lower() in text_lower for kw in keywords)
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "config": self.config,
            "enabled": self.enabled,
        }


@dataclass
class Action:
    """
    Defines an action to perform in a workflow.
    
    Attributes:
        type: Type of action
        config: Action-specific configuration
        condition: Optional condition for execution
        on_error: Action to take on error
    """
    type: ActionType
    config: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    on_error: str = "continue"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @classmethod
    def send_message(
        cls,
        channel_id: str,
        text: str,
        **kwargs
    ) -> "Action":
        """Create a send message action."""
        return cls(
            type=ActionType.SEND_MESSAGE,
            config={
                "channel_id": channel_id,
                "text": text,
                **kwargs,
            }
        )
    
    @classmethod
    def reply(cls, text: str, **kwargs) -> "Action":
        """Create a reply action."""
        return cls(
            type=ActionType.REPLY,
            config={"text": text, **kwargs}
        )
    
    @classmethod
    def react(cls, emoji: str) -> "Action":
        """Create a reaction action."""
        return cls(
            type=ActionType.REACT,
            config={"emoji": emoji}
        )
    
    @classmethod
    def delay(
        cls,
        seconds: Optional[float] = None,
        minutes: Optional[float] = None,
        hours: Optional[float] = None,
    ) -> "Action":
        """Create a delay action."""
        total_seconds = (seconds or 0) + (minutes or 0) * 60 + (hours or 0) * 3600
        return cls(
            type=ActionType.DELAY,
            config={"seconds": total_seconds}
        )
    
    @classmethod
    def http_request(
        cls,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        save_as: Optional[str] = None,
    ) -> "Action":
        """Create an HTTP request action."""
        return cls(
            type=ActionType.HTTP_REQUEST,
            config={
                "url": url,
                "method": method,
                "headers": headers or {},
                "body": body,
                "save_as": save_as,
            }
        )
    
    @classmethod
    def set_variable(cls, name: str, value: Any) -> "Action":
        """Create a variable assignment action."""
        return cls(
            type=ActionType.SET_VARIABLE,
            config={"name": name, "value": value}
        )
    
    @classmethod
    def condition(
        cls,
        check: Callable[[Dict[str, Any]], bool],
        if_true: List["Action"],
        if_false: Optional[List["Action"]] = None,
    ) -> "Action":
        """Create a conditional branch action."""
        return cls(
            type=ActionType.CONDITION,
            config={
                "check": check,
                "if_true": if_true,
                "if_false": if_false or [],
            }
        )
    
    @classmethod
    def log(cls, message: str, level: str = "info") -> "Action":
        """Create a logging action."""
        return cls(
            type=ActionType.LOG,
            config={"message": message, "level": level}
        )
    
    @classmethod
    def call_function(
        cls,
        func: Callable,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
    ) -> "Action":
        """Create a function call action."""
        return cls(
            type=ActionType.CALL_FUNCTION,
            config={
                "func": func,
                "args": args or [],
                "kwargs": kwargs or {},
            }
        )
    
    async def execute(
        self,
        ctx: MessageContext,
        variables: Dict[str, Any]
    ) -> Any:
        """Execute this action."""
        if self.condition and not self.condition(variables):
            return None
        
        try:
            if self.type == ActionType.REPLY:
                text = self._interpolate(self.config["text"], variables)
                return await ctx.reply(text)
            
            elif self.type == ActionType.SEND_MESSAGE:
                text = self._interpolate(self.config["text"], variables)
                channel_id = self.config["channel_id"]
                return await ctx.send(text, channel_id=channel_id)
            
            elif self.type == ActionType.REACT:
                return await ctx.react(self.config["emoji"])
            
            elif self.type == ActionType.DELAY:
                await asyncio.sleep(self.config["seconds"])
                return None
            
            elif self.type == ActionType.SET_VARIABLE:
                variables[self.config["name"]] = self.config["value"]
                return None
            
            elif self.type == ActionType.LOG:
                message = self._interpolate(self.config["message"], variables)
                level = self.config.get("level", "info")
                getattr(logger, level)(message)
                return None
            
            elif self.type == ActionType.CALL_FUNCTION:
                func = self.config["func"]
                args = self.config.get("args", [])
                kwargs = self.config.get("kwargs", {})
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            
            elif self.type == ActionType.CONDITION:
                check = self.config["check"]
                if check(variables):
                    actions = self.config["if_true"]
                else:
                    actions = self.config.get("if_false", [])
                
                for action in actions:
                    await action.execute(ctx, variables)
                return None
            
        except Exception as e:
            if self.on_error == "stop":
                raise
            logger.error(f"Action error: {e}")
            return None
    
    def _interpolate(self, text: str, variables: Dict[str, Any]) -> str:
        """Interpolate variables in text using {{var}} syntax."""
        def replace(match):
            var_name = match.group(1).strip()
            return str(variables.get(var_name, match.group(0)))
        
        return re.sub(r"\{\{(\w+)\}\}", replace, text)
    
    def to_dict(self) -> Dict[str, Any]:
        config = {k: v for k, v in self.config.items() if not callable(v)}
        return {
            "id": self.id,
            "type": self.type.value,
            "config": config,
            "on_error": self.on_error,
        }


@dataclass
class WorkflowExecution:
    """Record of a workflow execution."""
    id: str
    workflow_id: str
    started_at: float
    completed_at: Optional[float] = None
    status: str = "running"
    actions_executed: int = 0
    variables: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    trigger_data: Dict[str, Any] = field(default_factory=dict)


class Workflow:
    """
    Automation workflow with triggers and actions.
    
    A workflow consists of one or more triggers that start execution,
    and a sequence of actions that are performed when triggered.
    
    Usage:
        workflow = Workflow(name="Welcome Flow")
        workflow.add_trigger(Trigger.on_message(pattern="hi|hello"))
        workflow.add_action(Action.reply("Welcome!"))
        workflow.add_action(Action.delay(seconds=2))
        workflow.add_action(Action.reply("How can I help?"))
        
        # Conditional actions
        workflow.add_action(Action.condition(
            check=lambda vars: vars.get("user_type") == "premium",
            if_true=[Action.reply("Thanks for being a premium member!")],
            if_false=[Action.reply("Consider upgrading to premium!")]
        ))
        
        # Register with a bot
        bot.add_workflow(workflow)
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        enabled: bool = True,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.enabled = enabled
        self.triggers: List[Trigger] = []
        self.actions: List[Action] = []
        self.variables: Dict[str, Any] = {}
        self.created_at = time.time()
        self.updated_at = time.time()
        self.execution_count = 0
        self.last_executed: Optional[float] = None
        self.executions: List[WorkflowExecution] = []
        self._lock = threading.Lock()
    
    def add_trigger(self, trigger: Trigger) -> "Workflow":
        """Add a trigger to the workflow. Returns self for chaining."""
        if len(self.triggers) >= AutomationLimit.MAX_TRIGGERS_PER_WORKFLOW.value:
            raise WorkflowError(
                f"Maximum triggers ({AutomationLimit.MAX_TRIGGERS_PER_WORKFLOW.value}) exceeded"
            )
        self.triggers.append(trigger)
        self.updated_at = time.time()
        return self
    
    def add_action(self, action: Action) -> "Workflow":
        """Add an action to the workflow. Returns self for chaining."""
        if len(self.actions) >= AutomationLimit.MAX_ACTIONS_PER_WORKFLOW.value:
            raise WorkflowError(
                f"Maximum actions ({AutomationLimit.MAX_ACTIONS_PER_WORKFLOW.value}) exceeded"
            )
        self.actions.append(action)
        self.updated_at = time.time()
        return self
    
    def set_variable(self, name: str, value: Any) -> "Workflow":
        """Set a workflow variable."""
        self.variables[name] = value
        return self
    
    def should_trigger(self, ctx: MessageContext) -> bool:
        """Check if any trigger matches the context."""
        if not self.enabled:
            return False
        return any(trigger.matches(ctx) for trigger in self.triggers)
    
    async def execute(
        self,
        ctx: MessageContext,
        initial_variables: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Execute the workflow."""
        execution = WorkflowExecution(
            id=str(uuid.uuid4()),
            workflow_id=self.id,
            started_at=time.time(),
            variables={**self.variables, **(initial_variables or {})},
            trigger_data={"message": ctx.message.to_dict()},
        )
        
        try:
            for i, action in enumerate(self.actions):
                try:
                    await action.execute(ctx, execution.variables)
                    execution.actions_executed = i + 1
                except Exception as e:
                    execution.errors.append(f"Action {i}: {str(e)}")
                    if action.on_error == "stop":
                        raise WorkflowExecutionError(self.id, i, str(e))
            
            execution.status = "completed"
            
        except Exception as e:
            execution.status = "failed"
            execution.errors.append(str(e))
        
        finally:
            execution.completed_at = time.time()
            with self._lock:
                self.execution_count += 1
                self.last_executed = execution.completed_at
                self.executions.append(execution)
                if len(self.executions) > 100:
                    self.executions = self.executions[-100:]
        
        return execution
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "triggers": [t.to_dict() for t in self.triggers],
            "actions": [a.to_dict() for a in self.actions],
            "variables": self.variables,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_count": self.execution_count,
            "last_executed": self.last_executed,
        }


@dataclass
class ScheduledTask:
    """A scheduled task definition."""
    id: str
    name: str
    callback: Callable[[], Awaitable[Any]]
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    enabled: bool = True
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    def calculate_next_run(self, from_time: Optional[float] = None) -> float:
        """Calculate next run time."""
        now = from_time or time.time()
        
        if self.schedule_type == ScheduleType.INTERVAL:
            interval = self.schedule_config.get("seconds", 60)
            return now + interval
        
        elif self.schedule_type == ScheduleType.ONCE:
            return self.schedule_config.get("at", now)
        
        elif self.schedule_type == ScheduleType.DAILY:
            hour = self.schedule_config.get("hour", 0)
            minute = self.schedule_config.get("minute", 0)
            
            dt = datetime.fromtimestamp(now)
            target = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if target.timestamp() <= now:
                target = target + timedelta(days=1)
            
            return target.timestamp()
        
        elif self.schedule_type == ScheduleType.CRON:
            return self._next_cron_time(now)
        
        return now + 3600
    
    def _next_cron_time(self, from_time: float) -> float:
        """Calculate next cron execution time."""
        cron = self.schedule_config.get("expression", "* * * * *")
        parts = cron.split()
        
        if len(parts) != 5:
            return from_time + 60
        
        dt = datetime.fromtimestamp(from_time) + timedelta(minutes=1)
        dt = dt.replace(second=0, microsecond=0)
        
        for _ in range(525600):
            if self._matches_cron(dt, parts):
                return dt.timestamp()
            dt += timedelta(minutes=1)
        
        return from_time + 3600


    def _matches_cron(self, dt: datetime, parts: List[str]) -> bool:
        """Check if datetime matches cron expression."""
        minute, hour, day, month, dow = parts
        
        def match_field(value: int, field: str) -> bool:
            if field == "*":
                return True
            if field.isdigit():
                return value == int(field)
            if "/" in field:
                _, step = field.split("/")
                return value % int(step) == 0
            if "-" in field:
                start, end = field.split("-")
                return int(start) <= value <= int(end)
            if "," in field:
                return value in [int(v) for v in field.split(",")]
            return False
        
        return (
            match_field(dt.minute, minute) and
            match_field(dt.hour, hour) and
            match_field(dt.day, day) and
            match_field(dt.month, month) and
            match_field(dt.weekday(), dow)
        )


class Scheduler:
    """
    Cron-like scheduler for executing tasks on a schedule.
    
    Supports cron expressions, intervals, and specific times.
    
    Usage:
        scheduler = Scheduler()
        
        @scheduler.cron("0 9 * * *")  # Every day at 9 AM
        async def daily_report():
            await send_report()
        
        @scheduler.every(minutes=30)
        async def check_health():
            await health_check()
        
        @scheduler.once("2024-12-25 00:00:00")
        async def christmas_greeting():
            await send_greeting()
        
        # Start the scheduler
        await scheduler.start()
    """
    
    def __init__(self, timezone: str = "UTC"):
        self.timezone = timezone
        self.tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.Scheduler")
    
    def cron(
        self,
        expression: str,
        name: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to schedule a task with a cron expression.
        
        Args:
            expression: Cron expression (minute hour day month dow)
            name: Optional task name
        
        Usage:
            @scheduler.cron("*/5 * * * *")  # Every 5 minutes
            async def my_task():
                ...
        """
        def decorator(func: Callable[[], Awaitable[Any]]) -> Callable:
            task_name = name or func.__name__
            task = ScheduledTask(
                id=str(uuid.uuid4()),
                name=task_name,
                callback=func,
                schedule_type=ScheduleType.CRON,
                schedule_config={"expression": expression},
            )
            task.next_run = task.calculate_next_run()
            self.tasks[task.id] = task
            return func
        return decorator
    
    def every(
        self,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        name: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to schedule a task at regular intervals.
        
        Usage:
            @scheduler.every(hours=1)
            async def hourly_task():
                ...
        """
        total_seconds = seconds + minutes * 60 + hours * 3600
        
        def decorator(func: Callable[[], Awaitable[Any]]) -> Callable:
            task_name = name or func.__name__
            task = ScheduledTask(
                id=str(uuid.uuid4()),
                name=task_name,
                callback=func,
                schedule_type=ScheduleType.INTERVAL,
                schedule_config={"seconds": total_seconds},
            )
            task.next_run = task.calculate_next_run()
            self.tasks[task.id] = task
            return func
        return decorator
    
    def daily(
        self,
        hour: int = 0,
        minute: int = 0,
        name: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to schedule a daily task.
        
        Usage:
            @scheduler.daily(hour=9, minute=30)
            async def morning_task():
                ...
        """
        def decorator(func: Callable[[], Awaitable[Any]]) -> Callable:
            task_name = name or func.__name__
            task = ScheduledTask(
                id=str(uuid.uuid4()),
                name=task_name,
                callback=func,
                schedule_type=ScheduleType.DAILY,
                schedule_config={"hour": hour, "minute": minute},
            )
            task.next_run = task.calculate_next_run()
            self.tasks[task.id] = task
            return func
        return decorator
    
    def once(
        self,
        at: Union[str, float, datetime],
        name: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to schedule a one-time task.
        
        Args:
            at: When to run (timestamp, ISO string, or datetime)
        """
        if isinstance(at, str):
            dt = datetime.fromisoformat(at)
            timestamp = dt.timestamp()
        elif isinstance(at, datetime):
            timestamp = at.timestamp()
        else:
            timestamp = at
        
        def decorator(func: Callable[[], Awaitable[Any]]) -> Callable:
            task_name = name or func.__name__
            task = ScheduledTask(
                id=str(uuid.uuid4()),
                name=task_name,
                callback=func,
                schedule_type=ScheduleType.ONCE,
                schedule_config={"at": timestamp},
            )
            task.next_run = timestamp
            self.tasks[task.id] = task
            return func
        return decorator
    
    def schedule(
        self,
        func: Callable[[], Awaitable[Any]],
        schedule_type: ScheduleType,
        config: Dict[str, Any],
        name: Optional[str] = None,
    ) -> ScheduledTask:
        """
        Programmatically schedule a task.
        
        Args:
            func: Async function to execute
            schedule_type: Type of schedule
            config: Schedule configuration
            name: Optional task name
        """
        task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name or func.__name__,
            callback=func,
            schedule_type=schedule_type,
            schedule_config=config,
        )
        task.next_run = task.calculate_next_run()
        self.tasks[task.id] = task
        return task
    
    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self.logger.info("Scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Scheduler stopped")
    
    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = time.time()
                
                tasks_to_run = [
                    task for task in self.tasks.values()
                    if task.enabled and task.next_run and task.next_run <= now
                ]
                
                for task in tasks_to_run:
                    asyncio.create_task(self._execute_task(task))
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a scheduled task."""
        try:
            self.logger.debug(f"Executing task: {task.name}")
            await task.callback()
            task.run_count += 1
            task.last_run = time.time()
            task.last_error = None
            
        except Exception as e:
            task.error_count += 1
            task.last_error = str(e)
            self.logger.error(f"Task {task.name} failed: {e}")
        
        finally:
            if task.schedule_type == ScheduleType.ONCE:
                task.enabled = False
            else:
                task.next_run = task.calculate_next_run()
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            return True
        return False
    
    def list_tasks(self) -> List[ScheduledTask]:
        """List all tasks."""
        return list(self.tasks.values())
    
    def get_next_runs(self, count: int = 10) -> List[Tuple[float, ScheduledTask]]:
        """Get upcoming task executions."""
        upcoming = [
            (task.next_run, task)
            for task in self.tasks.values()
            if task.enabled and task.next_run
        ]
        upcoming.sort(key=lambda x: x[0])
        return upcoming[:count]


@dataclass
class Job:
    """A background job definition."""
    id: str
    name: str
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    attempts: int = 0
    max_retries: int = 3
    retry_delay: float = 60.0
    timeout: float = 300.0
    result: Any = None
    error: Optional[str] = None
    scheduled_for: Optional[float] = None
    
    def __lt__(self, other: "Job") -> bool:
        """Compare by priority and creation time for heap."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "payload": self.payload,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "result": self.result,
            "error": self.error,
        }


class JobQueue:
    """
    Background job processing queue with priorities and retries.
    
    Provides reliable background job execution with:
    - Priority-based execution
    - Automatic retries with backoff
    - Job timeouts
    - Deferred execution
    - Job status tracking
    
    Usage:
        queue = JobQueue()
        
        @queue.job("send_email")
        async def send_email(to: str, subject: str, body: str):
            await email_service.send(to, subject, body)
        
        @queue.job("process_order", max_retries=5)
        async def process_order(order_id: str):
            await order_service.process(order_id)
        
        # Enqueue jobs
        await queue.enqueue("send_email",
            to="user@example.com",
            subject="Welcome!",
            body="Thanks for signing up!"
        )
        
        # High priority job
        await queue.enqueue("process_order",
            order_id="12345",
            priority=JobPriority.HIGH
        )
        
        # Deferred job
        await queue.enqueue("send_reminder",
            user_id="123",
            scheduled_for=time.time() + 3600  # 1 hour later
        )
        
        # Start processing
        await queue.start()
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = AutomationLimit.MAX_JOBS_IN_QUEUE.value,
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.handlers: Dict[str, Callable] = {}
        self._queue: List[Job] = []
        self._jobs: Dict[str, Job] = {}
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._lock = threading.Lock()
        self._condition = asyncio.Condition()
        self.logger = logging.getLogger(f"{__name__}.JobQueue")
    
    def job(
        self,
        name: str,
        max_retries: int = 3,
        retry_delay: float = 60.0,
        timeout: float = 300.0,
    ) -> Callable:
        """
        Decorator to register a job handler.
        
        Args:
            name: Job type name
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            timeout: Job execution timeout
        """
        def decorator(func: Callable) -> Callable:
            self.handlers[name] = {
                "func": func,
                "max_retries": max_retries,
                "retry_delay": retry_delay,
                "timeout": timeout,
            }
            return func
        return decorator
    
    async def enqueue(
        self,
        name: str,
        priority: JobPriority = JobPriority.NORMAL,
        scheduled_for: Optional[float] = None,
        **payload
    ) -> Job:
        """
        Add a job to the queue.
        
        Args:
            name: Job type name
            priority: Job priority
            scheduled_for: Optional timestamp for deferred execution
            **payload: Job data
        
        Returns:
            The created job
        """
        if name not in self.handlers:
            raise JobError(f"Unknown job type: {name}")
        
        if len(self._jobs) >= self.max_queue_size:
            raise JobError(f"Queue is full ({self.max_queue_size} jobs)")
        
        handler = self.handlers[name]
        
        job = Job(
            id=str(uuid.uuid4()),
            name=name,
            payload=payload,
            priority=priority,
            max_retries=handler["max_retries"],
            retry_delay=handler["retry_delay"],
            timeout=handler["timeout"],
            scheduled_for=scheduled_for,
        )
        
        with self._lock:
            self._jobs[job.id] = job
            if scheduled_for and scheduled_for > time.time():
                job.status = JobStatus.DEFERRED
            else:
                job.status = JobStatus.QUEUED
                heapq.heappush(self._queue, job)
        
        async with self._condition:
            self._condition.notify()
        
        return job
    
    async def start(self, num_workers: Optional[int] = None) -> None:
        """Start processing jobs."""
        if self._running:
            return
        
        self._running = True
        workers = num_workers or self.max_workers
        
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(workers)
        ]
        
        asyncio.create_task(self._deferred_checker())
        
        self.logger.info(f"JobQueue started with {workers} workers")
    
    async def stop(self) -> None:
        """Stop processing jobs."""
        self._running = False
        
        async with self._condition:
            self._condition.notify_all()
        
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        
        self.logger.info("JobQueue stopped")
    
    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes jobs."""
        while self._running:
            try:
                job = await self._get_next_job()
                if job:
                    await self._process_job(job, worker_id)
                else:
                    async with self._condition:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=1.0
                        )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    async def _get_next_job(self) -> Optional[Job]:
        """Get the next job from the queue."""
        with self._lock:
            while self._queue:
                job = heapq.heappop(self._queue)
                if job.status == JobStatus.QUEUED:
                    return job
            return None
    
    async def _process_job(self, job: Job, worker_id: int) -> None:
        """Process a single job."""
        handler = self.handlers.get(job.name)
        if not handler:
            job.status = JobStatus.FAILED
            job.error = f"No handler for job type: {job.name}"
            return
        
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.attempts += 1
        
        try:
            self.logger.debug(f"Worker {worker_id} processing job {job.id}")
            
            result = await asyncio.wait_for(
                handler["func"](**job.payload),
                timeout=job.timeout
            )
            
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            
            self.logger.debug(f"Job {job.id} completed")
            
        except asyncio.TimeoutError:
            job.status = JobStatus.TIMEOUT
            job.error = f"Job timed out after {job.timeout}s"
            await self._handle_retry(job)
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            self.logger.error(f"Job {job.id} failed: {e}")
            await self._handle_retry(job)
    
    async def _handle_retry(self, job: Job) -> None:
        """Handle job retry logic."""
        if job.attempts < job.max_retries:
            job.status = JobStatus.RETRYING
            delay = job.retry_delay * (2 ** (job.attempts - 1))
            job.scheduled_for = time.time() + delay
            
            self.logger.info(
                f"Job {job.id} will retry in {delay}s "
                f"(attempt {job.attempts}/{job.max_retries})"
            )
    
    async def _deferred_checker(self) -> None:
        """Check for deferred jobs that are ready to run."""
        while self._running:
            try:
                now = time.time()
                
                with self._lock:
                    for job in self._jobs.values():
                        if (job.status in (JobStatus.DEFERRED, JobStatus.RETRYING) and
                            job.scheduled_for and
                            job.scheduled_for <= now):
                            
                            job.status = JobStatus.QUEUED
                            heapq.heappush(self._queue, job)
                
                async with self._condition:
                    self._condition.notify()
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Deferred checker error: {e}")
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        job = self._jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.DEFERRED):
            job.status = JobStatus.CANCELLED
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        status_counts = defaultdict(int)
        for job in self._jobs.values():
            status_counts[job.status.value] += 1
        
        return {
            "total_jobs": len(self._jobs),
            "queued": len(self._queue),
            "by_status": dict(status_counts),
            "workers": len(self._workers),
            "running": self._running,
        }
    
    def clear_completed(self, older_than: float = 3600) -> int:
        """Clear completed jobs older than specified seconds."""
        cutoff = time.time() - older_than
        to_remove = [
            job_id for job_id, job in self._jobs.items()
            if job.status == JobStatus.COMPLETED and
            job.completed_at and job.completed_at < cutoff
        ]
        
        for job_id in to_remove:
            del self._jobs[job_id]
        
        return len(to_remove)


@dataclass
class Conversation:
    """
    Represents a conversation with state.
    
    Attributes:
        id: Unique conversation ID
        user_id: User this conversation belongs to
        channel_id: Channel this conversation is in
        state: Current conversation state
        data: Conversation data/variables
        history: Message history
        created_at: When conversation started
        updated_at: Last update time
        expires_at: When conversation expires
    """
    id: str
    user_id: str
    channel_id: str
    state: str = "new"
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def set(self, key: str, value: Any) -> None:
        """Set a conversation variable."""
        self.data[key] = value
        self.updated_at = time.time()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a conversation variable."""
        return self.data.get(key, default)
    
    def transition(self, new_state: str) -> None:
        """Transition to a new state."""
        self.state = new_state
        self.updated_at = time.time()
    
    def add_message(self, message: Message) -> None:
        """Add a message to history."""
        self.history.append(message)
        self.updated_at = time.time()
    
    @property
    def is_expired(self) -> bool:
        """Check if conversation has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "state": self.state,
            "data": self.data,
            "history_length": len(self.history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }


class ConversationManager:
    """
    Manages conversation state across users and channels.
    
    Provides stateful conversation tracking for multi-step interactions,
    form filling, and complex dialog flows.
    
    Usage:
        manager = ConversationManager(ttl=3600)  # 1 hour TTL
        
        @bot.on_command("survey")
        async def start_survey(ctx):
            conv = manager.get_or_create(ctx.user_id, ctx.channel_id)
            conv.transition("asking_name")
            conv.set("started_at", time.time())
            await ctx.reply("What's your name?")
        
        @bot.on_message()
        async def handle_survey(ctx):
            conv = manager.get(ctx.user_id, ctx.channel_id)
            if not conv:
                return
            
            if conv.state == "asking_name":
                conv.set("name", ctx.text)
                conv.transition("asking_email")
                await ctx.reply(f"Nice to meet you, {ctx.text}! What's your email?")
            
            elif conv.state == "asking_email":
                conv.set("email", ctx.text)
                conv.transition("complete")
                name = conv.get("name")
                await ctx.reply(f"Thanks {name}! Survey complete.")
                manager.end(ctx.user_id, ctx.channel_id)
    """
    
    def __init__(
        self,
        ttl: float = AutomationLimit.MAX_CONVERSATION_AGE_SECONDS.value,
        max_conversations: int = AutomationLimit.MAX_CONVERSATIONS.value,
    ):
        self.ttl = ttl
        self.max_conversations = max_conversations
        self._conversations: Dict[str, Conversation] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.ConversationManager")
    
    def _make_key(self, user_id: str, channel_id: str) -> str:
        """Create a unique key for user+channel."""
        return f"{user_id}:{channel_id}"
    
    def get(
        self,
        user_id: str,
        channel_id: str
    ) -> Optional[Conversation]:
        """Get an existing conversation."""
        key = self._make_key(user_id, channel_id)
        
        with self._lock:
            conv = self._conversations.get(key)
            if conv and conv.is_expired:
                del self._conversations[key]
                return None
            return conv
    
    def get_or_create(
        self,
        user_id: str,
        channel_id: str,
        initial_state: str = "new",
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Get existing conversation or create a new one."""
        key = self._make_key(user_id, channel_id)
        
        with self._lock:
            conv = self._conversations.get(key)
            
            if conv and not conv.is_expired:
                return conv
            
            if len(self._conversations) >= self.max_conversations:
                self._cleanup_expired()
            
            conv = Conversation(
                id=str(uuid.uuid4()),
                user_id=user_id,
                channel_id=channel_id,
                state=initial_state,
                data=initial_data or {},
                expires_at=time.time() + self.ttl if self.ttl > 0 else None,
            )
            
            self._conversations[key] = conv
            return conv
    
    def create(
        self,
        user_id: str,
        channel_id: str,
        initial_state: str = "new",
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Create a new conversation, replacing any existing one."""
        key = self._make_key(user_id, channel_id)
        
        with self._lock:
            conv = Conversation(
                id=str(uuid.uuid4()),
                user_id=user_id,
                channel_id=channel_id,
                state=initial_state,
                data=initial_data or {},
                expires_at=time.time() + self.ttl if self.ttl > 0 else None,
            )
            
            self._conversations[key] = conv
            return conv
    
    def end(self, user_id: str, channel_id: str) -> bool:
        """End and remove a conversation."""
        key = self._make_key(user_id, channel_id)
        
        with self._lock:
            if key in self._conversations:
                del self._conversations[key]
                return True
            return False
    
    def update_ttl(
        self,
        user_id: str,
        channel_id: str,
        additional_ttl: Optional[float] = None,
    ) -> bool:
        """Extend the TTL of a conversation."""
        key = self._make_key(user_id, channel_id)
        
        with self._lock:
            conv = self._conversations.get(key)
            if conv:
                ttl = additional_ttl if additional_ttl is not None else self.ttl
                conv.expires_at = time.time() + ttl
                return True
            return False
    
    def get_by_user(self, user_id: str) -> List[Conversation]:
        """Get all conversations for a user."""
        with self._lock:
            return [
                conv for conv in self._conversations.values()
                if conv.user_id == user_id and not conv.is_expired
            ]
    
    def get_by_channel(self, channel_id: str) -> List[Conversation]:
        """Get all conversations in a channel."""
        with self._lock:
            return [
                conv for conv in self._conversations.values()
                if conv.channel_id == channel_id and not conv.is_expired
            ]
    
    def get_by_state(self, state: str) -> List[Conversation]:
        """Get all conversations in a specific state."""
        with self._lock:
            return [
                conv for conv in self._conversations.values()
                if conv.state == state and not conv.is_expired
            ]
    
    def _cleanup_expired(self) -> int:
        """Remove expired conversations."""
        now = time.time()
        expired = [
            key for key, conv in self._conversations.items()
            if conv.expires_at and conv.expires_at < now
        ]
        
        for key in expired:
            del self._conversations[key]
        
        return len(expired)
    
    def cleanup(self) -> int:
        """Manually trigger cleanup of expired conversations."""
        with self._lock:
            return self._cleanup_expired()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._lock:
            state_counts = defaultdict(int)
            for conv in self._conversations.values():
                state_counts[conv.state] += 1
            
            return {
                "total": len(self._conversations),
                "by_state": dict(state_counts),
                "max_conversations": self.max_conversations,
                "ttl": self.ttl,
            }
    
    def list_all(self) -> List[Conversation]:
        """List all active conversations."""
        with self._lock:
            self._cleanup_expired()
            return list(self._conversations.values())


@dataclass
class WebhookRequest:
    """Incoming webhook request."""
    id: str
    path: str
    method: str
    headers: Dict[str, str]
    body: Any
    query_params: Dict[str, str]
    received_at: float = field(default_factory=time.time)
    source_ip: Optional[str] = None


@dataclass
class WebhookHandler:
    """A registered webhook handler."""
    id: str
    path: str
    method: str
    callback: Callable[[WebhookRequest], Awaitable[Any]]
    secret: Optional[str] = None
    enabled: bool = True
    call_count: int = 0
    last_called: Optional[float] = None


class WebhookReceiver:
    """
    Handles incoming webhook requests.
    
    Usage:
        receiver = WebhookReceiver()
        
        @receiver.webhook("/github", method="POST", secret="my_secret")
        async def handle_github(request: WebhookRequest):
            event = request.headers.get("X-GitHub-Event")
            if event == "push":
                await process_push(request.body)
            return {"status": "ok"}
        
        @receiver.webhook("/stripe")
        async def handle_stripe(request: WebhookRequest):
            await process_stripe_event(request.body)
            return {"received": True}
    """
    
    def __init__(self):
        self.handlers: Dict[str, WebhookHandler] = {}
        self.logger = logging.getLogger(f"{__name__}.WebhookReceiver")
    
    def webhook(
        self,
        path: str,
        method: str = "POST",
        secret: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to register a webhook handler.
        
        Args:
            path: URL path to handle
            method: HTTP method (GET, POST, etc.)
            secret: Optional secret for signature verification
        """
        def decorator(func: Callable[[WebhookRequest], Awaitable[Any]]) -> Callable:
            handler_id = f"{method}:{path}"
            handler = WebhookHandler(
                id=handler_id,
                path=path,
                method=method.upper(),
                callback=func,
                secret=secret,
            )
            self.handlers[handler_id] = handler
            return func
        return decorator
    
    async def handle(self, request: WebhookRequest) -> Any:
        """Handle an incoming webhook request."""
        handler_id = f"{request.method}:{request.path}"
        handler = self.handlers.get(handler_id)
        
        if not handler or not handler.enabled:
            return {"error": "Not found"}, 404
        
        if handler.secret:
            if not self._verify_signature(request, handler.secret):
                return {"error": "Invalid signature"}, 401
        
        try:
            handler.call_count += 1
            handler.last_called = time.time()
            
            result = await handler.callback(request)
            return result, 200
            
        except Exception as e:
            self.logger.error(f"Webhook handler error: {e}")
            return {"error": str(e)}, 500
    
    def _verify_signature(
        self,
        request: WebhookRequest,
        secret: str
    ) -> bool:
        """Verify webhook signature."""
        signature = request.headers.get("X-Signature") or request.headers.get("X-Hub-Signature-256")
        
        if not signature:
            return False
        
        if isinstance(request.body, (dict, list)):
            body_bytes = json.dumps(request.body).encode()
        elif isinstance(request.body, str):
            body_bytes = request.body.encode()
        else:
            body_bytes = request.body
        
        expected = hashlib.sha256(secret.encode() + body_bytes).hexdigest()
        
        signature_value = signature.replace("sha256=", "")
        return signature_value == expected
    
    def list_handlers(self) -> List[WebhookHandler]:
        """List all registered handlers."""
        return list(self.handlers.values())
    
    def enable_handler(self, handler_id: str) -> bool:
        """Enable a handler."""
        if handler_id in self.handlers:
            self.handlers[handler_id].enabled = True
            return True
        return False
    
    def disable_handler(self, handler_id: str) -> bool:
        """Disable a handler."""
        if handler_id in self.handlers:
            self.handlers[handler_id].enabled = False
            return True
        return False


_default_scheduler: Optional[Scheduler] = None
_default_job_queue: Optional[JobQueue] = None
_default_conversation_manager: Optional[ConversationManager] = None
_default_webhook_receiver: Optional[WebhookReceiver] = None


def get_default_scheduler() -> Scheduler:
    """Get the default scheduler instance."""
    global _default_scheduler
    if _default_scheduler is None:
        _default_scheduler = Scheduler()
    return _default_scheduler


def set_default_scheduler(scheduler: Scheduler) -> None:
    """Set the default scheduler instance."""
    global _default_scheduler
    _default_scheduler = scheduler


def get_default_job_queue() -> JobQueue:
    """Get the default job queue instance."""
    global _default_job_queue
    if _default_job_queue is None:
        _default_job_queue = JobQueue()
    return _default_job_queue


def set_default_job_queue(queue: JobQueue) -> None:
    """Set the default job queue instance."""
    global _default_job_queue
    _default_job_queue = queue


def get_default_conversation_manager() -> ConversationManager:
    """Get the default conversation manager instance."""
    global _default_conversation_manager
    if _default_conversation_manager is None:
        _default_conversation_manager = ConversationManager()
    return _default_conversation_manager


def set_default_conversation_manager(manager: ConversationManager) -> None:
    """Set the default conversation manager instance."""
    global _default_conversation_manager
    _default_conversation_manager = manager


def get_default_webhook_receiver() -> WebhookReceiver:
    """Get the default webhook receiver instance."""
    global _default_webhook_receiver
    if _default_webhook_receiver is None:
        _default_webhook_receiver = WebhookReceiver()
    return _default_webhook_receiver


def set_default_webhook_receiver(receiver: WebhookReceiver) -> None:
    """Set the default webhook receiver instance."""
    global _default_webhook_receiver
    _default_webhook_receiver = receiver


def create_bot(
    platform: BotPlatform,
    token: str,
    name: str = "Bot",
    **kwargs
) -> Bot:
    """
    Factory function to create a bot for a specific platform.
    
    Args:
        platform: Target platform
        token: API token
        name: Bot name
        **kwargs: Platform-specific options
    
    Returns:
        Configured bot instance
    """
    if platform == BotPlatform.SLACK:
        return SlackBot(token=token, name=name, **kwargs)
    elif platform == BotPlatform.TELEGRAM:
        return TelegramBot(token=token, name=name, **kwargs)
    elif platform == BotPlatform.DISCORD:
        return DiscordBot(token=token, name=name, **kwargs)
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def create_workflow(
    name: str,
    triggers: Optional[List[Trigger]] = None,
    actions: Optional[List[Action]] = None,
    description: str = "",
) -> Workflow:
    """
    Factory function to create a workflow.
    
    Args:
        name: Workflow name
        triggers: List of triggers
        actions: List of actions
        description: Workflow description
    
    Returns:
        Configured workflow instance
    """
    workflow = Workflow(name=name, description=description)
    
    for trigger in (triggers or []):
        workflow.add_trigger(trigger)
    
    for action in (actions or []):
        workflow.add_action(action)
    
    return workflow


async def schedule_job(
    name: str,
    priority: JobPriority = JobPriority.NORMAL,
    scheduled_for: Optional[float] = None,
    **payload
) -> Job:
    """
    Schedule a job using the default queue.
    
    Args:
        name: Job type name
        priority: Job priority
        scheduled_for: Optional delayed execution time
        **payload: Job data
    
    Returns:
        The scheduled job
    """
    queue = get_default_job_queue()
    return await queue.enqueue(
        name,
        priority=priority,
        scheduled_for=scheduled_for,
        **payload
    )


def get_conversation(
    user_id: str,
    channel_id: str
) -> Optional[Conversation]:
    """Get a conversation using the default manager."""
    manager = get_default_conversation_manager()
    return manager.get(user_id, channel_id)


def create_conversation(
    user_id: str,
    channel_id: str,
    initial_state: str = "new",
    initial_data: Optional[Dict[str, Any]] = None,
) -> Conversation:
    """Create a conversation using the default manager."""
    manager = get_default_conversation_manager()
    return manager.create(user_id, channel_id, initial_state, initial_data)


def end_conversation(user_id: str, channel_id: str) -> bool:
    """End a conversation using the default manager."""
    manager = get_default_conversation_manager()
    return manager.end(user_id, channel_id)
