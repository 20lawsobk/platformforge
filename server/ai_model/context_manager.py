"""
Persistent Project Memory and Context Manager for Platform Forge

This module ensures the AI remembers project context and doesn't repeat mistakes.
It provides comprehensive context management, conversation memory, learning from
corrections, and efficient context window management.

Key Components:
- ProjectContext: Dataclass storing all project-related context
- ContextManager: Load, update, and query project context
- ConversationMemory: Store and manage conversation history with context
- LearningMemory: Learn from mistakes and user corrections
- InstructionTracker: Track and categorize user instructions (DO, DON'T, PREFER, AVOID)
- ContextWindow: Efficiently manage 16k token context window

Usage:
    from server.ai_model.context_manager import (
        ContextManager,
        ConversationMemory,
        LearningMemory,
        InstructionTracker,
        ContextWindow,
        get_project_summary,
        search_context,
        remember,
        recall,
    )
    
    # Load project context
    manager = ContextManager()
    context = manager.load_context("/path/to/project")
    
    # Add user instructions
    manager.add_user_instruction("Don't use var, always use const or let")
    
    # Track conversation
    memory = ConversationMemory()
    memory.add_message("user", "Fix the login bug")
    memory.add_message("assistant", "I'll update the auth module...")
    
    # Learn from corrections
    learning = LearningMemory()
    learning.record_mistake("Used var instead of const", "Use const for constants")
    
    # Manage context window
    window = ContextWindow(max_tokens=16384)
    window.add_context("system", system_prompt, priority=10)
    window.add_context("instructions", user_instructions, priority=9)
"""

import os
import re
import json
import time
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
import uuid


class InstructionCategory(Enum):
    """Categories for user instructions."""
    DO = "do"
    DONT = "dont"
    PREFER = "prefer"
    AVOID = "avoid"
    ALWAYS = "always"
    NEVER = "never"
    
    def __str__(self) -> str:
        return self.value


class ContextPriority(Enum):
    """Priority levels for context items."""
    CRITICAL = 10
    HIGH = 8
    MEDIUM = 5
    LOW = 3
    OPTIONAL = 1
    
    def __str__(self) -> str:
        return self.value


class PatternConfidence(Enum):
    """Confidence levels for learned patterns."""
    UNCERTAIN = "uncertain"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CERTAIN = "certain"
    
    @property
    def score(self) -> float:
        mapping = {
            PatternConfidence.UNCERTAIN: 0.2,
            PatternConfidence.LOW: 0.4,
            PatternConfidence.MEDIUM: 0.6,
            PatternConfidence.HIGH: 0.8,
            PatternConfidence.CERTAIN: 1.0,
        }
        return mapping[self]


class MessageRole(Enum):
    """Roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class CodingStyle:
    """Represents coding style preferences for a project."""
    indentation: str = "spaces"
    indent_size: int = 2
    quote_style: str = "single"
    semicolons: bool = False
    trailing_commas: bool = True
    max_line_length: int = 100
    bracket_style: str = "same_line"
    naming_conventions: Dict[str, str] = field(default_factory=lambda: {
        "variables": "camelCase",
        "functions": "camelCase",
        "classes": "PascalCase",
        "constants": "UPPER_SNAKE_CASE",
        "files": "kebab-case",
    })
    custom_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "indentation": self.indentation,
            "indent_size": self.indent_size,
            "quote_style": self.quote_style,
            "semicolons": self.semicolons,
            "trailing_commas": self.trailing_commas,
            "max_line_length": self.max_line_length,
            "bracket_style": self.bracket_style,
            "naming_conventions": self.naming_conventions,
            "custom_rules": self.custom_rules,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingStyle":
        return cls(
            indentation=data.get("indentation", "spaces"),
            indent_size=data.get("indent_size", 2),
            quote_style=data.get("quote_style", "single"),
            semicolons=data.get("semicolons", False),
            trailing_commas=data.get("trailing_commas", True),
            max_line_length=data.get("max_line_length", 100),
            bracket_style=data.get("bracket_style", "same_line"),
            naming_conventions=data.get("naming_conventions", {}),
            custom_rules=data.get("custom_rules", []),
        )
    
    def get_summary(self) -> str:
        return (
            f"{self.indentation} ({self.indent_size}), "
            f"{self.quote_style} quotes, "
            f"{'with' if self.semicolons else 'no'} semicolons"
        )


@dataclass
class UserPreference:
    """Represents a single user preference."""
    key: str
    value: Any
    category: str
    source: str
    timestamp: float
    confidence: float = 1.0
    times_applied: int = 0
    last_applied: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "source": self.source,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "times_applied": self.times_applied,
            "last_applied": self.last_applied,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        return cls(
            key=data["key"],
            value=data["value"],
            category=data.get("category", "general"),
            source=data.get("source", "user"),
            timestamp=data.get("timestamp", time.time()),
            confidence=data.get("confidence", 1.0),
            times_applied=data.get("times_applied", 0),
            last_applied=data.get("last_applied"),
        )


@dataclass
class ExplicitInstruction:
    """Represents an explicit user instruction."""
    id: str
    instruction: str
    category: InstructionCategory
    keywords: List[str]
    timestamp: float
    active: bool = True
    source: str = "user"
    priority: int = 5
    times_triggered: int = 0
    last_triggered: Optional[float] = None
    context: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "category": self.category.value,
            "keywords": self.keywords,
            "timestamp": self.timestamp,
            "active": self.active,
            "source": self.source,
            "priority": self.priority,
            "times_triggered": self.times_triggered,
            "last_triggered": self.last_triggered,
            "context": self.context,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplicitInstruction":
        return cls(
            id=data["id"],
            instruction=data["instruction"],
            category=InstructionCategory(data["category"]),
            keywords=data.get("keywords", []),
            timestamp=data.get("timestamp", time.time()),
            active=data.get("active", True),
            source=data.get("source", "user"),
            priority=data.get("priority", 5),
            times_triggered=data.get("times_triggered", 0),
            last_triggered=data.get("last_triggered"),
            context=data.get("context", ""),
        )
    
    def matches(self, action: str) -> Tuple[bool, float]:
        """Check if this instruction matches an action."""
        action_lower = action.lower()
        keyword_matches = sum(1 for kw in self.keywords if kw.lower() in action_lower)
        if keyword_matches == 0:
            return False, 0.0
        match_score = keyword_matches / len(self.keywords) if self.keywords else 0.0
        return True, match_score


@dataclass
class ProjectContext:
    """
    Complete context for a project including metadata, structure,
    and user preferences.
    
    Attributes:
        project_id: Unique identifier for the project
        project_name: Human-readable project name
        project_type: Type of project (web, cli, library, etc.)
        languages: Programming languages used
        frameworks: Frameworks and libraries in use
        dependencies: Project dependencies with versions
        file_structure: Key files and directories
        entry_points: Main entry points for the application
        coding_style: Coding style preferences
        naming_conventions: Naming convention rules
        user_preferences: User-specific preferences
        explicit_instructions: User's explicit instructions
        created_at: When context was first created
        updated_at: When context was last updated
    """
    project_id: str
    project_name: str
    project_type: str = "unknown"
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    file_structure: Dict[str, List[str]] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    coding_style: CodingStyle = field(default_factory=CodingStyle)
    naming_conventions: Dict[str, str] = field(default_factory=dict)
    user_preferences: Dict[str, UserPreference] = field(default_factory=dict)
    explicit_instructions: List[ExplicitInstruction] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_type": self.project_type,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "dependencies": self.dependencies,
            "file_structure": self.file_structure,
            "entry_points": self.entry_points,
            "coding_style": self.coding_style.to_dict(),
            "naming_conventions": self.naming_conventions,
            "user_preferences": {k: v.to_dict() for k, v in self.user_preferences.items()},
            "explicit_instructions": [i.to_dict() for i in self.explicit_instructions],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectContext":
        return cls(
            project_id=data["project_id"],
            project_name=data["project_name"],
            project_type=data.get("project_type", "unknown"),
            languages=data.get("languages", []),
            frameworks=data.get("frameworks", []),
            dependencies=data.get("dependencies", {}),
            file_structure=data.get("file_structure", {}),
            entry_points=data.get("entry_points", []),
            coding_style=CodingStyle.from_dict(data.get("coding_style", {})),
            naming_conventions=data.get("naming_conventions", {}),
            user_preferences={
                k: UserPreference.from_dict(v) 
                for k, v in data.get("user_preferences", {}).items()
            },
            explicit_instructions=[
                ExplicitInstruction.from_dict(i) 
                for i in data.get("explicit_instructions", [])
            ],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )
    
    def get_summary(self) -> str:
        """Generate a concise summary of the project context."""
        lang_str = ", ".join(self.languages[:3]) if self.languages else "Unknown"
        fw_str = ", ".join(self.frameworks[:3]) if self.frameworks else "None"
        return (
            f"Project: {self.project_name} ({self.project_type})\n"
            f"Languages: {lang_str}\n"
            f"Frameworks: {fw_str}\n"
            f"Dependencies: {len(self.dependencies)}\n"
            f"Instructions: {len([i for i in self.explicit_instructions if i.active])}"
        )
    
    def get_active_instructions(self) -> List[ExplicitInstruction]:
        """Get all active instructions."""
        return [i for i in self.explicit_instructions if i.active]


@dataclass
class ConversationMessage:
    """Represents a single message in conversation history."""
    id: str
    role: MessageRole
    content: str
    timestamp: float
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_references: List[str] = field(default_factory=list)
    was_successful: Optional[bool] = None
    correction: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "tokens": self.tokens,
            "metadata": self.metadata,
            "context_references": self.context_references,
            "was_successful": self.was_successful,
            "correction": self.correction,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        return cls(
            id=data["id"],
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            tokens=data.get("tokens", 0),
            metadata=data.get("metadata", {}),
            context_references=data.get("context_references", []),
            was_successful=data.get("was_successful"),
            correction=data.get("correction"),
        )


@dataclass
class LearnedPattern:
    """Represents a pattern learned from user corrections."""
    id: str
    pattern_type: str
    trigger: str
    correct_response: str
    incorrect_response: str
    confidence: PatternConfidence
    times_applied: int = 0
    times_successful: int = 0
    created_at: float = field(default_factory=time.time)
    last_applied: Optional[float] = None
    context: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "trigger": self.trigger,
            "correct_response": self.correct_response,
            "incorrect_response": self.incorrect_response,
            "confidence": self.confidence.value,
            "times_applied": self.times_applied,
            "times_successful": self.times_successful,
            "created_at": self.created_at,
            "last_applied": self.last_applied,
            "context": self.context,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedPattern":
        return cls(
            id=data["id"],
            pattern_type=data["pattern_type"],
            trigger=data["trigger"],
            correct_response=data["correct_response"],
            incorrect_response=data.get("incorrect_response", ""),
            confidence=PatternConfidence(data.get("confidence", "medium")),
            times_applied=data.get("times_applied", 0),
            times_successful=data.get("times_successful", 0),
            created_at=data.get("created_at", time.time()),
            last_applied=data.get("last_applied"),
            context=data.get("context", ""),
            tags=data.get("tags", []),
        )
    
    def get_success_rate(self) -> float:
        """Calculate success rate of this pattern."""
        if self.times_applied == 0:
            return 0.0
        return self.times_successful / self.times_applied
    
    def update_confidence(self):
        """Update confidence based on success rate."""
        success_rate = self.get_success_rate()
        if self.times_applied < 3:
            self.confidence = PatternConfidence.UNCERTAIN
        elif success_rate >= 0.9:
            self.confidence = PatternConfidence.CERTAIN
        elif success_rate >= 0.75:
            self.confidence = PatternConfidence.HIGH
        elif success_rate >= 0.5:
            self.confidence = PatternConfidence.MEDIUM
        else:
            self.confidence = PatternConfidence.LOW


@dataclass
class ContextItem:
    """Represents an item in the context window."""
    id: str
    category: str
    content: str
    priority: int
    tokens: int
    timestamp: float
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "priority": self.priority,
            "tokens": self.tokens,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextItem":
        return cls(
            id=data["id"],
            category=data["category"],
            content=data["content"],
            priority=data.get("priority", 5),
            tokens=data.get("tokens", 0),
            timestamp=data.get("timestamp", time.time()),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata", {}),
        )
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class ConversationMemory:
    """
    Manages conversation history with context tracking.
    
    Features:
    - Store messages with metadata
    - Track what worked and what didn't
    - Identify user corrections
    - Summarize long conversations
    - Maintain context references
    """
    
    def __init__(self, max_messages: int = 100, max_tokens: int = 8000):
        self.messages: List[ConversationMessage] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.corrections: List[Dict[str, Any]] = []
        self.successful_patterns: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def add_message(
        self,
        role: Union[str, MessageRole],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        context_references: Optional[List[str]] = None,
    ) -> ConversationMessage:
        """Add a message to the conversation history."""
        if isinstance(role, str):
            role = MessageRole(role)
        
        message = ConversationMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            tokens=self._estimate_tokens(content),
            metadata=metadata or {},
            context_references=context_references or [],
        )
        
        with self._lock:
            self.messages.append(message)
            self._trim_if_needed()
        
        return message
    
    def mark_success(self, message_id: str, was_successful: bool):
        """Mark whether a message/response was successful."""
        with self._lock:
            for msg in self.messages:
                if msg.id == message_id:
                    msg.was_successful = was_successful
                    if was_successful:
                        self._record_successful_pattern(msg)
                    break
    
    def add_correction(self, original_message_id: str, correction: str):
        """Record a user correction to a previous response."""
        with self._lock:
            for msg in self.messages:
                if msg.id == original_message_id:
                    msg.correction = correction
                    msg.was_successful = False
                    self.corrections.append({
                        "message_id": original_message_id,
                        "original": msg.content,
                        "correction": correction,
                        "timestamp": time.time(),
                    })
                    break
    
    def _record_successful_pattern(self, message: ConversationMessage):
        """Record a successful pattern for future reference."""
        if message.role == MessageRole.ASSISTANT:
            context = self._get_message_context(message)
            self.successful_patterns.append({
                "response": message.content[:500],
                "context": context,
                "timestamp": time.time(),
            })
    
    def _get_message_context(self, message: ConversationMessage) -> str:
        """Get the context around a message."""
        msg_index = next(
            (i for i, m in enumerate(self.messages) if m.id == message.id),
            -1
        )
        if msg_index <= 0:
            return ""
        
        prev_messages = self.messages[max(0, msg_index - 2):msg_index]
        return "\n".join(m.content[:200] for m in prev_messages)
    
    def get_corrections(self) -> List[Dict[str, Any]]:
        """Get all recorded corrections."""
        return self.corrections.copy()
    
    def detect_correction_in_message(self, content: str) -> Optional[Dict[str, str]]:
        """Detect if a message contains a correction."""
        correction_patterns = [
            r"(?:i said|i told you|i asked|don't|do not|stop|never|always)\s+(.+)",
            r"(?:that's wrong|incorrect|not what i wanted|please don't)\s*(.+)?",
            r"(?:instead|rather|actually)\s+(?:use|do|make)\s+(.+)",
        ]
        
        content_lower = content.lower()
        for pattern in correction_patterns:
            match = re.search(pattern, content_lower, re.IGNORECASE)
            if match:
                return {
                    "detected_correction": match.group(0),
                    "specific_instruction": match.group(1) if match.lastindex else None,
                }
        return None
    
    def get_recent_messages(self, count: int = 10) -> List[ConversationMessage]:
        """Get the most recent messages."""
        with self._lock:
            return self.messages[-count:]
    
    def get_messages_by_role(self, role: MessageRole) -> List[ConversationMessage]:
        """Get all messages from a specific role."""
        return [m for m in self.messages if m.role == role]
    
    def summarize(self, max_tokens: int = 500) -> str:
        """Summarize the conversation for context window efficiency."""
        if not self.messages:
            return "No conversation history."
        
        total_messages = len(self.messages)
        user_messages = len([m for m in self.messages if m.role == MessageRole.USER])
        corrections_count = len(self.corrections)
        
        summary_parts = [
            f"Conversation: {total_messages} messages, {user_messages} from user",
            f"Corrections made: {corrections_count}",
        ]
        
        recent = self.get_recent_messages(5)
        if recent:
            summary_parts.append("\nRecent topics:")
            for msg in recent:
                topic = msg.content[:100].replace("\n", " ")
                summary_parts.append(f"- [{msg.role.value}]: {topic}...")
        
        if self.corrections:
            summary_parts.append("\nKey corrections:")
            for corr in self.corrections[-3:]:
                summary_parts.append(f"- {corr['correction'][:100]}...")
        
        return "\n".join(summary_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text.split()) + len(text) // 4
    
    def _trim_if_needed(self):
        """Trim old messages if limits exceeded."""
        total_tokens = sum(m.tokens for m in self.messages)
        
        while (len(self.messages) > self.max_messages or 
               total_tokens > self.max_tokens) and len(self.messages) > 5:
            removed = self.messages.pop(0)
            total_tokens -= removed.tokens
    
    def clear(self):
        """Clear all conversation history."""
        with self._lock:
            self.messages.clear()
            self.corrections.clear()
            self.successful_patterns.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "corrections": self.corrections,
            "successful_patterns": self.successful_patterns,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        memory = cls()
        memory.messages = [
            ConversationMessage.from_dict(m) for m in data.get("messages", [])
        ]
        memory.corrections = data.get("corrections", [])
        memory.successful_patterns = data.get("successful_patterns", [])
        return memory


class LearningMemory:
    """
    Learns from mistakes and user corrections.
    
    Features:
    - Store mistakes and their corrections
    - Learn user preferences over time
    - Apply learned patterns to new situations
    - Track confidence scores for patterns
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.patterns: Dict[str, LearnedPattern] = {}
        self.mistakes: List[Dict[str, Any]] = []
        self.preferences: Dict[str, UserPreference] = {}
        self.storage_path = storage_path
        self._lock = threading.Lock()
        
        if storage_path and os.path.exists(storage_path):
            self._load()
    
    def record_mistake(
        self,
        mistake: str,
        correction: str,
        context: str = "",
        tags: Optional[List[str]] = None,
    ) -> LearnedPattern:
        """Record a mistake and its correction."""
        pattern_id = str(uuid.uuid4())
        
        pattern = LearnedPattern(
            id=pattern_id,
            pattern_type="correction",
            trigger=mistake,
            correct_response=correction,
            incorrect_response=mistake,
            confidence=PatternConfidence.UNCERTAIN,
            context=context,
            tags=tags or [],
        )
        
        with self._lock:
            self.patterns[pattern_id] = pattern
            self.mistakes.append({
                "pattern_id": pattern_id,
                "mistake": mistake,
                "correction": correction,
                "timestamp": time.time(),
                "context": context,
            })
        
        self._save()
        return pattern
    
    def add_learned_pattern(
        self,
        pattern_type: str,
        trigger: str,
        correct_response: str,
        confidence: PatternConfidence = PatternConfidence.MEDIUM,
        context: str = "",
        tags: Optional[List[str]] = None,
    ) -> LearnedPattern:
        """Add a learned pattern directly."""
        pattern_id = str(uuid.uuid4())
        
        pattern = LearnedPattern(
            id=pattern_id,
            pattern_type=pattern_type,
            trigger=trigger,
            correct_response=correct_response,
            incorrect_response="",
            confidence=confidence,
            context=context,
            tags=tags or [],
        )
        
        with self._lock:
            self.patterns[pattern_id] = pattern
        
        self._save()
        return pattern
    
    def record_pattern_application(self, pattern_id: str, was_successful: bool):
        """Record when a pattern was applied and whether it worked."""
        with self._lock:
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                pattern.times_applied += 1
                pattern.last_applied = time.time()
                if was_successful:
                    pattern.times_successful += 1
                pattern.update_confidence()
        
        self._save()
    
    def learn_preference(
        self,
        key: str,
        value: Any,
        category: str = "general",
        source: str = "observation",
    ):
        """Learn a user preference from observed behavior."""
        with self._lock:
            if key in self.preferences:
                existing = self.preferences[key]
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.value = value
                existing.timestamp = time.time()
            else:
                self.preferences[key] = UserPreference(
                    key=key,
                    value=value,
                    category=category,
                    source=source,
                    timestamp=time.time(),
                    confidence=0.5,
                )
        
        self._save()
    
    def find_applicable_patterns(
        self,
        situation: str,
        min_confidence: PatternConfidence = PatternConfidence.LOW,
    ) -> List[Tuple[LearnedPattern, float]]:
        """Find patterns that might apply to a situation."""
        applicable = []
        situation_lower = situation.lower()
        
        for pattern in self.patterns.values():
            if pattern.confidence.score < min_confidence.score:
                continue
            
            trigger_lower = pattern.trigger.lower()
            similarity = SequenceMatcher(
                None, situation_lower, trigger_lower
            ).ratio()
            
            word_overlap = len(
                set(situation_lower.split()) & set(trigger_lower.split())
            )
            
            if similarity > 0.3 or word_overlap >= 2:
                score = similarity * pattern.confidence.score
                applicable.append((pattern, score))
        
        applicable.sort(key=lambda x: x[1], reverse=True)
        return applicable[:5]
    
    def get_preference(self, key: str) -> Optional[UserPreference]:
        """Get a learned preference by key."""
        return self.preferences.get(key)
    
    def get_all_preferences(self) -> Dict[str, UserPreference]:
        """Get all learned preferences."""
        return self.preferences.copy()
    
    def get_mistakes_summary(self) -> str:
        """Get a summary of common mistakes and corrections."""
        if not self.mistakes:
            return "No mistakes recorded."
        
        summary_parts = [f"Total mistakes recorded: {len(self.mistakes)}"]
        
        recent = self.mistakes[-5:]
        summary_parts.append("\nRecent mistakes:")
        for m in recent:
            summary_parts.append(
                f"- Mistake: {m['mistake'][:50]}...\n"
                f"  Correction: {m['correction'][:50]}..."
            )
        
        return "\n".join(summary_parts)
    
    def get_high_confidence_patterns(self) -> List[LearnedPattern]:
        """Get patterns with high confidence."""
        return [
            p for p in self.patterns.values()
            if p.confidence.score >= PatternConfidence.HIGH.score
        ]
    
    def _save(self):
        """Save learning data to storage."""
        if not self.storage_path:
            return
        
        try:
            data = {
                "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
                "mistakes": self.mistakes,
                "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
            }
            
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def _load(self):
        """Load learning data from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.patterns = {
                k: LearnedPattern.from_dict(v) 
                for k, v in data.get("patterns", {}).items()
            }
            self.mistakes = data.get("mistakes", [])
            self.preferences = {
                k: UserPreference.from_dict(v)
                for k, v in data.get("preferences", {}).items()
            }
        except Exception:
            pass
    
    def clear(self):
        """Clear all learned data."""
        with self._lock:
            self.patterns.clear()
            self.mistakes.clear()
            self.preferences.clear()
        self._save()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
            "mistakes": self.mistakes,
            "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
        }


class InstructionTracker:
    """
    Tracks and manages user instructions.
    
    Features:
    - Parse instructions from user messages
    - Categorize as DO, DON'T, PREFER, AVOID
    - Detect conflicts with proposed actions
    - Persist instructions across sessions
    """
    
    CATEGORY_PATTERNS = {
        InstructionCategory.DO: [
            r"(?:always|make sure to|you should|please do|do use|use)\s+(.+)",
            r"(?:i want you to|you must|ensure that)\s+(.+)",
        ],
        InstructionCategory.DONT: [
            r"(?:don't|do not|never|stop|avoid|no)\s+(.+)",
            r"(?:i said don't|please don't|you shouldn't)\s+(.+)",
        ],
        InstructionCategory.PREFER: [
            r"(?:i prefer|i like|prefer to|rather have|ideally)\s+(.+)",
            r"(?:when possible|if you can)\s+(.+)",
        ],
        InstructionCategory.AVOID: [
            r"(?:try not to|avoid|minimize|reduce)\s+(.+)",
            r"(?:i don't like|not a fan of)\s+(.+)",
        ],
        InstructionCategory.ALWAYS: [
            r"(?:always|every time|without exception)\s+(.+)",
        ],
        InstructionCategory.NEVER: [
            r"(?:never ever|under no circumstances|absolutely never)\s+(.+)",
        ],
    }
    
    def __init__(self, storage_path: Optional[str] = None):
        self.instructions: Dict[str, ExplicitInstruction] = {}
        self.storage_path = storage_path
        self._lock = threading.Lock()
        
        if storage_path and os.path.exists(storage_path):
            self._load()
    
    def add_instruction(
        self,
        instruction: str,
        category: Optional[InstructionCategory] = None,
        priority: int = 5,
        context: str = "",
    ) -> ExplicitInstruction:
        """Add a new instruction."""
        if category is None:
            category = self._detect_category(instruction)
        
        keywords = self._extract_keywords(instruction)
        
        instruction_obj = ExplicitInstruction(
            id=str(uuid.uuid4()),
            instruction=instruction,
            category=category,
            keywords=keywords,
            timestamp=time.time(),
            priority=priority,
            context=context,
        )
        
        with self._lock:
            self.instructions[instruction_obj.id] = instruction_obj
        
        self._save()
        return instruction_obj
    
    def parse_and_add(self, message: str) -> List[ExplicitInstruction]:
        """Parse a message for instructions and add them."""
        found_instructions = []
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, message, re.IGNORECASE)
                for match in matches:
                    instruction_text = match.group(0)
                    instr = self.add_instruction(
                        instruction_text,
                        category=category,
                    )
                    found_instructions.append(instr)
        
        return found_instructions
    
    def check_conflict(self, proposed_action: str) -> List[Dict[str, Any]]:
        """Check if a proposed action conflicts with any instructions."""
        conflicts = []
        
        with self._lock:
            for instr in self.instructions.values():
                if not instr.active:
                    continue
                
                matches, score = instr.matches(proposed_action)
                if matches:
                    is_conflict = instr.category in [
                        InstructionCategory.DONT,
                        InstructionCategory.AVOID,
                        InstructionCategory.NEVER,
                    ]
                    
                    if is_conflict:
                        conflicts.append({
                            "instruction_id": instr.id,
                            "instruction": instr.instruction,
                            "category": instr.category.value,
                            "match_score": score,
                            "priority": instr.priority,
                        })
        
        conflicts.sort(key=lambda x: (x["priority"], x["match_score"]), reverse=True)
        return conflicts
    
    def get_relevant_instructions(
        self,
        context: str,
        categories: Optional[List[InstructionCategory]] = None,
    ) -> List[ExplicitInstruction]:
        """Get instructions relevant to a context."""
        relevant = []
        context_lower = context.lower()
        
        for instr in self.instructions.values():
            if not instr.active:
                continue
            
            if categories and instr.category not in categories:
                continue
            
            matches, score = instr.matches(context)
            if matches or any(kw.lower() in context_lower for kw in instr.keywords):
                relevant.append(instr)
        
        relevant.sort(key=lambda x: x.priority, reverse=True)
        return relevant
    
    def get_all_instructions(
        self,
        active_only: bool = True,
    ) -> List[ExplicitInstruction]:
        """Get all stored instructions."""
        instructions = list(self.instructions.values())
        if active_only:
            instructions = [i for i in instructions if i.active]
        return instructions
    
    def get_instructions_by_category(
        self,
        category: InstructionCategory,
    ) -> List[ExplicitInstruction]:
        """Get instructions by category."""
        return [
            i for i in self.instructions.values()
            if i.category == category and i.active
        ]
    
    def deactivate_instruction(self, instruction_id: str):
        """Deactivate an instruction without deleting it."""
        with self._lock:
            if instruction_id in self.instructions:
                self.instructions[instruction_id].active = False
        self._save()
    
    def activate_instruction(self, instruction_id: str):
        """Reactivate an instruction."""
        with self._lock:
            if instruction_id in self.instructions:
                self.instructions[instruction_id].active = True
        self._save()
    
    def delete_instruction(self, instruction_id: str):
        """Permanently delete an instruction."""
        with self._lock:
            if instruction_id in self.instructions:
                del self.instructions[instruction_id]
        self._save()
    
    def record_trigger(self, instruction_id: str):
        """Record that an instruction was triggered."""
        with self._lock:
            if instruction_id in self.instructions:
                instr = self.instructions[instruction_id]
                instr.times_triggered += 1
                instr.last_triggered = time.time()
        self._save()
    
    def get_summary(self) -> str:
        """Get a summary of all instructions."""
        by_category: Dict[str, List[str]] = {}
        
        for instr in self.instructions.values():
            if not instr.active:
                continue
            cat = instr.category.value.upper()
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(instr.instruction[:80])
        
        summary_parts = ["User Instructions:"]
        for cat, instrs in by_category.items():
            summary_parts.append(f"\n{cat}:")
            for i in instrs[:5]:
                summary_parts.append(f"  - {i}")
            if len(instrs) > 5:
                summary_parts.append(f"  ... and {len(instrs) - 5} more")
        
        return "\n".join(summary_parts)
    
    def _detect_category(self, instruction: str) -> InstructionCategory:
        """Detect the category of an instruction."""
        instruction_lower = instruction.lower()
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, instruction_lower, re.IGNORECASE):
                    return category
        
        return InstructionCategory.DO
    
    def _extract_keywords(self, instruction: str) -> List[str]:
        """Extract keywords from an instruction."""
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'if', 'then', 'than', 'so', 'as', 'that',
            'this', 'it', 'i', 'you', 'we', 'they', 'my', 'your', 'our',
            'please', 'dont', "don't", 'never', 'always', 'use', 'make',
        }
        
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', instruction.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(keywords))[:10]
    
    def _save(self):
        """Save instructions to storage."""
        if not self.storage_path:
            return
        
        try:
            data = {
                "instructions": {
                    k: v.to_dict() for k, v in self.instructions.items()
                },
            }
            
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def _load(self):
        """Load instructions from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.instructions = {
                k: ExplicitInstruction.from_dict(v)
                for k, v in data.get("instructions", {}).items()
            }
        except Exception:
            pass
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instructions": {k: v.to_dict() for k, v in self.instructions.items()},
        }


class ContextWindow:
    """
    Efficiently manages a 16k token context window.
    
    Features:
    - Prioritize most relevant context
    - Summarize less critical information
    - Always include recent user instructions
    - Dynamic context allocation
    """
    
    DEFAULT_MAX_TOKENS = 16384
    
    RESERVED_CATEGORIES = {
        "system": 2000,
        "instructions": 1500,
        "recent_context": 2000,
    }
    
    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.max_tokens = max_tokens
        self.items: Dict[str, ContextItem] = {}
        self._lock = threading.Lock()
    
    def add_context(
        self,
        category: str,
        content: str,
        priority: int = 5,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextItem:
        """Add content to the context window."""
        item_id = str(uuid.uuid4())
        tokens = self._estimate_tokens(content)
        
        expires_at = None
        if ttl_seconds:
            expires_at = time.time() + ttl_seconds
        
        item = ContextItem(
            id=item_id,
            category=category,
            content=content,
            priority=priority,
            tokens=tokens,
            timestamp=time.time(),
            expires_at=expires_at,
            metadata=metadata or {},
        )
        
        with self._lock:
            self.items[item_id] = item
            self._optimize()
        
        return item
    
    def update_context(
        self,
        item_id: str,
        content: Optional[str] = None,
        priority: Optional[int] = None,
    ):
        """Update an existing context item."""
        with self._lock:
            if item_id in self.items:
                item = self.items[item_id]
                if content is not None:
                    item.content = content
                    item.tokens = self._estimate_tokens(content)
                if priority is not None:
                    item.priority = priority
                item.timestamp = time.time()
                self._optimize()
    
    def remove_context(self, item_id: str):
        """Remove a context item."""
        with self._lock:
            if item_id in self.items:
                del self.items[item_id]
    
    def remove_by_category(self, category: str):
        """Remove all items in a category."""
        with self._lock:
            self.items = {
                k: v for k, v in self.items.items()
                if v.category != category
            }
    
    def get_context(self) -> str:
        """Get the assembled context within token limits."""
        self._cleanup_expired()
        
        with self._lock:
            sorted_items = sorted(
                self.items.values(),
                key=lambda x: (x.priority, x.timestamp),
                reverse=True,
            )
        
        assembled = []
        current_tokens = 0
        
        for item in sorted_items:
            if current_tokens + item.tokens <= self.max_tokens:
                assembled.append(item.content)
                current_tokens += item.tokens
            elif item.priority >= ContextPriority.HIGH.value:
                summarized = self._summarize(item.content, item.tokens // 2)
                summary_tokens = self._estimate_tokens(summarized)
                if current_tokens + summary_tokens <= self.max_tokens:
                    assembled.append(summarized)
                    current_tokens += summary_tokens
        
        return "\n\n".join(assembled)
    
    def get_by_category(self, category: str) -> List[ContextItem]:
        """Get all items in a category."""
        return [
            item for item in self.items.values()
            if item.category == category
        ]
    
    def get_usage(self) -> Dict[str, Any]:
        """Get context window usage statistics."""
        total_tokens = sum(item.tokens for item in self.items.values())
        by_category: Dict[str, int] = {}
        
        for item in self.items.values():
            if item.category not in by_category:
                by_category[item.category] = 0
            by_category[item.category] += item.tokens
        
        return {
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": (total_tokens / self.max_tokens) * 100,
            "items_count": len(self.items),
            "by_category": by_category,
            "available_tokens": self.max_tokens - total_tokens,
        }
    
    def has_capacity(self, tokens_needed: int) -> bool:
        """Check if there's capacity for additional tokens."""
        current_usage = sum(item.tokens for item in self.items.values())
        return (current_usage + tokens_needed) <= self.max_tokens
    
    def _optimize(self):
        """Optimize context window by removing low-priority items if over limit."""
        total_tokens = sum(item.tokens for item in self.items.values())
        
        if total_tokens <= self.max_tokens:
            return
        
        sorted_items = sorted(
            list(self.items.items()),
            key=lambda x: (x[1].priority, x[1].timestamp),
        )
        
        for item_id, item in sorted_items:
            if total_tokens <= self.max_tokens:
                break
            
            if item.category not in self.RESERVED_CATEGORIES:
                if item.priority < ContextPriority.HIGH.value:
                    total_tokens -= item.tokens
                    del self.items[item_id]
    
    def _cleanup_expired(self):
        """Remove expired context items."""
        current_time = time.time()
        with self._lock:
            self.items = {
                k: v for k, v in self.items.items()
                if not v.is_expired()
            }
    
    def _summarize(self, content: str, target_tokens: int) -> str:
        """Summarize content to fit within token limit."""
        words = content.split()
        target_words = target_tokens * 3 // 4
        
        if len(words) <= target_words:
            return content
        
        lines = content.split('\n')
        if len(lines) > 5:
            summary_lines = [
                lines[0],
                f"... ({len(lines) - 2} lines summarized) ...",
                lines[-1],
            ]
            return '\n'.join(summary_lines)
        
        return ' '.join(words[:target_words]) + "..."
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text.split()) + len(text) // 4
    
    def clear(self):
        """Clear all context items."""
        with self._lock:
            self.items.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "usage": self.get_usage(),
        }


class ContextManager:
    """
    Main context manager that orchestrates all context components.
    
    Features:
    - Load and update project context
    - Manage conversation memory
    - Track user instructions
    - Learn from corrections
    - Manage context window
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_context_tokens: int = 16384,
    ):
        self.storage_dir = storage_dir or ".platform_forge"
        self.max_context_tokens = max_context_tokens
        
        self.context: Optional[ProjectContext] = None
        self.conversation = ConversationMemory()
        self.learning = LearningMemory(
            storage_path=os.path.join(self.storage_dir, "learning.json")
            if storage_dir else None
        )
        self.instructions = InstructionTracker(
            storage_path=os.path.join(self.storage_dir, "instructions.json")
            if storage_dir else None
        )
        self.context_window = ContextWindow(max_tokens=max_context_tokens)
        self._memory_store: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def load_context(self, project_path: str) -> ProjectContext:
        """Load or create context for a project."""
        context_file = os.path.join(
            project_path, self.storage_dir, "context.json"
        )
        
        if os.path.exists(context_file):
            try:
                with open(context_file, 'r') as f:
                    data = json.load(f)
                self.context = ProjectContext.from_dict(data)
                self.context.updated_at = time.time()
            except Exception:
                self.context = self._create_new_context(project_path)
        else:
            self.context = self._create_new_context(project_path)
        
        self._initialize_context_window()
        return self.context
    
    def _create_new_context(self, project_path: str) -> ProjectContext:
        """Create a new project context by analyzing the project."""
        project_name = os.path.basename(project_path) or "Unnamed Project"
        project_id = hashlib.md5(project_path.encode()).hexdigest()[:12]
        
        context = ProjectContext(
            project_id=project_id,
            project_name=project_name,
        )
        
        context = self._analyze_project(project_path, context)
        
        self._save_context(project_path)
        return context
    
    def _analyze_project(
        self,
        project_path: str,
        context: ProjectContext,
    ) -> ProjectContext:
        """Analyze project structure to populate context."""
        if not os.path.exists(project_path):
            return context
        
        package_json = os.path.join(project_path, "package.json")
        if os.path.exists(package_json):
            try:
                with open(package_json, 'r') as f:
                    pkg = json.load(f)
                
                if "typescript" not in context.languages:
                    if os.path.exists(os.path.join(project_path, "tsconfig.json")):
                        context.languages.append("typescript")
                    else:
                        context.languages.append("javascript")
                
                deps = pkg.get("dependencies", {})
                dev_deps = pkg.get("devDependencies", {})
                all_deps = {**deps, **dev_deps}
                context.dependencies.update(all_deps)
                
                frameworks = []
                if "react" in all_deps:
                    frameworks.append("React")
                if "vue" in all_deps:
                    frameworks.append("Vue")
                if "express" in all_deps:
                    frameworks.append("Express")
                if "next" in all_deps:
                    frameworks.append("Next.js")
                context.frameworks.extend(frameworks)
                
                context.project_type = "web_app"
            except Exception:
                pass
        
        requirements_txt = os.path.join(project_path, "requirements.txt")
        pyproject = os.path.join(project_path, "pyproject.toml")
        if os.path.exists(requirements_txt) or os.path.exists(pyproject):
            if "python" not in context.languages:
                context.languages.append("python")
        
        file_structure: Dict[str, List[str]] = {
            "directories": [],
            "key_files": [],
        }
        
        for item in os.listdir(project_path):
            item_path = os.path.join(project_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                file_structure["directories"].append(item)
            elif os.path.isfile(item_path):
                if item in ["package.json", "tsconfig.json", "requirements.txt",
                           "pyproject.toml", "Cargo.toml", "go.mod", "main.py",
                           "main.ts", "index.ts", "App.tsx"]:
                    file_structure["key_files"].append(item)
        
        context.file_structure = file_structure
        
        return context
    
    def _save_context(self, project_path: str):
        """Save context to storage."""
        if not self.context:
            return
        
        storage_path = os.path.join(project_path, self.storage_dir)
        os.makedirs(storage_path, exist_ok=True)
        
        context_file = os.path.join(storage_path, "context.json")
        try:
            with open(context_file, 'w') as f:
                json.dump(self.context.to_dict(), f, indent=2)
        except Exception:
            pass
    
    def _initialize_context_window(self):
        """Initialize context window with project context."""
        if not self.context:
            return
        
        self.context_window.add_context(
            "project",
            self.context.get_summary(),
            priority=ContextPriority.HIGH.value,
        )
        
        instructions_summary = self.instructions.get_summary()
        if instructions_summary:
            self.context_window.add_context(
                "instructions",
                instructions_summary,
                priority=ContextPriority.CRITICAL.value,
            )
    
    def update_context(self, updates: Dict[str, Any]) -> ProjectContext:
        """Incrementally update project context."""
        if not self.context:
            raise ValueError("No context loaded. Call load_context first.")
        
        for key, value in updates.items():
            if hasattr(self.context, key):
                if isinstance(value, list) and hasattr(self.context, key):
                    current = getattr(self.context, key)
                    if isinstance(current, list):
                        for item in value:
                            if item not in current:
                                current.append(item)
                        continue
                elif isinstance(value, dict) and hasattr(self.context, key):
                    current = getattr(self.context, key)
                    if isinstance(current, dict):
                        current.update(value)
                        continue
                
                setattr(self.context, key, value)
        
        self.context.updated_at = time.time()
        return self.context
    
    def get_relevant_context(self, query: str) -> Dict[str, Any]:
        """Get context relevant to a query."""
        relevant = {
            "project": None,
            "instructions": [],
            "patterns": [],
            "conversation": None,
        }
        
        if self.context:
            relevant["project"] = {
                "name": self.context.project_name,
                "type": self.context.project_type,
                "languages": self.context.languages,
                "frameworks": self.context.frameworks,
            }
        
        relevant["instructions"] = [
            i.to_dict()
            for i in self.instructions.get_relevant_instructions(query)[:5]
        ]
        
        patterns = self.learning.find_applicable_patterns(query)
        relevant["patterns"] = [
            {"pattern": p.to_dict(), "score": s}
            for p, s in patterns[:3]
        ]
        
        relevant["conversation"] = self.conversation.summarize(max_tokens=200)
        
        return relevant
    
    def add_user_instruction(
        self,
        instruction: str,
        priority: int = 5,
    ) -> ExplicitInstruction:
        """Add an explicit user instruction."""
        instr = self.instructions.add_instruction(instruction, priority=priority)
        
        if self.context:
            self.context.explicit_instructions.append(instr)
        
        self.context_window.add_context(
            "instructions",
            self.instructions.get_summary(),
            priority=ContextPriority.CRITICAL.value,
        )
        
        return instr
    
    def add_learned_pattern(
        self,
        pattern_type: str,
        trigger: str,
        correct_response: str,
        context: str = "",
    ) -> LearnedPattern:
        """Add a learned pattern."""
        return self.learning.add_learned_pattern(
            pattern_type=pattern_type,
            trigger=trigger,
            correct_response=correct_response,
            context=context,
        )
    
    def get_user_instructions(
        self,
        active_only: bool = True,
    ) -> List[ExplicitInstruction]:
        """Get all stored user instructions."""
        return self.instructions.get_all_instructions(active_only=active_only)
    
    def clear_stale_context(self, max_age_days: int = 30):
        """Clear context older than max_age_days."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        with self._lock:
            self.instructions.instructions = {
                k: v for k, v in self.instructions.instructions.items()
                if v.timestamp > cutoff_time
            }
            
            self.learning.patterns = {
                k: v for k, v in self.learning.patterns.items()
                if v.created_at > cutoff_time or v.times_applied > 5
            }
            
            self.learning.mistakes = [
                m for m in self.learning.mistakes
                if m.get("timestamp", 0) > cutoff_time
            ]
    
    def check_action_conflicts(self, proposed_action: str) -> List[Dict[str, Any]]:
        """Check if a proposed action conflicts with user instructions."""
        return self.instructions.check_conflict(proposed_action)
    
    def record_interaction(
        self,
        role: str,
        content: str,
        was_successful: Optional[bool] = None,
    ) -> ConversationMessage:
        """Record an interaction in conversation memory."""
        message = self.conversation.add_message(role, content)
        
        if was_successful is not None:
            self.conversation.mark_success(message.id, was_successful)
        
        return message
    
    def get_full_context(self) -> str:
        """Get the full assembled context for AI consumption."""
        return self.context_window.get_context()
    
    def remember(self, key: str, value: Any):
        """Store a value for later retrieval."""
        with self._lock:
            self._memory_store[key] = {
                "value": value,
                "timestamp": time.time(),
            }
    
    def recall(self, key: str) -> Optional[Any]:
        """Retrieve a previously stored value."""
        with self._lock:
            if key in self._memory_store:
                return self._memory_store[key]["value"]
        return None
    
    def forget(self, key: str):
        """Remove a stored value."""
        with self._lock:
            if key in self._memory_store:
                del self._memory_store[key]
    
    def get_project_summary(self) -> str:
        """Get a quick overview of the project context."""
        if not self.context:
            return "No project context loaded."
        
        return self.context.get_summary()
    
    def search_context(self, query: str) -> Dict[str, Any]:
        """Search across all context for relevant information."""
        results = {
            "instructions": [],
            "patterns": [],
            "memory": [],
        }
        
        query_lower = query.lower()
        
        for instr in self.instructions.get_all_instructions():
            if query_lower in instr.instruction.lower():
                results["instructions"].append(instr.to_dict())
        
        patterns = self.learning.find_applicable_patterns(query)
        results["patterns"] = [
            {"pattern": p.to_dict(), "score": s}
            for p, s in patterns
        ]
        
        with self._lock:
            for key, data in self._memory_store.items():
                if query_lower in key.lower():
                    results["memory"].append({
                        "key": key,
                        "value": data["value"],
                    })
        
        return results
    
    def export_state(self) -> Dict[str, Any]:
        """Export the complete state for backup or transfer."""
        return {
            "context": self.context.to_dict() if self.context else None,
            "conversation": self.conversation.to_dict(),
            "learning": self.learning.to_dict(),
            "instructions": self.instructions.to_dict(),
            "context_window": self.context_window.to_dict(),
            "memory_store": self._memory_store,
            "exported_at": time.time(),
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import a previously exported state."""
        if state.get("context"):
            self.context = ProjectContext.from_dict(state["context"])
        
        if state.get("conversation"):
            self.conversation = ConversationMemory.from_dict(state["conversation"])
        
        if state.get("learning"):
            for k, v in state["learning"].get("patterns", {}).items():
                self.learning.patterns[k] = LearnedPattern.from_dict(v)
            self.learning.mistakes = state["learning"].get("mistakes", [])
            for k, v in state["learning"].get("preferences", {}).items():
                self.learning.preferences[k] = UserPreference.from_dict(v)
        
        if state.get("instructions"):
            for k, v in state["instructions"].get("instructions", {}).items():
                self.instructions.instructions[k] = ExplicitInstruction.from_dict(v)
        
        if state.get("memory_store"):
            self._memory_store = state["memory_store"]


_default_manager: Optional[ContextManager] = None
_manager_lock = threading.Lock()


def get_default_manager() -> ContextManager:
    """Get or create the default context manager."""
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = ContextManager()
        return _default_manager


def set_default_manager(manager: ContextManager):
    """Set the default context manager."""
    global _default_manager
    with _manager_lock:
        _default_manager = manager


def get_project_summary() -> str:
    """Quick overview of the current project."""
    return get_default_manager().get_project_summary()


def search_context(query: str) -> Dict[str, Any]:
    """Find relevant context for a query."""
    return get_default_manager().search_context(query)


def remember(key: str, value: Any):
    """Store a value for later retrieval."""
    get_default_manager().remember(key, value)


def recall(key: str) -> Optional[Any]:
    """Retrieve a previously stored value."""
    return get_default_manager().recall(key)


def load_project_context(project_path: str) -> ProjectContext:
    """Load context for a project."""
    return get_default_manager().load_context(project_path)


def add_instruction(instruction: str, priority: int = 5) -> ExplicitInstruction:
    """Add a user instruction."""
    return get_default_manager().add_user_instruction(instruction, priority)


def check_conflicts(proposed_action: str) -> List[Dict[str, Any]]:
    """Check for instruction conflicts with an action."""
    return get_default_manager().check_action_conflicts(proposed_action)


def learn_from_correction(
    mistake: str,
    correction: str,
    context: str = "",
) -> LearnedPattern:
    """Learn from a user correction."""
    return get_default_manager().learning.record_mistake(
        mistake=mistake,
        correction=correction,
        context=context,
    )


if __name__ == "__main__":
    print("Context Manager Demo")
    print("=" * 60)
    
    manager = ContextManager()
    
    manager.add_user_instruction("Don't use var, always use const or let")
    manager.add_user_instruction("Prefer async/await over callbacks")
    manager.add_user_instruction("Always add error handling")
    
    print("\nUser Instructions:")
    for instr in manager.get_user_instructions():
        print(f"  [{instr.category.value}] {instr.instruction}")
    
    conflicts = manager.check_action_conflicts("Use var for loop counter")
    print(f"\nConflicts found: {len(conflicts)}")
    for c in conflicts:
        print(f"  - {c['instruction']}")
    
    manager.learning.record_mistake(
        mistake="Used callback instead of async/await",
        correction="Use async/await for better readability",
    )
    
    patterns = manager.learning.find_applicable_patterns("callback function")
    print(f"\nApplicable patterns: {len(patterns)}")
    for p, score in patterns:
        print(f"  - {p.correct_response} (score: {score:.2f})")
    
    manager.remember("last_edited_file", "src/App.tsx")
    print(f"\nRecalled: {manager.recall('last_edited_file')}")
    
    manager.conversation.add_message("user", "Fix the login bug")
    manager.conversation.add_message("assistant", "I'll update the auth module...")
    print(f"\nConversation summary:\n{manager.conversation.summarize()}")
    
    print(f"\nContext window usage: {manager.context_window.get_usage()}")
