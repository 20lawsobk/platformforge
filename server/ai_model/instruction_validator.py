"""
Instruction Validation System for Platform Forge

This module ensures the AI follows user commands and doesn't do things the user
explicitly said NOT to do. It provides comprehensive instruction parsing, validation,
emphasis detection, and compliance checking.

Key Components:
- Instruction: Dataclass for representing user instructions with type, priority, scope
- InstructionParser: Parse natural language instructions detecting negations/imperatives
- InstructionValidator: Validate proposed actions against active instructions
- ValidationResult: Complete result of instruction validation
- EmphasisDetector: Detect user frustration and emphasis patterns
- ComplianceChecker: Check if generated code complies with all instructions

Usage:
    from server.ai_model.instruction_validator import (
        Instruction,
        InstructionType,
        InstructionScope,
        InstructionParser,
        InstructionValidator,
        ValidationResult,
        EmphasisDetector,
        ComplianceChecker,
        parse_instruction,
        check_compliance,
        get_prohibition_conflicts,
        extract_action_intent,
    )
    
    # Parse a user instruction
    instruction = parse_instruction("Don't use var, always use const or let")
    print(f"Type: {instruction.instruction_type}, Keywords: {instruction.keywords}")
    
    # Validate an action
    validator = InstructionValidator()
    result = validator.validate_action("var x = 5", [instruction])
    if not result.is_valid:
        print(f"Violation: {result.user_message}")
    
    # Detect emphasis/frustration
    detector = EmphasisDetector()
    emphasis = detector.detect("I ALREADY TOLD YOU not to use var!!!")
    print(f"Priority boost: {emphasis.priority_boost}")
    
    # Check code compliance
    checker = ComplianceChecker()
    compliance = checker.check_code("const x = 5; let y = 10;", [instruction])
    print(f"Compliance score: {compliance.score}%")
"""

import re
import time
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from difflib import SequenceMatcher


class InstructionType(Enum):
    """Types of user instructions."""
    COMMAND = "command"
    PROHIBITION = "prohibition"
    PREFERENCE = "preference"
    WARNING = "warning"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def priority_weight(self) -> float:
        """Default priority weight for each type."""
        weights = {
            InstructionType.COMMAND: 1.0,
            InstructionType.PROHIBITION: 1.2,
            InstructionType.PREFERENCE: 0.6,
            InstructionType.WARNING: 0.8,
        }
        return weights[self]


class InstructionScope(Enum):
    """Scope of instruction application."""
    GLOBAL = "global"
    FILE = "file"
    FUNCTION = "function"
    LINE = "line"
    SESSION = "session"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def specificity(self) -> int:
        """Specificity level (higher = more specific)."""
        levels = {
            InstructionScope.GLOBAL: 1,
            InstructionScope.SESSION: 2,
            InstructionScope.FILE: 3,
            InstructionScope.FUNCTION: 4,
            InstructionScope.LINE: 5,
        }
        return levels[self]


class ViolationSeverity(Enum):
    """Severity levels for instruction violations."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def __str__(self) -> str:
        return self.value
    
    def __lt__(self, other):
        order = [ViolationSeverity.INFO, ViolationSeverity.LOW, 
                 ViolationSeverity.MEDIUM, ViolationSeverity.HIGH, 
                 ViolationSeverity.CRITICAL]
        return order.index(self) < order.index(other)
    
    def __le__(self, other):
        return self == other or self < other
    
    @property
    def numeric_value(self) -> int:
        """Numeric value for calculations."""
        mapping = {
            ViolationSeverity.INFO: 5,
            ViolationSeverity.LOW: 15,
            ViolationSeverity.MEDIUM: 35,
            ViolationSeverity.HIGH: 60,
            ViolationSeverity.CRITICAL: 100
        }
        return mapping[self]


class EmphasisLevel(Enum):
    """Levels of emphasis detected in user messages."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"
    
    @property
    def priority_boost(self) -> float:
        """Priority boost multiplier for each emphasis level."""
        boosts = {
            EmphasisLevel.NONE: 1.0,
            EmphasisLevel.MILD: 1.2,
            EmphasisLevel.MODERATE: 1.5,
            EmphasisLevel.STRONG: 2.0,
            EmphasisLevel.EXTREME: 3.0,
        }
        return boosts[self]


@dataclass
class Instruction:
    """
    Represents a user instruction that the AI must follow.
    
    Attributes:
        id: Unique identifier for the instruction
        text: Original instruction text from user
        instruction_type: Type of instruction (COMMAND, PROHIBITION, etc.)
        priority: Priority level (1-10, higher = more important)
        created_at: Timestamp when instruction was created
        expires_at: Optional timestamp when instruction expires
        scope: Scope of application (global, file, function, etc.)
        keywords: Extracted keywords for matching
        target_file: Specific file if scope is FILE
        target_function: Specific function if scope is FUNCTION
        is_active: Whether the instruction is currently active
        times_violated: Number of times this instruction was violated
        last_violated: Timestamp of last violation
        source: Where the instruction came from (user, system, learned)
        emphasis_level: Detected emphasis level when instruction was given
        original_priority: Original priority before any boosts
    """
    id: str
    text: str
    instruction_type: InstructionType
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    scope: InstructionScope = InstructionScope.GLOBAL
    keywords: List[str] = field(default_factory=list)
    target_file: Optional[str] = None
    target_function: Optional[str] = None
    is_active: bool = True
    times_violated: int = 0
    last_violated: Optional[float] = None
    source: str = "user"
    emphasis_level: EmphasisLevel = EmphasisLevel.NONE
    original_priority: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.original_priority is None:
            self.original_priority = self.priority
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    def is_expired(self) -> bool:
        """Check if the instruction has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def applies_to_scope(self, file_path: Optional[str] = None, 
                        function_name: Optional[str] = None) -> bool:
        """Check if instruction applies to given scope."""
        if not self.is_active or self.is_expired():
            return False
        
        if self.scope == InstructionScope.GLOBAL:
            return True
        elif self.scope == InstructionScope.SESSION:
            return True
        elif self.scope == InstructionScope.FILE:
            if self.target_file and file_path:
                return self.target_file in file_path or file_path in self.target_file
            return True
        elif self.scope == InstructionScope.FUNCTION:
            if self.target_function and function_name:
                return self.target_function == function_name
            return True
        elif self.scope == InstructionScope.LINE:
            return True
        
        return True
    
    def matches_action(self, action: str) -> Tuple[bool, float]:
        """
        Check if this instruction matches a proposed action.
        
        Returns:
            Tuple of (matches, confidence_score)
        """
        action_lower = action.lower()
        
        if not self.keywords:
            return False, 0.0
        
        matched_keywords = sum(1 for kw in self.keywords if kw.lower() in action_lower)
        if matched_keywords == 0:
            return False, 0.0
        
        match_ratio = matched_keywords / len(self.keywords)
        
        text_similarity = SequenceMatcher(None, self.text.lower(), action_lower).ratio()
        
        confidence = (match_ratio * 0.7) + (text_similarity * 0.3)
        
        return True, confidence
    
    def record_violation(self):
        """Record that this instruction was violated."""
        self.times_violated += 1
        self.last_violated = time.time()
    
    def boost_priority(self, factor: float):
        """Boost priority by a factor (used for repeated instructions)."""
        self.priority = min(10, int(self.original_priority * factor))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "instruction_type": self.instruction_type.value,
            "priority": self.priority,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "scope": self.scope.value,
            "keywords": self.keywords,
            "target_file": self.target_file,
            "target_function": self.target_function,
            "is_active": self.is_active,
            "times_violated": self.times_violated,
            "last_violated": self.last_violated,
            "source": self.source,
            "emphasis_level": self.emphasis_level.value,
            "original_priority": self.original_priority,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Instruction":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            text=data["text"],
            instruction_type=InstructionType(data["instruction_type"]),
            priority=data.get("priority", 5),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            scope=InstructionScope(data.get("scope", "global")),
            keywords=data.get("keywords", []),
            target_file=data.get("target_file"),
            target_function=data.get("target_function"),
            is_active=data.get("is_active", True),
            times_violated=data.get("times_violated", 0),
            last_violated=data.get("last_violated"),
            source=data.get("source", "user"),
            emphasis_level=EmphasisLevel(data.get("emphasis_level", "none")),
            original_priority=data.get("original_priority"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ActionIntent:
    """Extracted intent from an instruction."""
    action_verb: str
    target: str
    modifiers: List[str] = field(default_factory=list)
    is_negated: bool = False
    is_conditional: bool = False
    condition: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_verb": self.action_verb,
            "target": self.target,
            "modifiers": self.modifiers,
            "is_negated": self.is_negated,
            "is_conditional": self.is_conditional,
            "condition": self.condition,
            "confidence": self.confidence,
        }


@dataclass
class Violation:
    """Represents a single instruction violation."""
    instruction: Instruction
    action: str
    description: str
    severity: ViolationSeverity
    confidence: float = 1.0
    location: Optional[str] = None
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_id": self.instruction.id,
            "instruction_text": self.instruction.text,
            "action": self.action,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "location": self.location,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class ValidationResult:
    """
    Complete result of validating an action against instructions.
    
    Attributes:
        is_valid: Whether the action is valid (no violations)
        violations: List of instruction violations found
        warnings: List of warning messages
        conflicting_instructions: Instructions that conflict with the action
        severity: Overall severity of violations
        suggested_modifications: Suggested ways to make action compliant
        user_message: Human-readable message for the user
        confidence: Overall confidence in the validation result
        checked_instructions: Number of instructions checked
        matched_instructions: Number of instructions that matched
    """
    is_valid: bool
    violations: List[Violation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    conflicting_instructions: List[Instruction] = field(default_factory=list)
    severity: ViolationSeverity = ViolationSeverity.INFO
    suggested_modifications: List[str] = field(default_factory=list)
    user_message: str = ""
    confidence: float = 1.0
    checked_instructions: int = 0
    matched_instructions: int = 0
    
    def __post_init__(self):
        if not self.user_message and self.violations:
            self.user_message = self._generate_user_message()
        if self.violations:
            self._calculate_severity()
    
    def _calculate_severity(self):
        """Calculate overall severity from individual violations."""
        if not self.violations:
            self.severity = ViolationSeverity.INFO
            return
        self.severity = max(v.severity for v in self.violations)
    
    def _generate_user_message(self) -> str:
        """Generate a user-friendly message about violations."""
        if not self.violations:
            return "Action is compliant with all instructions."
        
        messages = []
        for v in self.violations:
            severity_prefix = f"[{v.severity.value.upper()}]"
            messages.append(f"{severity_prefix} {v.description}")
            if v.suggested_fix:
                messages.append(f"  Suggestion: {v.suggested_fix}")
        
        return "\n".join(messages)
    
    def add_violation(self, violation: Violation):
        """Add a violation to the result."""
        self.violations.append(violation)
        self.conflicting_instructions.append(violation.instruction)
        self.is_valid = False
        self._calculate_severity()
        self.user_message = self._generate_user_message()
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "conflicting_instructions": [i.to_dict() for i in self.conflicting_instructions],
            "severity": self.severity.value,
            "suggested_modifications": self.suggested_modifications,
            "user_message": self.user_message,
            "confidence": self.confidence,
            "checked_instructions": self.checked_instructions,
            "matched_instructions": self.matched_instructions,
        }


@dataclass
class EmphasisResult:
    """Result of emphasis detection."""
    level: EmphasisLevel
    priority_boost: float
    detected_patterns: List[str] = field(default_factory=list)
    frustration_indicators: List[str] = field(default_factory=list)
    is_repeated: bool = False
    repeat_count: int = 0
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "priority_boost": self.priority_boost,
            "detected_patterns": self.detected_patterns,
            "frustration_indicators": self.frustration_indicators,
            "is_repeated": self.is_repeated,
            "repeat_count": self.repeat_count,
            "confidence": self.confidence,
        }


@dataclass
class ComplianceResult:
    """Result of compliance checking."""
    is_compliant: bool
    score: float
    violations: List[Violation] = field(default_factory=list)
    matched_prohibitions: List[Tuple[str, Instruction]] = field(default_factory=list)
    missing_requirements: List[Tuple[str, Instruction]] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    summary: str = ""
    
    def __post_init__(self):
        if not self.summary:
            self.summary = self._generate_summary()
    
    def _generate_summary(self) -> str:
        """Generate a compliance summary."""
        if self.is_compliant:
            return f"Code is compliant. Score: {self.score:.1f}%"
        
        issues = []
        if self.matched_prohibitions:
            issues.append(f"{len(self.matched_prohibitions)} prohibited patterns found")
        if self.missing_requirements:
            issues.append(f"{len(self.missing_requirements)} required patterns missing")
        
        return f"Compliance score: {self.score:.1f}%. Issues: {'; '.join(issues)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "score": self.score,
            "violations": [v.to_dict() for v in self.violations],
            "matched_prohibitions": [
                {"pattern": p, "instruction_id": i.id} 
                for p, i in self.matched_prohibitions
            ],
            "missing_requirements": [
                {"pattern": p, "instruction_id": i.id}
                for p, i in self.missing_requirements
            ],
            "passed_checks": self.passed_checks,
            "summary": self.summary,
        }


class NegationPatterns:
    """Patterns for detecting negation in instructions."""
    
    EXPLICIT_NEGATIONS = [
        r"\bdon'?t\b",
        r"\bdo\s+not\b",
        r"\bnever\b",
        r"\bstop\b",
        r"\bavoid\b",
        r"\bmust\s+not\b",
        r"\bmustn'?t\b",
        r"\bshouldn'?t\b",
        r"\bshould\s+not\b",
        r"\bcan'?t\b",
        r"\bcannot\b",
        r"\bno\s+\w+ing\b",
        r"\brefrain\s+from\b",
        r"\bstay\s+away\s+from\b",
        r"\bforget\s+about\b",
        r"\bquit\b",
        r"\bstop\s+\w+ing\b",
        r"\bnot\s+allowed\b",
        r"\bprohibit(?:ed)?\b",
        r"\bforbid(?:den)?\b",
        r"\bban(?:ned)?\b",
    ]
    
    IMPLICIT_NEGATIONS = [
        r"\binstead\s+of\b",
        r"\brather\s+than\b",
        r"\bwithout\b",
        r"\bexcept\b",
        r"\bunless\b",
        r"\bexcluding\b",
        r"\bomit\b",
        r"\bskip\b",
        r"\bignore\b",
        r"\bbypass\b",
        r"\breplace\b.*\bwith\b",
    ]
    
    @classmethod
    def get_all_patterns(cls) -> List[str]:
        return cls.EXPLICIT_NEGATIONS + cls.IMPLICIT_NEGATIONS


class ImperativePatterns:
    """Patterns for detecting imperative instructions."""
    
    STRONG_IMPERATIVES = [
        r"\balways\b",
        r"\bmust\b",
        r"\brequired?\b",
        r"\bmandatory\b",
        r"\bessential\b",
        r"\bforced?\b",
        r"\bobligatory\b",
        r"\bcompulsory\b",
        r"\bimperative\b",
        r"\bcritical(?:ly)?\b",
    ]
    
    MODERATE_IMPERATIVES = [
        r"\bshould\b",
        r"\bensure\b",
        r"\bmake\s+sure\b",
        r"\bverify\b",
        r"\bconfirm\b",
        r"\bguarantee\b",
        r"\bneed\s+to\b",
        r"\bhave\s+to\b",
        r"\bgot\s+to\b",
        r"\bought\s+to\b",
    ]
    
    MILD_IMPERATIVES = [
        r"\bprefer\b",
        r"\bfavor\b",
        r"\blike\b",
        r"\bwant\b",
        r"\bwish\b",
        r"\brecommend\b",
        r"\bsuggest\b",
        r"\badvise\b",
        r"\bconsider\b",
        r"\btry\s+to\b",
    ]
    
    @classmethod
    def get_all_patterns(cls) -> List[Tuple[str, str]]:
        """Returns (pattern, strength) tuples."""
        result = []
        for p in cls.STRONG_IMPERATIVES:
            result.append((p, "strong"))
        for p in cls.MODERATE_IMPERATIVES:
            result.append((p, "moderate"))
        for p in cls.MILD_IMPERATIVES:
            result.append((p, "mild"))
        return result


class EmphasisPatterns:
    """Patterns for detecting emphasis and frustration."""
    
    FRUSTRATION_PATTERNS = [
        (r"\bI\s+(already\s+)?told\s+you\b", "frustration"),
        (r"\bI\s+said\b", "frustration"),
        (r"\bhow\s+many\s+times\b", "frustration"),
        (r"\bagain\s*[!?]", "frustration"),
        (r"\bwhy\s+did\s+you\b", "frustration"),
        (r"\bwhy\s+are\s+you\b", "frustration"),
        (r"\bstop\s+doing\b", "frustration"),
        (r"\bplease\s+stop\b", "frustration"),
        (r"\bfor\s+the\s+last\s+time\b", "frustration"),
        (r"\bonce\s+and\s+for\s+all\b", "frustration"),
        (r"\bI\s+keep\s+telling\b", "frustration"),
        (r"\bhow\s+hard\s+is\s+it\b", "frustration"),
        (r"\bseriously\b", "frustration"),
        (r"\bunbelievable\b", "frustration"),
    ]
    
    EMPHASIS_PATTERNS = [
        (r"\bVERY\b", "emphasis"),
        (r"\bEXTREMELY\b", "emphasis"),
        (r"\bABSOLUTELY\b", "emphasis"),
        (r"\bDEFINITELY\b", "emphasis"),
        (r"\bCOMPLETELY\b", "emphasis"),
        (r"\bTOTALLY\b", "emphasis"),
        (r"\bURGENT(?:LY)?\b", "emphasis"),
        (r"\bIMPORTANT\b", "emphasis"),
        (r"\bCRITICAL\b", "emphasis"),
        (r"\bCRUCIAL\b", "emphasis"),
    ]
    
    CAPS_THRESHOLD = 0.5
    EXCLAMATION_THRESHOLD = 2


class InstructionParser:
    """
    Parses natural language instructions to extract structured information.
    
    Features:
    - Detect negations and prohibitions
    - Detect imperatives and commands
    - Extract action targets and keywords
    - Handle emphasis and frustration markers
    """
    
    def __init__(self):
        self.negation_patterns = [re.compile(p, re.IGNORECASE) 
                                   for p in NegationPatterns.get_all_patterns()]
        self.imperative_patterns = [(re.compile(p, re.IGNORECASE), s) 
                                     for p, s in ImperativePatterns.get_all_patterns()]
        self.emphasis_detector = EmphasisDetector()
    
    def parse(self, text: str) -> Instruction:
        """
        Parse a natural language instruction into an Instruction object.
        
        Args:
            text: The instruction text to parse
            
        Returns:
            Parsed Instruction object
        """
        text = text.strip()
        
        has_negation = self._detect_negation(text)
        imperative_strength = self._detect_imperative(text)
        keywords = self._extract_keywords(text)
        action_intent = self._extract_action_intent(text)
        emphasis_result = self.emphasis_detector.detect(text)
        scope = self._detect_scope(text)
        
        instruction_type = self._determine_type(has_negation, imperative_strength)
        
        base_priority = self._calculate_priority(
            instruction_type, imperative_strength, emphasis_result
        )
        
        instruction = Instruction(
            id=str(uuid.uuid4())[:8],
            text=text,
            instruction_type=instruction_type,
            priority=base_priority,
            scope=scope,
            keywords=keywords,
            emphasis_level=emphasis_result.level,
            metadata={
                "has_negation": has_negation,
                "imperative_strength": imperative_strength,
                "action_intent": action_intent.to_dict() if action_intent else None,
                "emphasis_patterns": emphasis_result.detected_patterns,
            }
        )
        
        return instruction
    
    def _detect_negation(self, text: str) -> bool:
        """Detect if the instruction contains negation."""
        for pattern in self.negation_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _detect_imperative(self, text: str) -> str:
        """Detect imperative strength: 'strong', 'moderate', 'mild', or 'none'."""
        for pattern, strength in self.imperative_patterns:
            if pattern.search(text):
                return strength
        return "none"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the instruction."""
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
            'if', 'or', 'because', 'until', 'while', 'although', 'though',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'this',
            'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'dont',
            "don't", 'never', 'always', 'make', 'sure', 'please', 'want',
        }
        
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        
        keywords = []
        for word in words:
            if word not in stopwords and len(word) > 2:
                keywords.append(word)
        
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]
    
    def _extract_action_intent(self, text: str) -> Optional[ActionIntent]:
        """Extract the action intent from the instruction."""
        action_verbs = [
            'use', 'add', 'remove', 'delete', 'create', 'update', 'change',
            'modify', 'write', 'read', 'call', 'invoke', 'execute', 'run',
            'import', 'export', 'include', 'exclude', 'enable', 'disable',
            'install', 'uninstall', 'deploy', 'build', 'test', 'debug',
            'format', 'lint', 'check', 'validate', 'verify', 'ensure',
            'set', 'get', 'put', 'post', 'patch', 'fetch', 'send', 'receive',
        ]
        
        text_lower = text.lower()
        
        for verb in action_verbs:
            pattern = rf'\b{verb}\s+(\w+(?:\s+\w+)?)'
            match = re.search(pattern, text_lower)
            if match:
                target = match.group(1)
                is_negated = self._detect_negation(text)
                
                return ActionIntent(
                    action_verb=verb,
                    target=target,
                    is_negated=is_negated,
                    confidence=0.8
                )
        
        return None
    
    def _detect_scope(self, text: str) -> InstructionScope:
        """Detect the scope of the instruction."""
        text_lower = text.lower()
        
        if re.search(r'\b(in|for)\s+(this\s+)?file\b', text_lower):
            return InstructionScope.FILE
        if re.search(r'\b(in|for)\s+(this\s+)?function\b', text_lower):
            return InstructionScope.FUNCTION
        if re.search(r'\b(on\s+)?this\s+line\b', text_lower):
            return InstructionScope.LINE
        if re.search(r'\b(for\s+)?(this\s+)?session\b', text_lower):
            return InstructionScope.SESSION
        if re.search(r'\beverywhere\b|\balways\b|\bglobal(ly)?\b', text_lower):
            return InstructionScope.GLOBAL
        
        return InstructionScope.GLOBAL
    
    def _determine_type(self, has_negation: bool, imperative_strength: str) -> InstructionType:
        """Determine the instruction type based on detected patterns."""
        if has_negation:
            return InstructionType.PROHIBITION
        elif imperative_strength == "strong":
            return InstructionType.COMMAND
        elif imperative_strength == "moderate":
            return InstructionType.COMMAND
        elif imperative_strength == "mild":
            return InstructionType.PREFERENCE
        else:
            return InstructionType.PREFERENCE
    
    def _calculate_priority(self, instruction_type: InstructionType,
                           imperative_strength: str,
                           emphasis_result: EmphasisResult) -> int:
        """Calculate the priority based on type, strength, and emphasis."""
        base_priority = {
            InstructionType.COMMAND: 7,
            InstructionType.PROHIBITION: 8,
            InstructionType.PREFERENCE: 4,
            InstructionType.WARNING: 6,
        }.get(instruction_type, 5)
        
        strength_boost = {
            "strong": 2,
            "moderate": 1,
            "mild": 0,
            "none": 0,
        }.get(imperative_strength, 0)
        
        emphasis_boost = int((emphasis_result.priority_boost - 1) * 2)
        
        return min(10, base_priority + strength_boost + emphasis_boost)


class EmphasisDetector:
    """
    Detects emphasis and frustration in user messages.
    
    Features:
    - Recognize frustration patterns like "I ALREADY TOLD YOU"
    - Detect CAPS usage for emphasis
    - Count exclamation marks
    - Track repeated instructions for escalating priority
    """
    
    def __init__(self):
        self.frustration_patterns = [
            (re.compile(p, re.IGNORECASE), t)
            for p, t in EmphasisPatterns.FRUSTRATION_PATTERNS
        ]
        self.emphasis_patterns = [
            (re.compile(p, re.IGNORECASE), t)
            for p, t in EmphasisPatterns.EMPHASIS_PATTERNS
        ]
        self.instruction_history: Dict[str, int] = {}
    
    def detect(self, text: str) -> EmphasisResult:
        """
        Detect emphasis and frustration in text.
        
        Args:
            text: The text to analyze
            
        Returns:
            EmphasisResult with detected patterns and priority boost
        """
        detected_patterns = []
        frustration_indicators = []
        
        for pattern, pattern_type in self.frustration_patterns:
            match = pattern.search(text)
            if match:
                detected_patterns.append(match.group())
                frustration_indicators.append(pattern_type)
        
        for pattern, pattern_type in self.emphasis_patterns:
            match = pattern.search(text)
            if match:
                detected_patterns.append(match.group())
        
        caps_ratio = self._calculate_caps_ratio(text)
        exclamation_count = text.count('!')
        
        is_repeated, repeat_count = self._check_repetition(text)
        
        level = self._determine_level(
            len(frustration_indicators),
            caps_ratio,
            exclamation_count,
            is_repeated,
            repeat_count
        )
        
        return EmphasisResult(
            level=level,
            priority_boost=level.priority_boost,
            detected_patterns=detected_patterns,
            frustration_indicators=frustration_indicators,
            is_repeated=is_repeated,
            repeat_count=repeat_count,
            confidence=0.8 + (len(detected_patterns) * 0.05)
        )
    
    def _calculate_caps_ratio(self, text: str) -> float:
        """Calculate the ratio of uppercase characters."""
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        uppercase = sum(1 for c in letters if c.isupper())
        return uppercase / len(letters)
    
    def _check_repetition(self, text: str) -> Tuple[bool, int]:
        """Check if this instruction has been given before."""
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()[:8]
        
        count = self.instruction_history.get(text_hash, 0)
        self.instruction_history[text_hash] = count + 1
        
        return count > 0, count + 1
    
    def _determine_level(self, frustration_count: int, caps_ratio: float,
                        exclamation_count: int, is_repeated: bool,
                        repeat_count: int) -> EmphasisLevel:
        """Determine the overall emphasis level."""
        score = 0
        
        score += frustration_count * 2
        
        if caps_ratio >= EmphasisPatterns.CAPS_THRESHOLD:
            score += 3
        elif caps_ratio >= 0.3:
            score += 1
        
        if exclamation_count >= EmphasisPatterns.EXCLAMATION_THRESHOLD:
            score += min(exclamation_count - 1, 3)
        
        if is_repeated:
            score += min(repeat_count, 4)
        
        if score >= 8:
            return EmphasisLevel.EXTREME
        elif score >= 5:
            return EmphasisLevel.STRONG
        elif score >= 3:
            return EmphasisLevel.MODERATE
        elif score >= 1:
            return EmphasisLevel.MILD
        else:
            return EmphasisLevel.NONE
    
    def record_instruction(self, text: str):
        """Record an instruction for repetition tracking."""
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()[:8]
        self.instruction_history[text_hash] = self.instruction_history.get(text_hash, 0) + 1
    
    def reset_history(self):
        """Reset instruction history."""
        self.instruction_history.clear()


class InstructionValidator:
    """
    Validates proposed actions against a set of user instructions.
    
    Features:
    - Check if actions conflict with instructions
    - Return severity of violations
    - Suggest compliant alternatives
    - Support for different scopes and priorities
    """
    
    def __init__(self):
        self.alternative_suggestions: Dict[str, List[str]] = {
            "var": ["const", "let"],
            "any": ["specific types", "unknown"],
            "console.log": ["logger.debug", "logger.info"],
            "==": ["==="],
            "!=": ["!=="],
            "eval": ["JSON.parse", "Function constructor (carefully)"],
            "innerHTML": ["textContent", "createElement"],
            "document.write": ["DOM manipulation methods"],
        }
    
    def validate_action(self, proposed_action: str, 
                       instructions: List[Instruction],
                       file_path: Optional[str] = None,
                       function_name: Optional[str] = None) -> ValidationResult:
        """
        Validate a proposed action against all relevant instructions.
        
        Args:
            proposed_action: The action or code to validate
            instructions: List of instructions to check against
            file_path: Optional file path for scope filtering
            function_name: Optional function name for scope filtering
            
        Returns:
            ValidationResult with all violations and suggestions
        """
        result = ValidationResult(
            is_valid=True,
            checked_instructions=len(instructions)
        )
        
        sorted_instructions = sorted(
            instructions,
            key=lambda i: (-i.priority, i.instruction_type.priority_weight)
        )
        
        for instruction in sorted_instructions:
            if not instruction.applies_to_scope(file_path, function_name):
                continue
            
            matches, confidence = instruction.matches_action(proposed_action)
            
            if matches:
                result.matched_instructions += 1
                
                if instruction.instruction_type == InstructionType.PROHIBITION:
                    violation = self._create_violation(
                        instruction, proposed_action, confidence
                    )
                    result.add_violation(violation)
                    
                    instruction.record_violation()
                    
                    suggestions = self._get_suggestions(instruction, proposed_action)
                    result.suggested_modifications.extend(suggestions)
                
                elif instruction.instruction_type == InstructionType.WARNING:
                    result.add_warning(
                        f"Warning from instruction '{instruction.text}': "
                        f"Action may conflict with user preferences"
                    )
        
        result.confidence = self._calculate_overall_confidence(result)
        
        return result
    
    def _create_violation(self, instruction: Instruction, 
                         action: str, confidence: float) -> Violation:
        """Create a Violation object for a detected conflict."""
        severity = self._determine_severity(instruction)
        
        description = f"Action conflicts with instruction: '{instruction.text}'"
        if instruction.times_violated > 0:
            description += f" (violated {instruction.times_violated + 1} times)"
        
        suggested_fix = self._get_primary_suggestion(instruction, action)
        
        return Violation(
            instruction=instruction,
            action=action,
            description=description,
            severity=severity,
            confidence=confidence,
            suggested_fix=suggested_fix,
        )
    
    def _determine_severity(self, instruction: Instruction) -> ViolationSeverity:
        """Determine the severity of violating an instruction."""
        base_severity = {
            1: ViolationSeverity.INFO,
            2: ViolationSeverity.INFO,
            3: ViolationSeverity.LOW,
            4: ViolationSeverity.LOW,
            5: ViolationSeverity.MEDIUM,
            6: ViolationSeverity.MEDIUM,
            7: ViolationSeverity.HIGH,
            8: ViolationSeverity.HIGH,
            9: ViolationSeverity.CRITICAL,
            10: ViolationSeverity.CRITICAL,
        }.get(instruction.priority, ViolationSeverity.MEDIUM)
        
        if instruction.emphasis_level in [EmphasisLevel.STRONG, EmphasisLevel.EXTREME]:
            if base_severity < ViolationSeverity.HIGH:
                return ViolationSeverity.HIGH
        
        if instruction.times_violated >= 2:
            if base_severity < ViolationSeverity.HIGH:
                return ViolationSeverity.HIGH
        
        return base_severity
    
    def _get_suggestions(self, instruction: Instruction, 
                        action: str) -> List[str]:
        """Get suggestions for making the action compliant."""
        suggestions = []
        
        for pattern, alternatives in self.alternative_suggestions.items():
            if pattern.lower() in action.lower():
                for alt in alternatives:
                    suggestions.append(f"Consider using '{alt}' instead of '{pattern}'")
        
        if instruction.instruction_type == InstructionType.PROHIBITION:
            suggestions.append(f"Remove or refactor the prohibited pattern")
        
        return suggestions
    
    def _get_primary_suggestion(self, instruction: Instruction, 
                               action: str) -> Optional[str]:
        """Get the primary suggestion for fixing the violation."""
        for pattern, alternatives in self.alternative_suggestions.items():
            if pattern.lower() in action.lower():
                return f"Use '{alternatives[0]}' instead of '{pattern}'"
        
        return "Refactor to comply with instruction"
    
    def _calculate_overall_confidence(self, result: ValidationResult) -> float:
        """Calculate overall confidence in the validation result."""
        if not result.violations:
            return 1.0
        
        avg_confidence = sum(v.confidence for v in result.violations) / len(result.violations)
        return avg_confidence
    
    def add_alternative_suggestion(self, pattern: str, alternatives: List[str]):
        """Add a custom alternative suggestion mapping."""
        self.alternative_suggestions[pattern] = alternatives


class ComplianceChecker:
    """
    Checks if generated code complies with all user instructions.
    
    Features:
    - Scan for prohibited patterns
    - Verify required patterns are present
    - Calculate compliance score
    - Generate detailed reports
    """
    
    def __init__(self):
        self.pattern_cache: Dict[str, re.Pattern] = {}
    
    def check_code(self, code: str, 
                  instructions: List[Instruction],
                  file_path: Optional[str] = None) -> ComplianceResult:
        """
        Check if code complies with all instructions.
        
        Args:
            code: The code to check
            instructions: List of instructions to validate against
            file_path: Optional file path for scope filtering
            
        Returns:
            ComplianceResult with score and details
        """
        violations = []
        matched_prohibitions = []
        missing_requirements = []
        passed_checks = []
        
        for instruction in instructions:
            if not instruction.is_active or instruction.is_expired():
                continue
            
            if instruction.scope == InstructionScope.FILE and file_path:
                if instruction.target_file and instruction.target_file not in file_path:
                    continue
            
            if instruction.instruction_type == InstructionType.PROHIBITION:
                prohibited_matches = self._scan_for_patterns(
                    code, instruction.keywords
                )
                
                if prohibited_matches:
                    for match in prohibited_matches:
                        matched_prohibitions.append((match, instruction))
                        violations.append(Violation(
                            instruction=instruction,
                            action=code[:100],
                            description=f"Prohibited pattern '{match}' found",
                            severity=ViolationSeverity.HIGH,
                            location=f"Pattern: {match}",
                        ))
                else:
                    passed_checks.append(f"No prohibited pattern from: {instruction.text[:50]}")
            
            elif instruction.instruction_type == InstructionType.COMMAND:
                required_patterns = self._extract_required_patterns(instruction)
                for pattern in required_patterns:
                    if not self._pattern_exists(code, pattern):
                        missing_requirements.append((pattern, instruction))
                    else:
                        passed_checks.append(f"Required pattern '{pattern}' found")
        
        score = self._calculate_score(
            len(violations),
            len(matched_prohibitions),
            len(missing_requirements),
            len(passed_checks)
        )
        
        return ComplianceResult(
            is_compliant=len(violations) == 0 and len(missing_requirements) == 0,
            score=score,
            violations=violations,
            matched_prohibitions=matched_prohibitions,
            missing_requirements=missing_requirements,
            passed_checks=passed_checks,
        )
    
    def _scan_for_patterns(self, code: str, keywords: List[str]) -> List[str]:
        """Scan code for matching keyword patterns."""
        matches = []
        code_lower = code.lower()
        
        for keyword in keywords:
            if keyword.lower() in code_lower:
                matches.append(keyword)
        
        return matches
    
    def _extract_required_patterns(self, instruction: Instruction) -> List[str]:
        """Extract patterns that should be required from a command instruction."""
        patterns = []
        
        require_keywords = ['use', 'always', 'must', 'required', 'ensure']
        text_lower = instruction.text.lower()
        
        for kw in require_keywords:
            if kw in text_lower:
                for instruction_kw in instruction.keywords:
                    if instruction_kw not in require_keywords:
                        patterns.append(instruction_kw)
                break
        
        return patterns
    
    def _pattern_exists(self, code: str, pattern: str) -> bool:
        """Check if a pattern exists in the code."""
        return pattern.lower() in code.lower()
    
    def _calculate_score(self, violations: int, prohibitions: int,
                        missing: int, passed: int) -> float:
        """Calculate compliance score (0-100)."""
        total_checks = violations + prohibitions + missing + passed
        
        if total_checks == 0:
            return 100.0
        
        penalty = (violations * 15) + (prohibitions * 10) + (missing * 5)
        
        max_penalty = total_checks * 15
        score = max(0, 100 - (penalty / max_penalty * 100))
        
        return round(score, 1)
    
    def check_single_instruction(self, code: str, 
                                instruction: Instruction) -> Tuple[bool, str]:
        """Check compliance with a single instruction."""
        if instruction.instruction_type == InstructionType.PROHIBITION:
            matches = self._scan_for_patterns(code, instruction.keywords)
            if matches:
                return False, f"Found prohibited: {', '.join(matches)}"
            return True, "No violations"
        
        elif instruction.instruction_type == InstructionType.COMMAND:
            required = self._extract_required_patterns(instruction)
            missing = [p for p in required if not self._pattern_exists(code, p)]
            if missing:
                return False, f"Missing required: {', '.join(missing)}"
            return True, "All requirements met"
        
        return True, "Preference/Warning - no strict check"


def parse_instruction(text: str) -> Instruction:
    """
    Convenience function to parse a single instruction.
    
    Args:
        text: The instruction text to parse
        
    Returns:
        Parsed Instruction object
    """
    parser = InstructionParser()
    return parser.parse(text)


def check_compliance(action: str, 
                    instructions: List[Instruction]) -> ComplianceResult:
    """
    Convenience function to check compliance of an action.
    
    Args:
        action: The action/code to check
        instructions: List of instructions to check against
        
    Returns:
        ComplianceResult with score and details
    """
    checker = ComplianceChecker()
    return checker.check_code(action, instructions)


def get_prohibition_conflicts(action: str, 
                             instructions: List[Instruction]) -> List[Instruction]:
    """
    Get all prohibition instructions that conflict with an action.
    
    Args:
        action: The action to check
        instructions: List of instructions to check against
        
    Returns:
        List of conflicting prohibition instructions
    """
    conflicts = []
    
    for instruction in instructions:
        if instruction.instruction_type != InstructionType.PROHIBITION:
            continue
        
        if not instruction.is_active or instruction.is_expired():
            continue
        
        matches, _ = instruction.matches_action(action)
        if matches:
            conflicts.append(instruction)
    
    return conflicts


def extract_action_intent(instruction_text: str) -> Optional[ActionIntent]:
    """
    Extract the action intent from an instruction text.
    
    Args:
        instruction_text: The instruction to analyze
        
    Returns:
        ActionIntent object or None if no intent detected
    """
    parser = InstructionParser()
    parsed = parser.parse(instruction_text)
    
    if parsed.metadata.get("action_intent"):
        data = parsed.metadata["action_intent"]
        return ActionIntent(
            action_verb=data["action_verb"],
            target=data["target"],
            modifiers=data.get("modifiers", []),
            is_negated=data.get("is_negated", False),
            is_conditional=data.get("is_conditional", False),
            condition=data.get("condition"),
            confidence=data.get("confidence", 0.5),
        )
    
    return None


def validate_action(proposed_action: str,
                   instructions: List[Instruction],
                   file_path: Optional[str] = None,
                   function_name: Optional[str] = None) -> ValidationResult:
    """
    Validate an action against a set of instructions.
    
    Args:
        proposed_action: The action to validate
        instructions: List of instructions to check
        file_path: Optional file context
        function_name: Optional function context
        
    Returns:
        ValidationResult with violations and suggestions
    """
    validator = InstructionValidator()
    return validator.validate_action(
        proposed_action, instructions, file_path, function_name
    )


def detect_emphasis(text: str) -> EmphasisResult:
    """
    Detect emphasis and frustration in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        EmphasisResult with level and patterns
    """
    detector = EmphasisDetector()
    return detector.detect(text)


class InstructionStore:
    """
    Storage and management for instructions.
    
    Features:
    - Add/remove/update instructions
    - Query by type, scope, or keywords
    - Persist to JSON file
    - Automatic expiration handling
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.instructions: Dict[str, Instruction] = {}
        self.storage_path = storage_path
        
        if storage_path:
            self._load_from_file()
    
    def add(self, instruction: Instruction) -> str:
        """Add an instruction to the store."""
        self.instructions[instruction.id] = instruction
        self._save_to_file()
        return instruction.id
    
    def remove(self, instruction_id: str) -> bool:
        """Remove an instruction by ID."""
        if instruction_id in self.instructions:
            del self.instructions[instruction_id]
            self._save_to_file()
            return True
        return False
    
    def get(self, instruction_id: str) -> Optional[Instruction]:
        """Get an instruction by ID."""
        return self.instructions.get(instruction_id)
    
    def get_all(self, active_only: bool = True) -> List[Instruction]:
        """Get all instructions, optionally filtering to active only."""
        instructions = list(self.instructions.values())
        
        if active_only:
            instructions = [i for i in instructions if i.is_active and not i.is_expired()]
        
        return sorted(instructions, key=lambda i: -i.priority)
    
    def get_by_type(self, instruction_type: InstructionType) -> List[Instruction]:
        """Get all instructions of a specific type."""
        return [
            i for i in self.instructions.values()
            if i.instruction_type == instruction_type and i.is_active
        ]
    
    def get_by_scope(self, scope: InstructionScope) -> List[Instruction]:
        """Get all instructions with a specific scope."""
        return [
            i for i in self.instructions.values()
            if i.scope == scope and i.is_active
        ]
    
    def search_by_keyword(self, keyword: str) -> List[Instruction]:
        """Search instructions by keyword."""
        keyword_lower = keyword.lower()
        return [
            i for i in self.instructions.values()
            if any(kw.lower() == keyword_lower for kw in i.keywords)
            and i.is_active
        ]
    
    def get_prohibitions(self) -> List[Instruction]:
        """Get all active prohibition instructions."""
        return self.get_by_type(InstructionType.PROHIBITION)
    
    def get_commands(self) -> List[Instruction]:
        """Get all active command instructions."""
        return self.get_by_type(InstructionType.COMMAND)
    
    def deactivate(self, instruction_id: str) -> bool:
        """Deactivate an instruction without removing it."""
        instruction = self.instructions.get(instruction_id)
        if instruction:
            instruction.is_active = False
            self._save_to_file()
            return True
        return False
    
    def activate(self, instruction_id: str) -> bool:
        """Reactivate a deactivated instruction."""
        instruction = self.instructions.get(instruction_id)
        if instruction:
            instruction.is_active = True
            self._save_to_file()
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """Remove all expired instructions."""
        expired_ids = [
            i.id for i in self.instructions.values() 
            if i.is_expired()
        ]
        
        for id in expired_ids:
            del self.instructions[id]
        
        if expired_ids:
            self._save_to_file()
        
        return len(expired_ids)
    
    def _load_from_file(self):
        """Load instructions from JSON file."""
        import json
        import os
        
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for item in data.get("instructions", []):
                instruction = Instruction.from_dict(item)
                self.instructions[instruction.id] = instruction
        except (json.JSONDecodeError, IOError):
            pass
    
    def _save_to_file(self):
        """Save instructions to JSON file."""
        import json
        import os
        
        if not self.storage_path:
            return
        
        dir_path = os.path.dirname(self.storage_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        data = {
            "instructions": [i.to_dict() for i in self.instructions.values()],
            "saved_at": time.time(),
        }
        
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored instructions."""
        all_instructions = list(self.instructions.values())
        active = [i for i in all_instructions if i.is_active]
        
        type_counts = {}
        for t in InstructionType:
            type_counts[t.value] = sum(1 for i in active if i.instruction_type == t)
        
        scope_counts = {}
        for s in InstructionScope:
            scope_counts[s.value] = sum(1 for i in active if i.scope == s)
        
        total_violations = sum(i.times_violated for i in all_instructions)
        
        return {
            "total": len(all_instructions),
            "active": len(active),
            "expired": sum(1 for i in all_instructions if i.is_expired()),
            "by_type": type_counts,
            "by_scope": scope_counts,
            "total_violations": total_violations,
            "most_violated": max(
                all_instructions, 
                key=lambda i: i.times_violated,
                default=None
            ),
        }


__all__ = [
    "Instruction",
    "InstructionType",
    "InstructionScope",
    "ViolationSeverity",
    "EmphasisLevel",
    "ActionIntent",
    "Violation",
    "ValidationResult",
    "EmphasisResult",
    "ComplianceResult",
    "InstructionParser",
    "EmphasisDetector",
    "InstructionValidator",
    "ComplianceChecker",
    "InstructionStore",
    "NegationPatterns",
    "ImperativePatterns",
    "EmphasisPatterns",
    "parse_instruction",
    "check_compliance",
    "get_prohibition_conflicts",
    "extract_action_intent",
    "validate_action",
    "detect_emphasis",
]
