"""
Comprehensive Error Handling System for Platform Forge

This module provides user-friendly error handling with clear messages and actionable
solutions instead of cryptic technical errors. It includes error translation,
pattern matching, automatic fix suggestions, and logging.

Key Components:
- UserFriendlyError: Main error class with title, description, and solutions
- Solution: Actionable fix suggestion with steps and auto-fix capability
- ErrorTranslator: Translates exceptions to user-friendly errors
- ErrorPatternMatcher: Matches error messages to 100+ known patterns
- SelfServiceResolver: Attempts automatic fixes for common issues
- ErrorLogger: Logs and tracks error patterns

Usage:
    from server.ai_model.error_handler import (
        handle_error,
        suggest_fixes,
        auto_fix,
        get_error_context,
        ErrorTranslator,
        SelfServiceResolver,
    )
    
    # Quick error handling
    try:
        risky_operation()
    except Exception as e:
        friendly_error = handle_error(e)
        print(f"{friendly_error.title}: {friendly_error.description}")
        for solution in friendly_error.solutions:
            print(f"  - {solution.title}: {solution.steps}")
    
    # Auto-fix attempt
    result = auto_fix(error)
    if result.success:
        print("Issue automatically resolved!")
    
    # Get suggestions
    solutions = suggest_fixes(error)
    for s in solutions:
        print(f"{s.title} (confidence: {s.confidence})")
"""

import re
import sys
import os
import traceback
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod
from difflib import SequenceMatcher, get_close_matches


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    def __lt__(self, other):
        order = [ErrorSeverity.INFO, ErrorSeverity.WARNING, ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
        return order.index(self) < order.index(other)
    
    @property
    def emoji(self) -> str:
        """Get emoji representation."""
        mapping = {
            ErrorSeverity.INFO: "[INFO]",
            ErrorSeverity.WARNING: "[WARNING]",
            ErrorSeverity.ERROR: "[ERROR]",
            ErrorSeverity.CRITICAL: "[CRITICAL]"
        }
        return mapping[self]
    
    @property
    def color(self) -> str:
        """Get ANSI color code."""
        mapping = {
            ErrorSeverity.INFO: "\033[36m",
            ErrorSeverity.WARNING: "\033[33m",
            ErrorSeverity.ERROR: "\033[31m",
            ErrorSeverity.CRITICAL: "\033[91m"
        }
        return mapping[self]


class ErrorCategory(Enum):
    """Categories of errors."""
    IMPORT = "import"
    SYNTAX = "syntax"
    TYPE = "type"
    VALUE = "value"
    FILE = "file"
    NETWORK = "network"
    DATABASE = "database"
    PERMISSION = "permission"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    RUNTIME = "runtime"
    SECURITY = "security"
    AUTHENTICATION = "authentication"
    API = "api"
    FRAMEWORK = "framework"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


class EffortLevel(Enum):
    """Effort required to fix an error."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    DIFFICULT = "difficult"
    COMPLEX = "complex"
    
    @property
    def minutes_estimate(self) -> Tuple[int, int]:
        """Estimated time range in minutes."""
        mapping = {
            EffortLevel.TRIVIAL: (1, 5),
            EffortLevel.EASY: (5, 15),
            EffortLevel.MODERATE: (15, 60),
            EffortLevel.DIFFICULT: (60, 240),
            EffortLevel.COMPLEX: (240, 480)
        }
        return mapping[self]


@dataclass
class Solution:
    """Represents a suggested solution for an error."""
    title: str
    steps: List[str]
    auto_fixable: bool = False
    fix_command: Optional[str] = None
    confidence: float = 0.8
    effort_level: EffortLevel = EffortLevel.EASY
    prerequisites: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    rollback_steps: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "steps": self.steps,
            "auto_fixable": self.auto_fixable,
            "fix_command": self.fix_command,
            "confidence": self.confidence,
            "effort_level": self.effort_level.value,
            "prerequisites": self.prerequisites,
            "side_effects": self.side_effects,
            "rollback_steps": self.rollback_steps,
            "documentation_url": self.documentation_url,
        }
    
    def format_human_readable(self) -> str:
        """Format solution for display."""
        lines = [f"Solution: {self.title}"]
        lines.append(f"Confidence: {self.confidence * 100:.0f}%")
        lines.append(f"Effort: {self.effort_level.value} ({self.effort_level.minutes_estimate[0]}-{self.effort_level.minutes_estimate[1]} min)")
        
        if self.auto_fixable and self.fix_command:
            lines.append(f"Quick fix: {self.fix_command}")
        
        if self.steps:
            lines.append("Steps:")
            for i, step in enumerate(self.steps, 1):
                lines.append(f"  {i}. {step}")
        
        return "\n".join(lines)


@dataclass
class FixResult:
    """Result of an automatic fix attempt."""
    success: bool
    message: str
    command_run: Optional[str] = None
    output: str = ""
    error_output: str = ""
    changes_made: List[str] = field(default_factory=list)
    rollback_info: Optional[str] = None
    follow_up_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "command_run": self.command_run,
            "output": self.output,
            "error_output": self.error_output,
            "changes_made": self.changes_made,
            "rollback_info": self.rollback_info,
            "follow_up_actions": self.follow_up_actions,
        }


@dataclass
class ErrorContext:
    """Context information about where an error occurred."""
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    code_snippet: str = ""
    surrounding_lines: List[str] = field(default_factory=list)
    local_variables: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "code_snippet": self.code_snippet,
            "surrounding_lines": self.surrounding_lines,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class UserFriendlyError:
    """User-friendly error with clear description and solutions."""
    error_code: str
    title: str
    description: str
    technical_details: Optional[str] = None
    solutions: List[Solution] = field(default_factory=list)
    related_docs: List[str] = field(default_factory=list)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    category: ErrorCategory = ErrorCategory.UNKNOWN
    context: Optional[ErrorContext] = None
    original_exception: Optional[Exception] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "title": self.title,
            "description": self.description,
            "technical_details": self.technical_details,
            "solutions": [s.to_dict() for s in self.solutions],
            "related_docs": self.related_docs,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict() if self.context else None,
            "tags": self.tags,
        }
    
    def format_human_readable(self, show_technical: bool = False, show_solutions: bool = True) -> str:
        """Format error for human-readable display."""
        lines = []
        lines.append(f"{self.severity.emoji} {self.title}")
        lines.append(f"Error Code: {self.error_code}")
        lines.append("")
        lines.append(self.description)
        
        if show_technical and self.technical_details:
            lines.append("")
            lines.append("Technical Details:")
            lines.append(self.technical_details)
        
        if self.context:
            lines.append("")
            lines.append("Location:")
            if self.context.file_path:
                loc = f"  File: {self.context.file_path}"
                if self.context.line_number:
                    loc += f", Line: {self.context.line_number}"
                lines.append(loc)
            if self.context.code_snippet:
                lines.append(f"  Code: {self.context.code_snippet}")
        
        if show_solutions and self.solutions:
            lines.append("")
            lines.append("Suggested Solutions:")
            for i, solution in enumerate(self.solutions, 1):
                lines.append(f"\n{i}. {solution.title} (Confidence: {solution.confidence * 100:.0f}%)")
                if solution.auto_fixable and solution.fix_command:
                    lines.append(f"   Quick fix: {solution.fix_command}")
                for step in solution.steps:
                    lines.append(f"   - {step}")
        
        if self.related_docs:
            lines.append("")
            lines.append("Related Documentation:")
            for doc in self.related_docs:
                lines.append(f"  - {doc}")
        
        return "\n".join(lines)
    
    def get_best_solution(self) -> Optional[Solution]:
        """Get the solution with highest confidence."""
        if not self.solutions:
            return None
        return max(self.solutions, key=lambda s: s.confidence)
    
    def get_auto_fixable_solutions(self) -> List[Solution]:
        """Get all solutions that can be auto-fixed."""
        return [s for s in self.solutions if s.auto_fixable and s.fix_command]


@dataclass
class ErrorPattern:
    """Pattern definition for matching errors."""
    pattern: str
    error_code: str
    title: str
    description_template: str
    category: ErrorCategory
    severity: ErrorSeverity
    solution_templates: List[Dict[str, Any]] = field(default_factory=list)
    is_regex: bool = True
    language: Optional[str] = None
    framework: Optional[str] = None
    extract_groups: Dict[str, int] = field(default_factory=dict)
    related_docs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


ERROR_PATTERNS: List[ErrorPattern] = [
    ErrorPattern(
        pattern=r"ModuleNotFoundError: No module named '([^']+)'",
        error_code="PY_IMPORT_001",
        title="Missing Python Package",
        description_template="The Python package '{0}' is not installed in your environment.",
        category=ErrorCategory.IMPORT,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Install with pip",
                "steps": ["Run: pip install {0}"],
                "auto_fixable": True,
                "fix_command": "pip install {0}",
                "confidence": 0.95,
                "effort_level": "trivial"
            },
            {
                "title": "Install with pip3",
                "steps": ["Run: pip3 install {0}"],
                "auto_fixable": True,
                "fix_command": "pip3 install {0}",
                "confidence": 0.9,
                "effort_level": "trivial"
            },
            {
                "title": "Add to requirements.txt",
                "steps": ["Add '{0}' to requirements.txt", "Run: pip install -r requirements.txt"],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        extract_groups={"module_name": 1},
        related_docs=["https://pip.pypa.io/en/stable/", "https://pypi.org/"],
        tags=["python", "import", "dependency"]
    ),
    ErrorPattern(
        pattern=r"ImportError: cannot import name '([^']+)' from '([^']+)'",
        error_code="PY_IMPORT_002",
        title="Cannot Import Name",
        description_template="Cannot import '{0}' from the module '{1}'. This might be due to a circular import, wrong package version, or the name doesn't exist in that module.",
        category=ErrorCategory.IMPORT,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check package version",
                "steps": ["Verify you have the correct version: pip show {1}", "Upgrade if needed: pip install --upgrade {1}"],
                "auto_fixable": True,
                "fix_command": "pip install --upgrade {1}",
                "confidence": 0.7,
                "effort_level": "easy"
            },
            {
                "title": "Check for circular imports",
                "steps": ["Review your import structure", "Move imports inside functions if needed", "Use lazy imports"],
                "auto_fixable": False,
                "confidence": 0.6,
                "effort_level": "moderate"
            }
        ],
        extract_groups={"name": 1, "module": 2},
        tags=["python", "import", "circular"]
    ),
    ErrorPattern(
        pattern=r"SyntaxError: invalid syntax.*\n.*File \"([^\"]+)\", line (\d+)",
        error_code="PY_SYNTAX_001",
        title="Python Syntax Error",
        description_template="There's a syntax error in your code at file '{0}', line {1}. This is usually caused by a typo, missing bracket, or incorrect indentation.",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check for common syntax issues",
                "steps": [
                    "Look for missing colons (:) after if, for, def, class statements",
                    "Check for unmatched parentheses, brackets, or braces",
                    "Verify proper indentation (use 4 spaces)",
                    "Look for unclosed quotes in strings"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            }
        ],
        extract_groups={"file": 1, "line": 2},
        tags=["python", "syntax"]
    ),
    ErrorPattern(
        pattern=r"IndentationError: (unexpected indent|expected an indented block|unindent does not match)",
        error_code="PY_SYNTAX_002",
        title="Indentation Error",
        description_template="Python found an indentation problem: {0}. Python uses indentation to define code blocks.",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Fix indentation",
                "steps": [
                    "Use consistent indentation (4 spaces recommended)",
                    "Don't mix tabs and spaces",
                    "Configure your editor to convert tabs to spaces"
                ],
                "auto_fixable": False,
                "confidence": 0.95,
                "effort_level": "trivial"
            }
        ],
        tags=["python", "syntax", "indentation"]
    ),
    ErrorPattern(
        pattern=r"TypeError: '?(\w+)'? object is not (callable|subscriptable|iterable)",
        error_code="PY_TYPE_001",
        title="Type Error - Invalid Operation",
        description_template="You're trying to use a {0} object in a way it doesn't support (not {1}). This means you're treating a variable as the wrong type.",
        category=ErrorCategory.TYPE,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check variable type",
                "steps": [
                    "Use type() to check the actual type of your variable",
                    "Verify you haven't accidentally reassigned the variable",
                    "Check if a function is returning the expected type"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["python", "type"]
    ),
    ErrorPattern(
        pattern=r"TypeError: (\w+)\(\) (takes|got) (\d+) (positional arguments?|keyword arguments?) but (\d+) (was|were) given",
        error_code="PY_TYPE_002",
        title="Wrong Number of Arguments",
        description_template="Function {0} expected {3} arguments but received {5}. Check the function signature.",
        category=ErrorCategory.TYPE,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check function signature",
                "steps": [
                    "Review the function definition to see expected arguments",
                    "Check if you're missing required arguments",
                    "Verify you're not passing extra arguments"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            }
        ],
        tags=["python", "type", "arguments"]
    ),
    ErrorPattern(
        pattern=r"KeyError: ['\"]?([^'\"]+)['\"]?",
        error_code="PY_KEY_001",
        title="Dictionary Key Not Found",
        description_template="The key '{0}' doesn't exist in the dictionary. The dictionary might be empty or the key might be misspelled.",
        category=ErrorCategory.VALUE,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Use .get() method",
                "steps": [
                    "Instead of dict[key], use dict.get(key, default_value)",
                    "This returns None (or your default) if key doesn't exist"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "trivial"
            },
            {
                "title": "Check key exists first",
                "steps": [
                    "Use 'if key in dict:' before accessing",
                    "Print dict.keys() to see available keys"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["python", "dictionary", "key"]
    ),
    ErrorPattern(
        pattern=r"IndexError: (list|tuple|string) index out of range",
        error_code="PY_INDEX_001",
        title="Index Out of Range",
        description_template="You're trying to access an index that doesn't exist in the {0}. The {0} has fewer elements than the index you're trying to access.",
        category=ErrorCategory.VALUE,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check length before accessing",
                "steps": [
                    "Use len() to check the length first",
                    "Ensure your loop doesn't go beyond the length",
                    "Use try/except for unknown length data"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            }
        ],
        tags=["python", "index", "list"]
    ),
    ErrorPattern(
        pattern=r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
        error_code="PY_ATTR_001",
        title="Attribute Not Found",
        description_template="The {0} object doesn't have an attribute called '{1}'. This could be a typo or you might be using the wrong variable type.",
        category=ErrorCategory.TYPE,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check attribute spelling",
                "steps": [
                    "Use dir(object) to see available attributes",
                    "Check for typos in the attribute name",
                    "Verify the object is the type you expect"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["python", "attribute"]
    ),
    ErrorPattern(
        pattern=r"ValueError: (could not convert|invalid literal for) .* ('.*'|\".*\")",
        error_code="PY_VALUE_001",
        title="Value Conversion Error",
        description_template="Cannot convert the value {2} to the expected type. The input format is incorrect.",
        category=ErrorCategory.VALUE,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Validate input before conversion",
                "steps": [
                    "Check if the input is in the expected format",
                    "Use try/except around type conversions",
                    "Clean/sanitize input data first"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "easy"
            }
        ],
        tags=["python", "value", "conversion"]
    ),
    ErrorPattern(
        pattern=r"FileNotFoundError: \[Errno 2\] No such file or directory: ['\"]?([^'\"]+)['\"]?",
        error_code="FILE_001",
        title="File Not Found",
        description_template="The file '{0}' doesn't exist or the path is incorrect. Check if the file exists and the path is correct.",
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check file path",
                "steps": [
                    "Verify the file exists at the specified location",
                    "Use absolute paths instead of relative paths",
                    "Check for typos in the filename"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            },
            {
                "title": "Create the file if needed",
                "steps": [
                    "Create the file manually if it should exist",
                    "Check if the parent directory exists",
                    "Use os.makedirs() for directory creation"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["file", "path"]
    ),
    ErrorPattern(
        pattern=r"PermissionError: \[Errno 13\] Permission denied: ['\"]?([^'\"]+)['\"]?",
        error_code="PERM_001",
        title="Permission Denied",
        description_template="You don't have permission to access '{0}'. This file or directory requires different permissions.",
        category=ErrorCategory.PERMISSION,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Change file permissions (Unix)",
                "steps": [
                    "Run: chmod 644 {0} (for files)",
                    "Run: chmod 755 {0} (for directories/executables)"
                ],
                "auto_fixable": True,
                "fix_command": "chmod 644 {0}",
                "confidence": 0.8,
                "effort_level": "trivial"
            },
            {
                "title": "Change file ownership (Unix)",
                "steps": [
                    "Run: chown $USER {0}",
                    "Or run with sudo if system file"
                ],
                "auto_fixable": True,
                "fix_command": "chown $USER {0}",
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["permission", "file", "unix"]
    ),
    ErrorPattern(
        pattern=r"ConnectionRefusedError: \[Errno 111\] Connection refused",
        error_code="NET_001",
        title="Connection Refused",
        description_template="Cannot connect to the server. The server might not be running or is refusing connections on that port.",
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Start the server",
                "steps": [
                    "Check if the server is running: ps aux | grep [server_name]",
                    "Start the server if not running",
                    "Verify the correct port is being used"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            },
            {
                "title": "Check firewall settings",
                "steps": [
                    "Verify the port is not blocked by firewall",
                    "Check if the server is bound to the correct interface"
                ],
                "auto_fixable": False,
                "confidence": 0.6,
                "effort_level": "moderate"
            }
        ],
        tags=["network", "connection", "server"]
    ),
    ErrorPattern(
        pattern=r"TimeoutError|socket\.timeout|requests\.exceptions\.Timeout",
        error_code="NET_002",
        title="Connection Timeout",
        description_template="The operation took too long and timed out. The server might be slow or network issues exist.",
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.WARNING,
        solution_templates=[
            {
                "title": "Increase timeout",
                "steps": [
                    "Increase the timeout value in your request",
                    "Add timeout parameter: requests.get(url, timeout=30)"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "trivial"
            },
            {
                "title": "Add retry logic",
                "steps": [
                    "Implement exponential backoff",
                    "Use a retry library like tenacity or retrying"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "moderate"
            }
        ],
        tags=["network", "timeout"]
    ),
    ErrorPattern(
        pattern=r"MemoryError|cannot allocate memory",
        error_code="MEM_001",
        title="Out of Memory",
        description_template="The system ran out of memory. You're trying to load more data than available RAM.",
        category=ErrorCategory.MEMORY,
        severity=ErrorSeverity.CRITICAL,
        solution_templates=[
            {
                "title": "Process data in chunks",
                "steps": [
                    "Use generators instead of loading all data at once",
                    "Process data in smaller batches",
                    "Use pandas chunksize parameter for large files"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "moderate"
            },
            {
                "title": "Increase system memory",
                "steps": [
                    "Close other memory-intensive applications",
                    "Consider increasing swap space",
                    "Upgrade to a machine with more RAM"
                ],
                "auto_fixable": False,
                "confidence": 0.6,
                "effort_level": "difficult"
            }
        ],
        tags=["memory", "performance"]
    ),
    ErrorPattern(
        pattern=r"RecursionError: maximum recursion depth exceeded",
        error_code="PY_REC_001",
        title="Maximum Recursion Depth Exceeded",
        description_template="A recursive function called itself too many times without reaching a base case, causing a stack overflow.",
        category=ErrorCategory.RUNTIME,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check base case",
                "steps": [
                    "Ensure your recursive function has a valid base case",
                    "Verify the base case is reachable with your input",
                    "Add print statements to debug recursion"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "moderate"
            },
            {
                "title": "Convert to iterative",
                "steps": [
                    "Consider converting recursion to iteration",
                    "Use an explicit stack data structure",
                    "Or increase recursion limit with sys.setrecursionlimit()"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "moderate"
            }
        ],
        tags=["python", "recursion", "stack"]
    ),
    ErrorPattern(
        pattern=r"psycopg2\.(OperationalError|ProgrammingError|IntegrityError): (.+)",
        error_code="DB_PG_001",
        title="PostgreSQL Database Error",
        description_template="PostgreSQL error: {1}",
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.ERROR,
        language="python",
        framework="postgresql",
        solution_templates=[
            {
                "title": "Check database connection",
                "steps": [
                    "Verify DATABASE_URL environment variable is set",
                    "Check if PostgreSQL server is running",
                    "Verify credentials and database name"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "easy"
            }
        ],
        tags=["database", "postgresql"]
    ),
    ErrorPattern(
        pattern=r"sqlite3\.(OperationalError|IntegrityError): (.+)",
        error_code="DB_SQLITE_001",
        title="SQLite Database Error",
        description_template="SQLite error: {1}",
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.ERROR,
        language="python",
        framework="sqlite",
        solution_templates=[
            {
                "title": "Check database file",
                "steps": [
                    "Verify the database file exists and is accessible",
                    "Check file permissions on the database",
                    "Ensure no other process has locked the database"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "easy"
            }
        ],
        tags=["database", "sqlite"]
    ),
    ErrorPattern(
        pattern=r"django\.db\.(OperationalError|IntegrityError|ProgrammingError): (.+)",
        error_code="DJANGO_DB_001",
        title="Django Database Error",
        description_template="Django database error: {1}",
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.ERROR,
        language="python",
        framework="django",
        solution_templates=[
            {
                "title": "Run migrations",
                "steps": [
                    "Run: python manage.py makemigrations",
                    "Run: python manage.py migrate"
                ],
                "auto_fixable": True,
                "fix_command": "python manage.py migrate",
                "confidence": 0.75,
                "effort_level": "easy"
            },
            {
                "title": "Check database settings",
                "steps": [
                    "Verify DATABASES setting in settings.py",
                    "Check database server is running"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["django", "database", "migrations"]
    ),
    ErrorPattern(
        pattern=r"npm ERR! code E404.*package '([^']+)'",
        error_code="NPM_001",
        title="NPM Package Not Found",
        description_template="The npm package '{0}' was not found. It may not exist or is spelled incorrectly.",
        category=ErrorCategory.DEPENDENCY,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        solution_templates=[
            {
                "title": "Check package name",
                "steps": [
                    "Search for the correct package name at https://www.npmjs.com/",
                    "Check for typos in the package name",
                    "Verify the package hasn't been deprecated or renamed"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            }
        ],
        tags=["npm", "javascript", "dependency"]
    ),
    ErrorPattern(
        pattern=r"npm ERR! peer dep missing: ([^,]+)",
        error_code="NPM_002",
        title="Missing Peer Dependency",
        description_template="Missing peer dependency: {0}. Some packages require other packages to be installed alongside them.",
        category=ErrorCategory.DEPENDENCY,
        severity=ErrorSeverity.WARNING,
        language="javascript",
        solution_templates=[
            {
                "title": "Install peer dependency",
                "steps": ["Run: npm install {0}"],
                "auto_fixable": True,
                "fix_command": "npm install {0}",
                "confidence": 0.9,
                "effort_level": "trivial"
            }
        ],
        tags=["npm", "javascript", "peer-dependency"]
    ),
    ErrorPattern(
        pattern=r"Cannot find module '([^']+)'",
        error_code="NODE_001",
        title="Node.js Module Not Found",
        description_template="Cannot find the module '{0}'. It might not be installed or the path is incorrect.",
        category=ErrorCategory.IMPORT,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        solution_templates=[
            {
                "title": "Install with npm",
                "steps": ["Run: npm install {0}"],
                "auto_fixable": True,
                "fix_command": "npm install {0}",
                "confidence": 0.9,
                "effort_level": "trivial"
            },
            {
                "title": "Install with yarn",
                "steps": ["Run: yarn add {0}"],
                "auto_fixable": True,
                "fix_command": "yarn add {0}",
                "confidence": 0.85,
                "effort_level": "trivial"
            },
            {
                "title": "Check relative path",
                "steps": [
                    "If it's a local module, verify the path is correct",
                    "Use './' for relative imports",
                    "Check file exists at the specified location"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["nodejs", "javascript", "module"]
    ),
    ErrorPattern(
        pattern=r"SyntaxError: Unexpected token ([^\s]+)",
        error_code="JS_SYNTAX_001",
        title="JavaScript Syntax Error",
        description_template="Unexpected token '{0}' in JavaScript. This is usually caused by a missing bracket, comma, or semicolon.",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        solution_templates=[
            {
                "title": "Check for common syntax issues",
                "steps": [
                    "Look for missing commas in object/array literals",
                    "Check for unclosed brackets, braces, or parentheses",
                    "Verify all strings are properly closed"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["javascript", "syntax"]
    ),
    ErrorPattern(
        pattern=r"TypeError: Cannot read propert(y|ies) ('?\w+'?) of (undefined|null)",
        error_code="JS_TYPE_001",
        title="Cannot Read Property of Undefined/Null",
        description_template="Trying to access property {1} on {2}. The object is undefined or null when you're trying to use it.",
        category=ErrorCategory.TYPE,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        solution_templates=[
            {
                "title": "Add null check",
                "steps": [
                    "Use optional chaining: object?.property",
                    "Check if object exists before accessing: if (object) {{ ... }}",
                    "Use nullish coalescing: object?.property ?? defaultValue"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            }
        ],
        tags=["javascript", "type", "null"]
    ),
    ErrorPattern(
        pattern=r"ReferenceError: (\w+) is not defined",
        error_code="JS_REF_001",
        title="Variable Not Defined",
        description_template="The variable '{0}' is not defined. It might be misspelled or not in scope.",
        category=ErrorCategory.VALUE,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        solution_templates=[
            {
                "title": "Check variable name",
                "steps": [
                    "Verify the variable is spelled correctly",
                    "Ensure the variable is declared with let, const, or var",
                    "Check if the variable is in the correct scope"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["javascript", "variable", "scope"]
    ),
    ErrorPattern(
        pattern=r"Error: listen EADDRINUSE.*:(\d+)",
        error_code="NET_003",
        title="Port Already in Use",
        description_template="Port {0} is already being used by another application. You need to either stop that application or use a different port.",
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Find and kill process using port",
                "steps": [
                    "Find process: lsof -i :{0}",
                    "Kill process: kill -9 <PID>"
                ],
                "auto_fixable": True,
                "fix_command": "kill -9 $(lsof -t -i:{0})",
                "confidence": 0.85,
                "effort_level": "trivial"
            },
            {
                "title": "Use a different port",
                "steps": [
                    "Change the port in your application configuration",
                    "Common alternatives: 3001, 8080, 8000"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "easy"
            }
        ],
        tags=["network", "port", "server"]
    ),
    ErrorPattern(
        pattern=r"Error: ENOENT: no such file or directory, open '([^']+)'",
        error_code="NODE_FILE_001",
        title="File Not Found (Node.js)",
        description_template="Cannot open file '{0}'. The file doesn't exist or the path is incorrect.",
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        solution_templates=[
            {
                "title": "Check file path",
                "steps": [
                    "Verify the file exists at the specified path",
                    "Use path.join() for cross-platform paths",
                    "Check if you're using the correct working directory"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            }
        ],
        tags=["nodejs", "file", "path"]
    ),
    ErrorPattern(
        pattern=r"react-dom\..*Error: Objects are not valid as a React child",
        error_code="REACT_001",
        title="Invalid React Child",
        description_template="You're trying to render an object directly in React. Objects must be converted to strings or rendered as components.",
        category=ErrorCategory.FRAMEWORK,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        framework="react",
        solution_templates=[
            {
                "title": "Convert object to string",
                "steps": [
                    "Use JSON.stringify(object) to convert to string",
                    "Or render specific properties: {object.property}",
                    "Map arrays: {array.map(item => <Component key={item.id} />)}"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            }
        ],
        tags=["react", "render", "object"]
    ),
    ErrorPattern(
        pattern=r"Warning: Each child in a list should have a unique \"key\" prop",
        error_code="REACT_002",
        title="Missing React Key Prop",
        description_template="When rendering lists in React, each item needs a unique 'key' prop to help React identify which items have changed.",
        category=ErrorCategory.FRAMEWORK,
        severity=ErrorSeverity.WARNING,
        language="javascript",
        framework="react",
        solution_templates=[
            {
                "title": "Add key prop to list items",
                "steps": [
                    "Add a unique key prop: <Item key={item.id} />",
                    "Use a unique identifier (not array index if list changes)",
                    "If no ID exists, create one based on item content"
                ],
                "auto_fixable": False,
                "confidence": 0.95,
                "effort_level": "trivial"
            }
        ],
        tags=["react", "list", "key"]
    ),
    ErrorPattern(
        pattern=r"Error: Invalid hook call\. Hooks can only be called inside",
        error_code="REACT_003",
        title="Invalid React Hook Call",
        description_template="React hooks must be called inside a function component or another hook, not in regular functions or class components.",
        category=ErrorCategory.FRAMEWORK,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        framework="react",
        solution_templates=[
            {
                "title": "Move hook inside component",
                "steps": [
                    "Ensure hooks are called at the top level of function components",
                    "Don't call hooks inside loops, conditions, or nested functions",
                    "Don't call hooks in class components"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "moderate"
            }
        ],
        related_docs=["https://reactjs.org/docs/hooks-rules.html"],
        tags=["react", "hooks"]
    ),
    ErrorPattern(
        pattern=r"CORS.*(?:blocked|policy).*(?:origin|'([^']+)')",
        error_code="CORS_001",
        title="CORS Policy Blocked",
        description_template="Cross-Origin Request blocked. The server doesn't allow requests from your application's origin.",
        category=ErrorCategory.SECURITY,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Enable CORS on server",
                "steps": [
                    "Add CORS headers to your server response",
                    "For Express: app.use(cors())",
                    "For Flask: from flask_cors import CORS; CORS(app)"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            },
            {
                "title": "Use a proxy during development",
                "steps": [
                    "Configure a proxy in your dev server",
                    "In package.json: \"proxy\": \"http://localhost:5000\"",
                    "Or use a CORS browser extension for testing"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["cors", "security", "api"]
    ),
    ErrorPattern(
        pattern=r"401 Unauthorized|Authentication required|Invalid credentials",
        error_code="AUTH_001",
        title="Authentication Failed",
        description_template="Authentication failed. Your credentials are invalid, expired, or missing.",
        category=ErrorCategory.AUTHENTICATION,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check credentials",
                "steps": [
                    "Verify API key or token is correct",
                    "Check if credentials have expired",
                    "Ensure credentials are included in request headers"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            },
            {
                "title": "Refresh authentication",
                "steps": [
                    "Log out and log back in",
                    "Refresh your access token",
                    "Generate new API credentials if expired"
                ],
                "auto_fixable": False,
                "confidence": 0.75,
                "effort_level": "easy"
            }
        ],
        tags=["authentication", "api", "credentials"]
    ),
    ErrorPattern(
        pattern=r"403 Forbidden|Access denied|Permission denied",
        error_code="AUTH_002",
        title="Access Forbidden",
        description_template="Access forbidden. You don't have permission to access this resource.",
        category=ErrorCategory.PERMISSION,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check permissions",
                "steps": [
                    "Verify your account has the required permissions",
                    "Contact the administrator to grant access",
                    "Check if you're using the correct API scope"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "moderate"
            }
        ],
        tags=["permission", "access", "authorization"]
    ),
    ErrorPattern(
        pattern=r"API rate limit exceeded|Too many requests|429",
        error_code="API_001",
        title="Rate Limit Exceeded",
        description_template="You've made too many API requests. Wait before making more requests.",
        category=ErrorCategory.API,
        severity=ErrorSeverity.WARNING,
        solution_templates=[
            {
                "title": "Implement rate limiting",
                "steps": [
                    "Add delays between requests",
                    "Implement exponential backoff",
                    "Cache responses to reduce API calls"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "moderate"
            },
            {
                "title": "Upgrade API plan",
                "steps": [
                    "Check if higher rate limits are available",
                    "Contact API provider for enterprise options"
                ],
                "auto_fixable": False,
                "confidence": 0.6,
                "effort_level": "easy"
            }
        ],
        tags=["api", "rate-limit"]
    ),
    ErrorPattern(
        pattern=r"error: failed to push some refs|rejected\].*\(fetch first\)",
        error_code="GIT_001",
        title="Git Push Rejected",
        description_template="Git push was rejected because the remote has changes you don't have locally. You need to pull first.",
        category=ErrorCategory.RUNTIME,
        severity=ErrorSeverity.WARNING,
        solution_templates=[
            {
                "title": "Pull and merge",
                "steps": [
                    "Run: git pull origin <branch>",
                    "Resolve any merge conflicts",
                    "Then push again"
                ],
                "auto_fixable": True,
                "fix_command": "git pull --rebase && git push",
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["git", "push", "pull"]
    ),
    ErrorPattern(
        pattern=r"MERGE CONFLICT|Automatic merge failed|fix conflicts",
        error_code="GIT_002",
        title="Git Merge Conflict",
        description_template="There are conflicting changes that Git cannot automatically merge. You need to resolve them manually.",
        category=ErrorCategory.RUNTIME,
        severity=ErrorSeverity.WARNING,
        solution_templates=[
            {
                "title": "Resolve conflicts manually",
                "steps": [
                    "Open conflicting files (marked with <<<<<<< and >>>>>>>)",
                    "Choose which changes to keep",
                    "Remove conflict markers",
                    "Stage and commit: git add . && git commit"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "moderate"
            }
        ],
        tags=["git", "merge", "conflict"]
    ),
    ErrorPattern(
        pattern=r"go: module .* not found|cannot find module|cannot find package",
        error_code="GO_001",
        title="Go Module Not Found",
        description_template="Go cannot find the required module. It may not be installed or the import path is incorrect.",
        category=ErrorCategory.IMPORT,
        severity=ErrorSeverity.ERROR,
        language="go",
        solution_templates=[
            {
                "title": "Install Go module",
                "steps": [
                    "Run: go mod tidy",
                    "Or: go get <module-path>"
                ],
                "auto_fixable": True,
                "fix_command": "go mod tidy",
                "confidence": 0.9,
                "effort_level": "trivial"
            },
            {
                "title": "Initialize Go modules",
                "steps": [
                    "If go.mod doesn't exist, run: go mod init <module-name>"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["go", "module", "import"]
    ),
    ErrorPattern(
        pattern=r"undefined: (\w+)|undeclared name: (\w+)",
        error_code="GO_002",
        title="Undefined Identifier (Go)",
        description_template="The identifier '{0}' is not defined. It might be misspelled or not imported.",
        category=ErrorCategory.VALUE,
        severity=ErrorSeverity.ERROR,
        language="go",
        solution_templates=[
            {
                "title": "Check import and spelling",
                "steps": [
                    "Verify the identifier is spelled correctly",
                    "Check if the package is imported",
                    "Run: goimports to auto-fix imports"
                ],
                "auto_fixable": True,
                "fix_command": "goimports -w .",
                "confidence": 0.8,
                "effort_level": "easy"
            }
        ],
        tags=["go", "undefined", "import"]
    ),
    ErrorPattern(
        pattern=r"rust-analyzer|error\[E\d+\]: (.+)",
        error_code="RUST_001",
        title="Rust Compiler Error",
        description_template="Rust compilation error: {0}",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.ERROR,
        language="rust",
        solution_templates=[
            {
                "title": "Check Rust documentation",
                "steps": [
                    "Run: rustc --explain E<error_code>",
                    "Check the Rust documentation for the specific error"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "moderate"
            },
            {
                "title": "Run cargo fix",
                "steps": [
                    "Run: cargo fix --allow-dirty",
                    "This may auto-fix some common issues"
                ],
                "auto_fixable": True,
                "fix_command": "cargo fix --allow-dirty",
                "confidence": 0.6,
                "effort_level": "easy"
            }
        ],
        tags=["rust", "compiler"]
    ),
    ErrorPattern(
        pattern=r"cargo.*error: could not find `([^`]+)` in",
        error_code="RUST_002",
        title="Rust Crate Not Found",
        description_template="Cannot find the Rust crate '{0}'. It might not be added to Cargo.toml or installed.",
        category=ErrorCategory.IMPORT,
        severity=ErrorSeverity.ERROR,
        language="rust",
        solution_templates=[
            {
                "title": "Add to Cargo.toml",
                "steps": [
                    "Add '{0}' to [dependencies] in Cargo.toml",
                    "Run: cargo build"
                ],
                "auto_fixable": True,
                "fix_command": "cargo add {0}",
                "confidence": 0.9,
                "effort_level": "trivial"
            }
        ],
        tags=["rust", "cargo", "dependency"]
    ),
    ErrorPattern(
        pattern=r"docker:.*Error response from daemon: (.+)",
        error_code="DOCKER_001",
        title="Docker Error",
        description_template="Docker error: {0}",
        category=ErrorCategory.ENVIRONMENT,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Check Docker service",
                "steps": [
                    "Verify Docker is running: docker info",
                    "Start Docker: sudo systemctl start docker",
                    "Add user to docker group: sudo usermod -aG docker $USER"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["docker", "container"]
    ),
    ErrorPattern(
        pattern=r"docker.*no space left on device",
        error_code="DOCKER_002",
        title="Docker - No Disk Space",
        description_template="Docker has run out of disk space. Old images and containers may be using up space.",
        category=ErrorCategory.ENVIRONMENT,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Clean up Docker",
                "steps": [
                    "Remove unused containers: docker container prune",
                    "Remove unused images: docker image prune -a",
                    "Remove all unused data: docker system prune -a"
                ],
                "auto_fixable": True,
                "fix_command": "docker system prune -af",
                "confidence": 0.9,
                "effort_level": "trivial"
            }
        ],
        tags=["docker", "disk", "cleanup"]
    ),
    ErrorPattern(
        pattern=r"SSL.*certificate.*expired|certificate has expired|CERTIFICATE_VERIFY_FAILED",
        error_code="SSL_001",
        title="SSL Certificate Issue",
        description_template="There's a problem with the SSL certificate. It may be expired, self-signed, or invalid.",
        category=ErrorCategory.SECURITY,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Update certificates",
                "steps": [
                    "Update system certificates: apt-get update && apt-get install ca-certificates",
                    "Or for Python: pip install --upgrade certifi"
                ],
                "auto_fixable": True,
                "fix_command": "pip install --upgrade certifi",
                "confidence": 0.7,
                "effort_level": "easy"
            },
            {
                "title": "Renew SSL certificate",
                "steps": [
                    "If you own the certificate, renew it",
                    "For Let's Encrypt: certbot renew"
                ],
                "auto_fixable": False,
                "confidence": 0.6,
                "effort_level": "moderate"
            }
        ],
        tags=["ssl", "certificate", "https"]
    ),
    ErrorPattern(
        pattern=r"OOM|Killed|out of memory|cannot allocate|memory allocation failed",
        error_code="MEM_002",
        title="Process Killed - Out of Memory",
        description_template="The process was killed because it used too much memory. The system ran out of available RAM.",
        category=ErrorCategory.MEMORY,
        severity=ErrorSeverity.CRITICAL,
        solution_templates=[
            {
                "title": "Reduce memory usage",
                "steps": [
                    "Process data in smaller batches",
                    "Use generators for large datasets",
                    "Clear unused variables with del",
                    "Use memory-efficient data structures"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "moderate"
            },
            {
                "title": "Monitor memory usage",
                "steps": [
                    "Use memory_profiler to identify memory leaks",
                    "Monitor with: watch -n 1 free -m"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "moderate"
            }
        ],
        tags=["memory", "oom", "performance"]
    ),
    ErrorPattern(
        pattern=r"Segmentation fault|SIGSEGV|core dumped",
        error_code="MEM_003",
        title="Segmentation Fault",
        description_template="The program tried to access memory it shouldn't. This is usually caused by a bug in native code or a library.",
        category=ErrorCategory.MEMORY,
        severity=ErrorSeverity.CRITICAL,
        solution_templates=[
            {
                "title": "Debug the issue",
                "steps": [
                    "Run with valgrind: valgrind ./program",
                    "Check for null pointer dereferences",
                    "Verify array bounds are correct"
                ],
                "auto_fixable": False,
                "confidence": 0.6,
                "effort_level": "difficult"
            },
            {
                "title": "Update libraries",
                "steps": [
                    "Update to the latest version of involved libraries",
                    "The bug may already be fixed"
                ],
                "auto_fixable": False,
                "confidence": 0.5,
                "effort_level": "easy"
            }
        ],
        tags=["segfault", "memory", "crash"]
    ),
    ErrorPattern(
        pattern=r"EACCES.*permission denied.*npm|npm ERR! code EACCES",
        error_code="NPM_003",
        title="NPM Permission Error",
        description_template="NPM doesn't have permission to install packages globally. This is a common issue with npm configuration.",
        category=ErrorCategory.PERMISSION,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        solution_templates=[
            {
                "title": "Fix npm permissions",
                "steps": [
                    "Create a directory for global packages: mkdir ~/.npm-global",
                    "Configure npm: npm config set prefix '~/.npm-global'",
                    "Add to PATH: export PATH=~/.npm-global/bin:$PATH"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            },
            {
                "title": "Use npx or local install",
                "steps": [
                    "Use npx instead of global install: npx <package>",
                    "Or install locally: npm install <package>"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "trivial"
            }
        ],
        tags=["npm", "permission", "global"]
    ),
    ErrorPattern(
        pattern=r"error: externally-managed-environment|This environment is externally managed",
        error_code="PY_ENV_001",
        title="Externally Managed Python Environment",
        description_template="This Python environment is managed by the system and doesn't allow direct pip installs. Use a virtual environment.",
        category=ErrorCategory.ENVIRONMENT,
        severity=ErrorSeverity.ERROR,
        language="python",
        solution_templates=[
            {
                "title": "Create a virtual environment",
                "steps": [
                    "Create venv: python -m venv .venv",
                    "Activate: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)",
                    "Then install packages: pip install <package>"
                ],
                "auto_fixable": True,
                "fix_command": "python -m venv .venv && source .venv/bin/activate",
                "confidence": 0.95,
                "effort_level": "easy"
            },
            {
                "title": "Use pipx for CLI tools",
                "steps": [
                    "Install pipx: pip install --user pipx",
                    "Install CLI tools: pipx install <package>"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["python", "venv", "pip"]
    ),
    ErrorPattern(
        pattern=r"OSError: \[Errno 28\] No space left on device",
        error_code="DISK_001",
        title="Disk Full",
        description_template="The disk is full. There's no space left to write files.",
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.CRITICAL,
        solution_templates=[
            {
                "title": "Free up disk space",
                "steps": [
                    "Check disk usage: df -h",
                    "Find large files: du -sh /* | sort -rh | head",
                    "Clean package caches: apt clean, npm cache clean --force, pip cache purge"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            },
            {
                "title": "Remove old files",
                "steps": [
                    "Remove old logs: rm /var/log/*.old",
                    "Clean temp files: rm -rf /tmp/*",
                    "Remove unused packages: apt autoremove"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["disk", "space", "storage"]
    ),
    ErrorPattern(
        pattern=r"JSONDecodeError|json\.decoder\.JSONDecodeError|Invalid JSON|Unexpected token.*JSON",
        error_code="JSON_001",
        title="Invalid JSON",
        description_template="The JSON data is invalid or malformed. There might be a syntax error in the JSON.",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Validate JSON",
                "steps": [
                    "Use a JSON validator to find the error",
                    "Check for trailing commas (not allowed in JSON)",
                    "Ensure all strings use double quotes",
                    "Verify no comments are in the JSON (not allowed)"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            }
        ],
        tags=["json", "parse", "syntax"]
    ),
    ErrorPattern(
        pattern=r"yaml\..*Error|YAMLError|could not find expected|mapping values are not allowed",
        error_code="YAML_001",
        title="Invalid YAML",
        description_template="The YAML file has a syntax error. YAML is sensitive to indentation and special characters.",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Fix YAML syntax",
                "steps": [
                    "Use consistent indentation (2 spaces recommended)",
                    "Quote strings with special characters",
                    "Use a YAML validator/linter"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["yaml", "syntax", "config"]
    ),
    ErrorPattern(
        pattern=r"toml.*Error|invalid value|expected.*got|invalid escape sequence",
        error_code="TOML_001",
        title="Invalid TOML",
        description_template="The TOML file has a syntax error. Check the format of your configuration file.",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Fix TOML syntax",
                "steps": [
                    "Use proper string quoting (double quotes)",
                    "Check date/time formats",
                    "Validate with a TOML parser"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["toml", "syntax", "config"]
    ),
    ErrorPattern(
        pattern=r"error TS\d+: (.+)",
        error_code="TS_001",
        title="TypeScript Error",
        description_template="TypeScript compilation error: {0}",
        category=ErrorCategory.TYPE,
        severity=ErrorSeverity.ERROR,
        language="typescript",
        solution_templates=[
            {
                "title": "Fix type error",
                "steps": [
                    "Check the type annotations match the values",
                    "Add proper type definitions",
                    "Use 'as' for type assertions when appropriate"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "moderate"
            }
        ],
        tags=["typescript", "type", "compile"]
    ),
    ErrorPattern(
        pattern=r"env.*not found|environment variable.*not set|missing.*env|undefined.*process\.env",
        error_code="ENV_001",
        title="Missing Environment Variable",
        description_template="A required environment variable is not set. Your application needs this configuration.",
        category=ErrorCategory.CONFIGURATION,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Set environment variable",
                "steps": [
                    "Create a .env file in your project root",
                    "Add the required variable: KEY=value",
                    "Load with dotenv or similar"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "trivial"
            },
            {
                "title": "Export in shell",
                "steps": [
                    "Export: export VAR_NAME=value",
                    "Or add to ~/.bashrc or ~/.zshrc"
                ],
                "auto_fixable": False,
                "confidence": 0.75,
                "effort_level": "trivial"
            }
        ],
        tags=["environment", "config", "dotenv"]
    ),
    ErrorPattern(
        pattern=r"UnicodeDecodeError|UnicodeEncodeError|codec can't (decode|encode)",
        error_code="ENCODING_001",
        title="Unicode/Encoding Error",
        description_template="There's an encoding problem with your text data. The file might use a different encoding than expected.",
        category=ErrorCategory.VALUE,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Specify encoding",
                "steps": [
                    "Open files with explicit encoding: open(file, encoding='utf-8')",
                    "Try other encodings: latin-1, cp1252",
                    "Use chardet to detect encoding"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "easy"
            }
        ],
        tags=["unicode", "encoding", "utf8"]
    ),
    ErrorPattern(
        pattern=r"deadlock|DeadlockDetected|concurrent modification",
        error_code="CONCURRENT_001",
        title="Deadlock Detected",
        description_template="A deadlock was detected. Multiple processes are waiting for each other, causing the program to hang.",
        category=ErrorCategory.RUNTIME,
        severity=ErrorSeverity.CRITICAL,
        solution_templates=[
            {
                "title": "Review locking order",
                "steps": [
                    "Ensure locks are always acquired in the same order",
                    "Use timeouts on lock acquisition",
                    "Consider using lock-free data structures"
                ],
                "auto_fixable": False,
                "confidence": 0.6,
                "effort_level": "difficult"
            }
        ],
        tags=["concurrency", "deadlock", "threading"]
    ),
    ErrorPattern(
        pattern=r"PoolError|pool.*exhausted|connection pool|too many connections",
        error_code="DB_POOL_001",
        title="Connection Pool Exhausted",
        description_template="All database connections are in use. The connection pool is exhausted.",
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Increase pool size",
                "steps": [
                    "Increase max_connections in pool configuration",
                    "Ensure connections are properly returned to pool",
                    "Use connection timeouts"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "moderate"
            },
            {
                "title": "Check for connection leaks",
                "steps": [
                    "Ensure all connections are closed after use",
                    "Use context managers (with statements)",
                    "Monitor active connections"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "moderate"
            }
        ],
        tags=["database", "pool", "connections"]
    ),
    ErrorPattern(
        pattern=r"FATAL ERROR:.*JavaScript heap out of memory",
        error_code="NODE_MEM_001",
        title="Node.js Out of Memory",
        description_template="Node.js ran out of memory. The JavaScript heap is exhausted.",
        category=ErrorCategory.MEMORY,
        severity=ErrorSeverity.CRITICAL,
        language="javascript",
        solution_templates=[
            {
                "title": "Increase Node.js memory limit",
                "steps": [
                    "Run: export NODE_OPTIONS=--max-old-space-size=4096",
                    "Or run: node --max-old-space-size=4096 app.js"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["nodejs", "memory", "heap"]
    ),
    ErrorPattern(
        pattern=r"Warning: Each child in a list should have a unique \"key\" prop",
        error_code="REACT_KEY_001",
        title="React Missing Key Prop",
        description_template="React list items need unique key props for proper rendering.",
        category=ErrorCategory.FRAMEWORK,
        severity=ErrorSeverity.WARNING,
        language="javascript",
        framework="react",
        solution_templates=[
            {
                "title": "Add unique key prop",
                "steps": [
                    "Add key={item.id} to each list item",
                    "Avoid using array index as key if list can reorder"
                ],
                "auto_fixable": False,
                "confidence": 0.95,
                "effort_level": "trivial"
            }
        ],
        tags=["react", "key", "list"]
    ),
    ErrorPattern(
        pattern=r"Vue warn.*Property.*was accessed during render but is not defined",
        error_code="VUE_001",
        title="Vue Undefined Property",
        description_template="Vue component is accessing a property that hasn't been defined.",
        category=ErrorCategory.FRAMEWORK,
        severity=ErrorSeverity.ERROR,
        language="javascript",
        framework="vue",
        solution_templates=[
            {
                "title": "Define the property",
                "steps": [
                    "Add the property to data() or computed",
                    "Check spelling of property name",
                    "Ensure property is declared before use"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["vue", "property", "undefined"]
    ),
    ErrorPattern(
        pattern=r"Angular.*ExpressionChangedAfterItHasBeenCheckedError",
        error_code="ANGULAR_001",
        title="Angular Expression Changed Error",
        description_template="Angular detected a value change after change detection finished.",
        category=ErrorCategory.FRAMEWORK,
        severity=ErrorSeverity.ERROR,
        language="typescript",
        framework="angular",
        solution_templates=[
            {
                "title": "Fix change detection",
                "steps": [
                    "Use setTimeout() or ngAfterViewInit()",
                    "Trigger change detection with ChangeDetectorRef",
                    "Review component lifecycle hooks"
                ],
                "auto_fixable": False,
                "confidence": 0.7,
                "effort_level": "moderate"
            }
        ],
        tags=["angular", "change-detection"]
    ),
    ErrorPattern(
        pattern=r"django\.core\.exceptions\.ImproperlyConfigured: (.+)",
        error_code="DJANGO_CONFIG_001",
        title="Django Configuration Error",
        description_template="Django is not properly configured: {0}",
        category=ErrorCategory.CONFIGURATION,
        severity=ErrorSeverity.ERROR,
        language="python",
        framework="django",
        solution_templates=[
            {
                "title": "Check Django settings",
                "steps": [
                    "Verify DJANGO_SETTINGS_MODULE is set",
                    "Check settings.py for missing configurations",
                    "Run: python manage.py check"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "moderate"
            }
        ],
        tags=["django", "configuration"]
    ),
    ErrorPattern(
        pattern=r"Flask.*could not locate.*application",
        error_code="FLASK_001",
        title="Flask Application Not Found",
        description_template="Flask cannot locate your application.",
        category=ErrorCategory.CONFIGURATION,
        severity=ErrorSeverity.ERROR,
        language="python",
        framework="flask",
        solution_templates=[
            {
                "title": "Set Flask environment",
                "steps": [
                    "Set FLASK_APP environment variable: export FLASK_APP=app.py",
                    "Ensure your app file exists",
                    "Check that the app object is correctly named"
                ],
                "auto_fixable": False,
                "confidence": 0.9,
                "effort_level": "easy"
            }
        ],
        tags=["flask", "configuration"]
    ),
    ErrorPattern(
        pattern=r"OSError.*\[WinError 10048\].*address already in use",
        error_code="WIN_PORT_001",
        title="Windows Port Already in Use",
        description_template="A Windows port is already in use by another application.",
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.ERROR,
        solution_templates=[
            {
                "title": "Find and kill process (Windows)",
                "steps": [
                    "Run: netstat -ano | findstr :PORT",
                    "Run: taskkill /PID <PID> /F"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["windows", "port", "network"]
    ),
    ErrorPattern(
        pattern=r"pip.*Could not find a version that satisfies the requirement",
        error_code="PIP_VERSION_001",
        title="Pip Version Not Found",
        description_template="Pip cannot find a compatible version of the requested package.",
        category=ErrorCategory.DEPENDENCY,
        severity=ErrorSeverity.ERROR,
        language="python",
        solution_templates=[
            {
                "title": "Check package availability",
                "steps": [
                    "Verify the package name is correct",
                    "Check if package supports your Python version",
                    "Try: pip install package==VERSION with a specific version"
                ],
                "auto_fixable": False,
                "confidence": 0.8,
                "effort_level": "easy"
            }
        ],
        tags=["pip", "version", "dependency"]
    ),
    ErrorPattern(
        pattern=r"error.*ENOSPC.*no space left on device|npm ERR!.*ENOSPC",
        error_code="ENOSPC_001",
        title="No Disk Space (ENOSPC)",
        description_template="The operation failed because there is no space left on the device.",
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.CRITICAL,
        solution_templates=[
            {
                "title": "Free up disk space",
                "steps": [
                    "Check disk usage: df -h",
                    "Remove node_modules and reinstall: rm -rf node_modules && npm install",
                    "Clear npm cache: npm cache clean --force"
                ],
                "auto_fixable": True,
                "fix_command": "npm cache clean --force",
                "confidence": 0.7,
                "effort_level": "easy"
            }
        ],
        tags=["disk", "space", "npm"]
    ),
    ErrorPattern(
        pattern=r"error\[E0599\]:.*method .* not found",
        error_code="RUST_METHOD_001",
        title="Rust Method Not Found",
        description_template="The method doesn't exist on the type you're calling it on.",
        category=ErrorCategory.TYPE,
        severity=ErrorSeverity.ERROR,
        language="rust",
        solution_templates=[
            {
                "title": "Check method implementation",
                "steps": [
                    "Verify the trait is in scope (use proper import)",
                    "Check if method exists on the type",
                    "Run: rustc --explain E0599"
                ],
                "auto_fixable": False,
                "confidence": 0.75,
                "effort_level": "moderate"
            }
        ],
        tags=["rust", "method", "type"]
    ),
    ErrorPattern(
        pattern=r"panic.*index out of bounds",
        error_code="RUST_PANIC_001",
        title="Rust Index Out of Bounds",
        description_template="Rust panicked due to an array index being out of bounds.",
        category=ErrorCategory.VALUE,
        severity=ErrorSeverity.ERROR,
        language="rust",
        solution_templates=[
            {
                "title": "Check array bounds",
                "steps": [
                    "Use .get() for safe access that returns Option",
                    "Check length before indexing",
                    "Use iterators instead of manual indexing"
                ],
                "auto_fixable": False,
                "confidence": 0.85,
                "effort_level": "easy"
            }
        ],
        tags=["rust", "panic", "index"]
    ),
]


COMMON_ERRORS: Dict[str, Dict[str, Any]] = {
    "ModuleNotFoundError": {
        "title": "Package Not Installed",
        "description": "A required Python package is not installed in your environment.",
        "category": ErrorCategory.IMPORT,
        "severity": ErrorSeverity.ERROR,
        "quick_fix": "pip install {module_name}",
        "solutions": [
            {"title": "Install with pip", "command": "pip install {module_name}", "confidence": 0.95},
            {"title": "Install with pip3", "command": "pip3 install {module_name}", "confidence": 0.9},
        ]
    },
    "ImportError": {
        "title": "Import Failed",
        "description": "Failed to import a module. The module might be installed incorrectly or have missing dependencies.",
        "category": ErrorCategory.IMPORT,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Reinstall package", "command": "pip install --force-reinstall {module}", "confidence": 0.8},
        ]
    },
    "SyntaxError": {
        "title": "Code Syntax Error",
        "description": "There's a typo or syntax mistake in your code.",
        "category": ErrorCategory.SYNTAX,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check syntax", "steps": ["Review the line mentioned in the error", "Look for missing colons, brackets, or quotes"], "confidence": 0.9},
        ]
    },
    "IndentationError": {
        "title": "Indentation Problem",
        "description": "Python code has incorrect indentation. Python uses indentation to define code blocks.",
        "category": ErrorCategory.SYNTAX,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Fix indentation", "steps": ["Use 4 spaces for each level", "Don't mix tabs and spaces"], "confidence": 0.95},
        ]
    },
    "TabError": {
        "title": "Mixed Tabs and Spaces",
        "description": "The code mixes tabs and spaces for indentation, which Python doesn't allow.",
        "category": ErrorCategory.SYNTAX,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Convert to spaces", "steps": ["Configure editor to use spaces", "Replace all tabs with 4 spaces"], "confidence": 0.95},
        ]
    },
    "TypeError": {
        "title": "Type Mismatch",
        "description": "You're using a value of the wrong type for an operation.",
        "category": ErrorCategory.TYPE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check types", "steps": ["Use type() to check variable types", "Convert types if needed"], "confidence": 0.8},
        ]
    },
    "ValueError": {
        "title": "Invalid Value",
        "description": "A value is in the wrong format or out of acceptable range.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Validate input", "steps": ["Check the input format", "Use try/except for conversions"], "confidence": 0.8},
        ]
    },
    "KeyError": {
        "title": "Key Not Found",
        "description": "The dictionary key you're looking for doesn't exist.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Use .get()", "steps": ["Use dict.get(key, default) instead of dict[key]"], "confidence": 0.9},
        ]
    },
    "IndexError": {
        "title": "Index Out of Range",
        "description": "You're trying to access an index that doesn't exist in the list.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check bounds", "steps": ["Use len() to check length first", "Verify loop boundaries"], "confidence": 0.85},
        ]
    },
    "AttributeError": {
        "title": "Attribute Not Found",
        "description": "The object doesn't have the attribute or method you're trying to use.",
        "category": ErrorCategory.TYPE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check attribute", "steps": ["Use dir(obj) to see available attributes", "Verify object type"], "confidence": 0.8},
        ]
    },
    "NameError": {
        "title": "Variable Not Defined",
        "description": "You're using a variable that hasn't been defined yet.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Define variable", "steps": ["Check spelling", "Ensure variable is defined before use"], "confidence": 0.9},
        ]
    },
    "FileNotFoundError": {
        "title": "File Not Found",
        "description": "The file you're trying to access doesn't exist.",
        "category": ErrorCategory.FILE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check path", "steps": ["Verify file exists", "Use absolute path", "Check for typos"], "confidence": 0.9},
        ]
    },
    "IsADirectoryError": {
        "title": "Expected File, Got Directory",
        "description": "You're trying to open a directory as if it were a file.",
        "category": ErrorCategory.FILE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check path", "steps": ["Ensure path points to a file, not a directory"], "confidence": 0.95},
        ]
    },
    "NotADirectoryError": {
        "title": "Expected Directory, Got File",
        "description": "You're trying to use a file as if it were a directory.",
        "category": ErrorCategory.FILE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check path", "steps": ["Ensure path points to a directory, not a file"], "confidence": 0.95},
        ]
    },
    "PermissionError": {
        "title": "Permission Denied",
        "description": "You don't have permission to access this file or directory.",
        "category": ErrorCategory.PERMISSION,
        "severity": ErrorSeverity.ERROR,
        "quick_fix": "chmod 644 {path}",
        "solutions": [
            {"title": "Change permissions", "command": "chmod 644 {path}", "confidence": 0.8},
            {"title": "Change ownership", "command": "chown $USER {path}", "confidence": 0.7},
        ]
    },
    "ConnectionRefusedError": {
        "title": "Server Not Running",
        "description": "Cannot connect to the server. It might not be running or is refusing connections.",
        "category": ErrorCategory.NETWORK,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Start server", "steps": ["Check if server is running", "Start the server"], "confidence": 0.85},
        ]
    },
    "ConnectionError": {
        "title": "Connection Failed",
        "description": "Failed to establish a network connection.",
        "category": ErrorCategory.NETWORK,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check network", "steps": ["Verify internet connection", "Check server address"], "confidence": 0.7},
        ]
    },
    "TimeoutError": {
        "title": "Operation Timed Out",
        "description": "The operation took too long and was cancelled.",
        "category": ErrorCategory.NETWORK,
        "severity": ErrorSeverity.WARNING,
        "solutions": [
            {"title": "Increase timeout", "steps": ["Add timeout parameter with higher value", "Retry operation"], "confidence": 0.8},
        ]
    },
    "MemoryError": {
        "title": "Out of Memory",
        "description": "The system ran out of memory.",
        "category": ErrorCategory.MEMORY,
        "severity": ErrorSeverity.CRITICAL,
        "solutions": [
            {"title": "Reduce memory", "steps": ["Process data in chunks", "Use generators"], "confidence": 0.7},
        ]
    },
    "RecursionError": {
        "title": "Maximum Recursion Exceeded",
        "description": "A function called itself too many times, causing a stack overflow.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Fix recursion", "steps": ["Add proper base case", "Consider iterative approach"], "confidence": 0.85},
        ]
    },
    "ZeroDivisionError": {
        "title": "Division by Zero",
        "description": "You're trying to divide by zero, which is mathematically undefined.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Add check", "steps": ["Check if divisor is zero before dividing", "Use try/except"], "confidence": 0.95},
        ]
    },
    "OverflowError": {
        "title": "Number Too Large",
        "description": "A calculation resulted in a number too large to represent.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Use larger type", "steps": ["Use Python's arbitrary precision integers", "Consider scientific notation"], "confidence": 0.7},
        ]
    },
    "AssertionError": {
        "title": "Assertion Failed",
        "description": "An assertion in the code failed, indicating an unexpected state.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check assumption", "steps": ["Review what the assertion is checking", "Debug why the condition is false"], "confidence": 0.8},
        ]
    },
    "StopIteration": {
        "title": "Iterator Exhausted",
        "description": "Tried to get more items from an iterator that has no more items.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.WARNING,
        "solutions": [
            {"title": "Check iterator", "steps": ["Use for loop instead of next()", "Handle StopIteration exception"], "confidence": 0.85},
        ]
    },
    "GeneratorExit": {
        "title": "Generator Closed",
        "description": "A generator was closed before it could finish.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.WARNING,
        "solutions": [
            {"title": "Handle closure", "steps": ["Ensure generator completes before closing", "Use try/finally for cleanup"], "confidence": 0.7},
        ]
    },
    "EOFError": {
        "title": "Unexpected End of Input",
        "description": "Reached the end of input unexpectedly when reading data.",
        "category": ErrorCategory.FILE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check input", "steps": ["Ensure input source has data", "Check for empty files"], "confidence": 0.8},
        ]
    },
    "OSError": {
        "title": "Operating System Error",
        "description": "An operating system-related error occurred.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check system resources", "steps": ["Verify file/directory permissions", "Check disk space"], "confidence": 0.6},
        ]
    },
    "RuntimeError": {
        "title": "Runtime Error",
        "description": "An error occurred during program execution.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Debug execution", "steps": ["Add print statements to trace execution", "Use a debugger"], "confidence": 0.5},
        ]
    },
    "NotImplementedError": {
        "title": "Feature Not Implemented",
        "description": "This feature or method hasn't been implemented yet.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.WARNING,
        "solutions": [
            {"title": "Implement method", "steps": ["Override the method in a subclass", "Implement the missing functionality"], "confidence": 0.9},
        ]
    },
    "BrokenPipeError": {
        "title": "Broken Pipe",
        "description": "Writing to a pipe whose other end has closed.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Handle pipe closure", "steps": ["Check if reader is still active", "Handle SIGPIPE signal"], "confidence": 0.7},
        ]
    },
    "ChildProcessError": {
        "title": "Child Process Error",
        "description": "A child process operation failed.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check subprocess", "steps": ["Verify command exists", "Check subprocess output for errors"], "confidence": 0.7},
        ]
    },
    "ProcessLookupError": {
        "title": "Process Not Found",
        "description": "The specified process does not exist.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check process", "steps": ["Verify process is running", "Use correct PID"], "confidence": 0.8},
        ]
    },
    "BlockingIOError": {
        "title": "Would Block",
        "description": "An operation would block in non-blocking mode.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.WARNING,
        "solutions": [
            {"title": "Handle async", "steps": ["Use proper async/await patterns", "Handle blocking in event loop"], "confidence": 0.7},
        ]
    },
    "InterruptedError": {
        "title": "Operation Interrupted",
        "description": "A system call was interrupted.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.WARNING,
        "solutions": [
            {"title": "Retry operation", "steps": ["Wrap in retry loop", "Handle keyboard interrupt separately"], "confidence": 0.75},
        ]
    },
    "SystemExit": {
        "title": "System Exit",
        "description": "The program requested to exit.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.INFO,
        "solutions": [
            {"title": "Normal exit", "steps": ["This is usually intentional", "Check exit code for reason"], "confidence": 0.9},
        ]
    },
    "KeyboardInterrupt": {
        "title": "User Interrupted",
        "description": "The user pressed Ctrl+C to stop the program.",
        "category": ErrorCategory.RUNTIME,
        "severity": ErrorSeverity.INFO,
        "solutions": [
            {"title": "Handle gracefully", "steps": ["Add signal handler for cleanup", "Use try/finally for resources"], "confidence": 0.9},
        ]
    },
    "UnicodeDecodeError": {
        "title": "Text Decoding Error",
        "description": "Cannot decode text with the specified encoding.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Fix encoding", "steps": ["Specify correct encoding", "Try utf-8, latin-1, or cp1252"], "confidence": 0.8},
        ]
    },
    "UnicodeEncodeError": {
        "title": "Text Encoding Error",
        "description": "Cannot encode text with the specified encoding.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Fix encoding", "steps": ["Use utf-8 encoding", "Handle special characters"], "confidence": 0.8},
        ]
    },
    "LookupError": {
        "title": "Lookup Failed",
        "description": "A lookup operation failed (key or index not found).",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check key/index", "steps": ["Verify the key or index exists", "Use .get() or try/except"], "confidence": 0.8},
        ]
    },
    "ArithmeticError": {
        "title": "Arithmetic Error",
        "description": "A mathematical operation produced an invalid result.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Validate input", "steps": ["Check for division by zero", "Handle edge cases"], "confidence": 0.7},
        ]
    },
    "FloatingPointError": {
        "title": "Floating Point Error",
        "description": "A floating point operation produced an invalid result.",
        "category": ErrorCategory.VALUE,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Handle precision", "steps": ["Use decimal module for precision", "Check for NaN or infinity"], "confidence": 0.6},
        ]
    },
    "EnvironmentError": {
        "title": "Environment Error",
        "description": "An error related to the system environment occurred.",
        "category": ErrorCategory.ENVIRONMENT,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check environment", "steps": ["Verify environment variables", "Check system configuration"], "confidence": 0.6},
        ]
    },
    "BufferError": {
        "title": "Buffer Error",
        "description": "A buffer-related operation could not be performed.",
        "category": ErrorCategory.MEMORY,
        "severity": ErrorSeverity.ERROR,
        "solutions": [
            {"title": "Check buffer", "steps": ["Ensure buffer is properly sized", "Handle memory constraints"], "confidence": 0.5},
        ]
    },
}


class ErrorPatternMatcher:
    """Matches error messages to known patterns for user-friendly translation."""
    
    def __init__(self, patterns: Optional[List[ErrorPattern]] = None):
        self.patterns = patterns or ERROR_PATTERNS
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        for pattern in self.patterns:
            if pattern.is_regex:
                try:
                    self._compiled_patterns[pattern.error_code] = re.compile(
                        pattern.pattern, re.IGNORECASE | re.MULTILINE
                    )
                except re.error:
                    pass
    
    def match(self, error_message: str, language: Optional[str] = None, 
              framework: Optional[str] = None) -> Optional[Tuple[ErrorPattern, Dict[str, str]]]:
        """Match error message against known patterns."""
        for pattern in self.patterns:
            if language and pattern.language and pattern.language != language:
                continue
            if framework and pattern.framework and pattern.framework != framework:
                continue
            
            compiled = self._compiled_patterns.get(pattern.error_code)
            if compiled:
                match = compiled.search(error_message)
                if match:
                    groups = {}
                    for i, group in enumerate(match.groups()):
                        groups[str(i)] = group or ""
                    for name, idx in pattern.extract_groups.items():
                        if idx <= len(match.groups()):
                            groups[name] = match.group(idx) or ""
                    return pattern, groups
            elif not pattern.is_regex:
                if pattern.pattern.lower() in error_message.lower():
                    return pattern, {}
        
        return None
    
    def get_all_patterns_for_category(self, category: ErrorCategory) -> List[ErrorPattern]:
        """Get all patterns for a specific category."""
        return [p for p in self.patterns if p.category == category]
    
    def get_all_patterns_for_language(self, language: str) -> List[ErrorPattern]:
        """Get all patterns for a specific language."""
        return [p for p in self.patterns if p.language == language or p.language is None]
    
    def add_pattern(self, pattern: ErrorPattern):
        """Add a new pattern."""
        self.patterns.append(pattern)
        if pattern.is_regex:
            try:
                self._compiled_patterns[pattern.error_code] = re.compile(
                    pattern.pattern, re.IGNORECASE | re.MULTILINE
                )
            except re.error:
                pass
    
    def suggest_similar_identifiers(self, identifier: str, available: List[str], 
                                    max_suggestions: int = 3) -> List[str]:
        """Suggest similar identifiers for 'did you mean?' feature."""
        return get_close_matches(identifier, available, n=max_suggestions, cutoff=0.6)


class ErrorTranslator:
    """Translates exceptions to user-friendly errors."""
    
    def __init__(self, pattern_matcher: Optional[ErrorPatternMatcher] = None):
        self.pattern_matcher = pattern_matcher or ErrorPatternMatcher()
    
    def translate(self, exception: Exception, 
                  context: Optional[ErrorContext] = None) -> UserFriendlyError:
        """Translate an exception to a user-friendly error."""
        error_message = str(exception)
        exception_type = type(exception).__name__
        
        match_result = self.pattern_matcher.match(error_message)
        if match_result:
            pattern, groups = match_result
            return self._create_from_pattern(exception, pattern, groups, context)
        
        if exception_type in COMMON_ERRORS:
            return self._create_from_common_error(exception, context)
        
        return self._create_generic_error(exception, context)
    
    def _create_from_pattern(self, exception: Exception, pattern: ErrorPattern,
                             groups: Dict[str, str], 
                             context: Optional[ErrorContext]) -> UserFriendlyError:
        """Create user-friendly error from matched pattern."""
        description = pattern.description_template.format(*[groups.get(str(i), "") for i in range(10)])
        
        solutions = []
        for sol_template in pattern.solution_templates:
            steps = [step.format(*[groups.get(str(i), "") for i in range(10)]) 
                     for step in sol_template.get("steps", [])]
            
            fix_command = sol_template.get("fix_command")
            if fix_command:
                fix_command = fix_command.format(*[groups.get(str(i), "") for i in range(10)])
            
            effort_str = sol_template.get("effort_level", "easy")
            effort = EffortLevel.EASY
            for level in EffortLevel:
                if level.value == effort_str:
                    effort = level
                    break
            
            solutions.append(Solution(
                title=sol_template.get("title", "Fix the issue"),
                steps=steps,
                auto_fixable=sol_template.get("auto_fixable", False),
                fix_command=fix_command,
                confidence=sol_template.get("confidence", 0.7),
                effort_level=effort
            ))
        
        return UserFriendlyError(
            error_code=pattern.error_code,
            title=pattern.title,
            description=description,
            technical_details=str(exception),
            solutions=solutions,
            related_docs=pattern.related_docs,
            severity=pattern.severity,
            category=pattern.category,
            context=context,
            original_exception=exception,
            tags=pattern.tags
        )
    
    def _create_from_common_error(self, exception: Exception,
                                  context: Optional[ErrorContext]) -> UserFriendlyError:
        """Create user-friendly error from common error mapping."""
        exception_type = type(exception).__name__
        error_info = COMMON_ERRORS[exception_type]
        
        solutions = []
        for sol_info in error_info.get("solutions", []):
            steps = sol_info.get("steps", [])
            if not steps and sol_info.get("command"):
                steps = [f"Run: {sol_info['command']}"]
            
            solutions.append(Solution(
                title=sol_info.get("title", "Fix the issue"),
                steps=steps,
                auto_fixable=bool(sol_info.get("command")),
                fix_command=sol_info.get("command"),
                confidence=sol_info.get("confidence", 0.7),
                effort_level=EffortLevel.EASY
            ))
        
        return UserFriendlyError(
            error_code=f"COMMON_{exception_type.upper()}",
            title=error_info["title"],
            description=error_info["description"],
            technical_details=str(exception),
            solutions=solutions,
            severity=error_info.get("severity", ErrorSeverity.ERROR),
            category=error_info.get("category", ErrorCategory.UNKNOWN),
            context=context,
            original_exception=exception
        )
    
    def _create_generic_error(self, exception: Exception,
                              context: Optional[ErrorContext]) -> UserFriendlyError:
        """Create a generic user-friendly error for unknown exceptions."""
        exception_type = type(exception).__name__
        
        return UserFriendlyError(
            error_code=f"UNKNOWN_{exception_type.upper()}",
            title=f"Unexpected {exception_type}",
            description=f"An unexpected error occurred: {str(exception)[:200]}",
            technical_details=str(exception),
            solutions=[
                Solution(
                    title="Search for the error",
                    steps=[
                        f"Search online for '{exception_type}'",
                        "Check Stack Overflow for similar issues",
                        "Review the stack trace for context"
                    ],
                    auto_fixable=False,
                    confidence=0.5,
                    effort_level=EffortLevel.MODERATE
                )
            ],
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.UNKNOWN,
            context=context,
            original_exception=exception
        )


class SelfServiceResolver:
    """Attempts automatic fixes for common issues."""
    
    def __init__(self):
        self._fix_handlers: Dict[str, Callable] = {
            "PY_IMPORT_001": self._fix_missing_module,
            "NODE_001": self._fix_missing_npm_module,
            "PERM_001": self._fix_permission,
            "NET_003": self._fix_port_in_use,
            "GO_001": self._fix_go_module,
            "RUST_002": self._fix_rust_crate,
            "DOCKER_002": self._fix_docker_space,
        }
    
    def can_auto_fix(self, error: UserFriendlyError) -> bool:
        """Check if this error can be automatically fixed."""
        if error.error_code in self._fix_handlers:
            return True
        return any(s.auto_fixable and s.fix_command for s in error.solutions)
    
    def attempt_fix(self, error: UserFriendlyError, dry_run: bool = False) -> FixResult:
        """Attempt to automatically fix the error."""
        if error.error_code in self._fix_handlers:
            handler = self._fix_handlers[error.error_code]
            return handler(error, dry_run)
        
        auto_fixable = error.get_auto_fixable_solutions()
        if auto_fixable:
            best = max(auto_fixable, key=lambda s: s.confidence)
            return self._execute_fix_command(best.fix_command, dry_run)
        
        return FixResult(
            success=False,
            message="No automatic fix available for this error"
        )
    
    def _execute_fix_command(self, command: str, dry_run: bool) -> FixResult:
        """Execute a fix command."""
        if dry_run:
            return FixResult(
                success=True,
                message=f"Would execute: {command}",
                command_run=command
            )
        
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return FixResult(
                    success=True,
                    message="Fix applied successfully",
                    command_run=command,
                    output=result.stdout,
                    changes_made=[f"Executed: {command}"]
                )
            else:
                return FixResult(
                    success=False,
                    message=f"Command failed with exit code {result.returncode}",
                    command_run=command,
                    output=result.stdout,
                    error_output=result.stderr
                )
        except subprocess.TimeoutExpired:
            return FixResult(
                success=False,
                message="Command timed out",
                command_run=command
            )
        except Exception as e:
            return FixResult(
                success=False,
                message=f"Error executing command: {str(e)}",
                command_run=command
            )
    
    def _fix_missing_module(self, error: UserFriendlyError, dry_run: bool) -> FixResult:
        """Fix missing Python module."""
        match = re.search(r"No module named '([^']+)'", str(error.original_exception))
        if match:
            module_name = match.group(1).split('.')[0]
            command = f"pip install {module_name}"
            return self._execute_fix_command(command, dry_run)
        return FixResult(success=False, message="Could not determine module name")
    
    def _fix_missing_npm_module(self, error: UserFriendlyError, dry_run: bool) -> FixResult:
        """Fix missing npm module."""
        match = re.search(r"Cannot find module '([^']+)'", str(error.original_exception))
        if match:
            module_name = match.group(1)
            if not module_name.startswith('.') and not module_name.startswith('/'):
                command = f"npm install {module_name}"
                return self._execute_fix_command(command, dry_run)
        return FixResult(success=False, message="Could not determine module name or it's a local module")
    
    def _fix_permission(self, error: UserFriendlyError, dry_run: bool) -> FixResult:
        """Fix permission error."""
        match = re.search(r"Permission denied: ['\"]?([^'\"]+)['\"]?", str(error.original_exception))
        if match:
            path = match.group(1)
            command = f"chmod 644 {path}"
            return self._execute_fix_command(command, dry_run)
        return FixResult(success=False, message="Could not determine file path")
    
    def _fix_port_in_use(self, error: UserFriendlyError, dry_run: bool) -> FixResult:
        """Fix port already in use."""
        match = re.search(r":(\d+)", str(error.original_exception))
        if match:
            port = match.group(1)
            command = f"kill -9 $(lsof -t -i:{port})"
            return self._execute_fix_command(command, dry_run)
        return FixResult(success=False, message="Could not determine port number")
    
    def _fix_go_module(self, error: UserFriendlyError, dry_run: bool) -> FixResult:
        """Fix Go module issues."""
        command = "go mod tidy"
        return self._execute_fix_command(command, dry_run)
    
    def _fix_rust_crate(self, error: UserFriendlyError, dry_run: bool) -> FixResult:
        """Fix missing Rust crate."""
        match = re.search(r"could not find `([^`]+)`", str(error.original_exception))
        if match:
            crate_name = match.group(1)
            command = f"cargo add {crate_name}"
            return self._execute_fix_command(command, dry_run)
        return FixResult(success=False, message="Could not determine crate name")
    
    def _fix_docker_space(self, error: UserFriendlyError, dry_run: bool) -> FixResult:
        """Fix Docker disk space issues."""
        command = "docker system prune -af"
        return self._execute_fix_command(command, dry_run)
    
    def suggest_similar(self, identifier: str, available: List[str]) -> List[str]:
        """Suggest similar identifiers for 'did you mean?' feature."""
        return get_close_matches(identifier, available, n=3, cutoff=0.6)
    
    def suggest_dependency_installation(self, module_name: str, 
                                         language: str = "python") -> List[Solution]:
        """Suggest dependency installation for a module."""
        solutions = []
        
        if language == "python":
            solutions.extend([
                Solution(
                    title=f"Install {module_name} with pip",
                    steps=[f"Run: pip install {module_name}"],
                    auto_fixable=True,
                    fix_command=f"pip install {module_name}",
                    confidence=0.95,
                    effort_level=EffortLevel.TRIVIAL
                ),
                Solution(
                    title=f"Install {module_name} with pip3",
                    steps=[f"Run: pip3 install {module_name}"],
                    auto_fixable=True,
                    fix_command=f"pip3 install {module_name}",
                    confidence=0.9,
                    effort_level=EffortLevel.TRIVIAL
                ),
                Solution(
                    title=f"Add {module_name} to requirements.txt",
                    steps=[
                        f"Add '{module_name}' to requirements.txt",
                        "Run: pip install -r requirements.txt"
                    ],
                    auto_fixable=False,
                    confidence=0.85,
                    effort_level=EffortLevel.EASY
                )
            ])
        elif language in ("javascript", "typescript"):
            solutions.extend([
                Solution(
                    title=f"Install {module_name} with npm",
                    steps=[f"Run: npm install {module_name}"],
                    auto_fixable=True,
                    fix_command=f"npm install {module_name}",
                    confidence=0.95,
                    effort_level=EffortLevel.TRIVIAL
                ),
                Solution(
                    title=f"Install {module_name} with yarn",
                    steps=[f"Run: yarn add {module_name}"],
                    auto_fixable=True,
                    fix_command=f"yarn add {module_name}",
                    confidence=0.85,
                    effort_level=EffortLevel.TRIVIAL
                )
            ])
        elif language == "go":
            solutions.append(Solution(
                title=f"Install {module_name}",
                steps=[f"Run: go get {module_name}"],
                auto_fixable=True,
                fix_command=f"go get {module_name}",
                confidence=0.9,
                effort_level=EffortLevel.TRIVIAL
            ))
        elif language == "rust":
            solutions.append(Solution(
                title=f"Add {module_name} to Cargo.toml",
                steps=[f"Run: cargo add {module_name}"],
                auto_fixable=True,
                fix_command=f"cargo add {module_name}",
                confidence=0.9,
                effort_level=EffortLevel.TRIVIAL
            ))
        
        return solutions


@dataclass
class ErrorLogEntry:
    """A single error log entry."""
    error_id: str
    error_code: str
    title: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: datetime
    context: Optional[ErrorContext] = None
    resolved: bool = False
    resolution: Optional[str] = None
    occurrence_count: int = 1
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.to_dict() if self.context else None,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "occurrence_count": self.occurrence_count,
            "tags": self.tags,
        }


@dataclass
class ErrorReport:
    """Summary report of errors."""
    period_start: datetime
    period_end: datetime
    total_errors: int
    errors_by_severity: Dict[str, int]
    errors_by_category: Dict[str, int]
    top_recurring_errors: List[Tuple[str, int]]
    unresolved_errors: int
    auto_fixed_count: int
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_errors": self.total_errors,
            "errors_by_severity": self.errors_by_severity,
            "errors_by_category": self.errors_by_category,
            "top_recurring_errors": self.top_recurring_errors,
            "unresolved_errors": self.unresolved_errors,
            "auto_fixed_count": self.auto_fixed_count,
            "recommendations": self.recommendations,
        }


class ErrorLogger:
    """Logs and tracks error patterns."""
    
    def __init__(self, log_file: Optional[str] = None, max_entries: int = 1000):
        self.log_file = log_file
        self.max_entries = max_entries
        self._entries: List[ErrorLogEntry] = []
        self._pattern_counts: Dict[str, int] = defaultdict(int)
        self._logger = logging.getLogger("error_handler")
        
        if log_file:
            self._load_from_file()
    
    def log(self, error: UserFriendlyError) -> str:
        """Log an error and return its ID."""
        error_id = self._generate_error_id(error)
        
        existing = self._find_similar(error)
        if existing:
            existing.occurrence_count += 1
            existing.timestamp = datetime.now()
            self._pattern_counts[error.error_code] += 1
            return existing.error_id
        
        entry = ErrorLogEntry(
            error_id=error_id,
            error_code=error.error_code,
            title=error.title,
            message=error.description,
            severity=error.severity,
            category=error.category,
            timestamp=datetime.now(),
            context=error.context,
            tags=error.tags
        )
        
        self._entries.append(entry)
        self._pattern_counts[error.error_code] += 1
        
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        
        self._log_to_file(entry)
        
        return error_id
    
    def mark_resolved(self, error_id: str, resolution: str):
        """Mark an error as resolved."""
        for entry in self._entries:
            if entry.error_id == error_id:
                entry.resolved = True
                entry.resolution = resolution
                break
    
    def get_entry(self, error_id: str) -> Optional[ErrorLogEntry]:
        """Get a specific error entry."""
        for entry in self._entries:
            if entry.error_id == error_id:
                return entry
        return None
    
    def get_recent_errors(self, count: int = 10, 
                          severity: Optional[ErrorSeverity] = None,
                          category: Optional[ErrorCategory] = None) -> List[ErrorLogEntry]:
        """Get recent errors with optional filtering."""
        filtered = self._entries
        
        if severity:
            filtered = [e for e in filtered if e.severity == severity]
        if category:
            filtered = [e for e in filtered if e.category == category]
        
        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)[:count]
    
    def get_recurring_errors(self, min_occurrences: int = 3) -> List[Tuple[str, int]]:
        """Get errors that occur frequently."""
        recurring = [
            (code, count) for code, count in self._pattern_counts.items()
            if count >= min_occurrences
        ]
        return sorted(recurring, key=lambda x: x[1], reverse=True)
    
    def generate_report(self, days: int = 7) -> ErrorReport:
        """Generate an error report for the specified period."""
        period_end = datetime.now()
        period_start = period_end.replace(hour=0, minute=0, second=0) 
        
        relevant = [
            e for e in self._entries
            if (period_end - e.timestamp).days <= days
        ]
        
        by_severity: Dict[str, int] = defaultdict(int)
        by_category: Dict[str, int] = defaultdict(int)
        
        for entry in relevant:
            by_severity[entry.severity.value] += entry.occurrence_count
            by_category[entry.category.value] += entry.occurrence_count
        
        recurring = self.get_recurring_errors(min_occurrences=2)
        unresolved = len([e for e in relevant if not e.resolved])
        
        recommendations = self._generate_recommendations(relevant, recurring)
        
        return ErrorReport(
            period_start=period_start,
            period_end=period_end,
            total_errors=sum(e.occurrence_count for e in relevant),
            errors_by_severity=dict(by_severity),
            errors_by_category=dict(by_category),
            top_recurring_errors=recurring[:10],
            unresolved_errors=unresolved,
            auto_fixed_count=0,
            recommendations=recommendations
        )
    
    def _generate_error_id(self, error: UserFriendlyError) -> str:
        """Generate a unique error ID."""
        content = f"{error.error_code}:{error.title}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _find_similar(self, error: UserFriendlyError) -> Optional[ErrorLogEntry]:
        """Find a similar recent error."""
        for entry in reversed(self._entries):
            if entry.error_code == error.error_code and not entry.resolved:
                time_diff = (datetime.now() - entry.timestamp).total_seconds()
                if time_diff < 3600:
                    return entry
        return None
    
    def _generate_recommendations(self, errors: List[ErrorLogEntry],
                                   recurring: List[Tuple[str, int]]) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        
        if recurring:
            top_error = recurring[0]
            recommendations.append(
                f"Address the recurring '{top_error[0]}' error which occurred {top_error[1]} times"
            )
        
        critical = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
        if critical:
            recommendations.append(
                f"Investigate {len(critical)} critical errors as top priority"
            )
        
        import_errors = [e for e in errors if e.category == ErrorCategory.IMPORT]
        if len(import_errors) > 5:
            recommendations.append(
                "Consider reviewing dependency management - many import errors detected"
            )
        
        permission_errors = [e for e in errors if e.category == ErrorCategory.PERMISSION]
        if permission_errors:
            recommendations.append(
                "Review file and directory permissions to prevent access issues"
            )
        
        return recommendations
    
    def _load_from_file(self):
        """Load entries from log file."""
        if not self.log_file or not os.path.exists(self.log_file):
            return
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        entry = ErrorLogEntry(
                            error_id=data.get("error_id", ""),
                            error_code=data.get("error_code", ""),
                            title=data.get("title", ""),
                            message=data.get("message", ""),
                            severity=ErrorSeverity(data.get("severity", "error")),
                            category=ErrorCategory(data.get("category", "unknown")),
                            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                            resolved=data.get("resolved", False),
                            occurrence_count=data.get("occurrence_count", 1),
                            tags=data.get("tags", [])
                        )
                        self._entries.append(entry)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except IOError:
            pass
    
    def _log_to_file(self, entry: ErrorLogEntry):
        """Log entry to file."""
        if not self.log_file:
            return
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except IOError:
            pass


_default_translator: Optional[ErrorTranslator] = None
_default_resolver: Optional[SelfServiceResolver] = None
_default_logger: Optional[ErrorLogger] = None


def get_default_translator() -> ErrorTranslator:
    """Get the default error translator."""
    global _default_translator
    if _default_translator is None:
        _default_translator = ErrorTranslator()
    return _default_translator


def get_default_resolver() -> SelfServiceResolver:
    """Get the default self-service resolver."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = SelfServiceResolver()
    return _default_resolver


def get_default_logger() -> ErrorLogger:
    """Get the default error logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = ErrorLogger()
    return _default_logger


def get_error_context(exception: Optional[Exception] = None) -> ErrorContext:
    """Extract context information from an exception or current state."""
    context = ErrorContext()
    
    if exception:
        tb = traceback.extract_tb(exception.__traceback__)
        if tb:
            last_frame = tb[-1]
            context.file_path = last_frame.filename
            context.line_number = last_frame.lineno
            context.function_name = last_frame.name
            context.code_snippet = last_frame.line or ""
        
        context.stack_trace = traceback.format_exc()
    else:
        frame = sys._getframe(1)
        context.file_path = frame.f_code.co_filename
        context.line_number = frame.f_lineno
        context.function_name = frame.f_code.co_name
    
    if context.file_path and context.line_number:
        try:
            with open(context.file_path, 'r') as f:
                lines = f.readlines()
                start = max(0, context.line_number - 3)
                end = min(len(lines), context.line_number + 2)
                context.surrounding_lines = [
                    line.rstrip() for line in lines[start:end]
                ]
        except (IOError, OSError):
            pass
    
    return context


def handle_error(error: Exception, 
                 log: bool = True,
                 attempt_fix: bool = False) -> UserFriendlyError:
    """Main entry point for handling errors.
    
    Args:
        error: The exception to handle
        log: Whether to log the error
        attempt_fix: Whether to attempt automatic fixes
    
    Returns:
        UserFriendlyError with details and solutions
    """
    translator = get_default_translator()
    context = get_error_context(error)
    friendly_error = translator.translate(error, context)
    
    if log:
        logger = get_default_logger()
        logger.log(friendly_error)
    
    if attempt_fix:
        resolver = get_default_resolver()
        if resolver.can_auto_fix(friendly_error):
            result = resolver.attempt_fix(friendly_error, dry_run=False)
            if result.success:
                friendly_error.solutions.insert(0, Solution(
                    title="[AUTO-FIXED]",
                    steps=[result.message],
                    auto_fixable=True,
                    confidence=1.0,
                    effort_level=EffortLevel.TRIVIAL
                ))
    
    return friendly_error


def suggest_fixes(error: Exception) -> List[Solution]:
    """Get suggested fixes for an error.
    
    Args:
        error: The exception to get fixes for
    
    Returns:
        List of Solution objects with fix suggestions
    """
    translator = get_default_translator()
    context = get_error_context(error)
    friendly_error = translator.translate(error, context)
    return friendly_error.solutions


def auto_fix(error: Exception, dry_run: bool = True) -> FixResult:
    """Attempt to automatically fix an error.
    
    Args:
        error: The exception to fix
        dry_run: If True, don't actually make changes
    
    Returns:
        FixResult with the outcome
    """
    translator = get_default_translator()
    resolver = get_default_resolver()
    
    context = get_error_context(error)
    friendly_error = translator.translate(error, context)
    
    if not resolver.can_auto_fix(friendly_error):
        return FixResult(
            success=False,
            message="No automatic fix available for this error type"
        )
    
    return resolver.attempt_fix(friendly_error, dry_run=dry_run)


def format_error_for_display(error: Exception, 
                             show_technical: bool = False,
                             show_solutions: bool = True) -> str:
    """Format an error for human-readable display.
    
    Args:
        error: The exception to format
        show_technical: Include technical details
        show_solutions: Include solution suggestions
    
    Returns:
        Formatted string for display
    """
    friendly_error = handle_error(error, log=False)
    return friendly_error.format_human_readable(
        show_technical=show_technical,
        show_solutions=show_solutions
    )


__all__ = [
    'ErrorSeverity',
    'ErrorCategory',
    'EffortLevel',
    'Solution',
    'FixResult',
    'ErrorContext',
    'UserFriendlyError',
    'ErrorPattern',
    'ErrorPatternMatcher',
    'ErrorTranslator',
    'SelfServiceResolver',
    'ErrorLogEntry',
    'ErrorReport',
    'ErrorLogger',
    'ERROR_PATTERNS',
    'COMMON_ERRORS',
    'get_default_translator',
    'get_default_resolver',
    'get_default_logger',
    'get_error_context',
    'handle_error',
    'suggest_fixes',
    'auto_fix',
    'format_error_for_display',
]
