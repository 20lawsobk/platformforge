"""
Comprehensive Code Analyzer for Platform Forge AI Model

This module provides advanced code analysis capabilities including:
- Static analysis (unused variables, imports, undefined references)
- Complexity metrics (cyclomatic, cognitive, LOC, maintainability index)
- Security vulnerability detection (SQL injection, XSS, command injection, etc.)
- Code smell detection (long methods, large classes, duplicate code, etc.)
- Dependency analysis (import graphs, circular dependencies)
- Performance analysis (N+1 queries, inefficient loops, memory leaks)

Usage:
    from server.ai_model.code_analyzer import CodeAnalyzer, analyze_code
    
    # Quick analysis
    report = analyze_code(code_string, language='python')
    
    # Detailed analysis with specific checks
    analyzer = CodeAnalyzer(language='python')
    static_results = analyzer.analyze_static(code_string)
    complexity = analyzer.analyze_complexity(code_string)
    vulnerabilities = analyzer.detect_vulnerabilities(code_string)
"""

import re
import ast
import hashlib
import math
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod


class Severity(Enum):
    """Severity levels for analysis findings."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    def __lt__(self, other):
        order = [Severity.INFO, Severity.WARNING, Severity.ERROR, Severity.CRITICAL]
        return order.index(self) < order.index(other)


@dataclass
class CodeLocation:
    """Represents a location in code."""
    line: int
    column: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    
    def __str__(self) -> str:
        if self.end_line and self.end_line != self.line:
            return f"lines {self.line}-{self.end_line}"
        return f"line {self.line}"


@dataclass
class AnalysisIssue:
    """Represents a single analysis finding."""
    rule_id: str
    message: str
    severity: Severity
    location: CodeLocation
    code_snippet: str = ""
    suggestion: str = ""
    category: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "severity": self.severity.value,
            "line": self.location.line,
            "column": self.location.column,
            "end_line": self.location.end_line,
            "end_column": self.location.end_column,
            "code_snippet": self.code_snippet,
            "suggestion": self.suggestion,
            "category": self.category,
            "confidence": self.confidence,
        }


@dataclass
class ComplexityMetrics:
    """Code complexity metrics."""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    source_lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    max_nesting_depth: int = 0
    average_function_length: float = 0.0
    max_function_length: int = 0
    maintainability_index: float = 100.0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    function_count: int = 0
    class_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "lines_of_code": self.lines_of_code,
            "source_lines_of_code": self.source_lines_of_code,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "max_nesting_depth": self.max_nesting_depth,
            "average_function_length": round(self.average_function_length, 2),
            "max_function_length": self.max_function_length,
            "maintainability_index": round(self.maintainability_index, 2),
            "halstead_volume": round(self.halstead_volume, 2),
            "halstead_difficulty": round(self.halstead_difficulty, 2),
            "halstead_effort": round(self.halstead_effort, 2),
            "function_count": self.function_count,
            "class_count": self.class_count,
        }


@dataclass
class DependencyInfo:
    """Dependency information."""
    name: str
    type: str  # 'internal', 'external', 'stdlib', 'relative'
    import_line: int
    alias: Optional[str] = None
    imported_names: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "import_line": self.import_line,
            "alias": self.alias,
            "imported_names": self.imported_names,
        }


@dataclass
class DependencyGraph:
    """Dependency analysis results."""
    imports: List[DependencyInfo] = field(default_factory=list)
    internal_dependencies: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    stdlib_dependencies: List[str] = field(default_factory=list)
    circular_dependencies: List[Tuple[str, str]] = field(default_factory=list)
    unused_imports: List[str] = field(default_factory=list)
    missing_imports: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "imports": [i.to_dict() for i in self.imports],
            "internal_dependencies": self.internal_dependencies,
            "external_dependencies": self.external_dependencies,
            "stdlib_dependencies": self.stdlib_dependencies,
            "circular_dependencies": self.circular_dependencies,
            "unused_imports": self.unused_imports,
            "missing_imports": self.missing_imports,
        }


@dataclass 
class AnalysisReport:
    """Complete analysis report."""
    language: str
    issues: List[AnalysisIssue] = field(default_factory=list)
    complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    dependencies: DependencyGraph = field(default_factory=DependencyGraph)
    score: float = 100.0
    grade: str = "A"
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: AnalysisIssue):
        self.issues.append(issue)
    
    def get_issues_by_severity(self, severity: Severity) -> List[AnalysisIssue]:
        return [i for i in self.issues if i.severity == severity]
    
    def get_issues_by_category(self, category: str) -> List[AnalysisIssue]:
        return [i for i in self.issues if i.category == category]
    
    def calculate_score(self):
        """Calculate overall code quality score."""
        base_score = 100.0
        
        for issue in self.issues:
            if issue.severity == Severity.CRITICAL:
                base_score -= 15 * issue.confidence
            elif issue.severity == Severity.ERROR:
                base_score -= 10 * issue.confidence
            elif issue.severity == Severity.WARNING:
                base_score -= 5 * issue.confidence
            elif issue.severity == Severity.INFO:
                base_score -= 1 * issue.confidence
        
        if self.complexity.cyclomatic_complexity > 20:
            base_score -= 10
        elif self.complexity.cyclomatic_complexity > 10:
            base_score -= 5
        
        if self.complexity.max_nesting_depth > 5:
            base_score -= 5
        
        if self.complexity.maintainability_index < 50:
            base_score -= 10
        elif self.complexity.maintainability_index < 65:
            base_score -= 5
        
        self.score = max(0, min(100, base_score))
        
        if self.score >= 90:
            self.grade = "A"
        elif self.score >= 80:
            self.grade = "B"
        elif self.score >= 70:
            self.grade = "C"
        elif self.score >= 60:
            self.grade = "D"
        else:
            self.grade = "F"
    
    def generate_summary(self):
        """Generate a human-readable summary."""
        critical = len(self.get_issues_by_severity(Severity.CRITICAL))
        errors = len(self.get_issues_by_severity(Severity.ERROR))
        warnings = len(self.get_issues_by_severity(Severity.WARNING))
        info = len(self.get_issues_by_severity(Severity.INFO))
        
        parts = []
        if critical:
            parts.append(f"{critical} critical")
        if errors:
            parts.append(f"{errors} errors")
        if warnings:
            parts.append(f"{warnings} warnings")
        if info:
            parts.append(f"{info} info")
        
        if parts:
            self.summary = f"Found {', '.join(parts)}. Grade: {self.grade} ({self.score:.1f}/100)"
        else:
            self.summary = f"No issues found. Grade: {self.grade} ({self.score:.1f}/100)"
    
    def generate_recommendations(self):
        """Generate actionable recommendations."""
        recs = []
        
        critical = self.get_issues_by_severity(Severity.CRITICAL)
        if critical:
            recs.append(f"Address {len(critical)} critical issues immediately - these may cause security vulnerabilities or runtime errors")
        
        vuln_issues = self.get_issues_by_category("security")
        if vuln_issues:
            recs.append(f"Review {len(vuln_issues)} security vulnerabilities - prioritize SQL injection and XSS fixes")
        
        if self.complexity.cyclomatic_complexity > 15:
            recs.append(f"Reduce cyclomatic complexity ({self.complexity.cyclomatic_complexity}) by extracting methods or simplifying conditionals")
        
        if self.complexity.max_nesting_depth > 4:
            recs.append(f"Reduce maximum nesting depth ({self.complexity.max_nesting_depth}) using early returns or guard clauses")
        
        if self.complexity.max_function_length > 50:
            recs.append(f"Break up long functions (max {self.complexity.max_function_length} lines) into smaller, focused functions")
        
        if self.dependencies.unused_imports:
            recs.append(f"Remove {len(self.dependencies.unused_imports)} unused imports: {', '.join(self.dependencies.unused_imports[:5])}")
        
        if self.dependencies.circular_dependencies:
            recs.append(f"Resolve {len(self.dependencies.circular_dependencies)} circular dependencies to improve modularity")
        
        smell_issues = self.get_issues_by_category("code_smell")
        if len(smell_issues) > 5:
            recs.append(f"Address {len(smell_issues)} code smells to improve maintainability")
        
        if self.complexity.maintainability_index < 65:
            recs.append(f"Improve maintainability index ({self.complexity.maintainability_index:.1f}) by adding documentation and simplifying code")
        
        self.recommendations = recs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "score": round(self.score, 2),
            "grade": self.grade,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "issues": [i.to_dict() for i in self.issues],
            "complexity": self.complexity.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "metadata": self.metadata,
            "issue_counts": {
                "critical": len(self.get_issues_by_severity(Severity.CRITICAL)),
                "error": len(self.get_issues_by_severity(Severity.ERROR)),
                "warning": len(self.get_issues_by_severity(Severity.WARNING)),
                "info": len(self.get_issues_by_severity(Severity.INFO)),
            }
        }


PYTHON_STDLIB = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins',
    'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs',
    'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser',
    'contextlib', 'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv',
    'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib',
    'dis', 'distutils', 'doctest', 'email', 'encodings', 'enum', 'errno', 'faulthandler',
    'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'fractions', 'ftplib', 'functools',
    'gc', 'getopt', 'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib',
    'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr', 'imp', 'importlib',
    'inspect', 'io', 'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache',
    'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes',
    'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis', 'nntplib', 'numbers',
    'operator', 'optparse', 'os', 'ossaudiodev', 'pathlib', 'pdb', 'pickle', 'pickletools',
    'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix', 'posixpath', 'pprint',
    'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri',
    'random', 're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched',
    'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site',
    'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl',
    'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess', 'sunau',
    'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile',
    'termios', 'test', 'textwrap', 'threading', 'time', 'timeit', 'tkinter', 'token',
    'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo',
    'types', 'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv',
    'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound', 'wsgiref',
    'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib', '_thread'
}


class VulnerabilityPatterns:
    """Security vulnerability detection patterns."""
    
    SQL_INJECTION = [
        (r'execute\s*\(\s*["\'].*%[sd].*["\'].*%', "SQL injection via string formatting"),
        (r'execute\s*\(\s*f["\'].*\{.*\}.*["\']', "SQL injection via f-string"),
        (r'execute\s*\(\s*["\'].*\+.*["\']', "SQL injection via string concatenation"),
        (r'cursor\.execute\s*\([^,)]*\+[^,)]*\)', "SQL injection in cursor.execute"),
        (r'\.query\s*\([^,)]*\+[^,)]*\)', "SQL injection in query method"),
        (r'\.raw\s*\([^,)]*\+[^,)]*\)', "SQL injection in raw query"),
        (r'SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*%s', "Potential SQL injection with %s"),
        (r'f["\']SELECT\s+.*FROM\s+.*WHERE\s+.*\{', "SQL injection in f-string query"),
    ]
    
    XSS = [
        (r'innerHTML\s*=\s*[^;]*\+', "XSS via innerHTML assignment"),
        (r'document\.write\s*\([^)]*\+', "XSS via document.write"),
        (r'\.html\s*\([^)]*\+', "XSS via jQuery .html()"),
        (r'eval\s*\([^)]*\+', "XSS/Code injection via eval"),
        (r'dangerouslySetInnerHTML', "Potential XSS via dangerouslySetInnerHTML"),
        (r'v-html\s*=', "Potential XSS via Vue v-html directive"),
        (r'\[innerHTML\]', "Potential XSS via Angular innerHTML binding"),
        (r'render_template_string\s*\([^)]*\+', "XSS via Flask template string"),
        (r'mark_safe\s*\([^)]*\+', "XSS via Django mark_safe"),
        (r'Markup\s*\([^)]*\+', "XSS via Jinja2 Markup"),
    ]
    
    COMMAND_INJECTION = [
        (r'os\.system\s*\([^)]*\+', "Command injection via os.system"),
        (r'os\.popen\s*\([^)]*\+', "Command injection via os.popen"),
        (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Command injection with shell=True"),
        (r'subprocess\.Popen\s*\([^)]*shell\s*=\s*True', "Command injection with shell=True"),
        (r'subprocess\.run\s*\([^)]*shell\s*=\s*True', "Command injection with shell=True"),
        (r'exec\s*\([^)]*\+', "Code injection via exec"),
        (r'eval\s*\([^)]*input', "Code injection via eval with user input"),
        (r'child_process\.exec\s*\([^)]*\+', "Command injection in Node.js"),
        (r'spawn\s*\([^)]*\+', "Command injection via spawn"),
    ]
    
    PATH_TRAVERSAL = [
        (r'open\s*\([^)]*\+[^)]*\)', "Path traversal in file open"),
        (r'open\s*\([^)]*%[sd]', "Path traversal via string formatting"),
        (r'open\s*\(f["\'][^"\']*\{', "Path traversal via f-string"),
        (r'\.read\s*\([^)]*\+', "Path traversal in file read"),
        (r'send_file\s*\([^)]*\+', "Path traversal in send_file"),
        (r'os\.path\.join\s*\([^)]*\.\.[^)]*\)', "Path traversal with .."),
        (r'readFile\s*\([^)]*\+', "Path traversal in Node.js readFile"),
        (r'require\s*\([^)]*\+', "Path traversal in require"),
    ]
    
    HARDCODED_SECRETS = [
        (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
        (r'secret\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret"),
        (r'api_key\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded API key"),
        (r'apikey\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded API key"),
        (r'api_secret\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded API secret"),
        (r'token\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded token"),
        (r'private_key\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded private key"),
        (r'aws_access_key\s*=\s*["\']AK[A-Z0-9]{18}["\']', "Hardcoded AWS access key"),
        (r'aws_secret\s*=\s*["\'][A-Za-z0-9/+=]{40}["\']', "Hardcoded AWS secret"),
        (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', "Hardcoded private key"),
        (r'-----BEGIN\s+CERTIFICATE-----', "Hardcoded certificate"),
        (r'Bearer\s+[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+', "Hardcoded JWT"),
        (r'sk_live_[A-Za-z0-9]{24,}', "Hardcoded Stripe secret key"),
        (r'ghp_[A-Za-z0-9]{36}', "Hardcoded GitHub token"),
    ]
    
    INSECURE_DESERIALIZATION = [
        (r'pickle\.loads?\s*\(', "Insecure deserialization via pickle"),
        (r'cPickle\.loads?\s*\(', "Insecure deserialization via cPickle"),
        (r'yaml\.load\s*\([^)]*Loader\s*=\s*None', "Insecure YAML loading"),
        (r'yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader)', "Insecure YAML loading without Loader"),
        (r'yaml\.unsafe_load', "Insecure YAML unsafe_load"),
        (r'marshal\.loads?\s*\(', "Insecure deserialization via marshal"),
        (r'shelve\.open', "Potentially insecure shelve usage"),
        (r'unserialize\s*\(', "PHP-style insecure deserialization"),
        (r'JSON\.parse\s*\([^)]*\)\s*;?\s*eval', "Insecure JSON parsing with eval"),
    ]
    
    CSRF = [
        (r'@csrf_exempt', "CSRF protection disabled"),
        (r'csrf_protect\s*=\s*False', "CSRF protection disabled"),
        (r'WTF_CSRF_ENABLED\s*=\s*False', "Flask-WTF CSRF disabled"),
        (r'disable_csrf', "CSRF protection disabled"),
        (r'verify\s*=\s*False', "SSL verification disabled (related to CSRF)"),
    ]
    
    BUFFER_OVERFLOW = [
        (r'strcpy\s*\(', "Unsafe strcpy - use strncpy"),
        (r'strcat\s*\(', "Unsafe strcat - use strncat"),
        (r'sprintf\s*\(', "Unsafe sprintf - use snprintf"),
        (r'gets\s*\(', "Extremely unsafe gets - use fgets"),
        (r'scanf\s*\([^,]*%s', "Unsafe scanf with %s - specify width"),
        (r'memcpy\s*\([^)]*sizeof\s*\*', "Potential buffer overflow in memcpy"),
        (r'malloc\s*\([^)]*\*[^)]*\+', "Potential integer overflow in malloc"),
    ]
    
    WEAK_CRYPTO = [
        (r'md5\s*\(', "Weak hash function MD5"),
        (r'sha1\s*\(', "Weak hash function SHA1"),
        (r'DES\s*\(', "Weak encryption DES"),
        (r'RC4\s*\(', "Weak encryption RC4"),
        (r'random\s*\(\)(?!.*secrets)', "Weak random number generator"),
        (r'Math\.random\s*\(\)', "Weak random in JavaScript"),
    ]


class CodeSmellPatterns:
    """Code smell detection patterns."""
    
    MAGIC_NUMBERS = [
        (r'(?<![A-Za-z0-9_])(?:if|while|for|elif|==|!=|<|>|<=|>=)\s*[^=<>!]*\b([2-9]\d{2,}|[1-9]\d{3,})\b', "Magic number in condition"),
        (r'range\s*\(\s*\d{2,}\s*\)', "Magic number in range"),
        (r'\[\s*\d{2,}\s*\]', "Magic number as index"),
        (r'sleep\s*\(\s*\d+\s*\)', "Magic number in sleep"),
        (r'timeout\s*=\s*\d{2,}', "Magic number in timeout"),
    ]
    
    LONG_PARAMETER_LIST = r'def\s+\w+\s*\([^)]{100,}\)'
    
    DEEP_NESTING_PATTERN = r'(\s{16,}|\t{4,})(if|for|while|try|with)'
    
    DUPLICATE_CODE_PATTERNS = [
        r'((?:.*\n){3,}).*\1',  # Repeated blocks
    ]
    
    GOD_CLASS_INDICATORS = [
        r'class\s+\w+.*:(?:.*\n){200,}',  # Very long class
    ]
    
    UNUSED_VARIABLE_PATTERN = r'\b(\w+)\s*=\s*[^=].*(?!\1)'


class BaseLanguageAnalyzer(ABC):
    """Abstract base class for language-specific analyzers."""
    
    @abstractmethod
    def parse(self, code: str) -> Any:
        """Parse code into AST or equivalent structure."""
        pass
    
    @abstractmethod
    def analyze_static(self, code: str) -> List[AnalysisIssue]:
        """Perform static analysis."""
        pass
    
    @abstractmethod
    def calculate_complexity(self, code: str) -> ComplexityMetrics:
        """Calculate complexity metrics."""
        pass
    
    @abstractmethod
    def analyze_dependencies(self, code: str) -> DependencyGraph:
        """Analyze code dependencies."""
        pass


class PythonAnalyzer(BaseLanguageAnalyzer):
    """Python-specific code analyzer using AST."""
    
    def __init__(self):
        self.tree: Optional[ast.AST] = None
        self.code_lines: List[str] = []
    
    def parse(self, code: str) -> Optional[ast.AST]:
        """Parse Python code into AST."""
        try:
            self.code_lines = code.split('\n')
            self.tree = ast.parse(code)
            return self.tree
        except SyntaxError as e:
            return None
    
    def analyze_static(self, code: str) -> List[AnalysisIssue]:
        """Perform static analysis on Python code."""
        issues: List[AnalysisIssue] = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(AnalysisIssue(
                rule_id="PY001",
                message=f"Syntax error: {e.msg}",
                severity=Severity.ERROR,
                location=CodeLocation(line=e.lineno or 1, column=e.offset or 0),
                code_snippet=self._get_line(code, e.lineno or 1),
                suggestion="Fix the syntax error to continue analysis",
                category="syntax"
            ))
            return issues
        
        defined_names: Set[str] = set()
        used_names: Set[str] = set()
        imported_names: Dict[str, int] = {}
        
        class NameVisitor(ast.NodeVisitor):
            def __init__(self, parent_analyzer):
                self.analyzer = parent_analyzer
                self.scope_stack: List[Set[str]] = [set()]
                
            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[0]
                    imported_names[name] = node.lineno
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name != '*':
                        imported_names[name] = node.lineno
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                defined_names.add(node.name)
                self.scope_stack.append(set())
                for arg in node.args.args:
                    self.scope_stack[-1].add(arg.arg)
                for arg in node.args.kwonlyargs:
                    self.scope_stack[-1].add(arg.arg)
                if node.args.vararg:
                    self.scope_stack[-1].add(node.args.vararg.arg)
                if node.args.kwarg:
                    self.scope_stack[-1].add(node.args.kwarg.arg)
                self.generic_visit(node)
                self.scope_stack.pop()
            
            visit_AsyncFunctionDef = visit_FunctionDef
            
            def visit_ClassDef(self, node):
                defined_names.add(node.name)
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if self.scope_stack:
                        self.scope_stack[-1].add(node.id)
                    defined_names.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                if isinstance(node.ctx, ast.Load):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)
                self.generic_visit(node)
        
        visitor = NameVisitor(self)
        visitor.visit(tree)
        
        builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
        builtins.update(['__name__', '__file__', '__doc__', '__package__', '__spec__', '__annotations__', '__builtins__', '__cached__'])
        
        for name, line in imported_names.items():
            if name not in used_names and name not in defined_names:
                issues.append(AnalysisIssue(
                    rule_id="PY002",
                    message=f"Unused import: '{name}'",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=line),
                    code_snippet=self._get_line(code, line),
                    suggestion=f"Remove the unused import '{name}'",
                    category="static",
                    confidence=0.9
                ))
        
        for name in used_names:
            if (name not in defined_names and 
                name not in imported_names and 
                name not in builtins and
                name not in PYTHON_STDLIB):
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id == name:
                        issues.append(AnalysisIssue(
                            rule_id="PY003",
                            message=f"Possibly undefined name: '{name}'",
                            severity=Severity.ERROR,
                            location=CodeLocation(line=node.lineno, column=node.col_offset),
                            code_snippet=self._get_line(code, node.lineno),
                            suggestion=f"Define or import '{name}' before use",
                            category="static",
                            confidence=0.8
                        ))
                        break
        
        issues.extend(self._check_pep8(code))
        issues.extend(self._check_type_hints(tree))
        issues.extend(self._check_exception_handling(tree, code))
        
        return issues
    
    def _check_pep8(self, code: str) -> List[AnalysisIssue]:
        """Check for common PEP8 violations."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(AnalysisIssue(
                    rule_id="PEP8-E501",
                    message=f"Line too long ({len(line)} > 120 characters)",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line[:80] + "..." if len(line) > 80 else line,
                    suggestion="Break line into multiple lines",
                    category="style"
                ))
            
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(AnalysisIssue(
                    rule_id="PEP8-W291",
                    message="Trailing whitespace",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line,
                    suggestion="Remove trailing whitespace",
                    category="style"
                ))
            
            if '\t' in line and '    ' in line:
                issues.append(AnalysisIssue(
                    rule_id="PEP8-E101",
                    message="Mixed tabs and spaces",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=i),
                    code_snippet=line,
                    suggestion="Use consistent indentation (prefer 4 spaces)",
                    category="style"
                ))
            
            if re.match(r'^class\s+[a-z]', line):
                issues.append(AnalysisIssue(
                    rule_id="PEP8-E742",
                    message="Class name should use CapWords convention",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line,
                    suggestion="Use CapWords convention for class names",
                    category="style"
                ))
            
            if re.match(r'^def\s+[A-Z]', line) and 'test' not in line.lower():
                issues.append(AnalysisIssue(
                    rule_id="PEP8-E743",
                    message="Function name should be lowercase",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line,
                    suggestion="Use snake_case for function names",
                    category="style"
                ))
        
        return issues
    
    def _check_type_hints(self, tree: ast.AST) -> List[AnalysisIssue]:
        """Check for missing type hints."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith('_'):
                    continue
                    
                missing_hints = []
                for arg in node.args.args:
                    if arg.arg != 'self' and arg.arg != 'cls' and not arg.annotation:
                        missing_hints.append(arg.arg)
                
                if not node.returns and not node.name.startswith('__'):
                    issues.append(AnalysisIssue(
                        rule_id="PY-TYPE-001",
                        message=f"Function '{node.name}' missing return type hint",
                        severity=Severity.INFO,
                        location=CodeLocation(line=node.lineno),
                        suggestion=f"Add return type annotation: def {node.name}(...) -> ReturnType:",
                        category="type_hints",
                        confidence=0.7
                    ))
                
                if missing_hints:
                    issues.append(AnalysisIssue(
                        rule_id="PY-TYPE-002",
                        message=f"Function '{node.name}' has parameters without type hints: {', '.join(missing_hints)}",
                        severity=Severity.INFO,
                        location=CodeLocation(line=node.lineno),
                        suggestion="Add type hints to function parameters",
                        category="type_hints",
                        confidence=0.7
                    ))
        
        return issues
    
    def _check_exception_handling(self, tree: ast.AST, code: str) -> List[AnalysisIssue]:
        """Check for exception handling issues."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(AnalysisIssue(
                        rule_id="PY-EXC-001",
                        message="Bare except clause catches all exceptions including KeyboardInterrupt",
                        severity=Severity.WARNING,
                        location=CodeLocation(line=node.lineno),
                        code_snippet=self._get_line(code, node.lineno),
                        suggestion="Use 'except Exception:' or a more specific exception type",
                        category="exception_handling"
                    ))
                elif isinstance(node.type, ast.Name) and node.type.id == 'Exception':
                    has_specific_handling = False
                    for child in ast.walk(node):
                        if isinstance(child, ast.Raise):
                            has_specific_handling = True
                            break
                    if not has_specific_handling:
                        issues.append(AnalysisIssue(
                            rule_id="PY-EXC-002",
                            message="Catching 'Exception' is too broad",
                            severity=Severity.INFO,
                            location=CodeLocation(line=node.lineno),
                            code_snippet=self._get_line(code, node.lineno),
                            suggestion="Consider catching more specific exceptions",
                            category="exception_handling"
                        ))
        
        return issues
    
    def calculate_complexity(self, code: str) -> ComplexityMetrics:
        """Calculate complexity metrics for Python code."""
        metrics = ComplexityMetrics()
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return metrics
        
        lines = code.split('\n')
        metrics.lines_of_code = len(lines)
        metrics.blank_lines = sum(1 for line in lines if not line.strip())
        metrics.comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        metrics.source_lines_of_code = metrics.lines_of_code - metrics.blank_lines - metrics.comment_lines
        
        metrics.cyclomatic_complexity = self._calculate_cyclomatic(tree)
        metrics.cognitive_complexity = self._calculate_cognitive(tree)
        metrics.max_nesting_depth = self._calculate_max_nesting(tree)
        
        function_lengths = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics.function_count += 1
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    length = node.end_lineno - node.lineno + 1
                else:
                    length = self._estimate_function_length(node)
                function_lengths.append(length)
            elif isinstance(node, ast.ClassDef):
                metrics.class_count += 1
        
        if function_lengths:
            metrics.average_function_length = sum(function_lengths) / len(function_lengths)
            metrics.max_function_length = max(function_lengths)
        
        halstead = self._calculate_halstead(tree)
        metrics.halstead_volume = halstead['volume']
        metrics.halstead_difficulty = halstead['difficulty']
        metrics.halstead_effort = halstead['effort']
        
        metrics.maintainability_index = self._calculate_maintainability_index(
            metrics.halstead_volume,
            metrics.cyclomatic_complexity,
            metrics.source_lines_of_code,
            metrics.comment_lines
        )
        
        return metrics
    
    def _calculate_cyclomatic(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                ast.ExceptHandler, ast.With, ast.AsyncWith,
                                ast.Assert)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
                if node.ifs:
                    complexity += len(node.ifs)
            elif isinstance(node, ast.Match):
                complexity += 1
            elif isinstance(node, ast.match_case):
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity."""
        complexity = 0
        
        def walk_with_nesting(node, nesting=0):
            nonlocal complexity
            
            increment = 0
            new_nesting = nesting
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                increment = 1 + nesting
                new_nesting = nesting + 1
            elif isinstance(node, (ast.ExceptHandler,)):
                increment = 1 + nesting
                new_nesting = nesting + 1
            elif isinstance(node, ast.BoolOp):
                increment = len(node.values) - 1
            elif isinstance(node, (ast.Break, ast.Continue)):
                increment = 1
            elif isinstance(node, ast.Lambda):
                increment = 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_nesting = 0
            
            complexity += increment
            
            for child in ast.iter_child_nodes(node):
                walk_with_nesting(child, new_nesting)
        
        walk_with_nesting(tree)
        return complexity
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        
        def walk_with_depth(node, depth=0):
            nonlocal max_depth
            
            current_depth = depth
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                ast.With, ast.AsyncWith, ast.Try,
                                ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                current_depth = depth + 1
                max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(node):
                walk_with_depth(child, current_depth)
        
        walk_with_depth(tree)
        return max_depth
    
    def _estimate_function_length(self, node: ast.AST) -> int:
        """Estimate function length when end_lineno is not available."""
        count = 0
        for _ in ast.walk(node):
            count += 1
        return max(1, count // 2)
    
    def _calculate_halstead(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate Halstead complexity metrics."""
        operators: Dict[str, int] = defaultdict(int)
        operands: Dict[str, int] = defaultdict(int)
        
        operator_types = (
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
            ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
            ast.FloorDiv, ast.And, ast.Or, ast.Not, ast.Invert,
            ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
            ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn
        )
        
        for node in ast.walk(tree):
            if isinstance(node, operator_types):
                operators[type(node).__name__] += 1
            elif isinstance(node, ast.Name):
                operands[node.id] += 1
            elif isinstance(node, ast.Constant):
                operands[str(node.value)] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                operators['def'] += 1
            elif isinstance(node, ast.ClassDef):
                operators['class'] += 1
            elif isinstance(node, ast.If):
                operators['if'] += 1
            elif isinstance(node, ast.For):
                operators['for'] += 1
            elif isinstance(node, ast.While):
                operators['while'] += 1
            elif isinstance(node, ast.Return):
                operators['return'] += 1
        
        n1 = len(operators)  # unique operators
        n2 = len(operands)   # unique operands
        N1 = sum(operators.values())  # total operators
        N2 = sum(operands.values())   # total operands
        
        if n1 == 0 or n2 == 0:
            return {'volume': 0, 'difficulty': 0, 'effort': 0}
        
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        
        return {
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort
        }
    
    def _calculate_maintainability_index(self, volume: float, complexity: int,
                                        sloc: int, comment_lines: int) -> float:
        """Calculate maintainability index (0-100)."""
        if sloc == 0:
            return 100.0
        
        if volume <= 0:
            volume = 1
        
        mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(sloc)
        
        if sloc > 0:
            comment_ratio = comment_lines / sloc
            mi += 50 * math.sin(math.sqrt(2.4 * comment_ratio))
        
        mi = max(0, min(100, mi))
        
        return mi
    
    def analyze_dependencies(self, code: str) -> DependencyGraph:
        """Analyze Python code dependencies."""
        graph = DependencyGraph()
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return graph
        
        used_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    dep_type = self._classify_dependency(alias.name)
                    
                    info = DependencyInfo(
                        name=alias.name,
                        type=dep_type,
                        import_line=node.lineno,
                        alias=alias.asname,
                        imported_names=[]
                    )
                    graph.imports.append(info)
                    
                    name = alias.asname or module_name
                    if name not in used_names:
                        graph.unused_imports.append(name)
                    
                    if dep_type == 'external':
                        graph.external_dependencies.append(alias.name)
                    elif dep_type == 'stdlib':
                        graph.stdlib_dependencies.append(alias.name)
                    else:
                        graph.internal_dependencies.append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    dep_type = self._classify_dependency(node.module)
                    
                    if node.level > 0:
                        dep_type = 'relative'
                    
                    imported_names = [
                        alias.asname or alias.name 
                        for alias in node.names 
                        if alias.name != '*'
                    ]
                    
                    info = DependencyInfo(
                        name=node.module,
                        type=dep_type,
                        import_line=node.lineno,
                        imported_names=imported_names
                    )
                    graph.imports.append(info)
                    
                    for alias in node.names:
                        name = alias.asname or alias.name
                        if name != '*' and name not in used_names:
                            graph.unused_imports.append(name)
                    
                    if dep_type == 'external':
                        graph.external_dependencies.append(node.module)
                    elif dep_type == 'stdlib':
                        graph.stdlib_dependencies.append(node.module)
                    else:
                        graph.internal_dependencies.append(node.module)
        
        graph.external_dependencies = list(set(graph.external_dependencies))
        graph.stdlib_dependencies = list(set(graph.stdlib_dependencies))
        graph.internal_dependencies = list(set(graph.internal_dependencies))
        
        return graph
    
    def _classify_dependency(self, module_name: str) -> str:
        """Classify a dependency as stdlib, external, or internal."""
        base_module = module_name.split('.')[0]
        
        if base_module in PYTHON_STDLIB:
            return 'stdlib'
        
        common_external = {
            'numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn', 'tensorflow',
            'torch', 'keras', 'flask', 'django', 'fastapi', 'requests', 'aiohttp',
            'sqlalchemy', 'celery', 'redis', 'pymongo', 'psycopg2', 'boto3',
            'pytest', 'nose', 'mock', 'black', 'flake8', 'mypy', 'pylint',
            'pydantic', 'attrs', 'click', 'typer', 'rich', 'httpx', 'starlette'
        }
        
        if base_module in common_external:
            return 'external'
        
        if base_module.startswith('_'):
            return 'internal'
        
        return 'external'
    
    def _get_line(self, code: str, lineno: int) -> str:
        """Get a specific line from code."""
        lines = code.split('\n')
        if 1 <= lineno <= len(lines):
            return lines[lineno - 1]
        return ""


class JavaScriptAnalyzer(BaseLanguageAnalyzer):
    """JavaScript/TypeScript code analyzer using regex patterns."""
    
    def __init__(self, is_typescript: bool = False):
        self.is_typescript = is_typescript
    
    def parse(self, code: str) -> str:
        """Return code as-is (no AST parsing for JS without external deps)."""
        return code
    
    def analyze_static(self, code: str) -> List[AnalysisIssue]:
        """Perform static analysis on JavaScript/TypeScript code."""
        issues: List[AnalysisIssue] = []
        lines = code.split('\n')
        
        issues.extend(self._check_var_usage(code, lines))
        issues.extend(self._check_js_patterns(code, lines))
        
        if self.is_typescript:
            issues.extend(self._check_typescript_patterns(code, lines))
        
        return issues
    
    def _check_var_usage(self, code: str, lines: List[str]) -> List[AnalysisIssue]:
        """Check for var usage and suggest let/const."""
        issues = []
        
        for i, line in enumerate(lines, 1):
            if re.search(r'\bvar\s+\w+', line):
                issues.append(AnalysisIssue(
                    rule_id="JS001",
                    message="Use 'let' or 'const' instead of 'var'",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Replace 'var' with 'const' for values that won't be reassigned, or 'let' otherwise",
                    category="style"
                ))
            
            if re.search(r'==(?!=)', line) and not re.search(r'===', line):
                issues.append(AnalysisIssue(
                    rule_id="JS002",
                    message="Use strict equality (===) instead of loose equality (==)",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Replace '==' with '===' for type-safe comparisons",
                    category="style"
                ))
        
        return issues
    
    def _check_js_patterns(self, code: str, lines: List[str]) -> List[AnalysisIssue]:
        """Check for common JavaScript antipatterns."""
        issues = []
        
        for i, line in enumerate(lines, 1):
            if re.search(r'\.then\s*\(\s*function', line):
                issues.append(AnalysisIssue(
                    rule_id="JS003",
                    message="Consider using async/await instead of .then() chains",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Refactor to use async/await for cleaner async code",
                    category="style"
                ))
            
            if re.search(r'console\.(log|warn|error|info|debug)\s*\(', line):
                issues.append(AnalysisIssue(
                    rule_id="JS004",
                    message="Console statement detected - remove before production",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Remove or replace with proper logging",
                    category="debug"
                ))
            
            if re.search(r'\bfor\s*\(\s*\w+\s+in\s+', line):
                issues.append(AnalysisIssue(
                    rule_id="JS005",
                    message="for...in iterates over enumerable properties, use for...of for arrays",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Use for...of, forEach, or a traditional for loop for arrays",
                    category="correctness"
                ))
            
            if re.search(r'new\s+Array\s*\(', line):
                issues.append(AnalysisIssue(
                    rule_id="JS006",
                    message="Use array literal [] instead of new Array()",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Replace new Array() with []",
                    category="style"
                ))
            
            if re.search(r'new\s+Object\s*\(', line):
                issues.append(AnalysisIssue(
                    rule_id="JS007",
                    message="Use object literal {} instead of new Object()",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Replace new Object() with {}",
                    category="style"
                ))
        
        return issues
    
    def _check_typescript_patterns(self, code: str, lines: List[str]) -> List[AnalysisIssue]:
        """Check for TypeScript-specific patterns."""
        issues = []
        
        for i, line in enumerate(lines, 1):
            if re.search(r':\s*any\b', line):
                issues.append(AnalysisIssue(
                    rule_id="TS001",
                    message="Avoid using 'any' type - it disables type checking",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Use a more specific type or 'unknown' if type is truly unknown",
                    category="type_safety"
                ))
            
            if re.search(r'as\s+any\b', line):
                issues.append(AnalysisIssue(
                    rule_id="TS002",
                    message="Type assertion to 'any' bypasses type safety",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Use a more specific type assertion or fix the underlying type issue",
                    category="type_safety"
                ))
            
            if re.search(r'!\s*\.', line) or re.search(r'!\s*\[', line):
                issues.append(AnalysisIssue(
                    rule_id="TS003",
                    message="Non-null assertion (!) can lead to runtime errors",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Consider using optional chaining (?.) or proper null checks",
                    category="type_safety"
                ))
            
            if re.search(r'@ts-ignore', line):
                issues.append(AnalysisIssue(
                    rule_id="TS004",
                    message="@ts-ignore suppresses TypeScript errors",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Fix the underlying type error instead of ignoring it",
                    category="type_safety"
                ))
        
        return issues
    
    def calculate_complexity(self, code: str) -> ComplexityMetrics:
        """Calculate complexity metrics for JavaScript code."""
        metrics = ComplexityMetrics()
        lines = code.split('\n')
        
        metrics.lines_of_code = len(lines)
        metrics.blank_lines = sum(1 for line in lines if not line.strip())
        
        comment_pattern = r'(//.*$|/\*[\s\S]*?\*/)'
        comments = re.findall(comment_pattern, code, re.MULTILINE)
        metrics.comment_lines = sum(c.count('\n') + 1 for c in comments)
        
        metrics.source_lines_of_code = metrics.lines_of_code - metrics.blank_lines - metrics.comment_lines
        
        complexity_patterns = [
            r'\bif\b', r'\belse\s+if\b', r'\bwhile\b', r'\bfor\b',
            r'\bcase\b', r'\bcatch\b', r'\?\s*.*\s*:', r'\&\&', r'\|\|',
            r'\?\?', r'\.filter\b', r'\.map\b', r'\.reduce\b', r'\.forEach\b'
        ]
        
        complexity = 1
        for pattern in complexity_patterns:
            complexity += len(re.findall(pattern, code))
        
        metrics.cyclomatic_complexity = complexity
        
        max_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                spaces = indent // 2 if '\t' not in line[:indent] else indent * 2
                max_indent = max(max_indent, spaces // 2)
        
        metrics.max_nesting_depth = max_indent
        
        function_pattern = r'(function\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|(?:async\s+)?function\s*\()'
        functions = re.findall(function_pattern, code)
        metrics.function_count = len(functions)
        
        class_pattern = r'\bclass\s+\w+'
        metrics.class_count = len(re.findall(class_pattern, code))
        
        if metrics.function_count > 0:
            metrics.average_function_length = metrics.source_lines_of_code / metrics.function_count
        
        metrics.maintainability_index = self._estimate_maintainability(metrics)
        
        return metrics
    
    def _estimate_maintainability(self, metrics: ComplexityMetrics) -> float:
        """Estimate maintainability index for JavaScript."""
        mi = 100.0
        
        if metrics.cyclomatic_complexity > 20:
            mi -= 20
        elif metrics.cyclomatic_complexity > 10:
            mi -= 10
        
        if metrics.max_nesting_depth > 5:
            mi -= 15
        elif metrics.max_nesting_depth > 3:
            mi -= 5
        
        if metrics.source_lines_of_code > 500:
            mi -= 15
        elif metrics.source_lines_of_code > 200:
            mi -= 5
        
        if metrics.source_lines_of_code > 0:
            comment_ratio = metrics.comment_lines / metrics.source_lines_of_code
            if comment_ratio > 0.1:
                mi += 5
        
        return max(0, min(100, mi))
    
    def analyze_dependencies(self, code: str) -> DependencyGraph:
        """Analyze JavaScript/TypeScript code dependencies."""
        graph = DependencyGraph()
        lines = code.split('\n')
        
        import_patterns = [
            r"import\s+(?:(?:\{[^}]+\}|\*\s+as\s+\w+|\w+)\s*,?\s*)+\s+from\s+['\"]([^'\"]+)['\"]",
            r"import\s+['\"]([^'\"]+)['\"]",
            r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in import_patterns:
                matches = re.findall(pattern, line)
                for module in matches:
                    dep_type = self._classify_js_dependency(module)
                    
                    info = DependencyInfo(
                        name=module,
                        type=dep_type,
                        import_line=i,
                    )
                    graph.imports.append(info)
                    
                    if dep_type == 'external':
                        graph.external_dependencies.append(module)
                    elif dep_type == 'internal':
                        graph.internal_dependencies.append(module)
        
        graph.external_dependencies = list(set(graph.external_dependencies))
        graph.internal_dependencies = list(set(graph.internal_dependencies))
        
        return graph
    
    def _classify_js_dependency(self, module: str) -> str:
        """Classify a JavaScript dependency."""
        if module.startswith('.') or module.startswith('/'):
            return 'internal'
        if module.startswith('@'):
            return 'external'
        
        node_builtins = {
            'fs', 'path', 'http', 'https', 'url', 'querystring', 'stream',
            'util', 'events', 'buffer', 'crypto', 'os', 'child_process',
            'cluster', 'dgram', 'dns', 'net', 'readline', 'tls', 'zlib',
            'assert', 'console', 'process', 'timers', 'module', 'v8'
        }
        
        if module in node_builtins or module.startswith('node:'):
            return 'stdlib'
        
        return 'external'


class GenericAnalyzer(BaseLanguageAnalyzer):
    """Generic analyzer for languages without specific support."""
    
    def __init__(self, language: str = "unknown"):
        self.language = language
    
    def parse(self, code: str) -> str:
        return code
    
    def analyze_static(self, code: str) -> List[AnalysisIssue]:
        """Basic static analysis using patterns."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(AnalysisIssue(
                    rule_id="GEN001",
                    message=f"Line too long ({len(line)} characters)",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line[:80] + "...",
                    suggestion="Consider breaking this line",
                    category="style"
                ))
            
            if line.rstrip() != line:
                issues.append(AnalysisIssue(
                    rule_id="GEN002",
                    message="Trailing whitespace",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line,
                    suggestion="Remove trailing whitespace",
                    category="style"
                ))
        
        return issues
    
    def calculate_complexity(self, code: str) -> ComplexityMetrics:
        """Calculate basic complexity metrics."""
        metrics = ComplexityMetrics()
        lines = code.split('\n')
        
        metrics.lines_of_code = len(lines)
        metrics.blank_lines = sum(1 for line in lines if not line.strip())
        
        comment_indicators = ['//', '#', '--', '/*', '*/', '"""', "'''", ';']
        comment_count = 0
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(c) for c in comment_indicators):
                comment_count += 1
        
        metrics.comment_lines = comment_count
        metrics.source_lines_of_code = metrics.lines_of_code - metrics.blank_lines - metrics.comment_lines
        
        control_patterns = [
            r'\bif\b', r'\belse\b', r'\bwhile\b', r'\bfor\b',
            r'\bswitch\b', r'\bcase\b', r'\bcatch\b', r'\btry\b'
        ]
        
        complexity = 1
        for pattern in control_patterns:
            complexity += len(re.findall(pattern, code, re.IGNORECASE))
        
        metrics.cyclomatic_complexity = complexity
        
        return metrics
    
    def analyze_dependencies(self, code: str) -> DependencyGraph:
        """Basic dependency detection."""
        graph = DependencyGraph()
        
        import_patterns = [
            r'import\s+[\w.]+',
            r'from\s+[\w.]+\s+import',
            r'require\s*\([\'"][^\'"]+[\'"]\)',
            r'#include\s*[<"][^>"]+[>"]',
            r'use\s+[\w:]+',
        ]
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern in import_patterns:
                if re.search(pattern, line):
                    graph.imports.append(DependencyInfo(
                        name=line.strip(),
                        type='unknown',
                        import_line=i
                    ))
        
        return graph


class CodeAnalyzer:
    """
    Main code analyzer class providing comprehensive analysis capabilities.
    
    Supports:
    - Python (with AST parsing)
    - JavaScript/TypeScript (with pattern matching)
    - Generic analysis for other languages
    """
    
    SUPPORTED_LANGUAGES = {
        'python': PythonAnalyzer,
        'javascript': lambda: JavaScriptAnalyzer(is_typescript=False),
        'typescript': lambda: JavaScriptAnalyzer(is_typescript=True),
        'js': lambda: JavaScriptAnalyzer(is_typescript=False),
        'ts': lambda: JavaScriptAnalyzer(is_typescript=True),
        'jsx': lambda: JavaScriptAnalyzer(is_typescript=False),
        'tsx': lambda: JavaScriptAnalyzer(is_typescript=True),
    }
    
    def __init__(self, language: str = "auto"):
        self.language = language.lower()
        self.analyzer = self._get_analyzer()
    
    def _get_analyzer(self) -> BaseLanguageAnalyzer:
        """Get the appropriate language analyzer."""
        if self.language in self.SUPPORTED_LANGUAGES:
            analyzer_class = self.SUPPORTED_LANGUAGES[self.language]
            if callable(analyzer_class) and not isinstance(analyzer_class, type):
                return analyzer_class()
            return analyzer_class()
        return GenericAnalyzer(self.language)
    
    def _detect_language(self, code: str) -> str:
        """Auto-detect programming language from code patterns."""
        patterns = {
            'python': [r'^import\s+\w+', r'^from\s+\w+\s+import', r'^def\s+\w+\s*\(', r'^class\s+\w+(\s*\(|\s*:)'],
            'javascript': [r'^const\s+\w+\s*=', r'^let\s+\w+\s*=', r'^function\s+\w+\s*\(', r'=>', r'require\s*\('],
            'typescript': [r':\s*(string|number|boolean|any|void)\b', r'^interface\s+\w+', r'^type\s+\w+\s*=', r'<[A-Z]\w*>'],
            'java': [r'^public\s+class', r'^private\s+', r'^package\s+[\w.]+;', r'System\.out\.print'],
            'cpp': [r'^#include\s*<', r'^using\s+namespace', r'std::', r'^int\s+main\s*\('],
            'c': [r'^#include\s*<', r'^int\s+main\s*\(', r'printf\s*\(', r'malloc\s*\('],
            'rust': [r'^fn\s+\w+', r'^let\s+mut', r'^impl\s+', r'^use\s+\w+::', r'->'],
            'go': [r'^package\s+\w+', r'^func\s+', r'^import\s+\(', r':='],
            'ruby': [r'^def\s+\w+', r'^class\s+\w+\s*<', r'^module\s+', r'\.each\s+do'],
            'php': [r'<\?php', r'^\$\w+\s*=', r'^function\s+\w+', r'->'],
        }
        
        scores: Dict[str, int] = defaultdict(int)
        
        for lang, lang_patterns in patterns.items():
            for pattern in lang_patterns:
                if re.search(pattern, code, re.MULTILINE):
                    scores[lang] += 1
        
        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        
        return 'unknown'
    
    def analyze_static(self, code: str) -> List[AnalysisIssue]:
        """
        Perform static analysis on code.
        
        Detects:
        - Unused variables and imports
        - Undefined references
        - Syntax issues
        - Style violations
        """
        if self.language == 'auto':
            self.language = self._detect_language(code)
            self.analyzer = self._get_analyzer()
        
        return self.analyzer.analyze_static(code)
    
    def analyze_complexity(self, code: str) -> ComplexityMetrics:
        """
        Calculate complexity metrics.
        
        Returns:
        - Cyclomatic complexity
        - Cognitive complexity
        - Lines of code metrics
        - Nesting depth
        - Maintainability index
        """
        if self.language == 'auto':
            self.language = self._detect_language(code)
            self.analyzer = self._get_analyzer()
        
        return self.analyzer.calculate_complexity(code)
    
    def detect_vulnerabilities(self, code: str) -> List[AnalysisIssue]:
        """
        Detect security vulnerabilities.
        
        Checks for:
        - SQL injection
        - XSS vulnerabilities
        - Command injection
        - Path traversal
        - Hardcoded secrets
        - Insecure deserialization
        - CSRF issues
        - Buffer overflow patterns
        - Weak cryptography
        """
        issues: List[AnalysisIssue] = []
        lines = code.split('\n')
        
        vulnerability_checks = [
            (VulnerabilityPatterns.SQL_INJECTION, "sql_injection", Severity.CRITICAL),
            (VulnerabilityPatterns.XSS, "xss", Severity.CRITICAL),
            (VulnerabilityPatterns.COMMAND_INJECTION, "command_injection", Severity.CRITICAL),
            (VulnerabilityPatterns.PATH_TRAVERSAL, "path_traversal", Severity.ERROR),
            (VulnerabilityPatterns.HARDCODED_SECRETS, "hardcoded_secret", Severity.ERROR),
            (VulnerabilityPatterns.INSECURE_DESERIALIZATION, "insecure_deserialization", Severity.CRITICAL),
            (VulnerabilityPatterns.CSRF, "csrf", Severity.ERROR),
            (VulnerabilityPatterns.BUFFER_OVERFLOW, "buffer_overflow", Severity.CRITICAL),
            (VulnerabilityPatterns.WEAK_CRYPTO, "weak_crypto", Severity.WARNING),
        ]
        
        for patterns, category, severity in vulnerability_checks:
            for pattern, description in patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(AnalysisIssue(
                            rule_id=f"SEC-{category.upper()[:3]}",
                            message=description,
                            severity=severity,
                            location=CodeLocation(line=i),
                            code_snippet=line.strip()[:100],
                            suggestion=self._get_security_suggestion(category),
                            category="security",
                            confidence=0.85
                        ))
        
        return issues
    
    def _get_security_suggestion(self, category: str) -> str:
        """Get remediation suggestion for security issue."""
        suggestions = {
            "sql_injection": "Use parameterized queries or an ORM",
            "xss": "Sanitize user input and use proper escaping/encoding",
            "command_injection": "Avoid shell=True, use subprocess with list arguments",
            "path_traversal": "Validate and sanitize file paths, use os.path.basename",
            "hardcoded_secret": "Move secrets to environment variables or a secrets manager",
            "insecure_deserialization": "Use safe deserialization methods (yaml.safe_load, json)",
            "csrf": "Enable and properly configure CSRF protection",
            "buffer_overflow": "Use safe string functions with size limits",
            "weak_crypto": "Use strong cryptographic algorithms (SHA-256+, AES)",
        }
        return suggestions.get(category, "Review and fix the security issue")
    
    def detect_code_smells(self, code: str) -> List[AnalysisIssue]:
        """
        Detect code smells.
        
        Checks for:
        - Long methods
        - Large classes
        - Duplicate code patterns
        - Dead code
        - Magic numbers
        - Deep nesting
        - Too many parameters
        - God classes/functions
        """
        issues: List[AnalysisIssue] = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern, description in CodeSmellPatterns.MAGIC_NUMBERS:
                if re.search(pattern, line):
                    issues.append(AnalysisIssue(
                        rule_id="SMELL-MAGIC",
                        message=description,
                        severity=Severity.INFO,
                        location=CodeLocation(line=i),
                        code_snippet=line.strip(),
                        suggestion="Extract magic numbers to named constants",
                        category="code_smell"
                    ))
                    break
        
        if re.search(CodeSmellPatterns.LONG_PARAMETER_LIST, code):
            match = re.search(CodeSmellPatterns.LONG_PARAMETER_LIST, code)
            if match:
                line_num = code[:match.start()].count('\n') + 1
                issues.append(AnalysisIssue(
                    rule_id="SMELL-PARAMS",
                    message="Function has too many parameters",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=line_num),
                    code_snippet=match.group()[:80],
                    suggestion="Consider using a parameter object or breaking up the function",
                    category="code_smell"
                ))
        
        for i, line in enumerate(lines, 1):
            if re.match(CodeSmellPatterns.DEEP_NESTING_PATTERN, line):
                issues.append(AnalysisIssue(
                    rule_id="SMELL-NESTING",
                    message="Deeply nested code block",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Use early returns or extract nested logic into separate functions",
                    category="code_smell"
                ))
        
        issues.extend(self._detect_duplicate_code(code))
        issues.extend(self._detect_long_functions(code))
        issues.extend(self._detect_dead_code(code))
        
        return issues
    
    def _detect_duplicate_code(self, code: str) -> List[AnalysisIssue]:
        """Detect potential duplicate code blocks."""
        issues = []
        lines = code.split('\n')
        
        block_hashes: Dict[str, List[int]] = defaultdict(list)
        window_size = 5
        
        for i in range(len(lines) - window_size + 1):
            block = '\n'.join(line.strip() for line in lines[i:i + window_size] if line.strip())
            if len(block) > 50:
                block_hash = hashlib.md5(block.encode()).hexdigest()
                block_hashes[block_hash].append(i + 1)
        
        for hash_val, line_nums in block_hashes.items():
            if len(line_nums) > 1:
                issues.append(AnalysisIssue(
                    rule_id="SMELL-DUP",
                    message=f"Potential duplicate code block found at lines: {line_nums}",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=line_nums[0]),
                    code_snippet=f"Block appears {len(line_nums)} times",
                    suggestion="Extract duplicated code into a reusable function",
                    category="code_smell",
                    confidence=0.7
                ))
        
        return issues
    
    def _detect_long_functions(self, code: str) -> List[AnalysisIssue]:
        """Detect functions that are too long."""
        issues = []
        
        function_pattern = r'(def\s+(\w+)\s*\([^)]*\)|function\s+(\w+)\s*\([^)]*\)|const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)'
        
        for match in re.finditer(function_pattern, code):
            func_name = match.group(2) or match.group(3) or match.group(4) or 'anonymous'
            start_pos = match.start()
            line_start = code[:start_pos].count('\n') + 1
            
            remaining = code[start_pos:]
            brace_count = 0
            paren_count = 0
            in_string = False
            string_char = None
            func_length = 0
            
            for char in remaining:
                if char == '\n':
                    func_length += 1
                if char in '"\'':
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    elif char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
            
            if func_length > 50:
                issues.append(AnalysisIssue(
                    rule_id="SMELL-LONG",
                    message=f"Function '{func_name}' is too long ({func_length} lines)",
                    severity=Severity.WARNING,
                    location=CodeLocation(line=line_start, end_line=line_start + func_length),
                    suggestion=f"Break down '{func_name}' into smaller, focused functions",
                    category="code_smell"
                ))
        
        return issues
    
    def _detect_dead_code(self, code: str) -> List[AnalysisIssue]:
        """Detect potential dead code."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if stripped.startswith('#') and 'TODO' not in stripped and 'FIXME' not in stripped:
                if re.match(r'^#\s*(def|class|if|for|while|return)\s+', stripped):
                    issues.append(AnalysisIssue(
                        rule_id="SMELL-DEAD",
                        message="Commented-out code detected",
                        severity=Severity.INFO,
                        location=CodeLocation(line=i),
                        code_snippet=stripped[:60],
                        suggestion="Remove commented-out code or restore if needed",
                        category="code_smell"
                    ))
            
            if re.match(r'^\s*(return|break|continue)\s', stripped):
                if i < len(lines):
                    next_line = lines[i].strip() if i < len(lines) else ""
                    if next_line and not next_line.startswith(('#', '//', '/*', '*', '"""', "'''")):
                        if not re.match(r'^(except|elif|else|finally|case|default|\})', next_line):
                            issues.append(AnalysisIssue(
                                rule_id="SMELL-UNREACHABLE",
                                message="Potentially unreachable code after return/break/continue",
                                severity=Severity.WARNING,
                                location=CodeLocation(line=i + 1),
                                code_snippet=next_line[:60],
                                suggestion="Remove unreachable code",
                                category="code_smell",
                                confidence=0.6
                            ))
        
        return issues
    
    def analyze_dependencies(self, code: str) -> DependencyGraph:
        """
        Analyze code dependencies.
        
        Returns:
        - Import graph
        - Circular dependency detection
        - External vs internal dependencies
        """
        if self.language == 'auto':
            self.language = self._detect_language(code)
            self.analyzer = self._get_analyzer()
        
        return self.analyzer.analyze_dependencies(code)
    
    def analyze_performance(self, code: str) -> List[AnalysisIssue]:
        """
        Detect performance issues.
        
        Checks for:
        - N+1 query patterns
        - Inefficient loops
        - Memory leak patterns
        - Blocking operations
        - Unnecessary computations
        """
        issues: List[AnalysisIssue] = []
        lines = code.split('\n')
        
        performance_patterns = [
            (r'for\s+.*\s+in\s+.*:\s*\n\s+.*\.get\(|\.query\(|\.execute\(', 
             "Potential N+1 query pattern - queries inside loop", "PERF-N+1", Severity.WARNING),
            (r'for\s+.*\s+in\s+.*:\s*\n\s+.*await\s+', 
             "Await inside loop - consider using asyncio.gather", "PERF-AWAIT", Severity.WARNING),
            (r'\+\s*=\s*["\'].*["\'].*for\s+', 
             "String concatenation in loop - use join() instead", "PERF-STRCAT", Severity.INFO),
            (r'\.append\(.*\)\s*.*for\s+', 
             "Consider using list comprehension instead of append in loop", "PERF-APPEND", Severity.INFO),
            (r'sleep\s*\(\s*0\s*\)', 
             "sleep(0) - consider using asyncio for non-blocking", "PERF-SLEEP", Severity.INFO),
            (r'global\s+\w+', 
             "Global variable usage may cause performance issues", "PERF-GLOBAL", Severity.INFO),
            (r'import\s+.*\*', 
             "Wildcard import loads unnecessary modules", "PERF-IMPORT", Severity.INFO),
        ]
        
        for pattern, message, rule_id, severity in performance_patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(AnalysisIssue(
                    rule_id=rule_id,
                    message=message,
                    severity=severity,
                    location=CodeLocation(line=line_num),
                    code_snippet=match.group()[:80],
                    suggestion=self._get_performance_suggestion(rule_id),
                    category="performance"
                ))
        
        for i, line in enumerate(lines, 1):
            if re.search(r'len\s*\([^)]+\)\s*==\s*0|len\s*\([^)]+\)\s*>\s*0', line):
                issues.append(AnalysisIssue(
                    rule_id="PERF-LEN",
                    message="Use 'if not seq' or 'if seq' instead of len() check",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Replace len(x) == 0 with 'not x', len(x) > 0 with 'x'",
                    category="performance"
                ))
            
            if re.search(r'\.keys\(\)\s*\)', line) and 'in' in line:
                issues.append(AnalysisIssue(
                    rule_id="PERF-KEYS",
                    message="Unnecessary .keys() call - iterate directly over dict",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Replace 'for k in d.keys()' with 'for k in d'",
                    category="performance"
                ))
            
            if re.search(r'range\s*\(\s*len\s*\(', line):
                issues.append(AnalysisIssue(
                    rule_id="PERF-RANGE",
                    message="Use enumerate() instead of range(len())",
                    severity=Severity.INFO,
                    location=CodeLocation(line=i),
                    code_snippet=line.strip(),
                    suggestion="Replace 'for i in range(len(x))' with 'for i, item in enumerate(x)'",
                    category="performance"
                ))
        
        issues.extend(self._detect_memory_leaks(code, lines))
        
        return issues
    
    def _get_performance_suggestion(self, rule_id: str) -> str:
        """Get performance improvement suggestion."""
        suggestions = {
            "PERF-N+1": "Fetch all required data in a single query before the loop",
            "PERF-AWAIT": "Collect coroutines and use asyncio.gather() for concurrent execution",
            "PERF-STRCAT": "Collect strings in a list and use ''.join() at the end",
            "PERF-APPEND": "Use a list comprehension: [expr for item in iterable]",
            "PERF-SLEEP": "Use asyncio.sleep() for async code or reconsider the need for sleep",
            "PERF-GLOBAL": "Pass values as function parameters instead of using globals",
            "PERF-IMPORT": "Import only the specific names you need",
        }
        return suggestions.get(rule_id, "Optimize the identified pattern")
    
    def _detect_memory_leaks(self, code: str, lines: List[str]) -> List[AnalysisIssue]:
        """Detect potential memory leak patterns."""
        issues = []
        
        leak_patterns = [
            (r'while\s+True\s*:', "Infinite loop may cause memory buildup if not managed"),
            (r'(\w+)\s*=\s*\[\s*\]\s*\n.*while.*\n.*\1\.append', "Growing list in infinite loop"),
            (r'cache\s*=\s*\{\}(?!.*@lru_cache)', "Unbounded cache dictionary"),
            (r'global\s+\w+.*=\s*\[\]', "Global mutable list may accumulate data"),
            (r'setInterval\s*\(', "setInterval without clearInterval may cause memory leak"),
            (r'addEventListener\s*\((?!.*removeEventListener)', "Event listener without removal"),
        ]
        
        for pattern, message in leak_patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                line_num = code[:match.start()].count('\n') + 1
                issues.append(AnalysisIssue(
                    rule_id="PERF-LEAK",
                    message=message,
                    severity=Severity.WARNING,
                    location=CodeLocation(line=line_num),
                    code_snippet=match.group()[:60],
                    suggestion="Ensure proper cleanup and bounds for data structures",
                    category="performance",
                    confidence=0.6
                ))
        
        return issues
    
    def analyze(self, code: str) -> AnalysisReport:
        """
        Perform comprehensive code analysis.
        
        Combines all analysis methods and returns a complete report.
        """
        if self.language == 'auto':
            self.language = self._detect_language(code)
            self.analyzer = self._get_analyzer()
        
        report = AnalysisReport(language=self.language)
        
        report.issues.extend(self.analyze_static(code))
        report.complexity = self.analyze_complexity(code)
        report.issues.extend(self.detect_vulnerabilities(code))
        report.issues.extend(self.detect_code_smells(code))
        report.dependencies = self.analyze_dependencies(code)
        report.issues.extend(self.analyze_performance(code))
        
        report.calculate_score()
        report.generate_summary()
        report.generate_recommendations()
        
        report.metadata = {
            "analyzer_version": "1.0.0",
            "language_detected": self.language,
            "total_issues": len(report.issues),
        }
        
        return report


def analyze_code(code: str, language: str = "auto") -> AnalysisReport:
    """
    Convenience function for quick code analysis.
    
    Args:
        code: Source code string to analyze
        language: Programming language (auto-detected if not specified)
    
    Returns:
        AnalysisReport with comprehensive findings
    
    Example:
        >>> report = analyze_code('''
        ... def example():
        ...     x = 1
        ...     return x
        ... ''', language='python')
        >>> print(report.score)
        95.5
        >>> print(report.grade)
        A
    """
    analyzer = CodeAnalyzer(language=language)
    return analyzer.analyze(code)


def analyze_file(filepath: str, language: str = "auto") -> AnalysisReport:
    """
    Analyze code from a file.
    
    Args:
        filepath: Path to the source file
        language: Programming language (auto-detected from extension if not specified)
    
    Returns:
        AnalysisReport with comprehensive findings
    """
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
    }
    
    if language == "auto":
        import os
        ext = os.path.splitext(filepath)[1].lower()
        language = extension_map.get(ext, "auto")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    
    analyzer = CodeAnalyzer(language=language)
    report = analyzer.analyze(code)
    report.metadata['filepath'] = filepath
    
    return report


if __name__ == "__main__":
    sample_python = '''
import os
import json
import unused_module

def process_data(data, config, options, settings, params, extra):
    password = "hardcoded123"
    
    for item in data:
        result = db.query("SELECT * FROM users WHERE id = " + str(item.id))
        
        if result:
            if result.active:
                if result.verified:
                    if result.premium:
                        process(result)
    
    exec(user_input)
    
    return 42

class VeryLongClassName:
    pass
'''
    
    print("=" * 60)
    print("Code Analyzer Demo")
    print("=" * 60)
    
    report = analyze_code(sample_python, language='python')
    
    print(f"\nLanguage: {report.language}")
    print(f"Score: {report.score:.1f}/100")
    print(f"Grade: {report.grade}")
    print(f"\nSummary: {report.summary}")
    
    print(f"\nComplexity Metrics:")
    print(f"  Cyclomatic Complexity: {report.complexity.cyclomatic_complexity}")
    print(f"  Cognitive Complexity: {report.complexity.cognitive_complexity}")
    print(f"  Max Nesting Depth: {report.complexity.max_nesting_depth}")
    print(f"  Maintainability Index: {report.complexity.maintainability_index:.1f}")
    
    print(f"\nIssues by Severity:")
    for severity in Severity:
        issues = report.get_issues_by_severity(severity)
        if issues:
            print(f"  {severity.value.upper()}: {len(issues)}")
    
    print(f"\nTop Issues:")
    sorted_issues = sorted(report.issues, key=lambda x: x.severity, reverse=True)
    for issue in sorted_issues[:5]:
        print(f"  [{issue.severity.value.upper()}] Line {issue.location.line}: {issue.message}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report.recommendations[:5], 1):
        print(f"  {i}. {rec}")
