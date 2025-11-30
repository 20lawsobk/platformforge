"""
Comprehensive App Testing System for Platform Forge

This module provides browser automation testing capabilities matching Replit Agent's
testing features. It includes browser control, test execution, screenshot capture,
video recording, accessibility testing, visual regression testing, and auto-fixing.

Key Components:
- BrowserController: Abstract browser automation interface (Playwright/Puppeteer-like API)
- TestSession: Represents a testing session with steps and results
- TestStep: Individual test action
- TestRunner: Orchestrates test execution
- TestReporter: Generates test reports
- VisualTester: Screenshot comparison for visual regression
- IssueDetector: Detects common issues (broken links, console errors, accessibility)
- AutoFixer: Suggests or auto-fixes detected issues

Usage:
    from server.ai_model.app_testing import (
        BrowserController,
        TestRunner,
        TestSession,
        TestStep,
        IssueDetector,
        AutoFixer,
        run_test_suite,
        detect_issues,
        generate_report,
    )
    
    # Create a test session
    session = TestSession(
        name="Login Flow Test",
        base_url="http://localhost:5000"
    )
    
    # Add test steps
    session.add_step(TestStep(
        action=ActionType.NAVIGATE,
        target="/login",
        description="Navigate to login page"
    ))
    session.add_step(TestStep(
        action=ActionType.TYPE,
        selector="#email",
        value="test@example.com"
    ))
    session.add_step(TestStep(
        action=ActionType.CLICK,
        selector="#submit-btn"
    ))
    
    # Run the test
    runner = TestRunner()
    result = await runner.run_session(session)
    
    # Generate report
    report = generate_report(result)
    print(report.summary)
    
    # Detect and fix issues
    issues = detect_issues(session)
    fixer = AutoFixer()
    for issue in issues:
        fix = fixer.suggest_fix(issue)
        print(f"Issue: {issue.title}, Fix: {fix.description}")
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from difflib import SequenceMatcher


class ActionType(Enum):
    """Types of browser actions."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    HOVER = "hover"
    SCROLL = "scroll"
    WAIT = "wait"
    WAIT_FOR = "wait_for"
    WAIT_FOR_SELECTOR = "wait_for_selector"
    WAIT_FOR_NAVIGATION = "wait_for_navigation"
    WAIT_FOR_NETWORK_IDLE = "wait_for_network_idle"
    SCREENSHOT = "screenshot"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    FOCUS = "focus"
    BLUR = "blur"
    PRESS = "press"
    FILL = "fill"
    CLEAR = "clear"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    DRAG_DROP = "drag_drop"
    EVALUATE = "evaluate"
    ASSERT = "assert"
    CUSTOM = "custom"


class TestStatus(Enum):
    """Status of a test or test step."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERRORED = "errored"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class IssueType(Enum):
    """Types of issues that can be detected."""
    BROKEN_LINK = "broken_link"
    CONSOLE_ERROR = "console_error"
    CONSOLE_WARNING = "console_warning"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SEO = "seo"
    NETWORK_ERROR = "network_error"
    MISSING_ELEMENT = "missing_element"
    SLOW_LOADING = "slow_loading"
    MEMORY_LEAK = "memory_leak"
    LAYOUT_SHIFT = "layout_shift"
    UNHANDLED_ERROR = "unhandled_error"
    MISSING_ALT_TEXT = "missing_alt_text"
    LOW_CONTRAST = "low_contrast"
    KEYBOARD_TRAP = "keyboard_trap"
    MISSING_LABEL = "missing_label"
    INVALID_HTML = "invalid_html"


class IssueSeverity(Enum):
    """Severity levels for detected issues."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    @property
    def weight(self) -> int:
        weights = {
            IssueSeverity.INFO: 1,
            IssueSeverity.LOW: 2,
            IssueSeverity.MEDIUM: 5,
            IssueSeverity.HIGH: 10,
            IssueSeverity.CRITICAL: 20,
        }
        return weights[self]


class AssertionType(Enum):
    """Types of assertions for testing."""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    TEXT_EQUALS = "text_equals"
    TEXT_CONTAINS = "text_contains"
    TEXT_MATCHES = "text_matches"
    VALUE_EQUALS = "value_equals"
    ATTRIBUTE_EQUALS = "attribute_equals"
    CSS_PROPERTY_EQUALS = "css_property_equals"
    URL_EQUALS = "url_equals"
    URL_CONTAINS = "url_contains"
    TITLE_EQUALS = "title_equals"
    ELEMENT_COUNT = "element_count"
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    ENABLED = "enabled"
    DISABLED = "disabled"
    FOCUSED = "focused"
    SCREENSHOT_MATCH = "screenshot_match"


class BrowserType(Enum):
    """Supported browser types."""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    CHROME = "chrome"
    EDGE = "edge"


class RecordingFormat(Enum):
    """Video recording formats."""
    WEBM = "webm"
    MP4 = "mp4"
    GIF = "gif"


class ReportFormat(Enum):
    """Test report output formats."""
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    JUNIT = "junit"
    TAP = "tap"


class WaitCondition(Enum):
    """Conditions for waiting."""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    ATTACHED = "attached"
    DETACHED = "detached"
    STABLE = "stable"
    ENABLED = "enabled"
    DISABLED = "disabled"


TIMEOUT_DEFAULTS = {
    "navigation": 30000,
    "action": 10000,
    "assertion": 5000,
    "screenshot": 5000,
    "network_idle": 30000,
    "element": 10000,
}


ACCESSIBILITY_RULES = {
    "alt_text": {
        "selector": "img:not([alt]), img[alt='']",
        "description": "Images must have alt text",
        "severity": IssueSeverity.HIGH,
        "wcag": "1.1.1",
    },
    "form_labels": {
        "selector": "input:not([aria-label]):not([aria-labelledby]):not([id])",
        "description": "Form inputs must have associated labels",
        "severity": IssueSeverity.HIGH,
        "wcag": "1.3.1",
    },
    "button_text": {
        "selector": "button:empty:not([aria-label])",
        "description": "Buttons must have accessible text",
        "severity": IssueSeverity.HIGH,
        "wcag": "4.1.2",
    },
    "heading_order": {
        "description": "Heading levels should be in order",
        "severity": IssueSeverity.MEDIUM,
        "wcag": "1.3.1",
    },
    "link_text": {
        "selector": "a:not([aria-label])",
        "description": "Links must have descriptive text",
        "severity": IssueSeverity.MEDIUM,
        "wcag": "2.4.4",
    },
    "contrast_ratio": {
        "description": "Text must have sufficient color contrast",
        "severity": IssueSeverity.HIGH,
        "wcag": "1.4.3",
    },
    "focus_visible": {
        "description": "Focus indicator must be visible",
        "severity": IssueSeverity.MEDIUM,
        "wcag": "2.4.7",
    },
    "keyboard_accessible": {
        "description": "All functionality must be keyboard accessible",
        "severity": IssueSeverity.HIGH,
        "wcag": "2.1.1",
    },
    "aria_valid": {
        "description": "ARIA attributes must be valid",
        "severity": IssueSeverity.HIGH,
        "wcag": "4.1.2",
    },
    "lang_attribute": {
        "selector": "html:not([lang])",
        "description": "Page must have a lang attribute",
        "severity": IssueSeverity.HIGH,
        "wcag": "3.1.1",
    },
}


@dataclass
class ElementLocator:
    """Represents a way to locate an element."""
    selector: str
    strategy: str = "css"
    description: Optional[str] = None
    timeout: int = TIMEOUT_DEFAULTS["element"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selector": self.selector,
            "strategy": self.strategy,
            "description": self.description,
            "timeout": self.timeout,
        }
    
    @classmethod
    def css(cls, selector: str, **kwargs) -> "ElementLocator":
        return cls(selector=selector, strategy="css", **kwargs)
    
    @classmethod
    def xpath(cls, selector: str, **kwargs) -> "ElementLocator":
        return cls(selector=selector, strategy="xpath", **kwargs)
    
    @classmethod
    def text(cls, text: str, exact: bool = True, **kwargs) -> "ElementLocator":
        strategy = "text_exact" if exact else "text_contains"
        return cls(selector=text, strategy=strategy, **kwargs)
    
    @classmethod
    def test_id(cls, test_id: str, **kwargs) -> "ElementLocator":
        return cls(selector=f'[data-testid="{test_id}"]', strategy="css", **kwargs)
    
    @classmethod
    def role(cls, role: str, name: Optional[str] = None, **kwargs) -> "ElementLocator":
        selector = f'[role="{role}"]'
        if name:
            selector = f'[role="{role}"][aria-label="{name}"]'
        return cls(selector=selector, strategy="css", **kwargs)


@dataclass
class BrowserConfig:
    """Configuration for browser instance."""
    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    device_scale_factor: float = 1.0
    is_mobile: bool = False
    has_touch: bool = False
    locale: str = "en-US"
    timezone: str = "America/New_York"
    color_scheme: str = "light"
    user_agent: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    ignore_https_errors: bool = False
    java_script_enabled: bool = True
    storage_state: Optional[str] = None
    proxy: Optional[Dict[str, str]] = None
    slow_mo: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "browser_type": self.browser_type.value,
            "headless": self.headless,
            "viewport": {
                "width": self.viewport_width,
                "height": self.viewport_height,
            },
            "device_scale_factor": self.device_scale_factor,
            "is_mobile": self.is_mobile,
            "has_touch": self.has_touch,
            "locale": self.locale,
            "timezone": self.timezone,
            "color_scheme": self.color_scheme,
            "user_agent": self.user_agent,
            "extra_headers": self.extra_headers,
            "ignore_https_errors": self.ignore_https_errors,
            "java_script_enabled": self.java_script_enabled,
            "storage_state": self.storage_state,
            "proxy": self.proxy,
            "slow_mo": self.slow_mo,
        }


@dataclass
class Screenshot:
    """Represents a captured screenshot."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    data: bytes = b""
    format: str = "png"
    width: int = 0
    height: int = 0
    url: str = ""
    page_title: str = ""
    full_page: bool = False
    clip: Optional[Dict[str, int]] = None
    file_path: Optional[str] = None
    
    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("utf-8")
    
    @property
    def data_url(self) -> str:
        return f"data:image/{self.format};base64,{self.base64}"
    
    @property
    def size_bytes(self) -> int:
        return len(self.data)
    
    def save(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.data)
        self.file_path = path
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "url": self.url,
            "page_title": self.page_title,
            "full_page": self.full_page,
            "clip": self.clip,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
        }


@dataclass
class VideoRecording:
    """Represents a recorded video of a test session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    format: RecordingFormat = RecordingFormat.WEBM
    width: int = 1280
    height: int = 720
    fps: int = 30
    file_path: Optional[str] = None
    file_size: int = 0
    duration_ms: int = 0
    frames: List[bytes] = field(default_factory=list)
    is_recording: bool = False
    
    def start(self) -> None:
        self.is_recording = True
        self.start_time = datetime.now()
    
    def stop(self) -> None:
        self.is_recording = False
        self.end_time = datetime.now()
        if self.start_time and self.end_time:
            self.duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)
    
    def add_frame(self, frame: bytes) -> None:
        if self.is_recording:
            self.frames.append(frame)
    
    def save(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file_path = path
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "format": self.format.value,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "duration_ms": self.duration_ms,
            "frame_count": len(self.frames),
        }


@dataclass
class ConsoleMessage:
    """Represents a browser console message."""
    type: str
    text: str
    url: str = ""
    line_number: int = 0
    column_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    args: List[Any] = field(default_factory=list)
    
    @property
    def is_error(self) -> bool:
        return self.type in ("error", "exception")
    
    @property
    def is_warning(self) -> bool:
        return self.type == "warning"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text,
            "url": self.url,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace,
        }


@dataclass
class NetworkRequest:
    """Represents a network request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resource_type: str = "document"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "method": self.method,
            "headers": self.headers,
            "timestamp": self.timestamp.isoformat(),
            "resource_type": self.resource_type,
        }


@dataclass
class NetworkResponse:
    """Represents a network response."""
    request_id: str = ""
    url: str = ""
    status: int = 0
    status_text: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    timestamp: datetime = field(default_factory=datetime.now)
    timing: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_error(self) -> bool:
        return self.status >= 400
    
    @property
    def is_success(self) -> bool:
        return 200 <= self.status < 300
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "url": self.url,
            "status": self.status,
            "status_text": self.status_text,
            "headers": self.headers,
            "timestamp": self.timestamp.isoformat(),
            "timing": self.timing,
            "is_error": self.is_error,
        }


@dataclass
class PerformanceMetrics:
    """Collected performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    url: str = ""
    time_to_first_byte: float = 0.0
    first_contentful_paint: float = 0.0
    largest_contentful_paint: float = 0.0
    first_input_delay: float = 0.0
    cumulative_layout_shift: float = 0.0
    time_to_interactive: float = 0.0
    total_blocking_time: float = 0.0
    dom_content_loaded: float = 0.0
    load_event: float = 0.0
    js_heap_size: int = 0
    dom_nodes: int = 0
    layout_count: int = 0
    recalc_style_count: int = 0
    script_duration: float = 0.0
    layout_duration: float = 0.0
    recalc_style_duration: float = 0.0
    
    @property
    def core_web_vitals(self) -> Dict[str, float]:
        return {
            "LCP": self.largest_contentful_paint,
            "FID": self.first_input_delay,
            "CLS": self.cumulative_layout_shift,
        }
    
    @property
    def is_healthy(self) -> bool:
        return (
            self.largest_contentful_paint <= 2500
            and self.first_input_delay <= 100
            and self.cumulative_layout_shift <= 0.1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "url": self.url,
            "time_to_first_byte": self.time_to_first_byte,
            "first_contentful_paint": self.first_contentful_paint,
            "largest_contentful_paint": self.largest_contentful_paint,
            "first_input_delay": self.first_input_delay,
            "cumulative_layout_shift": self.cumulative_layout_shift,
            "time_to_interactive": self.time_to_interactive,
            "total_blocking_time": self.total_blocking_time,
            "dom_content_loaded": self.dom_content_loaded,
            "load_event": self.load_event,
            "js_heap_size": self.js_heap_size,
            "dom_nodes": self.dom_nodes,
            "core_web_vitals": self.core_web_vitals,
            "is_healthy": self.is_healthy,
        }


@dataclass
class DetectedIssue:
    """Represents a detected issue."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: IssueType = IssueType.CONSOLE_ERROR
    severity: IssueSeverity = IssueSeverity.MEDIUM
    title: str = ""
    description: str = ""
    url: str = ""
    selector: Optional[str] = None
    element_html: Optional[str] = None
    screenshot: Optional[Screenshot] = None
    timestamp: datetime = field(default_factory=datetime.now)
    wcag_criterion: Optional[str] = None
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "selector": self.selector,
            "element_html": self.element_html,
            "timestamp": self.timestamp.isoformat(),
            "wcag_criterion": self.wcag_criterion,
            "suggested_fix": self.suggested_fix,
            "auto_fixable": self.auto_fixable,
            "context": self.context,
        }


@dataclass
class TestStep:
    """Represents a single test step."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: ActionType = ActionType.NAVIGATE
    target: Optional[str] = None
    selector: Optional[str] = None
    locator: Optional[ElementLocator] = None
    value: Optional[str] = None
    description: Optional[str] = None
    timeout: int = TIMEOUT_DEFAULTS["action"]
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    screenshot_before: Optional[Screenshot] = None
    screenshot_after: Optional[Screenshot] = None
    assertion_type: Optional[AssertionType] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    wait_condition: Optional[WaitCondition] = None
    modifiers: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 0
    
    @property
    def duration_ms(self) -> Optional[int]:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return None
    
    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED
    
    @property
    def failed(self) -> bool:
        return self.status in (TestStatus.FAILED, TestStatus.ERRORED)
    
    def start(self) -> None:
        self.status = TestStatus.RUNNING
        self.start_time = datetime.now()
    
    def complete(self, passed: bool, error: Optional[str] = None) -> None:
        self.status = TestStatus.PASSED if passed else TestStatus.FAILED
        self.end_time = datetime.now()
        if error:
            self.error = error
            self.status = TestStatus.ERRORED
    
    def skip(self, reason: Optional[str] = None) -> None:
        self.status = TestStatus.SKIPPED
        if reason:
            self.error = reason
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action.value,
            "target": self.target,
            "selector": self.selector,
            "locator": self.locator.to_dict() if self.locator else None,
            "value": self.value,
            "description": self.description or self._auto_description(),
            "timeout": self.timeout,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "assertion_type": self.assertion_type.value if self.assertion_type else None,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "retry_count": self.retry_count,
        }
    
    def _auto_description(self) -> str:
        target = self.target or self.selector or ""
        if self.action == ActionType.NAVIGATE:
            return f"Navigate to {target}"
        elif self.action == ActionType.CLICK:
            return f"Click on {target}"
        elif self.action == ActionType.TYPE:
            return f"Type '{self.value}' into {target}"
        elif self.action == ActionType.ASSERT:
            return f"Assert {self.assertion_type.value if self.assertion_type else 'condition'} on {target}"
        elif self.action == ActionType.WAIT:
            return f"Wait {self.value}ms"
        elif self.action == ActionType.WAIT_FOR:
            return f"Wait for {target}"
        elif self.action == ActionType.SCREENSHOT:
            return "Take screenshot"
        else:
            return f"{self.action.value} on {target}"


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_id: str = ""
    name: str = ""
    status: TestStatus = TestStatus.PENDING
    steps: List[TestStep] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    screenshots: List[Screenshot] = field(default_factory=list)
    console_messages: List[ConsoleMessage] = field(default_factory=list)
    network_requests: List[NetworkRequest] = field(default_factory=list)
    network_responses: List[NetworkResponse] = field(default_factory=list)
    performance_metrics: Optional[PerformanceMetrics] = None
    issues: List[DetectedIssue] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> Optional[int]:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return None
    
    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED
    
    @property
    def failed(self) -> bool:
        return self.status in (TestStatus.FAILED, TestStatus.ERRORED)
    
    @property
    def passed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == TestStatus.PASSED)
    
    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status in (TestStatus.FAILED, TestStatus.ERRORED))
    
    @property
    def console_errors(self) -> List[ConsoleMessage]:
        return [m for m in self.console_messages if m.is_error]
    
    @property
    def network_errors(self) -> List[NetworkResponse]:
        return [r for r in self.network_responses if r.is_error]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "steps": [s.to_dict() for s in self.steps],
            "passed_steps": self.passed_steps,
            "failed_steps": self.failed_steps,
            "screenshots": [s.to_dict() for s in self.screenshots],
            "console_errors_count": len(self.console_errors),
            "network_errors_count": len(self.network_errors),
            "issues_count": len(self.issues),
            "performance_metrics": self.performance_metrics.to_dict() if self.performance_metrics else None,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class TestSession:
    """Represents a complete testing session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    base_url: str = ""
    browser_config: BrowserConfig = field(default_factory=BrowserConfig)
    steps: List[TestStep] = field(default_factory=list)
    setup_steps: List[TestStep] = field(default_factory=list)
    teardown_steps: List[TestStep] = field(default_factory=list)
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[TestResult] = None
    video: Optional[VideoRecording] = None
    tags: List[str] = field(default_factory=list)
    timeout: int = 60000
    retry_on_failure: bool = False
    max_retries: int = 2
    parallel: bool = False
    skip_condition: Optional[Callable[[], bool]] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: TestStep) -> "TestSession":
        self.steps.append(step)
        return self
    
    def add_setup_step(self, step: TestStep) -> "TestSession":
        self.setup_steps.append(step)
        return self
    
    def add_teardown_step(self, step: TestStep) -> "TestSession":
        self.teardown_steps.append(step)
        return self
    
    def navigate(self, url: str, **kwargs) -> "TestSession":
        step = TestStep(action=ActionType.NAVIGATE, target=url, **kwargs)
        return self.add_step(step)
    
    def click(self, selector: str, **kwargs) -> "TestSession":
        step = TestStep(action=ActionType.CLICK, selector=selector, **kwargs)
        return self.add_step(step)
    
    def type(self, selector: str, value: str, **kwargs) -> "TestSession":
        step = TestStep(action=ActionType.TYPE, selector=selector, value=value, **kwargs)
        return self.add_step(step)
    
    def fill(self, selector: str, value: str, **kwargs) -> "TestSession":
        step = TestStep(action=ActionType.FILL, selector=selector, value=value, **kwargs)
        return self.add_step(step)
    
    def wait(self, ms: int) -> "TestSession":
        step = TestStep(action=ActionType.WAIT, value=str(ms))
        return self.add_step(step)
    
    def wait_for(self, selector: str, condition: WaitCondition = WaitCondition.VISIBLE, **kwargs) -> "TestSession":
        step = TestStep(
            action=ActionType.WAIT_FOR,
            selector=selector,
            wait_condition=condition,
            **kwargs
        )
        return self.add_step(step)
    
    def screenshot(self, **kwargs) -> "TestSession":
        step = TestStep(action=ActionType.SCREENSHOT, **kwargs)
        return self.add_step(step)
    
    def assert_visible(self, selector: str, **kwargs) -> "TestSession":
        step = TestStep(
            action=ActionType.ASSERT,
            selector=selector,
            assertion_type=AssertionType.VISIBLE,
            **kwargs
        )
        return self.add_step(step)
    
    def assert_text(self, selector: str, expected: str, **kwargs) -> "TestSession":
        step = TestStep(
            action=ActionType.ASSERT,
            selector=selector,
            assertion_type=AssertionType.TEXT_EQUALS,
            expected_value=expected,
            **kwargs
        )
        return self.add_step(step)
    
    def assert_url(self, expected: str, **kwargs) -> "TestSession":
        step = TestStep(
            action=ActionType.ASSERT,
            assertion_type=AssertionType.URL_EQUALS,
            expected_value=expected,
            **kwargs
        )
        return self.add_step(step)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "base_url": self.base_url,
            "browser_config": self.browser_config.to_dict(),
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "step_count": len(self.steps),
            "tags": self.tags,
            "timeout": self.timeout,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
        }


@dataclass
class TestSuite:
    """Collection of test sessions."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    sessions: List[TestSession] = field(default_factory=list)
    global_setup: List[TestStep] = field(default_factory=list)
    global_teardown: List[TestStep] = field(default_factory=list)
    parallel_execution: bool = False
    max_parallel: int = 4
    fail_fast: bool = False
    retry_failed: bool = False
    tags: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def add_session(self, session: TestSession) -> "TestSuite":
        self.sessions.append(session)
        return self
    
    @property
    def total_steps(self) -> int:
        return sum(len(s.steps) for s in self.sessions)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "session_count": len(self.sessions),
            "total_steps": self.total_steps,
            "parallel_execution": self.parallel_execution,
            "max_parallel": self.max_parallel,
            "fail_fast": self.fail_fast,
            "tags": self.tags,
        }


@dataclass
class VisualDiff:
    """Result of visual comparison between screenshots."""
    baseline_screenshot: Screenshot
    current_screenshot: Screenshot
    diff_percentage: float = 0.0
    diff_pixels: int = 0
    diff_image: Optional[bytes] = None
    passed: bool = True
    threshold: float = 0.01
    regions_changed: List[Dict[str, int]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def similarity(self) -> float:
        return 1.0 - self.diff_percentage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_id": self.baseline_screenshot.id,
            "current_id": self.current_screenshot.id,
            "diff_percentage": self.diff_percentage,
            "diff_pixels": self.diff_pixels,
            "passed": self.passed,
            "threshold": self.threshold,
            "similarity": self.similarity,
            "regions_changed": self.regions_changed,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Fix:
    """Represents a suggested or applied fix."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    issue_id: str = ""
    issue_type: IssueType = IssueType.CONSOLE_ERROR
    title: str = ""
    description: str = ""
    code_before: Optional[str] = None
    code_after: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    confidence: float = 0.0
    auto_applicable: bool = False
    applied: bool = False
    applied_at: Optional[datetime] = None
    rollback_possible: bool = True
    
    def apply(self) -> bool:
        if self.auto_applicable and not self.applied:
            self.applied = True
            self.applied_at = datetime.now()
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "issue_id": self.issue_id,
            "issue_type": self.issue_type.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "confidence": self.confidence,
            "auto_applicable": self.auto_applicable,
            "applied": self.applied,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }


@dataclass
class TestReport:
    """Comprehensive test report."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    suite_name: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    results: List[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    total_duration_ms: int = 0
    issues: List[DetectedIssue] = field(default_factory=list)
    performance_summary: Dict[str, float] = field(default_factory=dict)
    accessibility_score: float = 0.0
    visual_regression_results: List[VisualDiff] = field(default_factory=list)
    coverage: Dict[str, float] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100
    
    @property
    def summary(self) -> str:
        status = "PASSED" if self.failed_tests == 0 else "FAILED"
        return (
            f"Test Report: {status}\n"
            f"Total: {self.total_tests} | "
            f"Passed: {self.passed_tests} | "
            f"Failed: {self.failed_tests} | "
            f"Skipped: {self.skipped_tests}\n"
            f"Duration: {self.total_duration_ms}ms | "
            f"Pass Rate: {self.pass_rate:.1f}%\n"
            f"Issues Found: {len(self.issues)} | "
            f"Accessibility Score: {self.accessibility_score:.1f}/100"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "suite_name": self.suite_name,
            "generated_at": self.generated_at.isoformat(),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "pass_rate": self.pass_rate,
            "total_duration_ms": self.total_duration_ms,
            "issues": [i.to_dict() for i in self.issues],
            "issues_by_severity": self._issues_by_severity(),
            "performance_summary": self.performance_summary,
            "accessibility_score": self.accessibility_score,
            "visual_regression_passed": all(v.passed for v in self.visual_regression_results),
            "environment": self.environment,
        }
    
    def _issues_by_severity(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return dict(counts)


class BrowserError(Exception):
    """Base exception for browser-related errors."""
    pass


class ElementNotFoundError(BrowserError):
    """Raised when an element cannot be found."""
    def __init__(self, selector: str, timeout: int = 0):
        self.selector = selector
        self.timeout = timeout
        super().__init__(f"Element not found: {selector} (timeout: {timeout}ms)")


class NavigationError(BrowserError):
    """Raised when navigation fails."""
    def __init__(self, url: str, reason: str = ""):
        self.url = url
        self.reason = reason
        super().__init__(f"Navigation failed to {url}: {reason}")


class TimeoutError(BrowserError):
    """Raised when an operation times out."""
    def __init__(self, operation: str, timeout: int):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Timeout after {timeout}ms: {operation}")


class AssertionError(BrowserError):
    """Raised when an assertion fails."""
    def __init__(self, message: str, expected: Any = None, actual: Any = None):
        self.expected = expected
        self.actual = actual
        super().__init__(f"Assertion failed: {message}")


class BrowserController(ABC):
    """
    Abstract base class for browser automation.
    
    This provides a Playwright/Puppeteer-like API that can be implemented
    by different browser automation backends.
    """
    
    def __init__(self, config: Optional[BrowserConfig] = None):
        self.config = config or BrowserConfig()
        self.is_connected = False
        self.current_url = ""
        self.console_messages: List[ConsoleMessage] = []
        self.network_requests: List[NetworkRequest] = []
        self.network_responses: List[NetworkResponse] = []
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    @abstractmethod
    async def launch(self) -> None:
        """Launch the browser."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the browser."""
        pass
    
    @abstractmethod
    async def new_page(self) -> None:
        """Create a new page/tab."""
        pass
    
    @abstractmethod
    async def navigate(self, url: str, wait_until: str = "load", timeout: int = None) -> None:
        """Navigate to a URL."""
        pass
    
    @abstractmethod
    async def reload(self, wait_until: str = "load") -> None:
        """Reload the current page."""
        pass
    
    @abstractmethod
    async def go_back(self) -> None:
        """Navigate back."""
        pass
    
    @abstractmethod
    async def go_forward(self) -> None:
        """Navigate forward."""
        pass
    
    @abstractmethod
    async def click(
        self,
        selector: str,
        button: str = "left",
        click_count: int = 1,
        modifiers: List[str] = None,
        timeout: int = None
    ) -> None:
        """Click an element."""
        pass
    
    @abstractmethod
    async def double_click(self, selector: str, timeout: int = None) -> None:
        """Double-click an element."""
        pass
    
    @abstractmethod
    async def right_click(self, selector: str, timeout: int = None) -> None:
        """Right-click an element."""
        pass
    
    @abstractmethod
    async def hover(self, selector: str, timeout: int = None) -> None:
        """Hover over an element."""
        pass
    
    @abstractmethod
    async def type(
        self,
        selector: str,
        text: str,
        delay: int = 0,
        timeout: int = None
    ) -> None:
        """Type text into an element."""
        pass
    
    @abstractmethod
    async def fill(self, selector: str, value: str, timeout: int = None) -> None:
        """Fill an input field (clears first)."""
        pass
    
    @abstractmethod
    async def clear(self, selector: str, timeout: int = None) -> None:
        """Clear an input field."""
        pass
    
    @abstractmethod
    async def press(self, selector: str, key: str, timeout: int = None) -> None:
        """Press a key while focused on an element."""
        pass
    
    @abstractmethod
    async def select(
        self,
        selector: str,
        value: Optional[str] = None,
        label: Optional[str] = None,
        index: Optional[int] = None,
        timeout: int = None
    ) -> None:
        """Select an option from a dropdown."""
        pass
    
    @abstractmethod
    async def check(self, selector: str, timeout: int = None) -> None:
        """Check a checkbox."""
        pass
    
    @abstractmethod
    async def uncheck(self, selector: str, timeout: int = None) -> None:
        """Uncheck a checkbox."""
        pass
    
    @abstractmethod
    async def focus(self, selector: str, timeout: int = None) -> None:
        """Focus an element."""
        pass
    
    @abstractmethod
    async def blur(self, selector: str) -> None:
        """Remove focus from an element."""
        pass
    
    @abstractmethod
    async def scroll(
        self,
        selector: Optional[str] = None,
        x: int = 0,
        y: int = 0
    ) -> None:
        """Scroll the page or an element."""
        pass
    
    @abstractmethod
    async def scroll_into_view(self, selector: str, timeout: int = None) -> None:
        """Scroll an element into view."""
        pass
    
    @abstractmethod
    async def upload_file(
        self,
        selector: str,
        file_paths: List[str],
        timeout: int = None
    ) -> None:
        """Upload files to a file input."""
        pass
    
    @abstractmethod
    async def drag_and_drop(
        self,
        source_selector: str,
        target_selector: str,
        timeout: int = None
    ) -> None:
        """Drag and drop from source to target."""
        pass
    
    @abstractmethod
    async def wait_for(
        self,
        selector: str,
        state: WaitCondition = WaitCondition.VISIBLE,
        timeout: int = None
    ) -> None:
        """Wait for an element to reach a state."""
        pass
    
    @abstractmethod
    async def wait_for_navigation(self, timeout: int = None) -> None:
        """Wait for navigation to complete."""
        pass
    
    @abstractmethod
    async def wait_for_network_idle(self, timeout: int = None) -> None:
        """Wait for network to be idle."""
        pass
    
    @abstractmethod
    async def wait(self, ms: int) -> None:
        """Wait for a specified time."""
        pass
    
    @abstractmethod
    async def screenshot(
        self,
        full_page: bool = False,
        clip: Optional[Dict[str, int]] = None,
        selector: Optional[str] = None
    ) -> Screenshot:
        """Take a screenshot."""
        pass
    
    @abstractmethod
    async def evaluate(self, script: str, *args) -> Any:
        """Evaluate JavaScript in the page context."""
        pass
    
    @abstractmethod
    async def get_text(self, selector: str, timeout: int = None) -> str:
        """Get text content of an element."""
        pass
    
    @abstractmethod
    async def get_value(self, selector: str, timeout: int = None) -> str:
        """Get value of an input element."""
        pass
    
    @abstractmethod
    async def get_attribute(
        self,
        selector: str,
        attribute: str,
        timeout: int = None
    ) -> Optional[str]:
        """Get attribute value of an element."""
        pass
    
    @abstractmethod
    async def get_css_property(
        self,
        selector: str,
        property_name: str,
        timeout: int = None
    ) -> str:
        """Get CSS property value of an element."""
        pass
    
    @abstractmethod
    async def is_visible(self, selector: str, timeout: int = None) -> bool:
        """Check if an element is visible."""
        pass
    
    @abstractmethod
    async def is_enabled(self, selector: str, timeout: int = None) -> bool:
        """Check if an element is enabled."""
        pass
    
    @abstractmethod
    async def is_checked(self, selector: str, timeout: int = None) -> bool:
        """Check if a checkbox is checked."""
        pass
    
    @abstractmethod
    async def get_url(self) -> str:
        """Get current URL."""
        pass
    
    @abstractmethod
    async def get_title(self) -> str:
        """Get page title."""
        pass
    
    @abstractmethod
    async def get_html(self, selector: Optional[str] = None) -> str:
        """Get HTML content."""
        pass
    
    @abstractmethod
    async def query_selector(self, selector: str) -> bool:
        """Check if element exists."""
        pass
    
    @abstractmethod
    async def query_selector_all(self, selector: str) -> int:
        """Count matching elements."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        pass
    
    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        self._event_handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable) -> None:
        """Remove an event handler."""
        if handler in self._event_handlers[event]:
            self._event_handlers[event].remove(handler)
    
    def emit(self, event: str, *args, **kwargs) -> None:
        """Emit an event."""
        for handler in self._event_handlers[event]:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logging.warning(f"Event handler error: {e}")


class SimulatedBrowserController(BrowserController):
    """
    A simulated browser controller for testing without a real browser.
    
    This implementation simulates browser behavior and is useful for
    testing the testing framework itself or for dry runs.
    """
    
    def __init__(self, config: Optional[BrowserConfig] = None):
        super().__init__(config)
        self._pages: List[Dict[str, Any]] = []
        self._current_page: Optional[Dict[str, Any]] = None
        self._elements: Dict[str, Dict[str, Any]] = {}
        self._navigation_history: List[str] = []
        self._screenshot_counter = 0
    
    async def launch(self) -> None:
        self.is_connected = True
        self.emit("launched")
    
    async def close(self) -> None:
        self.is_connected = False
        self._pages = []
        self._current_page = None
        self.emit("closed")
    
    async def new_page(self) -> None:
        page = {
            "url": "about:blank",
            "title": "",
            "html": "<html><head></head><body></body></html>",
            "elements": {},
        }
        self._pages.append(page)
        self._current_page = page
        self.emit("page_created")
    
    async def navigate(self, url: str, wait_until: str = "load", timeout: int = None) -> None:
        if not self._current_page:
            await self.new_page()
        
        self._current_page["url"] = url
        self._navigation_history.append(url)
        self.current_url = url
        self.emit("navigated", url)
        
        await asyncio.sleep(0.01)
    
    async def reload(self, wait_until: str = "load") -> None:
        if self._current_page:
            await self.navigate(self._current_page["url"], wait_until)
    
    async def go_back(self) -> None:
        if len(self._navigation_history) > 1:
            self._navigation_history.pop()
            url = self._navigation_history[-1]
            self._current_page["url"] = url
            self.current_url = url
    
    async def go_forward(self) -> None:
        pass
    
    async def click(
        self,
        selector: str,
        button: str = "left",
        click_count: int = 1,
        modifiers: List[str] = None,
        timeout: int = None
    ) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self.emit("clicked", selector)
        await asyncio.sleep(0.01)
    
    async def double_click(self, selector: str, timeout: int = None) -> None:
        await self.click(selector, click_count=2, timeout=timeout)
    
    async def right_click(self, selector: str, timeout: int = None) -> None:
        await self.click(selector, button="right", timeout=timeout)
    
    async def hover(self, selector: str, timeout: int = None) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self.emit("hovered", selector)
    
    async def type(
        self,
        selector: str,
        text: str,
        delay: int = 0,
        timeout: int = None
    ) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self._elements.setdefault(selector, {})["value"] = text
        self.emit("typed", selector, text)
        if delay:
            await asyncio.sleep(delay * len(text) / 1000)
    
    async def fill(self, selector: str, value: str, timeout: int = None) -> None:
        await self.clear(selector, timeout)
        await self.type(selector, value, timeout=timeout)
    
    async def clear(self, selector: str, timeout: int = None) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self._elements.setdefault(selector, {})["value"] = ""
    
    async def press(self, selector: str, key: str, timeout: int = None) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self.emit("pressed", selector, key)
    
    async def select(
        self,
        selector: str,
        value: Optional[str] = None,
        label: Optional[str] = None,
        index: Optional[int] = None,
        timeout: int = None
    ) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self._elements.setdefault(selector, {})["selected"] = value or label or str(index)
        self.emit("selected", selector, value or label or index)
    
    async def check(self, selector: str, timeout: int = None) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self._elements.setdefault(selector, {})["checked"] = True
    
    async def uncheck(self, selector: str, timeout: int = None) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self._elements.setdefault(selector, {})["checked"] = False
    
    async def focus(self, selector: str, timeout: int = None) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self._elements.setdefault(selector, {})["focused"] = True
    
    async def blur(self, selector: str) -> None:
        self._elements.setdefault(selector, {})["focused"] = False
    
    async def scroll(
        self,
        selector: Optional[str] = None,
        x: int = 0,
        y: int = 0
    ) -> None:
        self.emit("scrolled", x, y)
    
    async def scroll_into_view(self, selector: str, timeout: int = None) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
    
    async def upload_file(
        self,
        selector: str,
        file_paths: List[str],
        timeout: int = None
    ) -> None:
        if not await self.query_selector(selector):
            raise ElementNotFoundError(selector, timeout or TIMEOUT_DEFAULTS["element"])
        self._elements.setdefault(selector, {})["files"] = file_paths
    
    async def drag_and_drop(
        self,
        source_selector: str,
        target_selector: str,
        timeout: int = None
    ) -> None:
        if not await self.query_selector(source_selector):
            raise ElementNotFoundError(source_selector, timeout or TIMEOUT_DEFAULTS["element"])
        if not await self.query_selector(target_selector):
            raise ElementNotFoundError(target_selector, timeout or TIMEOUT_DEFAULTS["element"])
        self.emit("drag_drop", source_selector, target_selector)
    
    async def wait_for(
        self,
        selector: str,
        state: WaitCondition = WaitCondition.VISIBLE,
        timeout: int = None
    ) -> None:
        await asyncio.sleep(0.01)
        self._elements.setdefault(selector, {})["visible"] = True
    
    async def wait_for_navigation(self, timeout: int = None) -> None:
        await asyncio.sleep(0.01)
    
    async def wait_for_network_idle(self, timeout: int = None) -> None:
        await asyncio.sleep(0.01)
    
    async def wait(self, ms: int) -> None:
        await asyncio.sleep(ms / 1000)
    
    async def screenshot(
        self,
        full_page: bool = False,
        clip: Optional[Dict[str, int]] = None,
        selector: Optional[str] = None
    ) -> Screenshot:
        self._screenshot_counter += 1
        
        fake_png = b"\x89PNG\r\n\x1a\n" + bytes(100)
        
        screenshot = Screenshot(
            data=fake_png,
            format="png",
            width=self.config.viewport_width,
            height=self.config.viewport_height,
            url=self.current_url,
            page_title=await self.get_title(),
            full_page=full_page,
            clip=clip,
        )
        
        self.emit("screenshot", screenshot)
        return screenshot
    
    async def evaluate(self, script: str, *args) -> Any:
        if "return" in script and "document.title" in script:
            return self._current_page.get("title", "") if self._current_page else ""
        return None
    
    async def get_text(self, selector: str, timeout: int = None) -> str:
        return self._elements.get(selector, {}).get("text", f"Text of {selector}")
    
    async def get_value(self, selector: str, timeout: int = None) -> str:
        return self._elements.get(selector, {}).get("value", "")
    
    async def get_attribute(
        self,
        selector: str,
        attribute: str,
        timeout: int = None
    ) -> Optional[str]:
        return self._elements.get(selector, {}).get(f"attr_{attribute}")
    
    async def get_css_property(
        self,
        selector: str,
        property_name: str,
        timeout: int = None
    ) -> str:
        return self._elements.get(selector, {}).get(f"css_{property_name}", "")
    
    async def is_visible(self, selector: str, timeout: int = None) -> bool:
        return self._elements.get(selector, {}).get("visible", True)
    
    async def is_enabled(self, selector: str, timeout: int = None) -> bool:
        return self._elements.get(selector, {}).get("enabled", True)
    
    async def is_checked(self, selector: str, timeout: int = None) -> bool:
        return self._elements.get(selector, {}).get("checked", False)
    
    async def get_url(self) -> str:
        return self.current_url
    
    async def get_title(self) -> str:
        return self._current_page.get("title", "") if self._current_page else ""
    
    async def get_html(self, selector: Optional[str] = None) -> str:
        if self._current_page:
            return self._current_page.get("html", "<html></html>")
        return "<html></html>"
    
    async def query_selector(self, selector: str) -> bool:
        special_selectors = ["body", "html", "#", ".", "[data-testid"]
        for s in special_selectors:
            if selector.startswith(s) or selector == s:
                return True
        return selector in self._elements or True
    
    async def query_selector_all(self, selector: str) -> int:
        return 1
    
    async def get_performance_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            url=self.current_url,
            time_to_first_byte=50.0,
            first_contentful_paint=200.0,
            largest_contentful_paint=500.0,
            first_input_delay=10.0,
            cumulative_layout_shift=0.05,
            time_to_interactive=800.0,
            total_blocking_time=100.0,
            dom_content_loaded=300.0,
            load_event=600.0,
            js_heap_size=10000000,
            dom_nodes=100,
        )
    
    def set_element(self, selector: str, properties: Dict[str, Any]) -> None:
        """Helper to set element properties for testing."""
        self._elements[selector] = properties


class TestRunner:
    """
    Orchestrates test execution.
    
    Handles running individual tests, test sessions, and test suites
    with proper setup/teardown, error handling, and result collection.
    """
    
    def __init__(
        self,
        browser: Optional[BrowserController] = None,
        capture_screenshots: bool = True,
        capture_console: bool = True,
        capture_network: bool = True,
        record_video: bool = False,
        detect_issues: bool = True,
    ):
        self.browser = browser or SimulatedBrowserController()
        self.capture_screenshots = capture_screenshots
        self.capture_console = capture_console
        self.capture_network = capture_network
        self.record_video = record_video
        self.detect_issues = detect_issues
        self._issue_detector: Optional["IssueDetector"] = None
        self._video_recorder: Optional[VideoRecording] = None
        self._current_result: Optional[TestResult] = None
    
    async def run_step(self, step: TestStep, base_url: str = "") -> TestStep:
        """Execute a single test step."""
        step.start()
        
        try:
            if self.capture_screenshots and step.action != ActionType.SCREENSHOT:
                step.screenshot_before = await self.browser.screenshot()
            
            await self._execute_action(step, base_url)
            
            if self.capture_screenshots:
                step.screenshot_after = await self.browser.screenshot()
            
            step.complete(passed=True)
            
        except Exception as e:
            step.complete(passed=False, error=str(e))
            
            if step.max_retries > 0 and step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = TestStatus.PENDING
                return await self.run_step(step, base_url)
        
        return step
    
    async def _execute_action(self, step: TestStep, base_url: str = "") -> None:
        """Execute the action for a test step."""
        selector = step.selector or (step.locator.selector if step.locator else None)
        target = step.target
        
        if step.action == ActionType.NAVIGATE:
            url = target if target.startswith("http") else f"{base_url}{target}"
            await self.browser.navigate(url, timeout=step.timeout)
        
        elif step.action == ActionType.CLICK:
            await self.browser.click(selector, timeout=step.timeout)
        
        elif step.action == ActionType.DOUBLE_CLICK:
            await self.browser.double_click(selector, timeout=step.timeout)
        
        elif step.action == ActionType.RIGHT_CLICK:
            await self.browser.right_click(selector, timeout=step.timeout)
        
        elif step.action == ActionType.HOVER:
            await self.browser.hover(selector, timeout=step.timeout)
        
        elif step.action == ActionType.TYPE:
            await self.browser.type(selector, step.value or "", timeout=step.timeout)
        
        elif step.action == ActionType.FILL:
            await self.browser.fill(selector, step.value or "", timeout=step.timeout)
        
        elif step.action == ActionType.CLEAR:
            await self.browser.clear(selector, timeout=step.timeout)
        
        elif step.action == ActionType.PRESS:
            await self.browser.press(selector, step.value or "", timeout=step.timeout)
        
        elif step.action == ActionType.SELECT:
            await self.browser.select(selector, value=step.value, timeout=step.timeout)
        
        elif step.action == ActionType.CHECK:
            await self.browser.check(selector, timeout=step.timeout)
        
        elif step.action == ActionType.UNCHECK:
            await self.browser.uncheck(selector, timeout=step.timeout)
        
        elif step.action == ActionType.FOCUS:
            await self.browser.focus(selector, timeout=step.timeout)
        
        elif step.action == ActionType.BLUR:
            await self.browser.blur(selector)
        
        elif step.action == ActionType.SCROLL:
            x = step.options.get("x", 0)
            y = step.options.get("y", 0)
            await self.browser.scroll(selector, x, y)
        
        elif step.action == ActionType.WAIT:
            await self.browser.wait(int(step.value or 1000))
        
        elif step.action == ActionType.WAIT_FOR:
            condition = step.wait_condition or WaitCondition.VISIBLE
            await self.browser.wait_for(selector, condition, timeout=step.timeout)
        
        elif step.action == ActionType.WAIT_FOR_NAVIGATION:
            await self.browser.wait_for_navigation(timeout=step.timeout)
        
        elif step.action == ActionType.WAIT_FOR_NETWORK_IDLE:
            await self.browser.wait_for_network_idle(timeout=step.timeout)
        
        elif step.action == ActionType.SCREENSHOT:
            full_page = step.options.get("full_page", False)
            screenshot = await self.browser.screenshot(full_page=full_page)
            step.screenshot_after = screenshot
        
        elif step.action == ActionType.UPLOAD:
            files = step.options.get("files", [])
            await self.browser.upload_file(selector, files, timeout=step.timeout)
        
        elif step.action == ActionType.DRAG_DROP:
            target_selector = step.options.get("target_selector", "")
            await self.browser.drag_and_drop(selector, target_selector, timeout=step.timeout)
        
        elif step.action == ActionType.EVALUATE:
            result = await self.browser.evaluate(step.value or "")
            step.actual_value = result
        
        elif step.action == ActionType.ASSERT:
            await self._execute_assertion(step)
    
    async def _execute_assertion(self, step: TestStep) -> None:
        """Execute an assertion step."""
        selector = step.selector
        assertion_type = step.assertion_type
        expected = step.expected_value
        
        if assertion_type == AssertionType.VISIBLE:
            actual = await self.browser.is_visible(selector, timeout=step.timeout)
            step.actual_value = actual
            if not actual:
                raise AssertionError(f"Element {selector} is not visible")
        
        elif assertion_type == AssertionType.HIDDEN:
            actual = await self.browser.is_visible(selector, timeout=step.timeout)
            step.actual_value = not actual
            if actual:
                raise AssertionError(f"Element {selector} is not hidden")
        
        elif assertion_type == AssertionType.EXISTS:
            actual = await self.browser.query_selector(selector)
            step.actual_value = actual
            if not actual:
                raise AssertionError(f"Element {selector} does not exist")
        
        elif assertion_type == AssertionType.NOT_EXISTS:
            actual = await self.browser.query_selector(selector)
            step.actual_value = not actual
            if actual:
                raise AssertionError(f"Element {selector} exists but should not")
        
        elif assertion_type == AssertionType.TEXT_EQUALS:
            actual = await self.browser.get_text(selector, timeout=step.timeout)
            step.actual_value = actual
            if actual != expected:
                raise AssertionError(f"Text mismatch", expected, actual)
        
        elif assertion_type == AssertionType.TEXT_CONTAINS:
            actual = await self.browser.get_text(selector, timeout=step.timeout)
            step.actual_value = actual
            if expected not in actual:
                raise AssertionError(f"Text does not contain expected", expected, actual)
        
        elif assertion_type == AssertionType.VALUE_EQUALS:
            actual = await self.browser.get_value(selector, timeout=step.timeout)
            step.actual_value = actual
            if actual != expected:
                raise AssertionError(f"Value mismatch", expected, actual)
        
        elif assertion_type == AssertionType.URL_EQUALS:
            actual = await self.browser.get_url()
            step.actual_value = actual
            if actual != expected:
                raise AssertionError(f"URL mismatch", expected, actual)
        
        elif assertion_type == AssertionType.URL_CONTAINS:
            actual = await self.browser.get_url()
            step.actual_value = actual
            if expected not in actual:
                raise AssertionError(f"URL does not contain expected", expected, actual)
        
        elif assertion_type == AssertionType.TITLE_EQUALS:
            actual = await self.browser.get_title()
            step.actual_value = actual
            if actual != expected:
                raise AssertionError(f"Title mismatch", expected, actual)
        
        elif assertion_type == AssertionType.CHECKED:
            actual = await self.browser.is_checked(selector, timeout=step.timeout)
            step.actual_value = actual
            if not actual:
                raise AssertionError(f"Element {selector} is not checked")
        
        elif assertion_type == AssertionType.UNCHECKED:
            actual = await self.browser.is_checked(selector, timeout=step.timeout)
            step.actual_value = not actual
            if actual:
                raise AssertionError(f"Element {selector} is checked but should not be")
        
        elif assertion_type == AssertionType.ENABLED:
            actual = await self.browser.is_enabled(selector, timeout=step.timeout)
            step.actual_value = actual
            if not actual:
                raise AssertionError(f"Element {selector} is not enabled")
        
        elif assertion_type == AssertionType.DISABLED:
            actual = await self.browser.is_enabled(selector, timeout=step.timeout)
            step.actual_value = not actual
            if actual:
                raise AssertionError(f"Element {selector} is enabled but should be disabled")
        
        elif assertion_type == AssertionType.ELEMENT_COUNT:
            actual = await self.browser.query_selector_all(selector)
            step.actual_value = actual
            if actual != expected:
                raise AssertionError(f"Element count mismatch", expected, actual)
    
    async def run_session(self, session: TestSession) -> TestResult:
        """Run a complete test session."""
        result = TestResult(
            test_id=session.id,
            name=session.name,
            tags=session.tags,
        )
        self._current_result = result
        
        session.status = TestStatus.RUNNING
        result.status = TestStatus.RUNNING
        result.start_time = datetime.now()
        session.start_time = result.start_time
        
        try:
            if not self.browser.is_connected:
                await self.browser.launch()
                await self.browser.new_page()
            
            if session.base_url:
                await self.browser.navigate(session.base_url)
            
            if self.record_video:
                self._video_recorder = VideoRecording(session_id=session.id)
                self._video_recorder.start()
            
            for setup_step in session.setup_steps:
                await self.run_step(setup_step, session.base_url)
                if setup_step.failed:
                    raise Exception(f"Setup failed: {setup_step.error}")
            
            for step in session.steps:
                executed_step = await self.run_step(step, session.base_url)
                result.steps.append(executed_step)
                
                if executed_step.screenshot_after:
                    result.screenshots.append(executed_step.screenshot_after)
                
                if executed_step.failed and not session.retry_on_failure:
                    break
            
            for teardown_step in session.teardown_steps:
                await self.run_step(teardown_step, session.base_url)
            
            if self.capture_console:
                result.console_messages = self.browser.console_messages.copy()
            
            if self.capture_network:
                result.network_requests = self.browser.network_requests.copy()
                result.network_responses = self.browser.network_responses.copy()
            
            result.performance_metrics = await self.browser.get_performance_metrics()
            
            if self.detect_issues and self._issue_detector:
                result.issues = await self._issue_detector.detect_all(self.browser)
            
            all_passed = all(s.passed for s in result.steps)
            result.status = TestStatus.PASSED if all_passed else TestStatus.FAILED
            
        except Exception as e:
            result.status = TestStatus.ERRORED
            result.error = str(e)
        
        finally:
            result.end_time = datetime.now()
            session.end_time = result.end_time
            session.status = result.status
            session.result = result
            
            if self._video_recorder:
                self._video_recorder.stop()
                session.video = self._video_recorder
        
        return result
    
    async def run_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run a complete test suite."""
        results: List[TestResult] = []
        
        for setup_step in suite.global_setup:
            await self.run_step(setup_step)
            if setup_step.failed:
                raise Exception(f"Global setup failed: {setup_step.error}")
        
        if suite.parallel_execution:
            tasks = [self.run_session(s) for s in suite.sessions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in results if isinstance(r, TestResult)]
        else:
            for session in suite.sessions:
                result = await self.run_session(session)
                results.append(result)
                
                if suite.fail_fast and result.failed:
                    break
        
        for teardown_step in suite.global_teardown:
            await self.run_step(teardown_step)
        
        return results
    
    def set_issue_detector(self, detector: "IssueDetector") -> None:
        """Set the issue detector to use."""
        self._issue_detector = detector


class IssueDetector:
    """
    Detects common issues in web applications.
    
    Checks for broken links, console errors, accessibility issues,
    performance problems, and security concerns.
    """
    
    def __init__(self, rules: Optional[Dict[str, Dict]] = None):
        self.rules = rules or ACCESSIBILITY_RULES
        self._custom_detectors: List[Callable] = []
    
    async def detect_all(self, browser: BrowserController) -> List[DetectedIssue]:
        """Run all detectors and return found issues."""
        issues = []
        
        issues.extend(await self.detect_console_errors(browser))
        issues.extend(await self.detect_broken_links(browser))
        issues.extend(await self.detect_accessibility_issues(browser))
        issues.extend(await self.detect_performance_issues(browser))
        issues.extend(await self.detect_seo_issues(browser))
        
        for detector in self._custom_detectors:
            try:
                custom_issues = await detector(browser)
                issues.extend(custom_issues)
            except Exception as e:
                logging.warning(f"Custom detector failed: {e}")
        
        return issues
    
    async def detect_console_errors(self, browser: BrowserController) -> List[DetectedIssue]:
        """Detect console errors and warnings."""
        issues = []
        
        for msg in browser.console_messages:
            if msg.is_error:
                issues.append(DetectedIssue(
                    type=IssueType.CONSOLE_ERROR,
                    severity=IssueSeverity.HIGH,
                    title="Console Error",
                    description=msg.text,
                    url=msg.url or browser.current_url,
                    context={
                        "line": msg.line_number,
                        "column": msg.column_number,
                        "stack": msg.stack_trace,
                    },
                ))
            elif msg.is_warning:
                issues.append(DetectedIssue(
                    type=IssueType.CONSOLE_WARNING,
                    severity=IssueSeverity.LOW,
                    title="Console Warning",
                    description=msg.text,
                    url=msg.url or browser.current_url,
                ))
        
        return issues
    
    async def detect_broken_links(self, browser: BrowserController) -> List[DetectedIssue]:
        """Detect broken links based on network responses."""
        issues = []
        
        for response in browser.network_responses:
            if response.is_error:
                issues.append(DetectedIssue(
                    type=IssueType.BROKEN_LINK,
                    severity=IssueSeverity.MEDIUM if response.status < 500 else IssueSeverity.HIGH,
                    title=f"Broken Link ({response.status})",
                    description=f"Request to {response.url} failed with status {response.status}",
                    url=response.url,
                    context={
                        "status": response.status,
                        "status_text": response.status_text,
                    },
                ))
        
        return issues
    
    async def detect_accessibility_issues(self, browser: BrowserController) -> List[DetectedIssue]:
        """Detect accessibility issues based on WCAG guidelines."""
        issues = []
        
        for rule_name, rule in self.rules.items():
            if "selector" in rule:
                count = await browser.query_selector_all(rule["selector"])
                if count > 0:
                    issues.append(DetectedIssue(
                        type=IssueType.ACCESSIBILITY,
                        severity=rule.get("severity", IssueSeverity.MEDIUM),
                        title=f"Accessibility: {rule_name.replace('_', ' ').title()}",
                        description=rule["description"],
                        url=browser.current_url,
                        selector=rule["selector"],
                        wcag_criterion=rule.get("wcag"),
                        auto_fixable=self._is_auto_fixable(rule_name),
                        context={"element_count": count},
                    ))
        
        return issues
    
    async def detect_performance_issues(self, browser: BrowserController) -> List[DetectedIssue]:
        """Detect performance issues based on metrics."""
        issues = []
        metrics = await browser.get_performance_metrics()
        
        if metrics.largest_contentful_paint > 2500:
            issues.append(DetectedIssue(
                type=IssueType.PERFORMANCE,
                severity=IssueSeverity.HIGH if metrics.largest_contentful_paint > 4000 else IssueSeverity.MEDIUM,
                title="Slow Largest Contentful Paint (LCP)",
                description=f"LCP is {metrics.largest_contentful_paint:.0f}ms, should be under 2500ms",
                url=metrics.url,
                suggested_fix="Optimize images, reduce render-blocking resources, use lazy loading",
                context={"value": metrics.largest_contentful_paint, "threshold": 2500},
            ))
        
        if metrics.first_input_delay > 100:
            issues.append(DetectedIssue(
                type=IssueType.PERFORMANCE,
                severity=IssueSeverity.MEDIUM,
                title="High First Input Delay (FID)",
                description=f"FID is {metrics.first_input_delay:.0f}ms, should be under 100ms",
                url=metrics.url,
                suggested_fix="Break up long tasks, optimize JavaScript execution",
                context={"value": metrics.first_input_delay, "threshold": 100},
            ))
        
        if metrics.cumulative_layout_shift > 0.1:
            issues.append(DetectedIssue(
                type=IssueType.LAYOUT_SHIFT,
                severity=IssueSeverity.MEDIUM,
                title="High Cumulative Layout Shift (CLS)",
                description=f"CLS is {metrics.cumulative_layout_shift:.3f}, should be under 0.1",
                url=metrics.url,
                suggested_fix="Set explicit dimensions on images and embeds, avoid inserting content above existing content",
                context={"value": metrics.cumulative_layout_shift, "threshold": 0.1},
            ))
        
        if metrics.total_blocking_time > 300:
            issues.append(DetectedIssue(
                type=IssueType.PERFORMANCE,
                severity=IssueSeverity.MEDIUM,
                title="High Total Blocking Time",
                description=f"TBT is {metrics.total_blocking_time:.0f}ms, should be under 300ms",
                url=metrics.url,
                suggested_fix="Split long JavaScript tasks, use web workers",
                context={"value": metrics.total_blocking_time, "threshold": 300},
            ))
        
        return issues
    
    async def detect_seo_issues(self, browser: BrowserController) -> List[DetectedIssue]:
        """Detect SEO-related issues."""
        issues = []
        
        title = await browser.get_title()
        if not title or len(title) < 10:
            issues.append(DetectedIssue(
                type=IssueType.SEO,
                severity=IssueSeverity.MEDIUM,
                title="Missing or Short Page Title",
                description="Page title should be descriptive and at least 10 characters",
                url=browser.current_url,
                suggested_fix="Add a descriptive <title> tag",
                auto_fixable=False,
            ))
        elif len(title) > 60:
            issues.append(DetectedIssue(
                type=IssueType.SEO,
                severity=IssueSeverity.LOW,
                title="Page Title Too Long",
                description="Page title should be under 60 characters for optimal SEO",
                url=browser.current_url,
                suggested_fix="Shorten the title tag to under 60 characters",
            ))
        
        return issues
    
    def _is_auto_fixable(self, rule_name: str) -> bool:
        """Check if an issue can be auto-fixed."""
        auto_fixable = {"alt_text", "lang_attribute"}
        return rule_name in auto_fixable
    
    def add_detector(self, detector: Callable) -> None:
        """Add a custom detector function."""
        self._custom_detectors.append(detector)


class AutoFixer:
    """
    Suggests and applies fixes for detected issues.
    
    Provides both suggestions and auto-fix capabilities for
    common issues like missing alt text, accessibility problems, etc.
    """
    
    def __init__(self):
        self._fix_strategies: Dict[IssueType, List[Callable]] = defaultdict(list)
        self._register_default_strategies()
    
    def _register_default_strategies(self) -> None:
        """Register default fix strategies."""
        self._fix_strategies[IssueType.MISSING_ALT_TEXT].append(self._fix_missing_alt)
        self._fix_strategies[IssueType.MISSING_LABEL].append(self._fix_missing_label)
        self._fix_strategies[IssueType.ACCESSIBILITY].append(self._fix_accessibility)
        self._fix_strategies[IssueType.CONSOLE_ERROR].append(self._suggest_console_fix)
        self._fix_strategies[IssueType.BROKEN_LINK].append(self._suggest_broken_link_fix)
        self._fix_strategies[IssueType.PERFORMANCE].append(self._suggest_performance_fix)
    
    def suggest_fix(self, issue: DetectedIssue) -> Optional[Fix]:
        """Suggest a fix for an issue."""
        strategies = self._fix_strategies.get(issue.type, [])
        
        for strategy in strategies:
            try:
                fix = strategy(issue)
                if fix:
                    return fix
            except Exception as e:
                logging.warning(f"Fix strategy failed: {e}")
        
        return self._generate_generic_fix(issue)
    
    def suggest_fixes(self, issues: List[DetectedIssue]) -> List[Fix]:
        """Suggest fixes for multiple issues."""
        fixes = []
        for issue in issues:
            fix = self.suggest_fix(issue)
            if fix:
                fixes.append(fix)
        return fixes
    
    async def apply_fix(self, fix: Fix) -> bool:
        """Apply an auto-fix."""
        if not fix.auto_applicable:
            return False
        
        try:
            return fix.apply()
        except Exception as e:
            logging.error(f"Failed to apply fix: {e}")
            return False
    
    async def apply_fixes(self, fixes: List[Fix]) -> List[Fix]:
        """Apply multiple fixes."""
        applied = []
        for fix in fixes:
            if await self.apply_fix(fix):
                applied.append(fix)
        return applied
    
    def _fix_missing_alt(self, issue: DetectedIssue) -> Optional[Fix]:
        """Generate fix for missing alt text."""
        if issue.selector:
            return Fix(
                issue_id=issue.id,
                issue_type=issue.type,
                title="Add alt text to image",
                description="Add descriptive alt text to improve accessibility",
                code_before=f'<img src="..." >',
                code_after=f'<img src="..." alt="Descriptive text here">',
                confidence=0.9,
                auto_applicable=False,
            )
        return None
    
    def _fix_missing_label(self, issue: DetectedIssue) -> Optional[Fix]:
        """Generate fix for missing form labels."""
        return Fix(
            issue_id=issue.id,
            issue_type=issue.type,
            title="Add label to form input",
            description="Associate a label with the form input",
            code_before='<input type="text" name="email">',
            code_after='<label for="email">Email</label>\n<input type="text" id="email" name="email">',
            confidence=0.85,
            auto_applicable=False,
        )
    
    def _fix_accessibility(self, issue: DetectedIssue) -> Optional[Fix]:
        """Generate fix for accessibility issues."""
        if "lang_attribute" in issue.title.lower() or "lang" in str(issue.selector or ""):
            return Fix(
                issue_id=issue.id,
                issue_type=issue.type,
                title="Add lang attribute to HTML",
                description="Add language attribute to the html element",
                code_before='<html>',
                code_after='<html lang="en">',
                confidence=0.95,
                auto_applicable=True,
            )
        return None
    
    def _suggest_console_fix(self, issue: DetectedIssue) -> Optional[Fix]:
        """Suggest fix for console errors."""
        description = issue.description.lower()
        
        if "undefined" in description or "null" in description:
            return Fix(
                issue_id=issue.id,
                issue_type=issue.type,
                title="Add null check",
                description="Add defensive null/undefined check before accessing property",
                code_after="if (variable != null) { /* access property */ }",
                confidence=0.7,
                auto_applicable=False,
            )
        elif "network" in description or "fetch" in description:
            return Fix(
                issue_id=issue.id,
                issue_type=issue.type,
                title="Add error handling to network request",
                description="Wrap network request in try-catch with proper error handling",
                code_after="try { await fetch(...) } catch (e) { handleError(e) }",
                confidence=0.75,
                auto_applicable=False,
            )
        return None
    
    def _suggest_broken_link_fix(self, issue: DetectedIssue) -> Optional[Fix]:
        """Suggest fix for broken links."""
        status = issue.context.get("status", 0)
        
        if status == 404:
            return Fix(
                issue_id=issue.id,
                issue_type=issue.type,
                title="Fix or remove broken link",
                description=f"The link to {issue.url} returns 404. Update the URL or remove the link.",
                confidence=0.9,
                auto_applicable=False,
            )
        elif status >= 500:
            return Fix(
                issue_id=issue.id,
                issue_type=issue.type,
                title="Server error on linked resource",
                description=f"The server at {issue.url} is returning errors. Check server logs or try again later.",
                confidence=0.6,
                auto_applicable=False,
            )
        return None
    
    def _suggest_performance_fix(self, issue: DetectedIssue) -> Optional[Fix]:
        """Suggest fix for performance issues."""
        if "LCP" in issue.title:
            return Fix(
                issue_id=issue.id,
                issue_type=issue.type,
                title="Optimize Largest Contentful Paint",
                description=(
                    "1. Optimize and compress images\n"
                    "2. Use modern image formats (WebP, AVIF)\n"
                    "3. Implement lazy loading for below-fold images\n"
                    "4. Preload critical resources\n"
                    "5. Reduce render-blocking CSS/JS"
                ),
                confidence=0.8,
                auto_applicable=False,
            )
        elif "CLS" in issue.title:
            return Fix(
                issue_id=issue.id,
                issue_type=issue.type,
                title="Reduce Cumulative Layout Shift",
                description=(
                    "1. Set explicit width/height on images and videos\n"
                    "2. Reserve space for ads and embeds\n"
                    "3. Avoid inserting content above existing content\n"
                    "4. Use CSS aspect-ratio or padding-top hack"
                ),
                confidence=0.8,
                auto_applicable=False,
            )
        return None
    
    def _generate_generic_fix(self, issue: DetectedIssue) -> Fix:
        """Generate a generic fix suggestion."""
        return Fix(
            issue_id=issue.id,
            issue_type=issue.type,
            title=f"Investigate {issue.type.value.replace('_', ' ').title()}",
            description=issue.suggested_fix or f"Review and address: {issue.description}",
            confidence=0.5,
            auto_applicable=False,
        )
    
    def register_strategy(self, issue_type: IssueType, strategy: Callable) -> None:
        """Register a custom fix strategy."""
        self._fix_strategies[issue_type].append(strategy)


class VisualTester:
    """
    Handles visual regression testing through screenshot comparison.
    
    Compares current screenshots against baseline images to detect
    visual changes and regressions.
    """
    
    def __init__(
        self,
        baseline_dir: str = "./baselines",
        diff_dir: str = "./diffs",
        threshold: float = 0.01,
    ):
        self.baseline_dir = Path(baseline_dir)
        self.diff_dir = Path(diff_dir)
        self.threshold = threshold
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.diff_dir.mkdir(parents=True, exist_ok=True)
    
    def compare(
        self,
        current: Screenshot,
        baseline: Screenshot,
        threshold: Optional[float] = None
    ) -> VisualDiff:
        """Compare two screenshots and return the diff result."""
        threshold = threshold or self.threshold
        
        diff_percentage = self._calculate_diff(current.data, baseline.data)
        diff_pixels = int(diff_percentage * current.width * current.height)
        
        passed = diff_percentage <= threshold
        
        diff_image = None
        if not passed:
            diff_image = self._generate_diff_image(current.data, baseline.data)
        
        return VisualDiff(
            baseline_screenshot=baseline,
            current_screenshot=current,
            diff_percentage=diff_percentage,
            diff_pixels=diff_pixels,
            diff_image=diff_image,
            passed=passed,
            threshold=threshold,
        )
    
    def _calculate_diff(self, current_data: bytes, baseline_data: bytes) -> float:
        """Calculate the difference percentage between two images."""
        if current_data == baseline_data:
            return 0.0
        
        current_hash = hashlib.md5(current_data).hexdigest()
        baseline_hash = hashlib.md5(baseline_data).hexdigest()
        
        if current_hash == baseline_hash:
            return 0.0
        
        similarity = SequenceMatcher(None, current_data, baseline_data).ratio()
        return 1.0 - similarity
    
    def _generate_diff_image(self, current: bytes, baseline: bytes) -> bytes:
        """Generate a diff image highlighting changes."""
        return current
    
    def save_baseline(self, screenshot: Screenshot, name: str) -> str:
        """Save a screenshot as a baseline."""
        path = self.baseline_dir / f"{name}.png"
        screenshot.save(str(path))
        return str(path)
    
    def load_baseline(self, name: str) -> Optional[Screenshot]:
        """Load a baseline screenshot."""
        path = self.baseline_dir / f"{name}.png"
        if not path.exists():
            return None
        
        with open(path, "rb") as f:
            data = f.read()
        
        return Screenshot(
            data=data,
            format="png",
            file_path=str(path),
        )
    
    def update_baseline(self, screenshot: Screenshot, name: str) -> str:
        """Update a baseline with a new screenshot."""
        return self.save_baseline(screenshot, name)
    
    async def compare_against_baseline(
        self,
        browser: BrowserController,
        baseline_name: str,
        full_page: bool = False
    ) -> VisualDiff:
        """Take a screenshot and compare against baseline."""
        current = await browser.screenshot(full_page=full_page)
        baseline = self.load_baseline(baseline_name)
        
        if baseline is None:
            self.save_baseline(current, baseline_name)
            return VisualDiff(
                baseline_screenshot=current,
                current_screenshot=current,
                diff_percentage=0.0,
                passed=True,
                threshold=self.threshold,
            )
        
        return self.compare(current, baseline)


class TestReporter:
    """
    Generates test reports in various formats.
    
    Supports HTML, JSON, Markdown, JUnit XML, and TAP formats.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        results: List[TestResult],
        suite_name: str = "Test Suite",
        format: ReportFormat = ReportFormat.HTML
    ) -> TestReport:
        """Generate a test report from results."""
        report = TestReport(
            suite_name=suite_name,
            results=results,
            total_tests=len(results),
            passed_tests=sum(1 for r in results if r.passed),
            failed_tests=sum(1 for r in results if r.failed),
            skipped_tests=sum(1 for r in results if r.status == TestStatus.SKIPPED),
            total_duration_ms=sum(r.duration_ms or 0 for r in results),
        )
        
        for result in results:
            report.issues.extend(result.issues)
            if result.performance_metrics:
                for key, value in result.performance_metrics.core_web_vitals.items():
                    if key not in report.performance_summary:
                        report.performance_summary[key] = []
                    report.performance_summary[key] = value
        
        if report.issues:
            accessibility_issues = [i for i in report.issues if i.type == IssueType.ACCESSIBILITY]
            if accessibility_issues:
                max_deduction = sum(i.severity.weight for i in accessibility_issues)
                report.accessibility_score = max(0, 100 - min(max_deduction, 100))
            else:
                report.accessibility_score = 100.0
        else:
            report.accessibility_score = 100.0
        
        return report
    
    def export(
        self,
        report: TestReport,
        format: ReportFormat = ReportFormat.HTML,
        filename: Optional[str] = None
    ) -> str:
        """Export report to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}"
        
        if format == ReportFormat.HTML:
            return self._export_html(report, filename)
        elif format == ReportFormat.JSON:
            return self._export_json(report, filename)
        elif format == ReportFormat.MARKDOWN:
            return self._export_markdown(report, filename)
        elif format == ReportFormat.JUNIT:
            return self._export_junit(report, filename)
        elif format == ReportFormat.TAP:
            return self._export_tap(report, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_html(self, report: TestReport, filename: str) -> str:
        """Export report as HTML."""
        path = self.output_dir / f"{filename}.html"
        
        status_color = "#4caf50" if report.failed_tests == 0 else "#f44336"
        status_text = "PASSED" if report.failed_tests == 0 else "FAILED"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {report.suite_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; }}
        h1 {{ margin-top: 0; color: #333; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat {{ background: #f9f9f9; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .status {{ display: inline-block; padding: 4px 12px; border-radius: 4px; color: white; font-weight: bold; }}
        .passed {{ background: #4caf50; }}
        .failed {{ background: #f44336; }}
        .skipped {{ background: #ff9800; }}
        .test {{ border: 1px solid #eee; border-radius: 6px; margin-bottom: 10px; overflow: hidden; }}
        .test-header {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 15px; background: #fafafa; cursor: pointer; }}
        .test-name {{ font-weight: 500; }}
        .test-duration {{ color: #666; font-size: 12px; }}
        .test-steps {{ padding: 15px; display: none; }}
        .test.expanded .test-steps {{ display: block; }}
        .step {{ display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid #eee; }}
        .step:last-child {{ border-bottom: none; }}
        .step-status {{ width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }}
        .step-status.passed {{ background: #4caf50; }}
        .step-status.failed {{ background: #f44336; }}
        .issues {{ margin-top: 20px; }}
        .issue {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 12px; margin-bottom: 10px; }}
        .issue.high {{ background: #f8d7da; border-color: #f44336; }}
        .issue.critical {{ background: #f8d7da; border-color: #d32f2f; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Report: {report.suite_name}</h1>
        <p>Generated at: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <div class="stat">
                <div class="stat-value" style="color: {status_color}">{status_text}</div>
                <div class="stat-label">Overall Status</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report.total_tests}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color: #4caf50">{report.passed_tests}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color: #f44336">{report.failed_tests}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report.total_duration_ms}ms</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report.pass_rate:.1f}%</div>
                <div class="stat-label">Pass Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report.accessibility_score:.0f}</div>
                <div class="stat-label">Accessibility Score</div>
            </div>
        </div>
        
        <h2>Test Results</h2>
        {"".join(self._render_test_html(r) for r in report.results)}
        
        {"<h2>Issues Found</h2>" if report.issues else ""}
        {"".join(self._render_issue_html(i) for i in report.issues)}
    </div>
    <script>
        document.querySelectorAll('.test-header').forEach(header => {{
            header.addEventListener('click', () => {{
                header.parentElement.classList.toggle('expanded');
            }});
        }});
    </script>
</body>
</html>"""
        
        with open(path, "w") as f:
            f.write(html)
        
        return str(path)
    
    def _render_test_html(self, result: TestResult) -> str:
        """Render a single test result as HTML."""
        status_class = "passed" if result.passed else "failed"
        steps_html = ""
        for step in result.steps:
            step_status = "passed" if step.passed else "failed"
            steps_html += f"""
            <div class="step">
                <div class="step-status {step_status}"></div>
                <div>{step.description or step.action.value}</div>
                <div style="margin-left: auto; color: #666;">{step.duration_ms or 0}ms</div>
            </div>"""
        
        return f"""
        <div class="test">
            <div class="test-header">
                <span class="test-name">{result.name}</span>
                <span>
                    <span class="test-duration">{result.duration_ms or 0}ms</span>
                    <span class="status {status_class}">{result.status.value.upper()}</span>
                </span>
            </div>
            <div class="test-steps">{steps_html}</div>
        </div>"""
    
    def _render_issue_html(self, issue: DetectedIssue) -> str:
        """Render a single issue as HTML."""
        severity_class = issue.severity.value if issue.severity.value in ("high", "critical") else ""
        return f"""
        <div class="issue {severity_class}">
            <strong>{issue.title}</strong> ({issue.severity.value.upper()})
            <p>{issue.description}</p>
            {f"<p><em>Fix: {issue.suggested_fix}</em></p>" if issue.suggested_fix else ""}
        </div>"""
    
    def _export_json(self, report: TestReport, filename: str) -> str:
        """Export report as JSON."""
        path = self.output_dir / f"{filename}.json"
        
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        return str(path)
    
    def _export_markdown(self, report: TestReport, filename: str) -> str:
        """Export report as Markdown."""
        path = self.output_dir / f"{filename}.md"
        
        status = "PASSED" if report.failed_tests == 0 else "FAILED"
        
        md = f"""# Test Report: {report.suite_name}

**Status:** {status}  
**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {report.total_tests} |
| Passed | {report.passed_tests} |
| Failed | {report.failed_tests} |
| Skipped | {report.skipped_tests} |
| Duration | {report.total_duration_ms}ms |
| Pass Rate | {report.pass_rate:.1f}% |
| Accessibility Score | {report.accessibility_score:.0f}/100 |

## Test Results

"""
        
        for result in report.results:
            status_emoji = "[PASS]" if result.passed else "[FAIL]"
            md += f"### {status_emoji} {result.name}\n\n"
            md += f"- Duration: {result.duration_ms or 0}ms\n"
            md += f"- Steps: {result.passed_steps} passed, {result.failed_steps} failed\n\n"
            
            if result.steps:
                md += "| Step | Status | Duration |\n"
                md += "|------|--------|----------|\n"
                for step in result.steps:
                    step_status = "Pass" if step.passed else "Fail"
                    md += f"| {step.description or step.action.value} | {step_status} | {step.duration_ms or 0}ms |\n"
                md += "\n"
        
        if report.issues:
            md += "## Issues Found\n\n"
            for issue in report.issues:
                md += f"### [{issue.severity.value.upper()}] {issue.title}\n\n"
                md += f"{issue.description}\n\n"
                if issue.suggested_fix:
                    md += f"**Suggested Fix:** {issue.suggested_fix}\n\n"
        
        with open(path, "w") as f:
            f.write(md)
        
        return str(path)
    
    def _export_junit(self, report: TestReport, filename: str) -> str:
        """Export report as JUnit XML."""
        path = self.output_dir / f"{filename}.xml"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="{report.suite_name}" tests="{report.total_tests}" failures="{report.failed_tests}" time="{report.total_duration_ms / 1000:.3f}">
    <testsuite name="{report.suite_name}" tests="{report.total_tests}" failures="{report.failed_tests}" time="{report.total_duration_ms / 1000:.3f}">
"""
        
        for result in report.results:
            duration = (result.duration_ms or 0) / 1000
            if result.passed:
                xml += f'        <testcase name="{result.name}" time="{duration:.3f}" />\n'
            else:
                xml += f"""        <testcase name="{result.name}" time="{duration:.3f}">
            <failure message="{result.error or 'Test failed'}" />
        </testcase>\n"""
        
        xml += """    </testsuite>
</testsuites>"""
        
        with open(path, "w") as f:
            f.write(xml)
        
        return str(path)
    
    def _export_tap(self, report: TestReport, filename: str) -> str:
        """Export report as TAP (Test Anything Protocol)."""
        path = self.output_dir / f"{filename}.tap"
        
        tap = f"TAP version 13\n1..{report.total_tests}\n"
        
        for i, result in enumerate(report.results, 1):
            status = "ok" if result.passed else "not ok"
            tap += f"{status} {i} - {result.name}\n"
            if not result.passed and result.error:
                tap += f"  ---\n  message: {result.error}\n  ---\n"
        
        with open(path, "w") as f:
            f.write(tap)
        
        return str(path)


_default_runner: Optional[TestRunner] = None
_default_issue_detector: Optional[IssueDetector] = None
_default_auto_fixer: Optional[AutoFixer] = None
_default_visual_tester: Optional[VisualTester] = None
_default_reporter: Optional[TestReporter] = None


def get_runner() -> TestRunner:
    """Get or create the default test runner."""
    global _default_runner
    if _default_runner is None:
        _default_runner = TestRunner()
    return _default_runner


def get_issue_detector() -> IssueDetector:
    """Get or create the default issue detector."""
    global _default_issue_detector
    if _default_issue_detector is None:
        _default_issue_detector = IssueDetector()
    return _default_issue_detector


def get_auto_fixer() -> AutoFixer:
    """Get or create the default auto fixer."""
    global _default_auto_fixer
    if _default_auto_fixer is None:
        _default_auto_fixer = AutoFixer()
    return _default_auto_fixer


def get_visual_tester() -> VisualTester:
    """Get or create the default visual tester."""
    global _default_visual_tester
    if _default_visual_tester is None:
        _default_visual_tester = VisualTester()
    return _default_visual_tester


def get_reporter() -> TestReporter:
    """Get or create the default reporter."""
    global _default_reporter
    if _default_reporter is None:
        _default_reporter = TestReporter()
    return _default_reporter


async def run_test_suite(suite: TestSuite) -> List[TestResult]:
    """Run a complete test suite using the default runner."""
    runner = get_runner()
    return await runner.run_suite(suite)


async def run_single_test(session: TestSession) -> TestResult:
    """Run a single test session using the default runner."""
    runner = get_runner()
    return await runner.run_session(session)


async def detect_issues(browser: BrowserController) -> List[DetectedIssue]:
    """Detect issues using the default detector."""
    detector = get_issue_detector()
    return await detector.detect_all(browser)


def suggest_fixes(issues: List[DetectedIssue]) -> List[Fix]:
    """Suggest fixes using the default auto fixer."""
    fixer = get_auto_fixer()
    return fixer.suggest_fixes(issues)


def generate_report(
    results: List[TestResult],
    suite_name: str = "Test Suite",
    format: ReportFormat = ReportFormat.HTML
) -> TestReport:
    """Generate a report using the default reporter."""
    reporter = get_reporter()
    return reporter.generate_report(results, suite_name, format)


async def record_video(session: TestSession) -> VideoRecording:
    """Run a session with video recording."""
    runner = TestRunner(record_video=True)
    await runner.run_session(session)
    return session.video


def create_test_session(
    name: str,
    base_url: str,
    steps: Optional[List[TestStep]] = None,
    **kwargs
) -> TestSession:
    """Helper to create a test session."""
    session = TestSession(name=name, base_url=base_url, **kwargs)
    if steps:
        for step in steps:
            session.add_step(step)
    return session


def navigate(url: str, **kwargs) -> TestStep:
    """Create a navigate step."""
    return TestStep(action=ActionType.NAVIGATE, target=url, **kwargs)


def click(selector: str, **kwargs) -> TestStep:
    """Create a click step."""
    return TestStep(action=ActionType.CLICK, selector=selector, **kwargs)


def type_text(selector: str, value: str, **kwargs) -> TestStep:
    """Create a type step."""
    return TestStep(action=ActionType.TYPE, selector=selector, value=value, **kwargs)


def fill(selector: str, value: str, **kwargs) -> TestStep:
    """Create a fill step."""
    return TestStep(action=ActionType.FILL, selector=selector, value=value, **kwargs)


def wait_for(selector: str, **kwargs) -> TestStep:
    """Create a wait_for step."""
    return TestStep(action=ActionType.WAIT_FOR, selector=selector, **kwargs)


def screenshot(**kwargs) -> TestStep:
    """Create a screenshot step."""
    return TestStep(action=ActionType.SCREENSHOT, **kwargs)


def assert_visible(selector: str, **kwargs) -> TestStep:
    """Create an assert visible step."""
    return TestStep(
        action=ActionType.ASSERT,
        selector=selector,
        assertion_type=AssertionType.VISIBLE,
        **kwargs
    )


def assert_text(selector: str, expected: str, **kwargs) -> TestStep:
    """Create an assert text step."""
    return TestStep(
        action=ActionType.ASSERT,
        selector=selector,
        assertion_type=AssertionType.TEXT_EQUALS,
        expected_value=expected,
        **kwargs
    )


__all__ = [
    'ActionType',
    'TestStatus',
    'IssueType',
    'IssueSeverity',
    'AssertionType',
    'BrowserType',
    'RecordingFormat',
    'ReportFormat',
    'WaitCondition',
    'TIMEOUT_DEFAULTS',
    'ACCESSIBILITY_RULES',
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
    'Fix',
    'TestReport',
    'BrowserError',
    'ElementNotFoundError',
    'NavigationError',
    'TimeoutError',
    'AssertionError',
    'BrowserController',
    'SimulatedBrowserController',
    'TestRunner',
    'IssueDetector',
    'AutoFixer',
    'VisualTester',
    'TestReporter',
    'get_runner',
    'get_issue_detector',
    'get_auto_fixer',
    'get_visual_tester',
    'get_reporter',
    'run_test_suite',
    'run_single_test',
    'detect_issues',
    'suggest_fixes',
    'generate_report',
    'record_video',
    'create_test_session',
    'navigate',
    'click',
    'type_text',
    'fill',
    'wait_for',
    'screenshot',
    'assert_visible',
    'assert_text',
]
