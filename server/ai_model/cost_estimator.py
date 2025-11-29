"""
Transparent Cost Estimation System for Platform Forge

This module provides clear cost visibility before any action, including
real-time spending tracking, budget management, and cost optimization.

Key Components:
- CostEstimate: Dataclass for detailed cost estimates with breakdowns
- CostEstimator: Estimate costs for various action types
- CostTracker: Real-time spending tracking with alerts
- BudgetManager: Set and enforce spending limits
- CostOptimizer: Suggestions for reducing costs
- PricingModels: Token, compute, storage, and action-based pricing

Usage:
    from server.ai_model.cost_estimator import (
        CostEstimator,
        CostTracker,
        BudgetManager,
        CostOptimizer,
        estimate_action_cost,
        format_cost,
        compare_costs,
        get_cost_breakdown,
    )
    
    # Estimate action cost
    estimator = CostEstimator()
    estimate = estimator.estimate_action_cost("code_generation", "medium", 4096)
    print(f"Expected cost: {format_cost(estimate.expected_cost)}")
    
    # Track spending
    tracker = CostTracker(daily_budget=10.0)
    tracker.record_cost("code_generation", 0.05)
    if tracker.is_approaching_limit():
        print(tracker.get_budget_alert())
    
    # Manage budgets
    budget_manager = BudgetManager(monthly_limit=100.0)
    can_proceed, message = budget_manager.check_action("expensive_operation", 5.0)
    
    # Optimize costs
    optimizer = CostOptimizer()
    suggestions = optimizer.get_suggestions(tracker.get_history())
"""

import re
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json


class CostUnit(Enum):
    """Units for cost measurement."""
    CREDITS = "credits"
    DOLLARS = "dollars"
    TOKENS = "tokens"
    COMPUTE_SECONDS = "compute_seconds"


class ActionType(Enum):
    """Types of billable actions."""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_REFACTORING = "code_refactoring"
    DEBUGGING = "debugging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    INFRASTRUCTURE = "infrastructure"
    SECURITY_SCAN = "security_scan"
    EMBEDDING = "embedding"
    CHAT = "chat"
    COMPLETION = "completion"
    VISION = "vision"
    AUDIO = "audio"
    CUSTOM = "custom"


class Complexity(Enum):
    """Complexity levels for cost estimation."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"
    
    @property
    def multiplier(self) -> float:
        """Get complexity multiplier for cost calculations."""
        multipliers = {
            Complexity.TRIVIAL: 0.5,
            Complexity.SIMPLE: 0.75,
            Complexity.MEDIUM: 1.0,
            Complexity.COMPLEX: 1.5,
            Complexity.VERY_COMPLEX: 2.5,
        }
        return multipliers[self]


class AlertLevel(Enum):
    """Budget alert levels."""
    NORMAL = "normal"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


@dataclass
class CostBreakdown:
    """Detailed breakdown of cost components."""
    compute: float = 0.0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    storage: float = 0.0
    api_calls: float = 0.0
    network: float = 0.0
    overhead: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "compute": self.compute,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "storage": self.storage,
            "api_calls": self.api_calls,
            "network": self.network,
            "overhead": self.overhead,
        }
    
    @property
    def total(self) -> float:
        """Calculate total cost from breakdown."""
        return (
            self.compute +
            self.input_tokens +
            self.output_tokens +
            self.storage +
            self.api_calls +
            self.network +
            self.overhead
        )


@dataclass
class CostEstimate:
    """
    Detailed cost estimate for an action.
    
    Attributes:
        min_cost: Minimum expected cost (optimistic)
        max_cost: Maximum expected cost (pessimistic)
        expected_cost: Most likely cost
        breakdown: Detailed breakdown by component
        confidence: How accurate the estimate is (0.0-1.0)
        factors: List of factors affecting the cost
        unit: Currency/unit for the costs
        action_type: Type of action being estimated
        warnings: Any cost-related warnings
        recommendations: Suggestions for cost optimization
    """
    min_cost: float
    max_cost: float
    expected_cost: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.8
    factors: List[str] = field(default_factory=list)
    unit: CostUnit = CostUnit.DOLLARS
    action_type: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.min_cost > self.expected_cost:
            self.min_cost = self.expected_cost * 0.7
        if self.max_cost < self.expected_cost:
            self.max_cost = self.expected_cost * 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "expected_cost": self.expected_cost,
            "breakdown": self.breakdown,
            "confidence": self.confidence,
            "factors": self.factors,
            "unit": self.unit.value,
            "action_type": self.action_type,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "formatted": {
                "min": format_cost(self.min_cost),
                "max": format_cost(self.max_cost),
                "expected": format_cost(self.expected_cost),
            },
            "metadata": self.metadata,
        }
    
    def get_range_string(self) -> str:
        """Get formatted cost range string."""
        if self.min_cost == self.max_cost:
            return format_cost(self.expected_cost)
        return f"{format_cost(self.min_cost)} - {format_cost(self.max_cost)}"
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Cost Estimate: {self.get_range_string()}",
            f"Expected: {format_cost(self.expected_cost)}",
            f"Confidence: {self.confidence:.0%}",
        ]
        if self.factors:
            lines.append(f"Factors: {', '.join(self.factors[:3])}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        return "\n".join(lines)


@dataclass
class CostRecord:
    """Record of a single cost event."""
    timestamp: datetime
    action_type: str
    amount: float
    unit: CostUnit
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "amount": self.amount,
            "unit": self.unit.value,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class BudgetAlert:
    """Budget alert information."""
    level: AlertLevel
    current_spending: float
    budget_limit: float
    percentage_used: float
    message: str
    time_period: str
    remaining: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "current_spending": self.current_spending,
            "budget_limit": self.budget_limit,
            "percentage_used": self.percentage_used,
            "message": self.message,
            "time_period": self.time_period,
            "remaining": self.remaining,
        }


@dataclass
class OptimizationSuggestion:
    """Cost optimization suggestion."""
    title: str
    description: str
    estimated_savings: float
    effort_level: str
    priority: int
    action_items: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "estimated_savings": self.estimated_savings,
            "estimated_savings_formatted": format_cost(self.estimated_savings),
            "effort_level": self.effort_level,
            "priority": self.priority,
            "action_items": self.action_items,
        }


class PricingModel(ABC):
    """Abstract base class for pricing models."""
    
    @abstractmethod
    def calculate_cost(self, **kwargs) -> float:
        """Calculate cost based on model-specific parameters."""
        pass
    
    @abstractmethod
    def get_rate_card(self) -> Dict[str, Any]:
        """Get the pricing rate card."""
        pass


class TokenPricingModel(PricingModel):
    """Token-based pricing model (e.g., for LLM APIs)."""
    
    DEFAULT_RATES = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "platform-forge": {"input": 0.001, "output": 0.002},
        "embedding": {"input": 0.0001, "output": 0.0},
    }
    
    def __init__(self, model: str = "platform-forge", custom_rates: Optional[Dict[str, float]] = None):
        self.model = model
        self.rates = custom_rates or self.DEFAULT_RATES.get(model, {"input": 0.001, "output": 0.002})
    
    def calculate_cost(self, input_tokens: int = 0, output_tokens: int = 0, **kwargs) -> float:
        """Calculate cost based on token counts."""
        input_cost = (input_tokens / 1000) * self.rates.get("input", 0.001)
        output_cost = (output_tokens / 1000) * self.rates.get("output", 0.002)
        return input_cost + output_cost
    
    def get_rate_card(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "unit": "per 1K tokens",
            "input_rate": self.rates.get("input", 0.001),
            "output_rate": self.rates.get("output", 0.002),
        }
    
    def estimate_tokens(self, text: str, is_code: bool = False) -> int:
        """Estimate token count for text."""
        if is_code:
            return len(text) // 3
        return len(text) // 4


class ComputePricingModel(PricingModel):
    """Compute-based pricing model (CPU/GPU time)."""
    
    DEFAULT_RATES = {
        "cpu_standard": 0.00001,
        "cpu_high": 0.00003,
        "gpu_t4": 0.0002,
        "gpu_a10": 0.0006,
        "gpu_a100": 0.002,
        "gpu_h100": 0.004,
    }
    
    def __init__(self, compute_type: str = "cpu_standard", custom_rate: Optional[float] = None):
        self.compute_type = compute_type
        self.rate_per_second = custom_rate or self.DEFAULT_RATES.get(compute_type, 0.00001)
    
    def calculate_cost(self, compute_seconds: float = 0, **kwargs) -> float:
        """Calculate cost based on compute time."""
        return compute_seconds * self.rate_per_second
    
    def get_rate_card(self) -> Dict[str, Any]:
        return {
            "compute_type": self.compute_type,
            "unit": "per second",
            "rate": self.rate_per_second,
            "hourly_rate": self.rate_per_second * 3600,
        }


class StoragePricingModel(PricingModel):
    """Storage-based pricing model."""
    
    DEFAULT_RATES = {
        "standard": 0.023,
        "infrequent": 0.0125,
        "archive": 0.004,
        "premium": 0.05,
    }
    
    def __init__(self, storage_tier: str = "standard", custom_rate: Optional[float] = None):
        self.storage_tier = storage_tier
        self.rate_per_gb_month = custom_rate or self.DEFAULT_RATES.get(storage_tier, 0.023)
    
    def calculate_cost(self, storage_gb: float = 0, duration_days: float = 30, **kwargs) -> float:
        """Calculate cost based on storage usage."""
        monthly_fraction = duration_days / 30
        return storage_gb * self.rate_per_gb_month * monthly_fraction
    
    def get_rate_card(self) -> Dict[str, Any]:
        return {
            "storage_tier": self.storage_tier,
            "unit": "per GB per month",
            "rate": self.rate_per_gb_month,
        }


class ActionPricingModel(PricingModel):
    """Action-based pricing model (per operation type)."""
    
    DEFAULT_RATES = {
        ActionType.CODE_GENERATION.value: 0.02,
        ActionType.CODE_ANALYSIS.value: 0.01,
        ActionType.CODE_REFACTORING.value: 0.03,
        ActionType.DEBUGGING.value: 0.025,
        ActionType.TESTING.value: 0.015,
        ActionType.DOCUMENTATION.value: 0.01,
        ActionType.INFRASTRUCTURE.value: 0.05,
        ActionType.SECURITY_SCAN.value: 0.04,
        ActionType.EMBEDDING.value: 0.0001,
        ActionType.CHAT.value: 0.005,
        ActionType.COMPLETION.value: 0.01,
        ActionType.VISION.value: 0.02,
        ActionType.AUDIO.value: 0.015,
        ActionType.CUSTOM.value: 0.01,
    }
    
    def __init__(self, custom_rates: Optional[Dict[str, float]] = None):
        self.rates = {**self.DEFAULT_RATES}
        if custom_rates:
            self.rates.update(custom_rates)
    
    def calculate_cost(self, action_type: str = "custom", count: int = 1, 
                       complexity_multiplier: float = 1.0, **kwargs) -> float:
        """Calculate cost based on action type and count."""
        base_rate = self.rates.get(action_type, self.rates[ActionType.CUSTOM.value])
        return base_rate * count * complexity_multiplier
    
    def get_rate_card(self) -> Dict[str, Any]:
        return {
            "unit": "per action",
            "rates": {k: v for k, v in self.rates.items()},
        }


class CostEstimator:
    """
    Estimates costs for various Platform Forge actions.
    
    Provides detailed cost estimates with breakdowns, confidence levels,
    and factors affecting the final cost.
    """
    
    def __init__(
        self,
        token_model: Optional[TokenPricingModel] = None,
        compute_model: Optional[ComputePricingModel] = None,
        storage_model: Optional[StoragePricingModel] = None,
        action_model: Optional[ActionPricingModel] = None,
    ):
        self.token_pricing = token_model or TokenPricingModel()
        self.compute_pricing = compute_model or ComputePricingModel()
        self.storage_pricing = storage_model or StoragePricingModel()
        self.action_pricing = action_model or ActionPricingModel()
        
        self._complexity_context_factors = {
            "small": (0, 1000),
            "medium": (1001, 8000),
            "large": (8001, 32000),
            "very_large": (32001, float("inf")),
        }
    
    def estimate_action_cost(
        self,
        action_type: str,
        complexity: Union[str, Complexity],
        context_size: int,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> CostEstimate:
        """
        Estimate cost for a general action.
        
        Args:
            action_type: Type of action (code_generation, debugging, etc.)
            complexity: Complexity level (trivial, simple, medium, complex, very_complex)
            context_size: Size of context in tokens/characters
            additional_context: Optional additional factors
        
        Returns:
            CostEstimate with detailed breakdown
        """
        if isinstance(complexity, str):
            complexity = Complexity(complexity.lower())
        
        multiplier = complexity.multiplier
        context_multiplier = self._get_context_multiplier(context_size)
        
        base_cost = self.action_pricing.calculate_cost(
            action_type=action_type,
            complexity_multiplier=multiplier * context_multiplier,
        )
        
        estimated_input_tokens = min(context_size, 16000)
        estimated_output_tokens = self._estimate_output_tokens(action_type, context_size)
        
        token_cost = self.token_pricing.calculate_cost(
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
        )
        
        compute_seconds = self._estimate_compute_time(action_type, complexity, context_size)
        compute_cost = self.compute_pricing.calculate_cost(compute_seconds=compute_seconds)
        
        total_expected = base_cost + token_cost + compute_cost
        
        factors = self._identify_cost_factors(action_type, complexity, context_size, additional_context)
        confidence = self._calculate_confidence(action_type, context_size, additional_context)
        
        variance = 0.2 + (1 - confidence) * 0.3
        min_cost = total_expected * (1 - variance)
        max_cost = total_expected * (1 + variance)
        
        breakdown = {
            "action_base": base_cost,
            "tokens": token_cost,
            "compute": compute_cost,
            "input_tokens_count": estimated_input_tokens,
            "output_tokens_count": estimated_output_tokens,
            "compute_seconds": compute_seconds,
        }
        
        warnings = self._generate_warnings(action_type, total_expected, context_size)
        recommendations = self._generate_recommendations(action_type, complexity, context_size)
        
        return CostEstimate(
            min_cost=round(min_cost, 6),
            max_cost=round(max_cost, 6),
            expected_cost=round(total_expected, 6),
            breakdown=breakdown,
            confidence=confidence,
            factors=factors,
            action_type=action_type,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "complexity": complexity.value,
                "context_size": context_size,
                "timestamp": datetime.now().isoformat(),
            },
        )
    
    def estimate_generation_cost(
        self,
        prompt_tokens: int,
        expected_output_tokens: int,
        model: str = "platform-forge",
    ) -> CostEstimate:
        """
        Estimate cost for text/code generation.
        
        Args:
            prompt_tokens: Number of input tokens
            expected_output_tokens: Expected number of output tokens
            model: Model to use for pricing
        
        Returns:
            CostEstimate with token-based breakdown
        """
        pricing = TokenPricingModel(model=model)
        
        token_cost = pricing.calculate_cost(
            input_tokens=prompt_tokens,
            output_tokens=expected_output_tokens,
        )
        
        total_tokens = prompt_tokens + expected_output_tokens
        compute_seconds = total_tokens * 0.001
        compute_cost = self.compute_pricing.calculate_cost(compute_seconds=compute_seconds)
        
        total_expected = token_cost + compute_cost
        
        confidence = 0.95 if prompt_tokens < 4000 else 0.85 if prompt_tokens < 16000 else 0.7
        variance = 0.1 + (1 - confidence) * 0.2
        
        factors = [
            f"Input tokens: {prompt_tokens:,}",
            f"Expected output tokens: {expected_output_tokens:,}",
            f"Model: {model}",
        ]
        
        if prompt_tokens > 8000:
            factors.append("Large context window")
        if expected_output_tokens > 2000:
            factors.append("Long expected output")
        
        breakdown = {
            "input_tokens": (prompt_tokens / 1000) * pricing.rates.get("input", 0.001),
            "output_tokens": (expected_output_tokens / 1000) * pricing.rates.get("output", 0.002),
            "compute": compute_cost,
        }
        
        return CostEstimate(
            min_cost=round(total_expected * (1 - variance), 6),
            max_cost=round(total_expected * (1 + variance), 6),
            expected_cost=round(total_expected, 6),
            breakdown=breakdown,
            confidence=confidence,
            factors=factors,
            action_type="generation",
            metadata={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "expected_output_tokens": expected_output_tokens,
            },
        )
    
    def estimate_analysis_cost(
        self,
        code_size: int,
        analysis_depth: str = "standard",
        include_security: bool = False,
        include_performance: bool = False,
    ) -> CostEstimate:
        """
        Estimate cost for code analysis.
        
        Args:
            code_size: Size of code in bytes/characters
            analysis_depth: Depth of analysis (quick, standard, deep, comprehensive)
            include_security: Include security analysis
            include_performance: Include performance analysis
        
        Returns:
            CostEstimate with analysis breakdown
        """
        depth_multipliers = {
            "quick": 0.5,
            "standard": 1.0,
            "deep": 2.0,
            "comprehensive": 3.5,
        }
        
        depth_multiplier = depth_multipliers.get(analysis_depth, 1.0)
        
        estimated_tokens = code_size // 4
        base_analysis_cost = self.action_pricing.calculate_cost(
            action_type=ActionType.CODE_ANALYSIS.value,
            complexity_multiplier=depth_multiplier,
        )
        
        token_cost = self.token_pricing.calculate_cost(
            input_tokens=estimated_tokens,
            output_tokens=estimated_tokens // 4,
        )
        
        additional_cost = 0.0
        additional_factors = []
        
        if include_security:
            security_cost = self.action_pricing.calculate_cost(
                action_type=ActionType.SECURITY_SCAN.value,
            )
            additional_cost += security_cost
            additional_factors.append("Security analysis included")
        
        if include_performance:
            perf_cost = base_analysis_cost * 0.5
            additional_cost += perf_cost
            additional_factors.append("Performance analysis included")
        
        total_expected = base_analysis_cost + token_cost + additional_cost
        
        confidence = 0.9 if code_size < 10000 else 0.8 if code_size < 100000 else 0.65
        
        factors = [
            f"Code size: {code_size:,} bytes",
            f"Analysis depth: {analysis_depth}",
            f"Estimated tokens: {estimated_tokens:,}",
        ] + additional_factors
        
        breakdown = {
            "base_analysis": base_analysis_cost,
            "token_processing": token_cost,
            "security_scan": additional_cost if include_security else 0.0,
            "performance_analysis": base_analysis_cost * 0.5 if include_performance else 0.0,
        }
        
        variance = 0.15 + (1 - confidence) * 0.25
        
        return CostEstimate(
            min_cost=round(total_expected * (1 - variance), 6),
            max_cost=round(total_expected * (1 + variance), 6),
            expected_cost=round(total_expected, 6),
            breakdown=breakdown,
            confidence=confidence,
            factors=factors,
            action_type="analysis",
            metadata={
                "code_size": code_size,
                "analysis_depth": analysis_depth,
                "include_security": include_security,
                "include_performance": include_performance,
            },
        )
    
    def _get_context_multiplier(self, context_size: int) -> float:
        """Calculate multiplier based on context size."""
        if context_size < 1000:
            return 0.8
        elif context_size < 4000:
            return 1.0
        elif context_size < 8000:
            return 1.3
        elif context_size < 16000:
            return 1.6
        else:
            return 2.0 + (context_size - 16000) / 16000 * 0.5
    
    def _estimate_output_tokens(self, action_type: str, context_size: int) -> int:
        """Estimate expected output tokens based on action type."""
        ratios = {
            ActionType.CODE_GENERATION.value: 1.5,
            ActionType.CODE_ANALYSIS.value: 0.3,
            ActionType.DEBUGGING.value: 0.4,
            ActionType.DOCUMENTATION.value: 0.8,
            ActionType.TESTING.value: 1.2,
            ActionType.CODE_REFACTORING.value: 1.0,
        }
        ratio = ratios.get(action_type, 0.5)
        return min(int(context_size * ratio), 8000)
    
    def _estimate_compute_time(
        self,
        action_type: str,
        complexity: Complexity,
        context_size: int,
    ) -> float:
        """Estimate compute time in seconds."""
        base_times = {
            ActionType.CODE_GENERATION.value: 2.0,
            ActionType.CODE_ANALYSIS.value: 1.5,
            ActionType.DEBUGGING.value: 3.0,
            ActionType.INFRASTRUCTURE.value: 5.0,
            ActionType.SECURITY_SCAN.value: 4.0,
        }
        base_time = base_times.get(action_type, 1.0)
        
        context_factor = 1 + (context_size / 10000)
        
        return base_time * complexity.multiplier * context_factor
    
    def _identify_cost_factors(
        self,
        action_type: str,
        complexity: Complexity,
        context_size: int,
        additional_context: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Identify factors that affect the cost estimate."""
        factors = []
        
        factors.append(f"Action type: {action_type}")
        factors.append(f"Complexity: {complexity.value}")
        
        if context_size < 1000:
            factors.append("Small context (optimized)")
        elif context_size > 8000:
            factors.append("Large context (higher cost)")
        
        if complexity in [Complexity.COMPLEX, Complexity.VERY_COMPLEX]:
            factors.append("High complexity multiplier applied")
        
        if additional_context:
            if additional_context.get("multi_file"):
                factors.append("Multi-file operation")
            if additional_context.get("requires_external_api"):
                factors.append("External API calls required")
        
        return factors
    
    def _calculate_confidence(
        self,
        action_type: str,
        context_size: int,
        additional_context: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate confidence level for the estimate."""
        base_confidence = 0.85
        
        well_understood_actions = [
            ActionType.CODE_GENERATION.value,
            ActionType.CODE_ANALYSIS.value,
            ActionType.EMBEDDING.value,
        ]
        
        if action_type in well_understood_actions:
            base_confidence += 0.05
        
        if context_size > 16000:
            base_confidence -= 0.15
        elif context_size > 8000:
            base_confidence -= 0.05
        
        if additional_context:
            if additional_context.get("multi_file"):
                base_confidence -= 0.1
            if additional_context.get("dynamic_output"):
                base_confidence -= 0.1
        
        return max(0.5, min(0.95, base_confidence))
    
    def _generate_warnings(
        self,
        action_type: str,
        expected_cost: float,
        context_size: int,
    ) -> List[str]:
        """Generate warnings about the cost estimate."""
        warnings = []
        
        if expected_cost > 0.5:
            warnings.append(f"High cost action: {format_cost(expected_cost)}")
        
        if context_size > 12000:
            warnings.append("Large context may result in truncation")
        
        expensive_actions = [ActionType.INFRASTRUCTURE.value, ActionType.SECURITY_SCAN.value]
        if action_type in expensive_actions:
            warnings.append(f"'{action_type}' is a premium operation")
        
        return warnings
    
    def _generate_recommendations(
        self,
        action_type: str,
        complexity: Complexity,
        context_size: int,
    ) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        if context_size > 8000:
            recommendations.append("Consider reducing context size by removing unnecessary code")
        
        if complexity in [Complexity.COMPLEX, Complexity.VERY_COMPLEX]:
            recommendations.append("Break down into smaller, simpler operations to reduce costs")
        
        if action_type == ActionType.CODE_GENERATION.value:
            recommendations.append("Use cached templates for common patterns")
        
        return recommendations


class CostTracker:
    """
    Tracks spending in real-time with budget alerts.
    
    Features:
    - Real-time cost recording
    - Budget threshold alerts (50%, 75%, 90%, 100%)
    - Daily/weekly/monthly summaries
    - Cost breakdown by action type
    """
    
    ALERT_THRESHOLDS = [
        (0.50, AlertLevel.WARNING, "50% of budget used"),
        (0.75, AlertLevel.HIGH, "75% of budget used"),
        (0.90, AlertLevel.CRITICAL, "90% of budget used"),
        (1.00, AlertLevel.EXCEEDED, "Budget exceeded"),
    ]
    
    def __init__(
        self,
        daily_budget: Optional[float] = None,
        weekly_budget: Optional[float] = None,
        monthly_budget: Optional[float] = None,
        alert_callback: Optional[Callable[[BudgetAlert], None]] = None,
    ):
        self.daily_budget = daily_budget
        self.weekly_budget = weekly_budget
        self.monthly_budget = monthly_budget
        self.alert_callback = alert_callback
        
        self._records: List[CostRecord] = []
        self._last_alert_level: Dict[str, AlertLevel] = {}
    
    def record_cost(
        self,
        action_type: str,
        amount: float,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        unit: CostUnit = CostUnit.DOLLARS,
    ) -> CostRecord:
        """
        Record a cost event.
        
        Args:
            action_type: Type of action performed
            amount: Cost amount
            description: Optional description
            metadata: Optional additional data
            unit: Currency unit
        
        Returns:
            The recorded CostRecord
        """
        record = CostRecord(
            timestamp=datetime.now(),
            action_type=action_type,
            amount=amount,
            unit=unit,
            description=description,
            metadata=metadata or {},
        )
        
        self._records.append(record)
        
        self._check_and_trigger_alerts()
        
        return record
    
    def get_daily_spending(self, date: Optional[datetime] = None) -> float:
        """Get total spending for a specific day."""
        target_date = (date or datetime.now()).date()
        return sum(
            r.amount for r in self._records
            if r.timestamp.date() == target_date
        )
    
    def get_weekly_spending(self, week_start: Optional[datetime] = None) -> float:
        """Get total spending for a week."""
        if week_start is None:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())
        
        week_start_date = week_start.date()
        week_end_date = week_start_date + timedelta(days=7)
        
        return sum(
            r.amount for r in self._records
            if week_start_date <= r.timestamp.date() < week_end_date
        )
    
    def get_monthly_spending(self, year: Optional[int] = None, month: Optional[int] = None) -> float:
        """Get total spending for a month."""
        now = datetime.now()
        target_year = year or now.year
        target_month = month or now.month
        
        return sum(
            r.amount for r in self._records
            if r.timestamp.year == target_year and r.timestamp.month == target_month
        )
    
    def get_spending_by_action_type(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get spending breakdown by action type."""
        filtered = self._filter_records(start_date, end_date)
        
        breakdown: Dict[str, float] = {}
        for record in filtered:
            breakdown[record.action_type] = breakdown.get(record.action_type, 0) + record.amount
        
        return dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get detailed daily summary."""
        target_date = (date or datetime.now()).date()
        daily_records = [r for r in self._records if r.timestamp.date() == target_date]
        
        total = sum(r.amount for r in daily_records)
        by_type = {}
        for r in daily_records:
            by_type[r.action_type] = by_type.get(r.action_type, 0) + r.amount
        
        return {
            "date": target_date.isoformat(),
            "total_spending": total,
            "total_formatted": format_cost(total),
            "action_count": len(daily_records),
            "by_action_type": by_type,
            "budget": self.daily_budget,
            "budget_remaining": (self.daily_budget - total) if self.daily_budget else None,
            "budget_percentage": (total / self.daily_budget * 100) if self.daily_budget else None,
        }
    
    def get_weekly_summary(self, week_start: Optional[datetime] = None) -> Dict[str, Any]:
        """Get detailed weekly summary."""
        if week_start is None:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())
        
        week_start_date = week_start.date()
        week_end_date = week_start_date + timedelta(days=7)
        
        weekly_records = [
            r for r in self._records
            if week_start_date <= r.timestamp.date() < week_end_date
        ]
        
        total = sum(r.amount for r in weekly_records)
        
        daily_breakdown = {}
        for r in weekly_records:
            day_str = r.timestamp.strftime("%A")
            daily_breakdown[day_str] = daily_breakdown.get(day_str, 0) + r.amount
        
        return {
            "week_start": week_start_date.isoformat(),
            "week_end": (week_end_date - timedelta(days=1)).isoformat(),
            "total_spending": total,
            "total_formatted": format_cost(total),
            "action_count": len(weekly_records),
            "daily_breakdown": daily_breakdown,
            "budget": self.weekly_budget,
            "budget_remaining": (self.weekly_budget - total) if self.weekly_budget else None,
        }
    
    def get_monthly_summary(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get detailed monthly summary."""
        now = datetime.now()
        target_year = year or now.year
        target_month = month or now.month
        
        monthly_records = [
            r for r in self._records
            if r.timestamp.year == target_year and r.timestamp.month == target_month
        ]
        
        total = sum(r.amount for r in monthly_records)
        
        by_type = {}
        for r in monthly_records:
            by_type[r.action_type] = by_type.get(r.action_type, 0) + r.amount
        
        weekly_breakdown = {}
        for r in monthly_records:
            week_num = r.timestamp.isocalendar()[1]
            weekly_breakdown[f"Week {week_num}"] = weekly_breakdown.get(f"Week {week_num}", 0) + r.amount
        
        return {
            "year": target_year,
            "month": target_month,
            "total_spending": total,
            "total_formatted": format_cost(total),
            "action_count": len(monthly_records),
            "by_action_type": by_type,
            "weekly_breakdown": weekly_breakdown,
            "budget": self.monthly_budget,
            "budget_remaining": (self.monthly_budget - total) if self.monthly_budget else None,
            "budget_percentage": (total / self.monthly_budget * 100) if self.monthly_budget else None,
        }
    
    def is_approaching_limit(self, threshold: float = 0.75) -> bool:
        """Check if spending is approaching any budget limit."""
        checks = [
            (self.daily_budget, self.get_daily_spending()),
            (self.weekly_budget, self.get_weekly_spending()),
            (self.monthly_budget, self.get_monthly_spending()),
        ]
        
        for budget, spending in checks:
            if budget and spending / budget >= threshold:
                return True
        return False
    
    def get_budget_alert(self) -> Optional[BudgetAlert]:
        """Get current budget alert if any threshold is exceeded."""
        alerts = []
        
        if self.daily_budget:
            daily = self.get_daily_spending()
            percentage = daily / self.daily_budget
            for threshold, level, msg in self.ALERT_THRESHOLDS:
                if percentage >= threshold:
                    alerts.append(BudgetAlert(
                        level=level,
                        current_spending=daily,
                        budget_limit=self.daily_budget,
                        percentage_used=percentage * 100,
                        message=f"Daily budget: {msg}",
                        time_period="daily",
                        remaining=max(0, self.daily_budget - daily),
                    ))
        
        if self.monthly_budget:
            monthly = self.get_monthly_spending()
            percentage = monthly / self.monthly_budget
            for threshold, level, msg in self.ALERT_THRESHOLDS:
                if percentage >= threshold:
                    alerts.append(BudgetAlert(
                        level=level,
                        current_spending=monthly,
                        budget_limit=self.monthly_budget,
                        percentage_used=percentage * 100,
                        message=f"Monthly budget: {msg}",
                        time_period="monthly",
                        remaining=max(0, self.monthly_budget - monthly),
                    ))
        
        if alerts:
            return max(alerts, key=lambda a: list(AlertLevel).index(a.level))
        return None
    
    def get_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action_type: Optional[str] = None,
    ) -> List[CostRecord]:
        """Get filtered cost history."""
        filtered = self._filter_records(start_date, end_date)
        
        if action_type:
            filtered = [r for r in filtered if r.action_type == action_type]
        
        return filtered
    
    def _filter_records(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> List[CostRecord]:
        """Filter records by date range."""
        filtered = self._records
        
        if start_date:
            filtered = [r for r in filtered if r.timestamp >= start_date]
        if end_date:
            filtered = [r for r in filtered if r.timestamp <= end_date]
        
        return filtered
    
    def _check_and_trigger_alerts(self):
        """Check budgets and trigger alerts if needed."""
        alert = self.get_budget_alert()
        if alert and self.alert_callback:
            period_key = alert.time_period
            last_level = self._last_alert_level.get(period_key)
            
            if last_level is None or list(AlertLevel).index(alert.level) > list(AlertLevel).index(last_level):
                self._last_alert_level[period_key] = alert.level
                self.alert_callback(alert)


class BudgetManager:
    """
    Manages spending limits and blocks actions that exceed limits.
    
    Features:
    - Per-action limits
    - Daily/monthly limits
    - Blocking with override option
    - Cheaper alternative suggestions
    """
    
    def __init__(
        self,
        per_action_limit: Optional[float] = None,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None,
        tracker: Optional[CostTracker] = None,
    ):
        self.per_action_limit = per_action_limit
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.tracker = tracker or CostTracker(
            daily_budget=daily_limit,
            monthly_budget=monthly_limit,
        )
        
        self._overrides: Dict[str, datetime] = {}
        self._action_limits: Dict[str, float] = {}
    
    def set_action_limit(self, action_type: str, limit: float):
        """Set a specific limit for an action type."""
        self._action_limits[action_type] = limit
    
    def check_action(
        self,
        action_type: str,
        estimated_cost: float,
        allow_override: bool = True,
    ) -> Tuple[bool, str, Optional[List[str]]]:
        """
        Check if an action can proceed within budget limits.
        
        Args:
            action_type: Type of action
            estimated_cost: Estimated cost of the action
            allow_override: Whether to allow override if blocked
        
        Returns:
            Tuple of (can_proceed, message, alternatives)
        """
        action_limit = self._action_limits.get(action_type, self.per_action_limit)
        if action_limit and estimated_cost > action_limit:
            alternatives = self._suggest_alternatives(action_type, estimated_cost)
            msg = f"Action cost ({format_cost(estimated_cost)}) exceeds per-action limit ({format_cost(action_limit)})"
            return (False, msg, alternatives)
        
        if self.daily_limit:
            daily_spending = self.tracker.get_daily_spending()
            if daily_spending + estimated_cost > self.daily_limit:
                remaining = self.daily_limit - daily_spending
                msg = f"Action would exceed daily budget. Remaining: {format_cost(remaining)}"
                alternatives = self._suggest_alternatives(action_type, estimated_cost)
                return (False, msg, alternatives)
        
        if self.monthly_limit:
            monthly_spending = self.tracker.get_monthly_spending()
            if monthly_spending + estimated_cost > self.monthly_limit:
                remaining = self.monthly_limit - monthly_spending
                msg = f"Action would exceed monthly budget. Remaining: {format_cost(remaining)}"
                alternatives = self._suggest_alternatives(action_type, estimated_cost)
                return (False, msg, alternatives)
        
        return (True, "Action approved within budget", None)
    
    def override_limit(self, action_type: str, reason: str) -> bool:
        """
        Override budget limits for a specific action type.
        
        Args:
            action_type: Type of action to override
            reason: Reason for the override
        
        Returns:
            True if override was registered
        """
        self._overrides[action_type] = datetime.now()
        return True
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status overview."""
        daily_spending = self.tracker.get_daily_spending()
        monthly_spending = self.tracker.get_monthly_spending()
        
        return {
            "daily": {
                "limit": self.daily_limit,
                "spent": daily_spending,
                "remaining": (self.daily_limit - daily_spending) if self.daily_limit else None,
                "percentage": (daily_spending / self.daily_limit * 100) if self.daily_limit else None,
            },
            "monthly": {
                "limit": self.monthly_limit,
                "spent": monthly_spending,
                "remaining": (self.monthly_limit - monthly_spending) if self.monthly_limit else None,
                "percentage": (monthly_spending / self.monthly_limit * 100) if self.monthly_limit else None,
            },
            "per_action": {
                "default_limit": self.per_action_limit,
                "custom_limits": self._action_limits,
            },
            "active_overrides": len(self._overrides),
        }
    
    def _suggest_alternatives(
        self,
        action_type: str,
        target_cost: float,
    ) -> List[str]:
        """Suggest cheaper alternatives for an action."""
        alternatives = []
        
        alternative_actions = {
            ActionType.CODE_GENERATION.value: [
                "Use a simpler model (30-50% cheaper)",
                "Reduce context size",
                "Use cached templates",
            ],
            ActionType.CODE_ANALYSIS.value: [
                "Use quick analysis mode",
                "Analyze specific files instead of entire codebase",
                "Skip security analysis for non-production code",
            ],
            ActionType.INFRASTRUCTURE.value: [
                "Generate basic configuration first",
                "Use predefined templates",
                "Generate for single cloud provider",
            ],
            ActionType.DEBUGGING.value: [
                "Start with quick diagnosis",
                "Focus on specific error messages",
                "Use lighter analysis depth",
            ],
        }
        
        alternatives = alternative_actions.get(action_type, [
            "Break down into smaller operations",
            "Use a lighter model",
            "Reduce input size",
        ])
        
        return alternatives


class CostOptimizer:
    """
    Provides suggestions for reducing costs and identifies expensive patterns.
    
    Features:
    - Cost reduction suggestions
    - Pattern identification
    - Batching recommendations
    - Cache utilization analysis
    """
    
    def __init__(self, tracker: Optional[CostTracker] = None):
        self.tracker = tracker
        self._cache_stats: Dict[str, int] = {"hits": 0, "misses": 0}
    
    def analyze_usage(
        self,
        history: Optional[List[CostRecord]] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze usage patterns for optimization opportunities.
        
        Args:
            history: Cost history to analyze (uses tracker if not provided)
            days: Number of days to analyze
        
        Returns:
            Analysis results with patterns and recommendations
        """
        if history is None and self.tracker:
            start_date = datetime.now() - timedelta(days=days)
            history = self.tracker.get_history(start_date=start_date)
        
        if not history:
            return {"status": "no_data", "suggestions": []}
        
        total_cost = sum(r.amount for r in history)
        action_counts: Dict[str, int] = {}
        action_costs: Dict[str, float] = {}
        
        for record in history:
            action_counts[record.action_type] = action_counts.get(record.action_type, 0) + 1
            action_costs[record.action_type] = action_costs.get(record.action_type, 0) + record.amount
        
        expensive_patterns = []
        for action, cost in sorted(action_costs.items(), key=lambda x: x[1], reverse=True):
            if cost > total_cost * 0.2:
                expensive_patterns.append({
                    "action": action,
                    "total_cost": cost,
                    "percentage": cost / total_cost * 100,
                    "count": action_counts[action],
                    "avg_cost": cost / action_counts[action],
                })
        
        suggestions = self._generate_optimization_suggestions(
            action_counts, action_costs, total_cost, history
        )
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "total_actions": len(history),
            "avg_cost_per_action": total_cost / len(history) if history else 0,
            "by_action_type": {
                action: {
                    "count": action_counts[action],
                    "total_cost": action_costs[action],
                    "avg_cost": action_costs[action] / action_counts[action],
                    "percentage": action_costs[action] / total_cost * 100,
                }
                for action in action_counts
            },
            "expensive_patterns": expensive_patterns,
            "suggestions": suggestions,
        }
    
    def get_suggestions(
        self,
        history: Optional[List[CostRecord]] = None,
    ) -> List[OptimizationSuggestion]:
        """Get prioritized list of optimization suggestions."""
        analysis = self.analyze_usage(history)
        return analysis.get("suggestions", [])
    
    def recommend_batching(
        self,
        pending_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Recommend whether to batch actions or execute individually.
        
        Args:
            pending_actions: List of pending actions with their types and sizes
        
        Returns:
            Recommendation with cost comparison
        """
        if not pending_actions:
            return {"recommendation": "no_actions", "batching_benefit": 0}
        
        individual_cost = 0.0
        action_pricing = ActionPricingModel()
        
        for action in pending_actions:
            individual_cost += action_pricing.calculate_cost(
                action_type=action.get("type", "custom"),
            )
        
        batchable_types: Dict[str, List[Dict]] = {}
        for action in pending_actions:
            action_type = action.get("type", "custom")
            if action_type not in batchable_types:
                batchable_types[action_type] = []
            batchable_types[action_type].append(action)
        
        batched_cost = 0.0
        batch_discount = 0.7
        
        for action_type, actions in batchable_types.items():
            base_cost = action_pricing.calculate_cost(action_type=action_type)
            if len(actions) > 1:
                batched_cost += base_cost * len(actions) * batch_discount
            else:
                batched_cost += base_cost
        
        savings = individual_cost - batched_cost
        recommendation = "batch" if savings > 0 else "individual"
        
        return {
            "recommendation": recommendation,
            "individual_cost": individual_cost,
            "batched_cost": batched_cost,
            "savings": savings,
            "savings_percentage": (savings / individual_cost * 100) if individual_cost > 0 else 0,
            "batch_groups": {k: len(v) for k, v in batchable_types.items()},
        }
    
    def analyze_cache_efficiency(self) -> Dict[str, Any]:
        """Analyze cache hit/miss ratio and suggest improvements."""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        if total == 0:
            return {
                "status": "no_data",
                "hit_ratio": 0,
                "recommendations": ["Enable caching for repeated operations"],
            }
        
        hit_ratio = self._cache_stats["hits"] / total
        
        recommendations = []
        if hit_ratio < 0.3:
            recommendations.append("Consider caching frequently used patterns")
            recommendations.append("Increase cache TTL for stable content")
        elif hit_ratio < 0.6:
            recommendations.append("Optimize cache key strategy")
        
        return {
            "total_requests": total,
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "hit_ratio": hit_ratio,
            "estimated_savings": self._cache_stats["hits"] * 0.01,
            "recommendations": recommendations,
        }
    
    def record_cache_access(self, hit: bool):
        """Record a cache hit or miss."""
        if hit:
            self._cache_stats["hits"] += 1
        else:
            self._cache_stats["misses"] += 1
    
    def identify_expensive_patterns(
        self,
        history: List[CostRecord],
        threshold_percentage: float = 20.0,
    ) -> List[Dict[str, Any]]:
        """Identify patterns that consume significant budget."""
        if not history:
            return []
        
        total_cost = sum(r.amount for r in history)
        threshold_cost = total_cost * (threshold_percentage / 100)
        
        action_costs: Dict[str, float] = {}
        action_counts: Dict[str, int] = {}
        
        for record in history:
            action_costs[record.action_type] = action_costs.get(record.action_type, 0) + record.amount
            action_counts[record.action_type] = action_counts.get(record.action_type, 0) + 1
        
        expensive = []
        for action, cost in action_costs.items():
            if cost >= threshold_cost:
                expensive.append({
                    "action_type": action,
                    "total_cost": cost,
                    "count": action_counts[action],
                    "avg_cost": cost / action_counts[action],
                    "percentage_of_total": cost / total_cost * 100,
                })
        
        return sorted(expensive, key=lambda x: x["total_cost"], reverse=True)
    
    def _generate_optimization_suggestions(
        self,
        action_counts: Dict[str, int],
        action_costs: Dict[str, float],
        total_cost: float,
        history: List[CostRecord],
    ) -> List[OptimizationSuggestion]:
        """Generate prioritized optimization suggestions."""
        suggestions = []
        
        for action, cost in action_costs.items():
            if cost > total_cost * 0.3:
                estimated_savings = cost * 0.2
                suggestions.append(OptimizationSuggestion(
                    title=f"Optimize {action} usage",
                    description=f"'{action}' accounts for {cost/total_cost*100:.1f}% of costs",
                    estimated_savings=estimated_savings,
                    effort_level="medium",
                    priority=1,
                    action_items=[
                        f"Review {action} operations for necessity",
                        "Consider caching results",
                        "Batch similar operations together",
                    ],
                ))
        
        for action, count in action_counts.items():
            if count > 10:
                suggestions.append(OptimizationSuggestion(
                    title=f"Batch {action} operations",
                    description=f"Found {count} individual '{action}' operations that could be batched",
                    estimated_savings=action_costs.get(action, 0) * 0.3,
                    effort_level="low",
                    priority=2,
                    action_items=[
                        "Group similar operations",
                        "Use batch API when available",
                        "Implement request queuing",
                    ],
                ))
        
        cache_efficiency = self.analyze_cache_efficiency()
        if cache_efficiency.get("hit_ratio", 1) < 0.5:
            suggestions.append(OptimizationSuggestion(
                title="Improve cache utilization",
                description=f"Current cache hit ratio is {cache_efficiency.get('hit_ratio', 0)*100:.1f}%",
                estimated_savings=total_cost * 0.15,
                effort_level="medium",
                priority=3,
                action_items=[
                    "Cache frequently used results",
                    "Implement result memoization",
                    "Use persistent cache for stable data",
                ],
            ))
        
        return sorted(suggestions, key=lambda s: s.priority)


def format_cost(amount: float, unit: str = "$", decimals: int = 4) -> str:
    """
    Format a cost amount as a currency string.
    
    Args:
        amount: The amount to format
        unit: Currency symbol (default: $)
        decimals: Number of decimal places (default: 4 for small amounts)
    
    Returns:
        Formatted string like "$0.0123" or "$1.50"
    """
    if amount == 0:
        return f"{unit}0.00"
    
    if amount >= 1:
        return f"{unit}{amount:,.2f}"
    elif amount >= 0.01:
        return f"{unit}{amount:.4f}"
    else:
        return f"{unit}{amount:.6f}"


def compare_costs(
    options: List[Dict[str, Any]],
    include_savings: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compare multiple options and rank by cost with savings calculation.
    
    Args:
        options: List of options with 'name' and 'cost' keys
        include_savings: Whether to include savings vs most expensive
    
    Returns:
        Ranked list with savings information
    """
    if not options:
        return []
    
    sorted_options = sorted(options, key=lambda x: x.get("cost", 0))
    
    max_cost = max(o.get("cost", 0) for o in options)
    min_cost = min(o.get("cost", 0) for o in options)
    
    result = []
    for i, option in enumerate(sorted_options):
        cost = option.get("cost", 0)
        entry = {
            "rank": i + 1,
            "name": option.get("name", f"Option {i+1}"),
            "cost": cost,
            "cost_formatted": format_cost(cost),
            **{k: v for k, v in option.items() if k not in ["name", "cost"]},
        }
        
        if include_savings and max_cost > 0:
            savings = max_cost - cost
            entry["savings_vs_most_expensive"] = savings
            entry["savings_formatted"] = format_cost(savings)
            entry["savings_percentage"] = (savings / max_cost * 100) if max_cost > 0 else 0
        
        result.append(entry)
    
    return result


def get_cost_breakdown(
    action_history: List[Union[CostRecord, Dict[str, Any]]],
    group_by: str = "action_type",
    include_daily: bool = True,
) -> Dict[str, Any]:
    """
    Generate a detailed cost breakdown report from action history.
    
    Args:
        action_history: List of cost records or dicts
        group_by: How to group costs (action_type, date, week)
        include_daily: Include daily breakdown
    
    Returns:
        Detailed breakdown report
    """
    if not action_history:
        return {
            "total": 0,
            "total_formatted": "$0.00",
            "action_count": 0,
            "breakdown": {},
            "summary": "No cost data available",
        }
    
    records = []
    for item in action_history:
        if isinstance(item, CostRecord):
            records.append(item)
        elif isinstance(item, dict):
            records.append(CostRecord(
                timestamp=datetime.fromisoformat(item.get("timestamp", datetime.now().isoformat())),
                action_type=item.get("action_type", "unknown"),
                amount=item.get("amount", 0),
                unit=CostUnit(item.get("unit", "dollars")),
            ))
    
    total = sum(r.amount for r in records)
    
    breakdown: Dict[str, Any] = {}
    
    if group_by == "action_type":
        by_type: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for r in records:
            by_type[r.action_type] = by_type.get(r.action_type, 0) + r.amount
            counts[r.action_type] = counts.get(r.action_type, 0) + 1
        
        breakdown = {
            action: {
                "total": cost,
                "total_formatted": format_cost(cost),
                "count": counts[action],
                "avg_cost": cost / counts[action],
                "percentage": (cost / total * 100) if total > 0 else 0,
            }
            for action, cost in sorted(by_type.items(), key=lambda x: x[1], reverse=True)
        }
    
    elif group_by == "date":
        by_date: Dict[str, float] = {}
        for r in records:
            date_str = r.timestamp.strftime("%Y-%m-%d")
            by_date[date_str] = by_date.get(date_str, 0) + r.amount
        
        breakdown = {
            date: {
                "total": cost,
                "total_formatted": format_cost(cost),
                "percentage": (cost / total * 100) if total > 0 else 0,
            }
            for date, cost in sorted(by_date.items())
        }
    
    daily_breakdown = None
    if include_daily:
        by_day: Dict[str, float] = {}
        for r in records:
            date_str = r.timestamp.strftime("%Y-%m-%d")
            by_day[date_str] = by_day.get(date_str, 0) + r.amount
        
        daily_breakdown = {
            date: {"total": cost, "formatted": format_cost(cost)}
            for date, cost in sorted(by_day.items())
        }
    
    avg_per_action = total / len(records) if records else 0
    
    return {
        "total": total,
        "total_formatted": format_cost(total),
        "action_count": len(records),
        "avg_per_action": avg_per_action,
        "avg_per_action_formatted": format_cost(avg_per_action),
        "breakdown": breakdown,
        "daily_breakdown": daily_breakdown,
        "period": {
            "start": min(r.timestamp for r in records).isoformat() if records else None,
            "end": max(r.timestamp for r in records).isoformat() if records else None,
            "days": (max(r.timestamp for r in records) - min(r.timestamp for r in records)).days + 1 if records else 0,
        },
        "summary": f"Total: {format_cost(total)} across {len(records)} actions",
    }


def estimate_action_cost(
    action_type: str,
    complexity: str = "medium",
    context_size: int = 4096,
) -> CostEstimate:
    """
    Quick helper to estimate action cost.
    
    Args:
        action_type: Type of action
        complexity: Complexity level
        context_size: Context size in tokens/characters
    
    Returns:
        CostEstimate with full details
    """
    estimator = CostEstimator()
    return estimator.estimate_action_cost(action_type, complexity, context_size)


def create_budget_manager(
    daily_limit: Optional[float] = None,
    monthly_limit: Optional[float] = None,
    per_action_limit: Optional[float] = None,
) -> BudgetManager:
    """
    Create a configured BudgetManager instance.
    
    Args:
        daily_limit: Daily spending limit
        monthly_limit: Monthly spending limit
        per_action_limit: Per-action spending limit
    
    Returns:
        Configured BudgetManager
    """
    return BudgetManager(
        daily_limit=daily_limit,
        monthly_limit=monthly_limit,
        per_action_limit=per_action_limit,
    )


__all__ = [
    "CostUnit",
    "ActionType",
    "Complexity",
    "AlertLevel",
    "CostBreakdown",
    "CostEstimate",
    "CostRecord",
    "BudgetAlert",
    "OptimizationSuggestion",
    "PricingModel",
    "TokenPricingModel",
    "ComputePricingModel",
    "StoragePricingModel",
    "ActionPricingModel",
    "CostEstimator",
    "CostTracker",
    "BudgetManager",
    "CostOptimizer",
    "format_cost",
    "compare_costs",
    "get_cost_breakdown",
    "estimate_action_cost",
    "create_budget_manager",
]
