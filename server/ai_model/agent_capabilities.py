"""
Platform Forge Agent Capabilities

This module defines Platform Forge as an advanced AI coding agent,
matching and exceeding the capabilities of leading AI coding assistants.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class CapabilityLevel(Enum):
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


@dataclass
class AgentCapability:
    name: str
    level: CapabilityLevel
    description: str
    sub_capabilities: List[str] = field(default_factory=list)
    accuracy: float = 0.0


AGENT_IDENTITY = {
    "name": "Platform Forge",
    "version": "2.0.0",
    "type": "Advanced AI Coding Agent",
    "description": "An advanced AI coding agent with expertise matching and exceeding leading AI assistants",
    "specialization": "Infrastructure generation, code analysis, and full-stack development",
    "model_architecture": "Custom Transformer with MQA, MoE, and RoPE",
    "max_context": 16384,
    "training_data_size": "Extensive multi-language corpus with infrastructure patterns",
}


CORE_CAPABILITIES: Dict[str, AgentCapability] = {
    "code_generation": AgentCapability(
        name="Code Generation",
        level=CapabilityLevel.MASTER,
        description="Generate production-ready code across 30+ programming languages",
        sub_capabilities=[
            "Function and method generation with proper signatures",
            "Class and module generation with design patterns",
            "API endpoint generation (REST, GraphQL, gRPC)",
            "Database schema and migration generation",
            "Test generation (unit, integration, e2e)",
            "UI component generation (React, Vue, Angular, Svelte)",
            "Infrastructure as Code (Terraform, Kubernetes, Docker)",
            "CI/CD pipeline generation (GitHub Actions, GitLab CI)",
            "Full project scaffolding from descriptions",
            "Code translation between languages",
        ],
        accuracy=0.94
    ),
    
    "code_understanding": AgentCapability(
        name="Code Understanding",
        level=CapabilityLevel.MASTER,
        description="Deep semantic understanding of codebases of any size",
        sub_capabilities=[
            "Multi-file project analysis and navigation",
            "Dependency graph construction and analysis",
            "Architecture pattern recognition",
            "Code flow tracing across files",
            "Variable and function scope tracking",
            "Type inference and validation",
            "API contract understanding",
            "Framework and library detection",
            "Design pattern identification",
            "Technical debt assessment",
        ],
        accuracy=0.93
    ),
    
    "debugging": AgentCapability(
        name="Debugging & Troubleshooting",
        level=CapabilityLevel.EXPERT,
        description="Identify, diagnose, and fix bugs across the full stack",
        sub_capabilities=[
            "Error message interpretation and root cause analysis",
            "Stack trace analysis and navigation",
            "Runtime error diagnosis",
            "Logic error detection",
            "Memory leak identification",
            "Performance bottleneck detection",
            "Race condition identification",
            "API error troubleshooting",
            "Database query optimization",
            "Network issue diagnosis",
        ],
        accuracy=0.91
    ),
    
    "refactoring": AgentCapability(
        name="Code Refactoring",
        level=CapabilityLevel.EXPERT,
        description="Transform and improve code while preserving behavior",
        sub_capabilities=[
            "Extract method/function refactoring",
            "Extract class/module refactoring",
            "Rename with all references updated",
            "Move code between files/modules",
            "Inline variable/function",
            "Convert to design patterns",
            "Modernize legacy code",
            "Performance optimization",
            "Security hardening",
            "Code style normalization",
        ],
        accuracy=0.92
    ),
    
    "security_analysis": AgentCapability(
        name="Security Analysis",
        level=CapabilityLevel.EXPERT,
        description="Identify and remediate security vulnerabilities",
        sub_capabilities=[
            "SQL injection detection and prevention",
            "XSS vulnerability identification",
            "CSRF protection verification",
            "Command injection detection",
            "Path traversal vulnerability detection",
            "Insecure deserialization detection",
            "Hardcoded secrets/credentials detection",
            "Authentication/authorization flaws",
            "OWASP Top 10 coverage",
            "Secure coding recommendations",
        ],
        accuracy=0.94
    ),
    
    "infrastructure": AgentCapability(
        name="Infrastructure & DevOps",
        level=CapabilityLevel.MASTER,
        description="Design and generate production-ready infrastructure",
        sub_capabilities=[
            "Terraform configuration generation",
            "Kubernetes manifests and Helm charts",
            "Docker and Docker Compose files",
            "AWS/GCP/Azure resource provisioning",
            "CI/CD pipeline design",
            "Monitoring and observability setup",
            "Auto-scaling configuration",
            "Load balancing and networking",
            "Secret management",
            "Disaster recovery planning",
        ],
        accuracy=0.95
    ),
    
    "database": AgentCapability(
        name="Database Operations",
        level=CapabilityLevel.EXPERT,
        description="Design schemas, write queries, optimize performance",
        sub_capabilities=[
            "Schema design and normalization",
            "Migration generation and management",
            "Complex query writing (SQL, NoSQL)",
            "Query optimization and indexing",
            "ORM integration (Drizzle, Prisma, SQLAlchemy)",
            "Database connection pooling",
            "Transaction management",
            "Data modeling for different use cases",
            "PostgreSQL, MySQL, MongoDB, Redis expertise",
            "Database scaling strategies",
        ],
        accuracy=0.93
    ),
    
    "api_development": AgentCapability(
        name="API Development",
        level=CapabilityLevel.MASTER,
        description="Design and implement APIs following best practices",
        sub_capabilities=[
            "RESTful API design",
            "GraphQL schema and resolvers",
            "gRPC service definitions",
            "WebSocket real-time APIs",
            "API authentication (JWT, OAuth, API keys)",
            "Rate limiting and throttling",
            "API versioning strategies",
            "OpenAPI/Swagger documentation",
            "Error handling and status codes",
            "API testing and mocking",
        ],
        accuracy=0.94
    ),
    
    "frontend": AgentCapability(
        name="Frontend Development",
        level=CapabilityLevel.MASTER,
        description="Build modern, responsive user interfaces",
        sub_capabilities=[
            "React/Vue/Angular/Svelte components",
            "State management (Redux, Zustand, Pinia)",
            "Responsive design with Tailwind/CSS",
            "Form handling and validation",
            "Client-side routing",
            "API integration and data fetching",
            "Performance optimization",
            "Accessibility (a11y) compliance",
            "Animation and transitions",
            "Progressive Web Apps (PWA)",
        ],
        accuracy=0.94
    ),
    
    "testing": AgentCapability(
        name="Testing & Quality Assurance",
        level=CapabilityLevel.EXPERT,
        description="Comprehensive testing strategies and implementation",
        sub_capabilities=[
            "Unit test generation",
            "Integration test design",
            "End-to-end test automation",
            "Test coverage analysis",
            "Mocking and stubbing",
            "Property-based testing",
            "Performance/load testing",
            "Security testing",
            "Accessibility testing",
            "CI/CD test integration",
        ],
        accuracy=0.90
    ),
    
    "documentation": AgentCapability(
        name="Documentation",
        level=CapabilityLevel.EXPERT,
        description="Generate clear, comprehensive documentation",
        sub_capabilities=[
            "Code comments and docstrings",
            "API documentation generation",
            "README and setup guides",
            "Architecture documentation",
            "User guides and tutorials",
            "Changelog generation",
            "Inline code explanations",
            "Technical specification writing",
            "Diagram generation descriptions",
            "Knowledge base articles",
        ],
        accuracy=0.92
    ),
    
    "project_management": AgentCapability(
        name="Project Understanding",
        level=CapabilityLevel.EXPERT,
        description="Understand and navigate complex project structures",
        sub_capabilities=[
            "Monorepo navigation",
            "Microservices architecture understanding",
            "Package and dependency management",
            "Build system configuration",
            "Development environment setup",
            "Configuration file management",
            "Version control operations",
            "Branch strategy recommendations",
            "Code review assistance",
            "Technical planning",
        ],
        accuracy=0.91
    ),
}


LANGUAGE_EXPERTISE: Dict[str, Dict[str, Any]] = {
    "tier1_expert": {
        "languages": ["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust"],
        "level": CapabilityLevel.MASTER,
        "accuracy": 0.95,
        "features": [
            "Complete language mastery",
            "Idiomatic code generation",
            "Advanced pattern recognition",
            "Performance optimization",
            "Security best practices",
        ]
    },
    "tier2_advanced": {
        "languages": ["C", "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala"],
        "level": CapabilityLevel.EXPERT,
        "accuracy": 0.90,
        "features": [
            "Strong language proficiency",
            "Framework integration",
            "Best practice adherence",
            "Error handling patterns",
        ]
    },
    "tier3_proficient": {
        "languages": ["R", "Lua", "Perl", "Haskell", "Elixir", "Clojure", "Dart", 
                     "Julia", "Shell/Bash", "SQL", "Assembly", "OCaml", "F#", 
                     "Erlang", "Zig", "Nim"],
        "level": CapabilityLevel.ADVANCED,
        "accuracy": 0.85,
        "features": [
            "Solid understanding",
            "Common patterns",
            "Basic optimization",
        ]
    },
}


ADVANCED_FEATURES: Dict[str, Dict[str, Any]] = {
    "multi_file_context": {
        "name": "Multi-File Context Understanding",
        "description": "Understand relationships across multiple files in a project",
        "max_files": 100,
        "capabilities": [
            "Cross-file reference tracking",
            "Import/export graph analysis",
            "Shared type inference",
            "Consistent refactoring across files",
        ]
    },
    "incremental_development": {
        "name": "Incremental Development",
        "description": "Build features step-by-step with continuous testing",
        "capabilities": [
            "Progressive feature implementation",
            "Checkpoint creation",
            "Rollback support",
            "Iterative refinement",
        ]
    },
    "intelligent_suggestions": {
        "name": "Intelligent Suggestions",
        "description": "Proactive recommendations for code improvements",
        "capabilities": [
            "Performance optimization suggestions",
            "Security hardening recommendations",
            "Best practice enforcement",
            "Modern syntax updates",
        ]
    },
    "context_awareness": {
        "name": "Context Awareness",
        "description": "Remember and apply project-specific patterns",
        "capabilities": [
            "Coding style matching",
            "Framework convention adherence",
            "Team pattern recognition",
            "Consistent naming",
        ]
    },
}


def get_all_capabilities() -> Dict[str, Any]:
    """Get complete capability overview."""
    return {
        "identity": AGENT_IDENTITY,
        "core_capabilities": {
            name: {
                "name": cap.name,
                "level": cap.level.name,
                "description": cap.description,
                "sub_capabilities": cap.sub_capabilities,
                "accuracy": cap.accuracy,
            }
            for name, cap in CORE_CAPABILITIES.items()
        },
        "language_expertise": LANGUAGE_EXPERTISE,
        "advanced_features": ADVANCED_FEATURES,
    }


def get_capability(name: str) -> AgentCapability:
    """Get a specific capability by name."""
    if name in CORE_CAPABILITIES:
        return CORE_CAPABILITIES[name]
    raise KeyError(f"Capability '{name}' not found")


def get_supported_languages() -> List[str]:
    """Get all supported programming languages."""
    languages = []
    for tier in LANGUAGE_EXPERTISE.values():
        languages.extend(tier["languages"])
    return languages


def get_accuracy_summary() -> Dict[str, float]:
    """Get accuracy summary for all capabilities."""
    return {
        name: cap.accuracy
        for name, cap in CORE_CAPABILITIES.items()
    }


def compare_to_baseline() -> Dict[str, Any]:
    """Compare Platform Forge capabilities to baseline AI assistants."""
    return {
        "comparison": "Platform Forge vs Standard AI Assistants",
        "advantages": [
            "30 programming languages (vs typical 10-15)",
            "55 frameworks with deep integration knowledge",
            "72 infrastructure tools with Terraform/K8s generation",
            "Advanced security analysis (OWASP Top 10 coverage)",
            "Custom transformer with 16k context window",
            "Multi-Query Attention for efficient long-context",
            "Mixture of Experts for specialized knowledge",
            "Production-ready infrastructure generation",
            "Real-time vulnerability detection",
            "Comprehensive code smell detection",
        ],
        "unique_features": [
            "Infrastructure-as-Code generation from any codebase",
            "Automatic scaling configuration",
            "Multi-cloud deployment patterns",
            "Security-first code analysis",
            "Performance bottleneck detection",
        ],
        "accuracy_comparison": {
            "code_generation": {"platform_forge": 0.94, "baseline": 0.85},
            "security_analysis": {"platform_forge": 0.94, "baseline": 0.75},
            "infrastructure": {"platform_forge": 0.95, "baseline": 0.70},
            "debugging": {"platform_forge": 0.91, "baseline": 0.82},
        }
    }


def get_task_capabilities(task_type: str) -> List[str]:
    """Get capabilities relevant to a specific task type."""
    task_mapping = {
        "build_app": ["code_generation", "frontend", "api_development", "database"],
        "fix_bug": ["debugging", "code_understanding", "testing"],
        "refactor": ["refactoring", "code_understanding", "testing"],
        "security_audit": ["security_analysis", "code_understanding"],
        "deploy": ["infrastructure", "documentation"],
        "optimize": ["debugging", "database", "code_understanding"],
        "document": ["documentation", "code_understanding"],
        "test": ["testing", "code_generation"],
    }
    
    task_type = task_type.lower().replace(" ", "_")
    if task_type in task_mapping:
        return task_mapping[task_type]
    return list(CORE_CAPABILITIES.keys())
