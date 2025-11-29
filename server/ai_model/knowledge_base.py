"""
Comprehensive Knowledge Base for Platform Forge AI Model

This module provides a complete catalog of programming languages, frameworks,
infrastructure tools, design patterns, and AI capabilities that the model
understands and can work with.

Usage:
    from server.ai_model.knowledge_base import (
        get_language_info,
        get_frameworks_for_language,
        get_infrastructure_by_category,
        get_patterns_by_type,
        get_all_skills,
        search_knowledge_base,
    )
    
    # Get Python language details
    python_info = get_language_info('python')
    
    # Get all Python frameworks
    python_frameworks = get_frameworks_for_language('python')
    
    # Search across all databases
    results = search_knowledge_base('kubernetes deployment')
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class LanguageInfo:
    """Programming language information."""
    name: str
    extensions: List[str]
    paradigms: List[str]
    common_frameworks: List[str]
    package_managers: List[str]
    expertise_level: int
    description: str
    typing: str
    compilation: str


@dataclass
class FrameworkInfo:
    """Framework information."""
    name: str
    language: str
    category: str
    version: str
    features: List[str]
    expertise_level: int
    description: str
    use_cases: List[str]


@dataclass
class InfrastructureInfo:
    """Infrastructure tool information."""
    name: str
    category: str
    features: List[str]
    terraform_support: bool
    kubernetes_integration: bool
    description: str
    alternatives: List[str]


@dataclass
class PatternInfo:
    """Design pattern information."""
    name: str
    category: str
    description: str
    languages_applicable: List[str]
    code_example: str
    use_cases: List[str]


@dataclass
class SkillInfo:
    """AI skill/capability information."""
    name: str
    category: str
    description: str
    accuracy_level: float
    capabilities: List[str]


LANGUAGES_DATABASE: Dict[str, Dict[str, Any]] = {
    "python": {
        "name": "Python",
        "extensions": [".py", ".pyw", ".pyi", ".pyx"],
        "paradigms": ["object-oriented", "functional", "procedural", "imperative"],
        "common_frameworks": ["Django", "Flask", "FastAPI", "Celery", "SQLAlchemy", "Pandas", "NumPy", "TensorFlow", "PyTorch"],
        "package_managers": ["pip", "poetry", "conda", "pipenv", "uv"],
        "expertise_level": 10,
        "description": "High-level, interpreted, general-purpose programming language emphasizing code readability",
        "typing": "dynamic",
        "compilation": "interpreted"
    },
    "javascript": {
        "name": "JavaScript",
        "extensions": [".js", ".mjs", ".cjs", ".jsx"],
        "paradigms": ["object-oriented", "functional", "event-driven", "imperative"],
        "common_frameworks": ["React", "Vue", "Angular", "Next.js", "Express", "NestJS", "Svelte", "Remix"],
        "package_managers": ["npm", "yarn", "pnpm", "bun"],
        "expertise_level": 10,
        "description": "Dynamic programming language for web development, both client and server-side",
        "typing": "dynamic",
        "compilation": "interpreted"
    },
    "typescript": {
        "name": "TypeScript",
        "extensions": [".ts", ".tsx", ".mts", ".cts"],
        "paradigms": ["object-oriented", "functional", "imperative"],
        "common_frameworks": ["React", "Vue", "Angular", "Next.js", "Express", "NestJS", "Svelte", "Remix"],
        "package_managers": ["npm", "yarn", "pnpm", "bun"],
        "expertise_level": 10,
        "description": "Typed superset of JavaScript that compiles to plain JavaScript",
        "typing": "static",
        "compilation": "transpiled"
    },
    "java": {
        "name": "Java",
        "extensions": [".java", ".jar", ".class"],
        "paradigms": ["object-oriented", "imperative", "concurrent"],
        "common_frameworks": ["Spring Boot", "Hibernate", "Maven", "Gradle", "JUnit", "Quarkus", "Micronaut"],
        "package_managers": ["maven", "gradle"],
        "expertise_level": 9,
        "description": "Object-oriented, class-based language designed for portability and enterprise applications",
        "typing": "static",
        "compilation": "compiled"
    },
    "c": {
        "name": "C",
        "extensions": [".c", ".h"],
        "paradigms": ["procedural", "imperative"],
        "common_frameworks": ["libc", "glib", "SDL", "OpenGL", "POSIX"],
        "package_managers": ["vcpkg", "conan", "pkg-config"],
        "expertise_level": 8,
        "description": "Low-level, procedural language for system programming and embedded systems",
        "typing": "static",
        "compilation": "compiled"
    },
    "cpp": {
        "name": "C++",
        "extensions": [".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"],
        "paradigms": ["object-oriented", "procedural", "functional", "generic"],
        "common_frameworks": ["Qt", "Boost", "STL", "OpenCV", "Eigen", "POCO"],
        "package_managers": ["vcpkg", "conan", "cmake"],
        "expertise_level": 8,
        "description": "General-purpose language with object-oriented, generic, and functional features",
        "typing": "static",
        "compilation": "compiled"
    },
    "csharp": {
        "name": "C#",
        "extensions": [".cs", ".csx"],
        "paradigms": ["object-oriented", "functional", "imperative", "component-oriented"],
        "common_frameworks": [".NET", "ASP.NET Core", "Entity Framework", "Blazor", "MAUI", "Unity"],
        "package_managers": ["nuget", "dotnet"],
        "expertise_level": 9,
        "description": "Modern, object-oriented language for the .NET platform",
        "typing": "static",
        "compilation": "compiled"
    },
    "go": {
        "name": "Go",
        "extensions": [".go"],
        "paradigms": ["procedural", "concurrent", "imperative"],
        "common_frameworks": ["Gin", "Echo", "Fiber", "GORM", "Chi", "Buffalo", "Beego"],
        "package_managers": ["go modules"],
        "expertise_level": 9,
        "description": "Statically typed, compiled language with garbage collection and CSP-style concurrency",
        "typing": "static",
        "compilation": "compiled"
    },
    "rust": {
        "name": "Rust",
        "extensions": [".rs"],
        "paradigms": ["functional", "imperative", "concurrent", "generic"],
        "common_frameworks": ["Actix", "Rocket", "Tokio", "Serde", "Axum", "Warp", "Diesel"],
        "package_managers": ["cargo"],
        "expertise_level": 9,
        "description": "Systems programming language focused on safety, concurrency, and performance",
        "typing": "static",
        "compilation": "compiled"
    },
    "ruby": {
        "name": "Ruby",
        "extensions": [".rb", ".erb", ".rake"],
        "paradigms": ["object-oriented", "functional", "imperative"],
        "common_frameworks": ["Rails", "Sinatra", "Sidekiq", "RSpec", "Hanami", "Grape"],
        "package_managers": ["gem", "bundler"],
        "expertise_level": 8,
        "description": "Dynamic, object-oriented language designed for simplicity and productivity",
        "typing": "dynamic",
        "compilation": "interpreted"
    },
    "php": {
        "name": "PHP",
        "extensions": [".php", ".phtml", ".php3", ".php4", ".php5", ".phps"],
        "paradigms": ["object-oriented", "procedural", "functional"],
        "common_frameworks": ["Laravel", "Symfony", "CodeIgniter", "Yii", "CakePHP", "Slim"],
        "package_managers": ["composer"],
        "expertise_level": 8,
        "description": "Server-side scripting language designed for web development",
        "typing": "dynamic",
        "compilation": "interpreted"
    },
    "swift": {
        "name": "Swift",
        "extensions": [".swift"],
        "paradigms": ["object-oriented", "functional", "protocol-oriented", "imperative"],
        "common_frameworks": ["SwiftUI", "UIKit", "Combine", "Vapor", "Perfect", "Kitura"],
        "package_managers": ["swift package manager", "cocoapods", "carthage"],
        "expertise_level": 8,
        "description": "General-purpose, compiled language developed by Apple for iOS and macOS",
        "typing": "static",
        "compilation": "compiled"
    },
    "kotlin": {
        "name": "Kotlin",
        "extensions": [".kt", ".kts"],
        "paradigms": ["object-oriented", "functional", "imperative"],
        "common_frameworks": ["Ktor", "Spring Boot", "Exposed", "Koin", "Android SDK"],
        "package_managers": ["gradle", "maven"],
        "expertise_level": 8,
        "description": "Modern, concise language that runs on JVM and Android",
        "typing": "static",
        "compilation": "compiled"
    },
    "scala": {
        "name": "Scala",
        "extensions": [".scala", ".sc"],
        "paradigms": ["object-oriented", "functional", "imperative"],
        "common_frameworks": ["Akka", "Play", "Spark", "Cats", "ZIO", "Slick"],
        "package_managers": ["sbt", "maven"],
        "expertise_level": 7,
        "description": "Multi-paradigm language combining object-oriented and functional programming",
        "typing": "static",
        "compilation": "compiled"
    },
    "r": {
        "name": "R",
        "extensions": [".r", ".R", ".rds", ".rda"],
        "paradigms": ["functional", "procedural", "object-oriented"],
        "common_frameworks": ["tidyverse", "ggplot2", "dplyr", "shiny", "caret", "mlr3"],
        "package_managers": ["cran", "devtools"],
        "expertise_level": 7,
        "description": "Language for statistical computing and graphics",
        "typing": "dynamic",
        "compilation": "interpreted"
    },
    "lua": {
        "name": "Lua",
        "extensions": [".lua"],
        "paradigms": ["procedural", "object-oriented", "functional"],
        "common_frameworks": ["LOVE2D", "Corona", "OpenResty", "Lapis", "Torch"],
        "package_managers": ["luarocks"],
        "expertise_level": 7,
        "description": "Lightweight, embeddable scripting language",
        "typing": "dynamic",
        "compilation": "interpreted"
    },
    "perl": {
        "name": "Perl",
        "extensions": [".pl", ".pm", ".t"],
        "paradigms": ["procedural", "object-oriented", "functional"],
        "common_frameworks": ["Mojolicious", "Catalyst", "Dancer", "CGI", "DBI"],
        "package_managers": ["cpan", "cpanm"],
        "expertise_level": 6,
        "description": "High-level, interpreted language for text processing and system administration",
        "typing": "dynamic",
        "compilation": "interpreted"
    },
    "haskell": {
        "name": "Haskell",
        "extensions": [".hs", ".lhs"],
        "paradigms": ["functional", "lazy evaluation", "pure"],
        "common_frameworks": ["Yesod", "Servant", "Scotty", "Aeson", "Lens", "Parsec"],
        "package_managers": ["cabal", "stack"],
        "expertise_level": 7,
        "description": "Purely functional programming language with strong static typing",
        "typing": "static",
        "compilation": "compiled"
    },
    "elixir": {
        "name": "Elixir",
        "extensions": [".ex", ".exs"],
        "paradigms": ["functional", "concurrent", "distributed"],
        "common_frameworks": ["Phoenix", "Ecto", "Nerves", "Absinthe", "Broadway"],
        "package_managers": ["mix", "hex"],
        "expertise_level": 7,
        "description": "Functional language for scalable and maintainable applications on the Erlang VM",
        "typing": "dynamic",
        "compilation": "compiled"
    },
    "clojure": {
        "name": "Clojure",
        "extensions": [".clj", ".cljs", ".cljc", ".edn"],
        "paradigms": ["functional", "concurrent", "lisp"],
        "common_frameworks": ["Ring", "Compojure", "Luminus", "Re-frame", "Pedestal"],
        "package_managers": ["leiningen", "deps.edn"],
        "expertise_level": 7,
        "description": "Modern Lisp dialect for the JVM emphasizing immutability and concurrency",
        "typing": "dynamic",
        "compilation": "compiled"
    },
    "dart": {
        "name": "Dart",
        "extensions": [".dart"],
        "paradigms": ["object-oriented", "functional", "imperative"],
        "common_frameworks": ["Flutter", "AngularDart", "Aqueduct", "Shelf"],
        "package_managers": ["pub"],
        "expertise_level": 8,
        "description": "Client-optimized language for fast apps on multiple platforms",
        "typing": "static",
        "compilation": "compiled"
    },
    "julia": {
        "name": "Julia",
        "extensions": [".jl"],
        "paradigms": ["procedural", "functional", "object-oriented", "multiple dispatch"],
        "common_frameworks": ["Flux", "DifferentialEquations", "Plots", "Genie", "JuMP"],
        "package_managers": ["pkg"],
        "expertise_level": 7,
        "description": "High-performance language for technical and scientific computing",
        "typing": "dynamic",
        "compilation": "compiled"
    },
    "shell": {
        "name": "Shell/Bash",
        "extensions": [".sh", ".bash", ".zsh", ".fish"],
        "paradigms": ["procedural", "scripting"],
        "common_frameworks": ["coreutils", "GNU tools", "POSIX"],
        "package_managers": ["apt", "yum", "brew", "pacman"],
        "expertise_level": 9,
        "description": "Command-line interpreter and scripting language for Unix/Linux systems",
        "typing": "untyped",
        "compilation": "interpreted"
    },
    "sql": {
        "name": "SQL",
        "extensions": [".sql", ".psql", ".mysql"],
        "paradigms": ["declarative", "set-based"],
        "common_frameworks": ["PostgreSQL", "MySQL", "SQLite", "SQL Server", "Oracle"],
        "package_managers": [],
        "expertise_level": 9,
        "description": "Domain-specific language for managing and querying relational databases",
        "typing": "static",
        "compilation": "interpreted"
    },
    "assembly": {
        "name": "Assembly",
        "extensions": [".asm", ".s", ".S"],
        "paradigms": ["imperative", "low-level"],
        "common_frameworks": ["NASM", "MASM", "GAS", "FASM"],
        "package_managers": [],
        "expertise_level": 6,
        "description": "Low-level programming language with processor-specific instructions",
        "typing": "untyped",
        "compilation": "assembled"
    },
    "ocaml": {
        "name": "OCaml",
        "extensions": [".ml", ".mli"],
        "paradigms": ["functional", "imperative", "object-oriented"],
        "common_frameworks": ["Dream", "Lwt", "Core", "Dune", "Mirage"],
        "package_managers": ["opam"],
        "expertise_level": 6,
        "description": "Industrial-strength functional programming language with pattern matching",
        "typing": "static",
        "compilation": "compiled"
    },
    "fsharp": {
        "name": "F#",
        "extensions": [".fs", ".fsi", ".fsx"],
        "paradigms": ["functional", "object-oriented", "imperative"],
        "common_frameworks": ["Giraffe", "Suave", "Saturn", "Fable", "FAKE"],
        "package_managers": ["nuget", "paket"],
        "expertise_level": 7,
        "description": "Functional-first language for the .NET platform",
        "typing": "static",
        "compilation": "compiled"
    },
    "erlang": {
        "name": "Erlang",
        "extensions": [".erl", ".hrl"],
        "paradigms": ["functional", "concurrent", "distributed"],
        "common_frameworks": ["OTP", "Cowboy", "Rebar3", "Mnesia"],
        "package_managers": ["rebar3", "hex"],
        "expertise_level": 6,
        "description": "Language for building massively scalable soft real-time systems",
        "typing": "dynamic",
        "compilation": "compiled"
    },
    "zig": {
        "name": "Zig",
        "extensions": [".zig"],
        "paradigms": ["procedural", "imperative", "generic"],
        "common_frameworks": ["std", "zls"],
        "package_managers": ["zig package manager"],
        "expertise_level": 6,
        "description": "Systems programming language designed as a C replacement",
        "typing": "static",
        "compilation": "compiled"
    },
    "nim": {
        "name": "Nim",
        "extensions": [".nim", ".nims"],
        "paradigms": ["procedural", "functional", "object-oriented", "metaprogramming"],
        "common_frameworks": ["Jester", "Karax", "Prologue", "Arraymancer"],
        "package_managers": ["nimble"],
        "expertise_level": 6,
        "description": "Statically typed compiled language with Python-like syntax",
        "typing": "static",
        "compilation": "compiled"
    },
}


FRAMEWORKS_DATABASE: Dict[str, Dict[str, Any]] = {
    "django": {
        "name": "Django",
        "language": "python",
        "category": "web_framework",
        "version": "5.0",
        "features": ["ORM", "admin_panel", "authentication", "templating", "REST_framework", "migrations", "middleware"],
        "expertise_level": 10,
        "description": "High-level Python web framework that encourages rapid development and clean design",
        "use_cases": ["web_applications", "CMS", "e-commerce", "APIs", "admin_dashboards"]
    },
    "flask": {
        "name": "Flask",
        "language": "python",
        "category": "web_framework",
        "version": "3.0",
        "features": ["routing", "templating", "WSGI", "extensions", "blueprints", "RESTful"],
        "expertise_level": 10,
        "description": "Lightweight WSGI web application framework in Python",
        "use_cases": ["microservices", "APIs", "web_applications", "prototyping"]
    },
    "fastapi": {
        "name": "FastAPI",
        "language": "python",
        "category": "web_framework",
        "version": "0.109",
        "features": ["async", "OpenAPI", "type_hints", "validation", "dependency_injection", "OAuth2"],
        "expertise_level": 10,
        "description": "Modern, fast (high-performance) web framework for building APIs with Python 3.7+",
        "use_cases": ["APIs", "microservices", "async_services", "ML_serving"]
    },
    "celery": {
        "name": "Celery",
        "language": "python",
        "category": "task_queue",
        "version": "5.3",
        "features": ["distributed_tasks", "scheduling", "retries", "monitoring", "workflows"],
        "expertise_level": 9,
        "description": "Distributed task queue for real-time processing and task scheduling",
        "use_cases": ["background_tasks", "scheduled_jobs", "distributed_processing"]
    },
    "sqlalchemy": {
        "name": "SQLAlchemy",
        "language": "python",
        "category": "orm",
        "version": "2.0",
        "features": ["ORM", "query_builder", "connection_pooling", "migrations", "async_support"],
        "expertise_level": 10,
        "description": "SQL toolkit and Object-Relational Mapping library for Python",
        "use_cases": ["database_access", "data_modeling", "query_building"]
    },
    "pandas": {
        "name": "Pandas",
        "language": "python",
        "category": "data_science",
        "version": "2.2",
        "features": ["dataframes", "data_manipulation", "time_series", "CSV_handling", "groupby"],
        "expertise_level": 10,
        "description": "Data analysis and manipulation library for Python",
        "use_cases": ["data_analysis", "ETL", "data_cleaning", "reporting"]
    },
    "numpy": {
        "name": "NumPy",
        "language": "python",
        "category": "data_science",
        "version": "1.26",
        "features": ["arrays", "linear_algebra", "FFT", "random", "broadcasting"],
        "expertise_level": 10,
        "description": "Fundamental package for scientific computing in Python",
        "use_cases": ["numerical_computing", "scientific_computing", "ML_preprocessing"]
    },
    "tensorflow": {
        "name": "TensorFlow",
        "language": "python",
        "category": "machine_learning",
        "version": "2.15",
        "features": ["neural_networks", "GPU_support", "distributed_training", "TensorBoard", "Keras"],
        "expertise_level": 9,
        "description": "End-to-end open source platform for machine learning",
        "use_cases": ["deep_learning", "neural_networks", "ML_models", "production_ML"]
    },
    "pytorch": {
        "name": "PyTorch",
        "language": "python",
        "category": "machine_learning",
        "version": "2.2",
        "features": ["tensors", "autograd", "GPU_support", "distributed", "TorchScript"],
        "expertise_level": 9,
        "description": "Open source machine learning framework for research and production",
        "use_cases": ["deep_learning", "research", "computer_vision", "NLP"]
    },
    "scikit_learn": {
        "name": "Scikit-learn",
        "language": "python",
        "category": "machine_learning",
        "version": "1.4",
        "features": ["classification", "regression", "clustering", "preprocessing", "model_selection"],
        "expertise_level": 9,
        "description": "Machine learning library for classical ML algorithms",
        "use_cases": ["classification", "regression", "clustering", "feature_engineering"]
    },
    "react": {
        "name": "React",
        "language": "javascript",
        "category": "frontend_framework",
        "version": "18.2",
        "features": ["components", "hooks", "virtual_DOM", "JSX", "context", "suspense"],
        "expertise_level": 10,
        "description": "JavaScript library for building user interfaces",
        "use_cases": ["SPAs", "web_applications", "mobile_apps", "dashboards"]
    },
    "vue": {
        "name": "Vue.js",
        "language": "javascript",
        "category": "frontend_framework",
        "version": "3.4",
        "features": ["reactivity", "components", "composition_API", "directives", "transitions"],
        "expertise_level": 9,
        "description": "Progressive JavaScript framework for building user interfaces",
        "use_cases": ["SPAs", "web_applications", "progressive_enhancement"]
    },
    "angular": {
        "name": "Angular",
        "language": "typescript",
        "category": "frontend_framework",
        "version": "17.0",
        "features": ["components", "dependency_injection", "routing", "forms", "RxJS", "signals"],
        "expertise_level": 9,
        "description": "Platform for building mobile and desktop web applications",
        "use_cases": ["enterprise_apps", "SPAs", "progressive_web_apps"]
    },
    "nextjs": {
        "name": "Next.js",
        "language": "javascript",
        "category": "fullstack_framework",
        "version": "14.1",
        "features": ["SSR", "SSG", "API_routes", "app_router", "image_optimization", "edge_runtime"],
        "expertise_level": 10,
        "description": "React framework for production with hybrid static & server rendering",
        "use_cases": ["web_applications", "e-commerce", "marketing_sites", "dashboards"]
    },
    "express": {
        "name": "Express.js",
        "language": "javascript",
        "category": "backend_framework",
        "version": "4.18",
        "features": ["routing", "middleware", "templating", "static_files", "error_handling"],
        "expertise_level": 10,
        "description": "Fast, unopinionated, minimalist web framework for Node.js",
        "use_cases": ["APIs", "web_applications", "microservices"]
    },
    "nestjs": {
        "name": "NestJS",
        "language": "typescript",
        "category": "backend_framework",
        "version": "10.3",
        "features": ["modules", "dependency_injection", "decorators", "GraphQL", "WebSockets", "microservices"],
        "expertise_level": 9,
        "description": "Progressive Node.js framework for building efficient, scalable server-side applications",
        "use_cases": ["enterprise_APIs", "microservices", "GraphQL_APIs"]
    },
    "svelte": {
        "name": "Svelte",
        "language": "javascript",
        "category": "frontend_framework",
        "version": "4.2",
        "features": ["reactivity", "components", "stores", "transitions", "compile_time"],
        "expertise_level": 8,
        "description": "Compiler that generates minimal and highly optimized JavaScript",
        "use_cases": ["web_applications", "SPAs", "interactive_components"]
    },
    "remix": {
        "name": "Remix",
        "language": "javascript",
        "category": "fullstack_framework",
        "version": "2.6",
        "features": ["nested_routing", "loaders", "actions", "error_boundaries", "progressive_enhancement"],
        "expertise_level": 8,
        "description": "Full stack web framework focused on web fundamentals",
        "use_cases": ["web_applications", "e-commerce", "content_sites"]
    },
    "spring_boot": {
        "name": "Spring Boot",
        "language": "java",
        "category": "backend_framework",
        "version": "3.2",
        "features": ["auto_configuration", "dependency_injection", "security", "data_JPA", "actuator"],
        "expertise_level": 9,
        "description": "Framework for building production-ready Spring applications",
        "use_cases": ["enterprise_applications", "microservices", "APIs"]
    },
    "hibernate": {
        "name": "Hibernate",
        "language": "java",
        "category": "orm",
        "version": "6.4",
        "features": ["ORM", "HQL", "caching", "lazy_loading", "transactions"],
        "expertise_level": 8,
        "description": "Object-relational mapping framework for Java",
        "use_cases": ["database_access", "data_modeling", "enterprise_apps"]
    },
    "maven": {
        "name": "Maven",
        "language": "java",
        "category": "build_tool",
        "version": "3.9",
        "features": ["dependency_management", "build_lifecycle", "plugins", "profiles"],
        "expertise_level": 9,
        "description": "Build automation and project management tool for Java",
        "use_cases": ["build_automation", "dependency_management", "project_structure"]
    },
    "gradle": {
        "name": "Gradle",
        "language": "java",
        "category": "build_tool",
        "version": "8.5",
        "features": ["build_scripts", "incremental_builds", "dependency_management", "plugins"],
        "expertise_level": 9,
        "description": "Build automation tool with Groovy/Kotlin DSL",
        "use_cases": ["build_automation", "multi_project_builds", "Android_development"]
    },
    "junit": {
        "name": "JUnit",
        "language": "java",
        "category": "testing",
        "version": "5.10",
        "features": ["assertions", "test_lifecycle", "parameterized_tests", "extensions"],
        "expertise_level": 9,
        "description": "Unit testing framework for Java applications",
        "use_cases": ["unit_testing", "integration_testing", "TDD"]
    },
    "quarkus": {
        "name": "Quarkus",
        "language": "java",
        "category": "backend_framework",
        "version": "3.7",
        "features": ["native_compilation", "reactive", "dev_mode", "extensions", "CDI"],
        "expertise_level": 8,
        "description": "Kubernetes-native Java framework tailored for GraalVM and OpenJDK",
        "use_cases": ["microservices", "serverless", "cloud_native"]
    },
    "gin": {
        "name": "Gin",
        "language": "go",
        "category": "web_framework",
        "version": "1.9",
        "features": ["routing", "middleware", "validation", "rendering", "crash_free"],
        "expertise_level": 9,
        "description": "High-performance HTTP web framework for Go",
        "use_cases": ["APIs", "microservices", "web_applications"]
    },
    "echo": {
        "name": "Echo",
        "language": "go",
        "category": "web_framework",
        "version": "4.11",
        "features": ["routing", "middleware", "data_binding", "templating", "WebSocket"],
        "expertise_level": 8,
        "description": "High performance, extensible, minimalist Go web framework",
        "use_cases": ["APIs", "microservices", "RESTful_services"]
    },
    "fiber": {
        "name": "Fiber",
        "language": "go",
        "category": "web_framework",
        "version": "2.52",
        "features": ["routing", "middleware", "zero_allocation", "WebSocket", "rate_limiting"],
        "expertise_level": 8,
        "description": "Express-inspired web framework built on Fasthttp",
        "use_cases": ["high_performance_APIs", "microservices"]
    },
    "gorm": {
        "name": "GORM",
        "language": "go",
        "category": "orm",
        "version": "1.25",
        "features": ["ORM", "associations", "hooks", "transactions", "migrations"],
        "expertise_level": 9,
        "description": "The fantastic ORM library for Go",
        "use_cases": ["database_access", "data_modeling", "CRUD_operations"]
    },
    "actix": {
        "name": "Actix Web",
        "language": "rust",
        "category": "web_framework",
        "version": "4.4",
        "features": ["async", "middleware", "extractors", "WebSocket", "HTTP2"],
        "expertise_level": 9,
        "description": "Powerful, pragmatic, and extremely fast web framework for Rust",
        "use_cases": ["high_performance_APIs", "microservices", "web_applications"]
    },
    "rocket": {
        "name": "Rocket",
        "language": "rust",
        "category": "web_framework",
        "version": "0.5",
        "features": ["routing", "guards", "responders", "fairings", "type_safe"],
        "expertise_level": 8,
        "description": "Web framework for Rust that makes it easy to write fast, secure apps",
        "use_cases": ["web_applications", "APIs", "secure_services"]
    },
    "tokio": {
        "name": "Tokio",
        "language": "rust",
        "category": "runtime",
        "version": "1.35",
        "features": ["async_runtime", "networking", "file_IO", "timers", "synchronization"],
        "expertise_level": 9,
        "description": "Asynchronous runtime for Rust providing networking and concurrency",
        "use_cases": ["async_applications", "networking", "concurrent_services"]
    },
    "serde": {
        "name": "Serde",
        "language": "rust",
        "category": "serialization",
        "version": "1.0",
        "features": ["serialization", "deserialization", "derive_macros", "custom_formats"],
        "expertise_level": 9,
        "description": "Framework for serializing and deserializing Rust data structures",
        "use_cases": ["JSON_handling", "configuration", "data_interchange"]
    },
    "axum": {
        "name": "Axum",
        "language": "rust",
        "category": "web_framework",
        "version": "0.7",
        "features": ["routing", "extractors", "middleware", "tower_integration", "async"],
        "expertise_level": 8,
        "description": "Ergonomic and modular web framework built with Tokio and Tower",
        "use_cases": ["web_applications", "APIs", "microservices"]
    },
    "rails": {
        "name": "Ruby on Rails",
        "language": "ruby",
        "category": "fullstack_framework",
        "version": "7.1",
        "features": ["MVC", "ActiveRecord", "scaffolding", "migrations", "Turbo", "Hotwire"],
        "expertise_level": 9,
        "description": "Full-stack framework emphasizing convention over configuration",
        "use_cases": ["web_applications", "e-commerce", "SaaS", "MVPs"]
    },
    "sinatra": {
        "name": "Sinatra",
        "language": "ruby",
        "category": "web_framework",
        "version": "4.0",
        "features": ["routing", "templates", "filters", "extensions"],
        "expertise_level": 8,
        "description": "DSL for quickly creating web applications in Ruby",
        "use_cases": ["microservices", "APIs", "simple_web_apps"]
    },
    "sidekiq": {
        "name": "Sidekiq",
        "language": "ruby",
        "category": "task_queue",
        "version": "7.2",
        "features": ["background_jobs", "scheduling", "retries", "batches", "rate_limiting"],
        "expertise_level": 8,
        "description": "Simple, efficient background processing for Ruby",
        "use_cases": ["background_jobs", "async_processing", "scheduled_tasks"]
    },
    "laravel": {
        "name": "Laravel",
        "language": "php",
        "category": "fullstack_framework",
        "version": "11.0",
        "features": ["Eloquent_ORM", "Blade_templating", "routing", "authentication", "queues", "Livewire"],
        "expertise_level": 9,
        "description": "PHP framework for artisans with expressive, elegant syntax",
        "use_cases": ["web_applications", "APIs", "e-commerce", "SaaS"]
    },
    "symfony": {
        "name": "Symfony",
        "language": "php",
        "category": "fullstack_framework",
        "version": "7.0",
        "features": ["bundles", "Doctrine", "Twig", "forms", "security", "events"],
        "expertise_level": 8,
        "description": "Set of reusable PHP components and a web application framework",
        "use_cases": ["enterprise_applications", "APIs", "CMS"]
    },
    "phoenix": {
        "name": "Phoenix",
        "language": "elixir",
        "category": "fullstack_framework",
        "version": "1.7",
        "features": ["LiveView", "channels", "Ecto", "PubSub", "presence"],
        "expertise_level": 8,
        "description": "Web framework for Elixir that enables building rich, interactive apps",
        "use_cases": ["real_time_apps", "web_applications", "APIs"]
    },
    "flutter": {
        "name": "Flutter",
        "language": "dart",
        "category": "mobile_framework",
        "version": "3.16",
        "features": ["widgets", "hot_reload", "cross_platform", "Material_Design", "Cupertino"],
        "expertise_level": 9,
        "description": "UI toolkit for building natively compiled applications",
        "use_cases": ["mobile_apps", "web_apps", "desktop_apps"]
    },
    "react_native": {
        "name": "React Native",
        "language": "javascript",
        "category": "mobile_framework",
        "version": "0.73",
        "features": ["native_components", "hot_reloading", "bridge", "Expo", "new_architecture"],
        "expertise_level": 9,
        "description": "Framework for building native mobile apps using React",
        "use_cases": ["mobile_apps", "cross_platform_apps"]
    },
    "electron": {
        "name": "Electron",
        "language": "javascript",
        "category": "desktop_framework",
        "version": "28.0",
        "features": ["cross_platform", "auto_update", "native_APIs", "Chromium", "Node_integration"],
        "expertise_level": 8,
        "description": "Build cross-platform desktop apps with JavaScript, HTML, and CSS",
        "use_cases": ["desktop_apps", "IDEs", "productivity_tools"]
    },
    "unity": {
        "name": "Unity",
        "language": "csharp",
        "category": "game_engine",
        "version": "2023.2",
        "features": ["3D_rendering", "physics", "animation", "scripting", "asset_pipeline"],
        "expertise_level": 8,
        "description": "Cross-platform game engine and development platform",
        "use_cases": ["games", "simulations", "VR_AR", "interactive_media"]
    },
    "dotnet_core": {
        "name": ".NET Core",
        "language": "csharp",
        "category": "runtime",
        "version": "8.0",
        "features": ["cross_platform", "high_performance", "dependency_injection", "minimal_APIs"],
        "expertise_level": 9,
        "description": "Open-source, cross-platform framework for building modern applications",
        "use_cases": ["web_applications", "APIs", "microservices", "cloud_apps"]
    },
    "aspnet_core": {
        "name": "ASP.NET Core",
        "language": "csharp",
        "category": "web_framework",
        "version": "8.0",
        "features": ["MVC", "Razor_Pages", "Blazor", "SignalR", "minimal_APIs", "gRPC"],
        "expertise_level": 9,
        "description": "Cross-platform, high-performance framework for building web applications",
        "use_cases": ["web_applications", "APIs", "real_time_apps"]
    },
    "entity_framework": {
        "name": "Entity Framework Core",
        "language": "csharp",
        "category": "orm",
        "version": "8.0",
        "features": ["ORM", "LINQ", "migrations", "change_tracking", "lazy_loading"],
        "expertise_level": 9,
        "description": "Modern object-database mapper for .NET",
        "use_cases": ["database_access", "data_modeling", "CRUD_operations"]
    },
    "tailwindcss": {
        "name": "Tailwind CSS",
        "language": "css",
        "category": "css_framework",
        "version": "3.4",
        "features": ["utility_first", "JIT_compiler", "responsive", "dark_mode", "plugins"],
        "expertise_level": 10,
        "description": "Utility-first CSS framework for rapidly building custom designs",
        "use_cases": ["web_styling", "responsive_design", "component_styling"]
    },
    "graphql": {
        "name": "GraphQL",
        "language": "multi",
        "category": "api_specification",
        "version": "2023",
        "features": ["queries", "mutations", "subscriptions", "schema", "resolvers"],
        "expertise_level": 9,
        "description": "Query language for APIs and runtime for executing those queries",
        "use_cases": ["API_development", "data_fetching", "real_time_updates"]
    },
    "prisma": {
        "name": "Prisma",
        "language": "typescript",
        "category": "orm",
        "version": "5.9",
        "features": ["type_safe", "migrations", "studio", "query_engine", "introspection"],
        "expertise_level": 9,
        "description": "Next-generation Node.js and TypeScript ORM",
        "use_cases": ["database_access", "type_safe_queries", "schema_management"]
    },
    "drizzle": {
        "name": "Drizzle ORM",
        "language": "typescript",
        "category": "orm",
        "version": "0.29",
        "features": ["type_safe", "SQL_like", "migrations", "zero_dependencies", "edge_ready"],
        "expertise_level": 8,
        "description": "TypeScript ORM that's simple, performant and edge-ready",
        "use_cases": ["database_access", "serverless", "edge_computing"]
    },
    "pytest": {
        "name": "Pytest",
        "language": "python",
        "category": "testing",
        "version": "8.0",
        "features": ["fixtures", "parametrization", "plugins", "assertions", "parallel_execution"],
        "expertise_level": 10,
        "description": "Full-featured Python testing framework",
        "use_cases": ["unit_testing", "integration_testing", "functional_testing"]
    },
    "jest": {
        "name": "Jest",
        "language": "javascript",
        "category": "testing",
        "version": "29.7",
        "features": ["snapshots", "mocking", "coverage", "parallel_execution", "watch_mode"],
        "expertise_level": 10,
        "description": "Delightful JavaScript Testing Framework",
        "use_cases": ["unit_testing", "component_testing", "integration_testing"]
    },
    "vitest": {
        "name": "Vitest",
        "language": "typescript",
        "category": "testing",
        "version": "1.2",
        "features": ["vite_integration", "ESM_native", "TypeScript", "snapshots", "coverage"],
        "expertise_level": 9,
        "description": "Blazing fast unit test framework powered by Vite",
        "use_cases": ["unit_testing", "component_testing", "Vite_projects"]
    },
    "cypress": {
        "name": "Cypress",
        "language": "javascript",
        "category": "testing",
        "version": "13.6",
        "features": ["e2e_testing", "component_testing", "time_travel", "real_browser", "screenshots"],
        "expertise_level": 9,
        "description": "Fast, easy and reliable testing for anything that runs in a browser",
        "use_cases": ["e2e_testing", "component_testing", "integration_testing"]
    },
    "playwright": {
        "name": "Playwright",
        "language": "multi",
        "category": "testing",
        "version": "1.41",
        "features": ["cross_browser", "auto_wait", "tracing", "screenshots", "video"],
        "expertise_level": 9,
        "description": "End-to-end testing framework for modern web apps",
        "use_cases": ["e2e_testing", "browser_automation", "scraping"]
    },
}


INFRASTRUCTURE_DATABASE: Dict[str, Dict[str, Any]] = {
    "aws_ec2": {
        "name": "AWS EC2",
        "category": "compute",
        "features": ["virtual_machines", "auto_scaling", "spot_instances", "EBS", "placement_groups"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Scalable virtual servers in the cloud",
        "alternatives": ["Azure VMs", "GCP Compute Engine", "DigitalOcean Droplets"]
    },
    "aws_lambda": {
        "name": "AWS Lambda",
        "category": "serverless",
        "features": ["event_driven", "auto_scaling", "pay_per_use", "layers", "container_support"],
        "terraform_support": True,
        "kubernetes_integration": False,
        "description": "Serverless compute service for running code",
        "alternatives": ["Azure Functions", "GCP Cloud Functions", "Cloudflare Workers"]
    },
    "aws_s3": {
        "name": "AWS S3",
        "category": "storage",
        "features": ["object_storage", "versioning", "lifecycle_policies", "encryption", "replication"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Scalable object storage in the cloud",
        "alternatives": ["Azure Blob Storage", "GCP Cloud Storage", "MinIO"]
    },
    "aws_rds": {
        "name": "AWS RDS",
        "category": "database",
        "features": ["managed_databases", "auto_backups", "read_replicas", "multi_AZ", "encryption"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Managed relational database service",
        "alternatives": ["Azure SQL Database", "GCP Cloud SQL", "PlanetScale"]
    },
    "aws_dynamodb": {
        "name": "AWS DynamoDB",
        "category": "database",
        "features": ["NoSQL", "auto_scaling", "global_tables", "DAX_caching", "streams"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Fast, flexible NoSQL database service",
        "alternatives": ["Azure Cosmos DB", "MongoDB Atlas", "GCP Firestore"]
    },
    "aws_eks": {
        "name": "AWS EKS",
        "category": "container",
        "features": ["managed_kubernetes", "fargate", "node_groups", "add_ons", "service_mesh"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Managed Kubernetes service on AWS",
        "alternatives": ["GKE", "AKS", "DigitalOcean Kubernetes"]
    },
    "aws_ecs": {
        "name": "AWS ECS",
        "category": "container",
        "features": ["container_orchestration", "fargate", "EC2_launch", "service_discovery"],
        "terraform_support": True,
        "kubernetes_integration": False,
        "description": "Fully managed container orchestration service",
        "alternatives": ["Azure Container Instances", "GCP Cloud Run"]
    },
    "aws_cloudfront": {
        "name": "AWS CloudFront",
        "category": "cdn",
        "features": ["global_CDN", "edge_locations", "Lambda@Edge", "WAF_integration"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Fast content delivery network service",
        "alternatives": ["Cloudflare", "Azure CDN", "Fastly"]
    },
    "aws_route53": {
        "name": "AWS Route 53",
        "category": "dns",
        "features": ["DNS_management", "health_checks", "traffic_routing", "domain_registration"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Scalable domain name system web service",
        "alternatives": ["Cloudflare DNS", "Azure DNS", "GCP Cloud DNS"]
    },
    "aws_sqs": {
        "name": "AWS SQS",
        "category": "messaging",
        "features": ["message_queues", "FIFO", "dead_letter_queues", "batch_operations"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Fully managed message queuing service",
        "alternatives": ["Azure Service Bus", "GCP Pub/Sub", "RabbitMQ"]
    },
    "aws_sns": {
        "name": "AWS SNS",
        "category": "messaging",
        "features": ["pub_sub", "push_notifications", "SMS", "email", "fanout"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Fully managed pub/sub messaging service",
        "alternatives": ["Azure Event Grid", "GCP Pub/Sub"]
    },
    "aws_api_gateway": {
        "name": "AWS API Gateway",
        "category": "api",
        "features": ["REST_APIs", "WebSocket", "throttling", "caching", "authorization"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Create, publish, and manage APIs at any scale",
        "alternatives": ["Azure API Management", "Kong", "Apigee"]
    },
    "aws_cognito": {
        "name": "AWS Cognito",
        "category": "identity",
        "features": ["user_pools", "identity_pools", "OAuth", "MFA", "social_sign_in"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "User identity and access management",
        "alternatives": ["Auth0", "Azure AD B2C", "Firebase Auth"]
    },
    "aws_elasticache": {
        "name": "AWS ElastiCache",
        "category": "caching",
        "features": ["Redis", "Memcached", "cluster_mode", "replication", "auto_failover"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "In-memory caching service",
        "alternatives": ["Azure Cache for Redis", "GCP Memorystore"]
    },
    "aws_secrets_manager": {
        "name": "AWS Secrets Manager",
        "category": "security",
        "features": ["secret_storage", "rotation", "versioning", "cross_account"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Securely store and manage secrets",
        "alternatives": ["HashiCorp Vault", "Azure Key Vault"]
    },
    "aws_cloudwatch": {
        "name": "AWS CloudWatch",
        "category": "monitoring",
        "features": ["metrics", "logs", "alarms", "dashboards", "insights"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Monitoring and observability service",
        "alternatives": ["Datadog", "Prometheus", "Grafana Cloud"]
    },
    "aws_iam": {
        "name": "AWS IAM",
        "category": "security",
        "features": ["users", "roles", "policies", "MFA", "identity_federation"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Identity and access management for AWS resources",
        "alternatives": ["Azure AD", "GCP IAM"]
    },
    "aws_vpc": {
        "name": "AWS VPC",
        "category": "networking",
        "features": ["subnets", "security_groups", "NAT", "peering", "transit_gateway"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Isolated cloud networking environment",
        "alternatives": ["Azure VNet", "GCP VPC"]
    },
    "aws_fargate": {
        "name": "AWS Fargate",
        "category": "serverless",
        "features": ["serverless_containers", "auto_scaling", "ECS_integration", "EKS_integration"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Serverless compute for containers",
        "alternatives": ["Azure Container Instances", "GCP Cloud Run"]
    },
    "aws_step_functions": {
        "name": "AWS Step Functions",
        "category": "orchestration",
        "features": ["workflow_automation", "state_machines", "error_handling", "parallel_execution"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Serverless workflow orchestration service",
        "alternatives": ["Azure Logic Apps", "GCP Workflows", "Temporal"]
    },
    "gcp_compute_engine": {
        "name": "GCP Compute Engine",
        "category": "compute",
        "features": ["virtual_machines", "preemptible_VMs", "custom_machines", "sole_tenant"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Virtual machines on Google infrastructure",
        "alternatives": ["AWS EC2", "Azure VMs"]
    },
    "gcp_cloud_functions": {
        "name": "GCP Cloud Functions",
        "category": "serverless",
        "features": ["event_driven", "HTTP_triggers", "background_functions", "gen2"],
        "terraform_support": True,
        "kubernetes_integration": False,
        "description": "Serverless execution environment for building cloud services",
        "alternatives": ["AWS Lambda", "Azure Functions"]
    },
    "gcp_cloud_run": {
        "name": "GCP Cloud Run",
        "category": "serverless",
        "features": ["container_hosting", "auto_scaling", "custom_domains", "revision_management"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Fully managed serverless container platform",
        "alternatives": ["AWS Fargate", "Azure Container Apps"]
    },
    "gcp_gke": {
        "name": "GCP GKE",
        "category": "container",
        "features": ["managed_kubernetes", "autopilot", "node_pools", "workload_identity"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Managed Kubernetes service on GCP",
        "alternatives": ["AWS EKS", "AKS"]
    },
    "gcp_cloud_sql": {
        "name": "GCP Cloud SQL",
        "category": "database",
        "features": ["managed_MySQL", "PostgreSQL", "SQL_Server", "high_availability"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Fully managed relational database service",
        "alternatives": ["AWS RDS", "Azure SQL Database"]
    },
    "gcp_bigquery": {
        "name": "GCP BigQuery",
        "category": "analytics",
        "features": ["data_warehouse", "ML_integration", "streaming", "BI_Engine"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Serverless, highly scalable data warehouse",
        "alternatives": ["AWS Redshift", "Snowflake", "Azure Synapse"]
    },
    "gcp_cloud_storage": {
        "name": "GCP Cloud Storage",
        "category": "storage",
        "features": ["object_storage", "lifecycle_management", "versioning", "encryption"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Unified object storage for developers and enterprises",
        "alternatives": ["AWS S3", "Azure Blob Storage"]
    },
    "gcp_firestore": {
        "name": "GCP Firestore",
        "category": "database",
        "features": ["NoSQL", "real_time_sync", "offline_support", "serverless"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Flexible, scalable NoSQL cloud database",
        "alternatives": ["AWS DynamoDB", "MongoDB Atlas"]
    },
    "gcp_pubsub": {
        "name": "GCP Pub/Sub",
        "category": "messaging",
        "features": ["messaging", "event_streaming", "push_pull", "dead_letter"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Messaging and ingestion for event-driven systems",
        "alternatives": ["AWS SNS/SQS", "Azure Service Bus"]
    },
    "gcp_cloud_cdn": {
        "name": "GCP Cloud CDN",
        "category": "cdn",
        "features": ["global_CDN", "edge_caching", "load_balancer_integration"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Content delivery network for low-latency content delivery",
        "alternatives": ["AWS CloudFront", "Cloudflare"]
    },
    "gcp_vertex_ai": {
        "name": "GCP Vertex AI",
        "category": "ml",
        "features": ["ML_training", "model_serving", "AutoML", "pipelines", "feature_store"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Unified ML platform for building and deploying ML models",
        "alternatives": ["AWS SageMaker", "Azure ML"]
    },
    "gcp_cloud_armor": {
        "name": "GCP Cloud Armor",
        "category": "security",
        "features": ["DDoS_protection", "WAF", "rate_limiting", "geo_blocking"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Network security and DDoS protection service",
        "alternatives": ["AWS WAF", "Cloudflare"]
    },
    "gcp_secret_manager": {
        "name": "GCP Secret Manager",
        "category": "security",
        "features": ["secret_storage", "versioning", "rotation", "access_control"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Store API keys, passwords, and other sensitive data",
        "alternatives": ["AWS Secrets Manager", "HashiCorp Vault"]
    },
    "gcp_cloud_build": {
        "name": "GCP Cloud Build",
        "category": "cicd",
        "features": ["CI_CD", "container_builds", "triggers", "artifact_registry"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Continuous integration and delivery platform",
        "alternatives": ["GitHub Actions", "CircleCI"]
    },
    "azure_vms": {
        "name": "Azure Virtual Machines",
        "category": "compute",
        "features": ["virtual_machines", "scale_sets", "spot_VMs", "availability_zones"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "On-demand, scalable computing resources",
        "alternatives": ["AWS EC2", "GCP Compute Engine"]
    },
    "azure_functions": {
        "name": "Azure Functions",
        "category": "serverless",
        "features": ["event_driven", "durable_functions", "bindings", "multiple_languages"],
        "terraform_support": True,
        "kubernetes_integration": False,
        "description": "Event-driven serverless compute platform",
        "alternatives": ["AWS Lambda", "GCP Cloud Functions"]
    },
    "azure_aks": {
        "name": "Azure AKS",
        "category": "container",
        "features": ["managed_kubernetes", "node_pools", "AAD_integration", "GitOps"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Managed Kubernetes service on Azure",
        "alternatives": ["AWS EKS", "GKE"]
    },
    "azure_cosmos_db": {
        "name": "Azure Cosmos DB",
        "category": "database",
        "features": ["multi_model", "global_distribution", "multi_master", "serverless"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Globally distributed, multi-model database service",
        "alternatives": ["AWS DynamoDB", "MongoDB Atlas"]
    },
    "azure_sql_database": {
        "name": "Azure SQL Database",
        "category": "database",
        "features": ["managed_SQL", "auto_tuning", "elastic_pools", "hyperscale"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Intelligent, scalable cloud database service",
        "alternatives": ["AWS RDS", "GCP Cloud SQL"]
    },
    "azure_blob_storage": {
        "name": "Azure Blob Storage",
        "category": "storage",
        "features": ["object_storage", "hot_cool_archive", "immutability", "versioning"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Massively scalable object storage",
        "alternatives": ["AWS S3", "GCP Cloud Storage"]
    },
    "azure_container_apps": {
        "name": "Azure Container Apps",
        "category": "serverless",
        "features": ["serverless_containers", "auto_scaling", "Dapr", "revisions"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Fully managed serverless container service",
        "alternatives": ["AWS Fargate", "GCP Cloud Run"]
    },
    "azure_app_service": {
        "name": "Azure App Service",
        "category": "paas",
        "features": ["web_apps", "API_apps", "deployment_slots", "auto_scale"],
        "terraform_support": True,
        "kubernetes_integration": False,
        "description": "Fully managed platform for building web apps",
        "alternatives": ["AWS Elastic Beanstalk", "GCP App Engine"]
    },
    "azure_key_vault": {
        "name": "Azure Key Vault",
        "category": "security",
        "features": ["secrets", "keys", "certificates", "HSM_backed"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Safeguard cryptographic keys and secrets",
        "alternatives": ["AWS Secrets Manager", "HashiCorp Vault"]
    },
    "azure_ad": {
        "name": "Azure Active Directory",
        "category": "identity",
        "features": ["identity_management", "SSO", "MFA", "conditional_access"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Enterprise identity service with SSO",
        "alternatives": ["AWS IAM", "Okta"]
    },
    "azure_devops": {
        "name": "Azure DevOps",
        "category": "cicd",
        "features": ["pipelines", "repos", "boards", "artifacts", "test_plans"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "DevOps services for teams",
        "alternatives": ["GitHub", "GitLab"]
    },
    "azure_monitor": {
        "name": "Azure Monitor",
        "category": "monitoring",
        "features": ["metrics", "logs", "alerts", "application_insights", "workbooks"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Full-stack monitoring service",
        "alternatives": ["Datadog", "AWS CloudWatch"]
    },
    "docker": {
        "name": "Docker",
        "category": "container",
        "features": ["containerization", "images", "compose", "swarm", "buildx"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Platform for developing, shipping, and running applications in containers",
        "alternatives": ["Podman", "containerd", "LXC"]
    },
    "kubernetes": {
        "name": "Kubernetes",
        "category": "container_orchestration",
        "features": ["orchestration", "auto_scaling", "service_discovery", "rolling_updates", "secrets"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Open-source container orchestration platform",
        "alternatives": ["Docker Swarm", "Nomad", "ECS"]
    },
    "helm": {
        "name": "Helm",
        "category": "kubernetes_tooling",
        "features": ["package_manager", "charts", "releases", "repositories", "hooks"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Package manager for Kubernetes",
        "alternatives": ["Kustomize", "Jsonnet"]
    },
    "podman": {
        "name": "Podman",
        "category": "container",
        "features": ["daemonless", "rootless", "pods", "OCI_compliant"],
        "terraform_support": False,
        "kubernetes_integration": True,
        "description": "Daemonless container engine for managing OCI containers",
        "alternatives": ["Docker", "containerd"]
    },
    "github_actions": {
        "name": "GitHub Actions",
        "category": "cicd",
        "features": ["workflows", "matrix_builds", "artifacts", "environments", "reusable_workflows"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Automate, customize, and execute software development workflows",
        "alternatives": ["GitLab CI", "CircleCI", "Jenkins"]
    },
    "gitlab_ci": {
        "name": "GitLab CI/CD",
        "category": "cicd",
        "features": ["pipelines", "runners", "artifacts", "environments", "auto_devops"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Continuous integration and delivery built into GitLab",
        "alternatives": ["GitHub Actions", "Jenkins", "CircleCI"]
    },
    "jenkins": {
        "name": "Jenkins",
        "category": "cicd",
        "features": ["pipelines", "plugins", "distributed_builds", "Jenkinsfile"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Open source automation server",
        "alternatives": ["GitHub Actions", "GitLab CI", "CircleCI"]
    },
    "circleci": {
        "name": "CircleCI",
        "category": "cicd",
        "features": ["workflows", "orbs", "caching", "parallelism", "insights"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Continuous integration and delivery platform",
        "alternatives": ["GitHub Actions", "GitLab CI", "Jenkins"]
    },
    "argocd": {
        "name": "ArgoCD",
        "category": "cicd",
        "features": ["GitOps", "declarative", "multi_cluster", "SSO", "RBAC"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Declarative, GitOps continuous delivery tool for Kubernetes",
        "alternatives": ["Flux", "Jenkins X"]
    },
    "terraform": {
        "name": "Terraform",
        "category": "iac",
        "features": ["infrastructure_as_code", "providers", "state_management", "modules", "workspaces"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Infrastructure as Code tool for building, changing, and versioning infrastructure",
        "alternatives": ["Pulumi", "CloudFormation", "Ansible"]
    },
    "pulumi": {
        "name": "Pulumi",
        "category": "iac",
        "features": ["infrastructure_as_code", "multi_language", "state_management", "automation_API"],
        "terraform_support": False,
        "kubernetes_integration": True,
        "description": "Infrastructure as Code using familiar programming languages",
        "alternatives": ["Terraform", "CDK", "Crossplane"]
    },
    "cloudformation": {
        "name": "AWS CloudFormation",
        "category": "iac",
        "features": ["infrastructure_as_code", "stacks", "drift_detection", "change_sets"],
        "terraform_support": False,
        "kubernetes_integration": True,
        "description": "Infrastructure as Code service for AWS resources",
        "alternatives": ["Terraform", "CDK", "Pulumi"]
    },
    "ansible": {
        "name": "Ansible",
        "category": "configuration_management",
        "features": ["automation", "playbooks", "roles", "inventory", "modules"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Simple, agentless IT automation platform",
        "alternatives": ["Chef", "Puppet", "Salt"]
    },
    "prometheus": {
        "name": "Prometheus",
        "category": "monitoring",
        "features": ["metrics", "alerting", "PromQL", "service_discovery", "federation"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Open-source systems monitoring and alerting toolkit",
        "alternatives": ["Datadog", "InfluxDB", "Victoria Metrics"]
    },
    "grafana": {
        "name": "Grafana",
        "category": "monitoring",
        "features": ["dashboards", "visualization", "alerting", "plugins", "annotations"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Open source analytics and interactive visualization web application",
        "alternatives": ["Datadog", "Kibana", "New Relic"]
    },
    "datadog": {
        "name": "Datadog",
        "category": "monitoring",
        "features": ["APM", "logs", "metrics", "synthetics", "RUM"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Cloud monitoring as a service platform",
        "alternatives": ["New Relic", "Dynatrace", "Splunk"]
    },
    "elk_stack": {
        "name": "ELK Stack",
        "category": "logging",
        "features": ["Elasticsearch", "Logstash", "Kibana", "Beats", "APM"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Log management and analytics platform",
        "alternatives": ["Splunk", "Datadog", "Loki"]
    },
    "nginx": {
        "name": "NGINX",
        "category": "web_server",
        "features": ["reverse_proxy", "load_balancing", "caching", "SSL_termination"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "High-performance HTTP server and reverse proxy",
        "alternatives": ["Apache", "Caddy", "HAProxy"]
    },
    "redis": {
        "name": "Redis",
        "category": "caching",
        "features": ["in_memory_db", "pub_sub", "clustering", "persistence", "streams"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "In-memory data structure store",
        "alternatives": ["Memcached", "DragonflyDB", "KeyDB"]
    },
    "postgresql": {
        "name": "PostgreSQL",
        "category": "database",
        "features": ["ACID", "JSON", "full_text_search", "extensions", "replication"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Advanced open source relational database",
        "alternatives": ["MySQL", "MariaDB", "CockroachDB"]
    },
    "mongodb": {
        "name": "MongoDB",
        "category": "database",
        "features": ["document_store", "sharding", "aggregation", "transactions", "change_streams"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Document-oriented NoSQL database",
        "alternatives": ["CouchDB", "DynamoDB", "Firestore"]
    },
    "kafka": {
        "name": "Apache Kafka",
        "category": "messaging",
        "features": ["event_streaming", "pub_sub", "partitioning", "replication", "connect"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Distributed event streaming platform",
        "alternatives": ["RabbitMQ", "Pulsar", "AWS Kinesis"]
    },
    "rabbitmq": {
        "name": "RabbitMQ",
        "category": "messaging",
        "features": ["message_broker", "AMQP", "plugins", "clustering", "federation"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Open source message broker",
        "alternatives": ["Kafka", "ActiveMQ", "AWS SQS"]
    },
    "vault": {
        "name": "HashiCorp Vault",
        "category": "security",
        "features": ["secret_management", "encryption", "identity", "dynamic_secrets"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Secrets management and data protection",
        "alternatives": ["AWS Secrets Manager", "Azure Key Vault"]
    },
    "consul": {
        "name": "HashiCorp Consul",
        "category": "service_mesh",
        "features": ["service_discovery", "health_checking", "KV_store", "service_mesh"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Service networking platform",
        "alternatives": ["Istio", "Linkerd", "AWS App Mesh"]
    },
    "istio": {
        "name": "Istio",
        "category": "service_mesh",
        "features": ["traffic_management", "security", "observability", "sidecar_proxy"],
        "terraform_support": True,
        "kubernetes_integration": True,
        "description": "Open platform to connect, manage, and secure microservices",
        "alternatives": ["Linkerd", "Consul Connect", "AWS App Mesh"]
    },
}


PATTERNS_DATABASE: Dict[str, Dict[str, Any]] = {
    "singleton": {
        "name": "Singleton",
        "category": "creational",
        "description": "Ensure a class has only one instance and provide global point of access to it",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance''',
        "use_cases": ["database_connections", "configuration_managers", "logging", "caching"]
    },
    "factory": {
        "name": "Factory Method",
        "category": "creational",
        "description": "Define an interface for creating an object, but let subclasses decide which class to instantiate",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class PaymentFactory:
    @staticmethod
    def create_payment(payment_type: str) -> Payment:
        if payment_type == "credit":
            return CreditCardPayment()
        elif payment_type == "paypal":
            return PayPalPayment()
        raise ValueError(f"Unknown payment type: {payment_type}")''',
        "use_cases": ["object_creation", "plugin_systems", "framework_extensions"]
    },
    "abstract_factory": {
        "name": "Abstract Factory",
        "category": "creational",
        "description": "Provide an interface for creating families of related objects without specifying concrete classes",
        "languages_applicable": ["python", "java", "csharp", "typescript", "cpp"],
        "code_example": '''class UIFactory(ABC):
    @abstractmethod
    def create_button(self) -> Button:
        pass
    
    @abstractmethod
    def create_checkbox(self) -> Checkbox:
        pass

class WindowsFactory(UIFactory):
    def create_button(self) -> Button:
        return WindowsButton()
    
    def create_checkbox(self) -> Checkbox:
        return WindowsCheckbox()''',
        "use_cases": ["cross_platform_UI", "theme_systems", "database_drivers"]
    },
    "builder": {
        "name": "Builder",
        "category": "creational",
        "description": "Separate the construction of a complex object from its representation",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class QueryBuilder:
    def __init__(self):
        self._query = Query()
    
    def select(self, *fields):
        self._query.select_fields = fields
        return self
    
    def where(self, condition):
        self._query.conditions.append(condition)
        return self
    
    def build(self) -> Query:
        return self._query''',
        "use_cases": ["complex_object_construction", "query_builders", "configuration_objects"]
    },
    "prototype": {
        "name": "Prototype",
        "category": "creational",
        "description": "Create new objects by copying existing objects without depending on their classes",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "cpp"],
        "code_example": '''import copy

class Prototype:
    def clone(self):
        return copy.deepcopy(self)

class Document(Prototype):
    def __init__(self, content):
        self.content = content''',
        "use_cases": ["object_cloning", "caching", "undo_mechanisms"]
    },
    "observer": {
        "name": "Observer",
        "category": "behavioral",
        "description": "Define a one-to-many dependency so that when one object changes state, all dependents are notified",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class EventEmitter:
    def __init__(self):
        self._listeners = {}
    
    def on(self, event: str, callback):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def emit(self, event: str, *args):
        for callback in self._listeners.get(event, []):
            callback(*args)''',
        "use_cases": ["event_systems", "pub_sub", "reactive_programming", "UI_updates"]
    },
    "strategy": {
        "name": "Strategy",
        "category": "behavioral",
        "description": "Define a family of algorithms, encapsulate each one, and make them interchangeable",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount: float) -> bool:
        pass

class CreditCardStrategy(PaymentStrategy):
    def pay(self, amount: float) -> bool:
        return process_credit_card(amount)

class PayPalStrategy(PaymentStrategy):
    def pay(self, amount: float) -> bool:
        return process_paypal(amount)''',
        "use_cases": ["algorithm_selection", "payment_processing", "sorting_strategies"]
    },
    "decorator": {
        "name": "Decorator",
        "category": "structural",
        "description": "Attach additional responsibilities to an object dynamically",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)''',
        "use_cases": ["logging", "caching", "authentication", "rate_limiting"]
    },
    "adapter": {
        "name": "Adapter",
        "category": "structural",
        "description": "Convert the interface of a class into another interface clients expect",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class LegacyPaymentSystem:
    def process_payment(self, data):
        pass

class ModernPaymentAdapter:
    def __init__(self, legacy: LegacyPaymentSystem):
        self._legacy = legacy
    
    def pay(self, amount: float, currency: str):
        data = {"amount": amount, "currency": currency}
        return self._legacy.process_payment(data)''',
        "use_cases": ["legacy_integration", "third_party_libraries", "API_compatibility"]
    },
    "command": {
        "name": "Command",
        "category": "behavioral",
        "description": "Encapsulate a request as an object, allowing parameterization and queuing",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class InsertTextCommand(Command):
    def __init__(self, document, position, text):
        self.document = document
        self.position = position
        self.text = text
    
    def execute(self):
        self.document.insert(self.position, self.text)
    
    def undo(self):
        self.document.delete(self.position, len(self.text))''',
        "use_cases": ["undo_redo", "transaction_systems", "task_queues"]
    },
    "facade": {
        "name": "Facade",
        "category": "structural",
        "description": "Provide a unified interface to a set of interfaces in a subsystem",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class OrderFacade:
    def __init__(self):
        self._inventory = InventoryService()
        self._payment = PaymentService()
        self._shipping = ShippingService()
    
    def place_order(self, order):
        if not self._inventory.check_stock(order.items):
            raise OutOfStockError()
        
        payment_result = self._payment.process(order.total)
        if not payment_result.success:
            raise PaymentError()
        
        return self._shipping.schedule(order)''',
        "use_cases": ["complex_subsystems", "API_simplification", "library_wrappers"]
    },
    "proxy": {
        "name": "Proxy",
        "category": "structural",
        "description": "Provide a surrogate or placeholder for another object to control access",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class CachedAPIProxy:
    def __init__(self, api: ExternalAPI):
        self._api = api
        self._cache = {}
    
    def get_data(self, key: str):
        if key not in self._cache:
            self._cache[key] = self._api.get_data(key)
        return self._cache[key]''',
        "use_cases": ["caching", "lazy_loading", "access_control", "logging"]
    },
    "composite": {
        "name": "Composite",
        "category": "structural",
        "description": "Compose objects into tree structures to represent part-whole hierarchies",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class FileSystemComponent(ABC):
    @abstractmethod
    def get_size(self) -> int:
        pass

class File(FileSystemComponent):
    def __init__(self, size: int):
        self.size = size
    
    def get_size(self) -> int:
        return self.size

class Directory(FileSystemComponent):
    def __init__(self):
        self.children = []
    
    def get_size(self) -> int:
        return sum(child.get_size() for child in self.children)''',
        "use_cases": ["tree_structures", "UI_components", "file_systems"]
    },
    "state": {
        "name": "State",
        "category": "behavioral",
        "description": "Allow an object to alter its behavior when its internal state changes",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class OrderState(ABC):
    @abstractmethod
    def process(self, order):
        pass

class PendingState(OrderState):
    def process(self, order):
        order.state = ProcessingState()

class ProcessingState(OrderState):
    def process(self, order):
        order.state = ShippedState()''',
        "use_cases": ["state_machines", "workflow_systems", "game_development"]
    },
    "template_method": {
        "name": "Template Method",
        "category": "behavioral",
        "description": "Define the skeleton of an algorithm, deferring some steps to subclasses",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "cpp"],
        "code_example": '''class DataProcessor(ABC):
    def process(self, data):
        validated = self.validate(data)
        transformed = self.transform(validated)
        return self.save(transformed)
    
    @abstractmethod
    def validate(self, data):
        pass
    
    @abstractmethod
    def transform(self, data):
        pass
    
    def save(self, data):
        return database.save(data)''',
        "use_cases": ["algorithm_frameworks", "data_processing", "code_generation"]
    },
    "iterator": {
        "name": "Iterator",
        "category": "behavioral",
        "description": "Provide a way to access elements of a collection sequentially without exposing representation",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class TreeIterator:
    def __init__(self, root):
        self.stack = [root] if root else []
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.stack:
            raise StopIteration
        node = self.stack.pop()
        self.stack.extend(node.children)
        return node.value''',
        "use_cases": ["collection_traversal", "lazy_evaluation", "data_streams"]
    },
    "mediator": {
        "name": "Mediator",
        "category": "behavioral",
        "description": "Define an object that encapsulates how a set of objects interact",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class ChatMediator:
    def __init__(self):
        self.users = []
    
    def register(self, user):
        self.users.append(user)
        user.mediator = self
    
    def send_message(self, sender, message):
        for user in self.users:
            if user != sender:
                user.receive(message)''',
        "use_cases": ["chat_systems", "UI_components", "event_handling"]
    },
    "chain_of_responsibility": {
        "name": "Chain of Responsibility",
        "category": "behavioral",
        "description": "Pass requests along a chain of handlers until one handles it",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust", "cpp"],
        "code_example": '''class Handler(ABC):
    def __init__(self, next_handler=None):
        self.next_handler = next_handler
    
    def handle(self, request):
        if self.can_handle(request):
            return self.process(request)
        elif self.next_handler:
            return self.next_handler.handle(request)
        return None
    
    @abstractmethod
    def can_handle(self, request) -> bool:
        pass
    
    @abstractmethod
    def process(self, request):
        pass''',
        "use_cases": ["middleware", "validation", "logging", "error_handling"]
    },
    "mvc": {
        "name": "MVC",
        "category": "architectural",
        "description": "Separate application into Model, View, and Controller components",
        "languages_applicable": ["python", "java", "csharp", "typescript", "ruby", "php", "go"],
        "code_example": '''# Model
class UserModel:
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id):
        return self.db.query("SELECT * FROM users WHERE id = ?", user_id)

# View
class UserView:
    def render(self, user):
        return f"<h1>{user.name}</h1>"

# Controller
class UserController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
    
    def show_user(self, user_id):
        user = self.model.get_user(user_id)
        return self.view.render(user)''',
        "use_cases": ["web_applications", "GUI_applications", "separation_of_concerns"]
    },
    "mvvm": {
        "name": "MVVM",
        "category": "architectural",
        "description": "Model-View-ViewModel pattern for UI development with data binding",
        "languages_applicable": ["typescript", "csharp", "kotlin", "swift", "javascript"],
        "code_example": '''// ViewModel
class UserViewModel {
    private user = observable({ name: "", email: "" });
    
    get name() { return this.user.name; }
    set name(value) { this.user.name = value; }
    
    async loadUser(id: string) {
        const data = await api.getUser(id);
        runInAction(() => {
            this.user = data;
        });
    }
}''',
        "use_cases": ["mobile_apps", "desktop_apps", "SPA_applications"]
    },
    "clean_architecture": {
        "name": "Clean Architecture",
        "category": "architectural",
        "description": "Separate software into layers with dependencies pointing inward",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "kotlin"],
        "code_example": '''# Domain Layer (innermost)
class User:
    def __init__(self, id: str, email: str):
        self.id = id
        self.email = email

# Use Case Layer
class CreateUserUseCase:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
    
    def execute(self, email: str) -> User:
        user = User(uuid4(), email)
        return self.user_repo.save(user)

# Interface Adapters Layer
class UserController:
    def __init__(self, create_user: CreateUserUseCase):
        self.create_user = create_user''',
        "use_cases": ["enterprise_applications", "testable_code", "maintainable_systems"]
    },
    "microservices": {
        "name": "Microservices",
        "category": "architectural",
        "description": "Structure application as collection of loosely coupled services",
        "languages_applicable": ["python", "java", "go", "typescript", "csharp", "rust", "kotlin"],
        "code_example": '''# User Service
@app.route("/users/<id>")
def get_user(id):
    return user_repository.find(id)

# Order Service (separate service)
@app.route("/orders", methods=["POST"])
def create_order():
    user = requests.get(f"{USER_SERVICE}/users/{user_id}")
    order = Order(user_id=user["id"], items=request.json["items"])
    return order_repository.save(order)''',
        "use_cases": ["scalable_systems", "team_independence", "technology_diversity"]
    },
    "event_sourcing": {
        "name": "Event Sourcing",
        "category": "architectural",
        "description": "Store state changes as a sequence of events rather than current state",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "scala"],
        "code_example": '''class BankAccount:
    def __init__(self):
        self.events = []
        self.balance = 0
    
    def deposit(self, amount):
        event = DepositEvent(amount=amount, timestamp=datetime.now())
        self.events.append(event)
        self.apply(event)
    
    def apply(self, event):
        if isinstance(event, DepositEvent):
            self.balance += event.amount
    
    def rebuild(self):
        self.balance = 0
        for event in self.events:
            self.apply(event)''',
        "use_cases": ["audit_trails", "temporal_queries", "event_driven_systems"]
    },
    "cqrs": {
        "name": "CQRS",
        "category": "architectural",
        "description": "Separate read and write operations into different models",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "kotlin"],
        "code_example": '''# Command side
class CreateOrderCommand:
    def __init__(self, user_id, items):
        self.user_id = user_id
        self.items = items

class OrderCommandHandler:
    def handle(self, command: CreateOrderCommand):
        order = Order(command.user_id, command.items)
        self.repository.save(order)
        self.event_bus.publish(OrderCreatedEvent(order.id))

# Query side (optimized for reads)
class OrderQueryHandler:
    def get_user_orders(self, user_id):
        return self.read_db.query("SELECT * FROM order_view WHERE user_id = ?", user_id)''',
        "use_cases": ["high_performance_reads", "complex_domains", "event_sourcing"]
    },
    "repository": {
        "name": "Repository",
        "category": "architectural",
        "description": "Abstract the data layer with a collection-like interface",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "ruby", "php"],
        "code_example": '''class UserRepository(ABC):
    @abstractmethod
    def find(self, id: str) -> Optional[User]:
        pass
    
    @abstractmethod
    def save(self, user: User) -> User:
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

class PostgresUserRepository(UserRepository):
    def find(self, id: str) -> Optional[User]:
        row = self.db.query("SELECT * FROM users WHERE id = %s", id)
        return User(**row) if row else None''',
        "use_cases": ["data_access", "testability", "database_abstraction"]
    },
    "dependency_injection": {
        "name": "Dependency Injection",
        "category": "architectural",
        "description": "Pass dependencies to objects rather than having them create dependencies",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "kotlin", "swift"],
        "code_example": '''class OrderService:
    def __init__(
        self,
        order_repo: OrderRepository,
        payment_service: PaymentService,
        notification_service: NotificationService
    ):
        self.order_repo = order_repo
        self.payment_service = payment_service
        self.notification_service = notification_service
    
    def create_order(self, order_data):
        order = Order(**order_data)
        self.order_repo.save(order)
        self.payment_service.process(order)
        self.notification_service.notify(order)''',
        "use_cases": ["testability", "loose_coupling", "flexibility"]
    },
    "input_validation": {
        "name": "Input Validation",
        "category": "security",
        "description": "Validate and sanitize all user inputs before processing",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "php", "ruby"],
        "code_example": '''from pydantic import BaseModel, validator, EmailStr

class UserInput(BaseModel):
    email: EmailStr
    password: str
    age: int
    
    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        return v
    
    @validator("age")
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError("Invalid age")
        return v''',
        "use_cases": ["API_endpoints", "form_processing", "data_integrity"]
    },
    "authentication": {
        "name": "Authentication",
        "category": "security",
        "description": "Verify the identity of users attempting to access the system",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "php", "ruby"],
        "code_example": '''import jwt
from datetime import datetime, timedelta

class JWTAuthentication:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def create_token(self, user_id: str) -> str:
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token: str) -> dict:
        return jwt.decode(token, self.secret_key, algorithms=["HS256"])''',
        "use_cases": ["user_login", "API_security", "session_management"]
    },
    "authorization": {
        "name": "Authorization (RBAC)",
        "category": "security",
        "description": "Control access to resources based on user roles and permissions",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "php", "ruby"],
        "code_example": '''class Permission:
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class RBACAuthorization:
    ROLE_PERMISSIONS = {
        "viewer": [Permission.READ],
        "editor": [Permission.READ, Permission.WRITE],
        "admin": [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN]
    }
    
    def has_permission(self, user_role: str, required_permission: str) -> bool:
        permissions = self.ROLE_PERMISSIONS.get(user_role, [])
        return required_permission in permissions''',
        "use_cases": ["access_control", "resource_protection", "multi_tenant"]
    },
    "encryption": {
        "name": "Data Encryption",
        "category": "security",
        "description": "Protect sensitive data using encryption at rest and in transit",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "rust"],
        "code_example": '''from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class DataEncryption:
    def __init__(self, password: str, salt: bytes):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> bytes:
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, encrypted: bytes) -> str:
        return self.cipher.decrypt(encrypted).decode()''',
        "use_cases": ["sensitive_data", "PII_protection", "compliance"]
    },
    "rate_limiting": {
        "name": "Rate Limiting",
        "category": "security",
        "description": "Limit the number of requests a user can make in a given time period",
        "languages_applicable": ["python", "java", "typescript", "go", "rust", "ruby"],
        "code_example": '''from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window
        
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > window_start
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True''',
        "use_cases": ["API_protection", "DDoS_prevention", "fair_usage"]
    },
    "sql_injection_prevention": {
        "name": "SQL Injection Prevention",
        "category": "security",
        "description": "Protect against SQL injection attacks using parameterized queries",
        "languages_applicable": ["python", "java", "csharp", "typescript", "go", "php", "ruby"],
        "code_example": '''# BAD - Vulnerable to SQL injection
def get_user_unsafe(user_id):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    return db.execute(query)

# GOOD - Using parameterized queries
def get_user_safe(user_id):
    query = "SELECT * FROM users WHERE id = %s"
    return db.execute(query, (user_id,))

# GOOD - Using ORM
def get_user_orm(user_id):
    return User.query.filter_by(id=user_id).first()''',
        "use_cases": ["database_queries", "API_endpoints", "form_processing"]
    },
    "xss_prevention": {
        "name": "XSS Prevention",
        "category": "security",
        "description": "Protect against Cross-Site Scripting attacks by escaping output",
        "languages_applicable": ["python", "java", "typescript", "php", "ruby", "go"],
        "code_example": '''import html

def render_user_content(user_input: str) -> str:
    # Escape HTML special characters
    escaped = html.escape(user_input)
    return f"<div class='user-content'>{escaped}</div>"

# For JavaScript contexts, use JSON encoding
import json
def render_js_data(data: dict) -> str:
    safe_json = json.dumps(data)
    return f"<script>const data = {safe_json};</script>"''',
        "use_cases": ["web_applications", "user_generated_content", "templating"]
    },
    "csrf_protection": {
        "name": "CSRF Protection",
        "category": "security",
        "description": "Protect against Cross-Site Request Forgery attacks using tokens",
        "languages_applicable": ["python", "java", "typescript", "php", "ruby", "go"],
        "code_example": '''import secrets

class CSRFProtection:
    def __init__(self):
        self.tokens = {}
    
    def generate_token(self, session_id: str) -> str:
        token = secrets.token_urlsafe(32)
        self.tokens[session_id] = token
        return token
    
    def validate_token(self, session_id: str, token: str) -> bool:
        expected = self.tokens.get(session_id)
        if not expected:
            return False
        return secrets.compare_digest(expected, token)''',
        "use_cases": ["form_submissions", "state_changing_requests", "web_security"]
    },
}


SKILLS_DATABASE: Dict[str, Dict[str, Any]] = {
    "function_generation": {
        "name": "Function Generation",
        "category": "code_generation",
        "description": "Generate complete, working functions from natural language descriptions or signatures",
        "accuracy_level": 0.92,
        "capabilities": [
            "generate_from_description",
            "generate_from_signature",
            "add_type_hints",
            "add_docstrings",
            "handle_edge_cases",
            "optimize_performance"
        ]
    },
    "class_generation": {
        "name": "Class Generation",
        "category": "code_generation",
        "description": "Generate complete class implementations with methods, properties, and inheritance",
        "accuracy_level": 0.90,
        "capabilities": [
            "generate_from_spec",
            "implement_interfaces",
            "add_constructors",
            "generate_methods",
            "handle_inheritance",
            "add_documentation"
        ]
    },
    "api_endpoint_generation": {
        "name": "API Endpoint Generation",
        "category": "code_generation",
        "description": "Generate RESTful API endpoints with validation, error handling, and documentation",
        "accuracy_level": 0.91,
        "capabilities": [
            "REST_endpoints",
            "GraphQL_resolvers",
            "request_validation",
            "response_formatting",
            "error_handling",
            "OpenAPI_specs"
        ]
    },
    "test_generation": {
        "name": "Test Generation",
        "category": "code_generation",
        "description": "Generate unit tests, integration tests, and test fixtures for existing code",
        "accuracy_level": 0.88,
        "capabilities": [
            "unit_tests",
            "integration_tests",
            "edge_case_tests",
            "mock_generation",
            "fixture_creation",
            "parameterized_tests"
        ]
    },
    "database_schema_generation": {
        "name": "Database Schema Generation",
        "category": "code_generation",
        "description": "Generate database schemas, migrations, and ORM models",
        "accuracy_level": 0.89,
        "capabilities": [
            "SQL_DDL",
            "ORM_models",
            "migrations",
            "indexes",
            "constraints",
            "relationships"
        ]
    },
    "component_generation": {
        "name": "UI Component Generation",
        "category": "code_generation",
        "description": "Generate frontend components for React, Vue, Angular, and other frameworks",
        "accuracy_level": 0.90,
        "capabilities": [
            "React_components",
            "Vue_components",
            "styling",
            "state_management",
            "event_handling",
            "accessibility"
        ]
    },
    "static_analysis": {
        "name": "Static Analysis",
        "category": "code_analysis",
        "description": "Analyze code for potential bugs, anti-patterns, and style issues without execution",
        "accuracy_level": 0.94,
        "capabilities": [
            "syntax_checking",
            "type_checking",
            "dead_code_detection",
            "unreachable_code",
            "unused_variables",
            "style_violations"
        ]
    },
    "complexity_metrics": {
        "name": "Complexity Metrics",
        "category": "code_analysis",
        "description": "Calculate and report code complexity metrics and maintainability scores",
        "accuracy_level": 0.95,
        "capabilities": [
            "cyclomatic_complexity",
            "cognitive_complexity",
            "lines_of_code",
            "function_length",
            "class_size",
            "maintainability_index"
        ]
    },
    "vulnerability_detection": {
        "name": "Vulnerability Detection",
        "category": "code_analysis",
        "description": "Identify security vulnerabilities and potential attack vectors in code",
        "accuracy_level": 0.87,
        "capabilities": [
            "SQL_injection",
            "XSS_vulnerabilities",
            "insecure_dependencies",
            "hardcoded_secrets",
            "CSRF_issues",
            "authentication_flaws"
        ]
    },
    "code_smell_detection": {
        "name": "Code Smell Detection",
        "category": "code_analysis",
        "description": "Identify code smells and design issues that indicate deeper problems",
        "accuracy_level": 0.89,
        "capabilities": [
            "long_methods",
            "god_classes",
            "feature_envy",
            "data_clumps",
            "primitive_obsession",
            "duplicate_code"
        ]
    },
    "dependency_analysis": {
        "name": "Dependency Analysis",
        "category": "code_analysis",
        "description": "Analyze project dependencies for security, updates, and compatibility",
        "accuracy_level": 0.93,
        "capabilities": [
            "outdated_packages",
            "security_advisories",
            "license_compliance",
            "circular_dependencies",
            "unused_dependencies",
            "version_conflicts"
        ]
    },
    "performance_analysis": {
        "name": "Performance Analysis",
        "category": "code_analysis",
        "description": "Identify performance bottlenecks and optimization opportunities",
        "accuracy_level": 0.86,
        "capabilities": [
            "time_complexity",
            "space_complexity",
            "memory_leaks",
            "N+1_queries",
            "inefficient_algorithms",
            "caching_opportunities"
        ]
    },
    "extract_method": {
        "name": "Extract Method",
        "category": "refactoring",
        "description": "Extract code blocks into separate, reusable methods",
        "accuracy_level": 0.93,
        "capabilities": [
            "identify_extraction_points",
            "preserve_semantics",
            "handle_return_values",
            "manage_parameters",
            "update_callers"
        ]
    },
    "rename_refactor": {
        "name": "Rename Refactoring",
        "category": "refactoring",
        "description": "Safely rename identifiers across the codebase",
        "accuracy_level": 0.96,
        "capabilities": [
            "rename_variables",
            "rename_functions",
            "rename_classes",
            "rename_files",
            "update_references",
            "update_imports"
        ]
    },
    "move_refactor": {
        "name": "Move Refactoring",
        "category": "refactoring",
        "description": "Move code elements between files, classes, or modules",
        "accuracy_level": 0.88,
        "capabilities": [
            "move_methods",
            "move_classes",
            "move_files",
            "update_imports",
            "preserve_dependencies"
        ]
    },
    "inline_refactor": {
        "name": "Inline Refactoring",
        "category": "refactoring",
        "description": "Inline variables, methods, or classes to simplify code",
        "accuracy_level": 0.91,
        "capabilities": [
            "inline_variables",
            "inline_methods",
            "inline_temp",
            "preserve_behavior",
            "remove_indirection"
        ]
    },
    "optimize_code": {
        "name": "Code Optimization",
        "category": "refactoring",
        "description": "Optimize code for performance, readability, or maintainability",
        "accuracy_level": 0.87,
        "capabilities": [
            "algorithm_optimization",
            "loop_optimization",
            "memory_optimization",
            "query_optimization",
            "caching_implementation"
        ]
    },
    "modernize_code": {
        "name": "Code Modernization",
        "category": "refactoring",
        "description": "Update code to use modern language features and best practices",
        "accuracy_level": 0.89,
        "capabilities": [
            "use_modern_syntax",
            "update_patterns",
            "migrate_APIs",
            "async_conversion",
            "type_annotations"
        ]
    },
    "docstring_generation": {
        "name": "Docstring Generation",
        "category": "documentation",
        "description": "Generate comprehensive docstrings for functions, classes, and modules",
        "accuracy_level": 0.92,
        "capabilities": [
            "function_docstrings",
            "class_docstrings",
            "module_docstrings",
            "parameter_descriptions",
            "return_descriptions",
            "example_code"
        ]
    },
    "api_documentation": {
        "name": "API Documentation",
        "category": "documentation",
        "description": "Generate API documentation including OpenAPI/Swagger specifications",
        "accuracy_level": 0.91,
        "capabilities": [
            "OpenAPI_specs",
            "endpoint_documentation",
            "request_examples",
            "response_schemas",
            "authentication_docs",
            "error_documentation"
        ]
    },
    "readme_generation": {
        "name": "README Generation",
        "category": "documentation",
        "description": "Generate comprehensive README files for projects",
        "accuracy_level": 0.90,
        "capabilities": [
            "project_description",
            "installation_instructions",
            "usage_examples",
            "configuration_docs",
            "contributing_guidelines",
            "license_info"
        ]
    },
    "inline_comments": {
        "name": "Inline Comment Generation",
        "category": "documentation",
        "description": "Add meaningful inline comments to explain complex code",
        "accuracy_level": 0.88,
        "capabilities": [
            "explain_logic",
            "document_edge_cases",
            "clarify_intent",
            "mark_todos",
            "explain_algorithms"
        ]
    },
    "changelog_generation": {
        "name": "Changelog Generation",
        "category": "documentation",
        "description": "Generate changelogs from commit history and pull requests",
        "accuracy_level": 0.87,
        "capabilities": [
            "version_summaries",
            "breaking_changes",
            "new_features",
            "bug_fixes",
            "deprecations",
            "migration_guides"
        ]
    },
    "code_explanation": {
        "name": "Code Explanation",
        "category": "documentation",
        "description": "Explain code functionality in natural language",
        "accuracy_level": 0.93,
        "capabilities": [
            "line_by_line_explanation",
            "algorithm_explanation",
            "pattern_identification",
            "complexity_explanation",
            "usage_examples"
        ]
    },
    "infrastructure_generation": {
        "name": "Infrastructure Generation",
        "category": "code_generation",
        "description": "Generate Infrastructure as Code for cloud deployments",
        "accuracy_level": 0.88,
        "capabilities": [
            "terraform_configs",
            "kubernetes_manifests",
            "docker_files",
            "helm_charts",
            "cloudformation_templates",
            "ansible_playbooks"
        ]
    },
    "git_operations": {
        "name": "Git Operations",
        "category": "code_generation",
        "description": "Generate git commands and workflows for version control",
        "accuracy_level": 0.94,
        "capabilities": [
            "commit_messages",
            "branch_strategies",
            "merge_resolution",
            "gitignore_files",
            "git_hooks",
            "workflow_configs"
        ]
    },
    "ci_cd_generation": {
        "name": "CI/CD Generation",
        "category": "code_generation",
        "description": "Generate continuous integration and deployment configurations",
        "accuracy_level": 0.89,
        "capabilities": [
            "github_actions",
            "gitlab_ci",
            "jenkins_pipelines",
            "test_workflows",
            "deployment_configs",
            "artifact_handling"
        ]
    },
    "code_translation": {
        "name": "Code Translation",
        "category": "code_generation",
        "description": "Translate code between programming languages",
        "accuracy_level": 0.85,
        "capabilities": [
            "python_to_javascript",
            "javascript_to_typescript",
            "java_to_kotlin",
            "preserve_logic",
            "adapt_idioms",
            "handle_libraries"
        ]
    },
    "code_completion": {
        "name": "Code Completion",
        "category": "code_generation",
        "description": "Provide intelligent code completion suggestions",
        "accuracy_level": 0.94,
        "capabilities": [
            "context_aware",
            "type_aware",
            "import_suggestions",
            "method_signatures",
            "variable_names",
            "pattern_completion"
        ]
    },
}


def get_language_info(lang_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a programming language.
    
    Args:
        lang_name: Name of the language (case-insensitive)
        
    Returns:
        Dictionary with language details or None if not found
    """
    lang_key = lang_name.lower().replace(" ", "_").replace("-", "_")
    
    aliases = {
        "bash": "shell",
        "sh": "shell",
        "zsh": "shell",
        "c++": "cpp",
        "c#": "csharp",
        "f#": "fsharp",
        "js": "javascript",
        "ts": "typescript",
    }
    
    lang_key = aliases.get(lang_key, lang_key)
    
    return LANGUAGES_DATABASE.get(lang_key)


def get_frameworks_for_language(lang_name: str) -> List[Dict[str, Any]]:
    """
    Get all frameworks for a specific programming language.
    
    Args:
        lang_name: Name of the language (case-insensitive)
        
    Returns:
        List of framework dictionaries
    """
    lang_key = lang_name.lower().replace(" ", "_").replace("-", "_")
    
    aliases = {
        "bash": "shell",
        "js": "javascript",
        "ts": "typescript",
        "c#": "csharp",
    }
    
    lang_key = aliases.get(lang_key, lang_key)
    
    frameworks = []
    for fw_key, fw_info in FRAMEWORKS_DATABASE.items():
        if fw_info["language"].lower() == lang_key:
            frameworks.append({"id": fw_key, **fw_info})
    
    return sorted(frameworks, key=lambda x: x["expertise_level"], reverse=True)


def get_infrastructure_by_category(category: str) -> List[Dict[str, Any]]:
    """
    Get infrastructure tools by category.
    
    Args:
        category: Category name (e.g., 'compute', 'database', 'cicd')
        
    Returns:
        List of infrastructure tool dictionaries
    """
    category_key = category.lower().replace(" ", "_").replace("-", "_")
    
    tools = []
    for tool_key, tool_info in INFRASTRUCTURE_DATABASE.items():
        if tool_info["category"].lower() == category_key:
            tools.append({"id": tool_key, **tool_info})
    
    return tools


def get_patterns_by_type(pattern_type: str) -> List[Dict[str, Any]]:
    """
    Get patterns by their category/type.
    
    Args:
        pattern_type: Type of pattern (e.g., 'creational', 'behavioral', 'security')
        
    Returns:
        List of pattern dictionaries
    """
    type_key = pattern_type.lower().replace(" ", "_").replace("-", "_")
    
    patterns = []
    for pattern_key, pattern_info in PATTERNS_DATABASE.items():
        if pattern_info["category"].lower() == type_key:
            patterns.append({"id": pattern_key, **pattern_info})
    
    return patterns


def get_all_skills() -> List[Dict[str, Any]]:
    """
    Get all AI capabilities/skills.
    
    Returns:
        List of all skill dictionaries with their IDs
    """
    skills = []
    for skill_key, skill_info in SKILLS_DATABASE.items():
        skills.append({"id": skill_key, **skill_info})
    
    return sorted(skills, key=lambda x: x["accuracy_level"], reverse=True)


def get_skills_by_category(category: str) -> List[Dict[str, Any]]:
    """
    Get AI skills filtered by category.
    
    Args:
        category: Skill category (e.g., 'code_generation', 'code_analysis')
        
    Returns:
        List of skill dictionaries
    """
    category_key = category.lower().replace(" ", "_").replace("-", "_")
    
    skills = []
    for skill_key, skill_info in SKILLS_DATABASE.items():
        if skill_info["category"].lower() == category_key:
            skills.append({"id": skill_key, **skill_info})
    
    return sorted(skills, key=lambda x: x["accuracy_level"], reverse=True)


def _fuzzy_match(query: str, text: str) -> float:
    """Calculate fuzzy match score between query and text."""
    query = query.lower()
    text = text.lower()
    
    if query in text:
        return 1.0
    
    return SequenceMatcher(None, query, text).ratio()


def search_knowledge_base(query: str, min_score: float = 0.3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fuzzy search across all databases.
    
    Args:
        query: Search query string
        min_score: Minimum match score (0.0 to 1.0)
        
    Returns:
        Dictionary with results from each database
    """
    query_lower = query.lower()
    results = {
        "languages": [],
        "frameworks": [],
        "infrastructure": [],
        "patterns": [],
        "skills": [],
    }
    
    for lang_key, lang_info in LANGUAGES_DATABASE.items():
        searchable = f"{lang_key} {lang_info['name']} {' '.join(lang_info['paradigms'])} {' '.join(lang_info['common_frameworks'])}"
        score = _fuzzy_match(query_lower, searchable)
        if score >= min_score:
            results["languages"].append({
                "id": lang_key,
                "score": score,
                **lang_info
            })
    
    for fw_key, fw_info in FRAMEWORKS_DATABASE.items():
        searchable = f"{fw_key} {fw_info['name']} {fw_info['language']} {fw_info['category']} {' '.join(fw_info['features'])}"
        score = _fuzzy_match(query_lower, searchable)
        if score >= min_score:
            results["frameworks"].append({
                "id": fw_key,
                "score": score,
                **fw_info
            })
    
    for tool_key, tool_info in INFRASTRUCTURE_DATABASE.items():
        searchable = f"{tool_key} {tool_info['name']} {tool_info['category']} {' '.join(tool_info['features'])}"
        score = _fuzzy_match(query_lower, searchable)
        if score >= min_score:
            results["infrastructure"].append({
                "id": tool_key,
                "score": score,
                **tool_info
            })
    
    for pattern_key, pattern_info in PATTERNS_DATABASE.items():
        searchable = f"{pattern_key} {pattern_info['name']} {pattern_info['category']} {pattern_info['description']}"
        score = _fuzzy_match(query_lower, searchable)
        if score >= min_score:
            results["patterns"].append({
                "id": pattern_key,
                "score": score,
                **pattern_info
            })
    
    for skill_key, skill_info in SKILLS_DATABASE.items():
        searchable = f"{skill_key} {skill_info['name']} {skill_info['category']} {skill_info['description']} {' '.join(skill_info['capabilities'])}"
        score = _fuzzy_match(query_lower, searchable)
        if score >= min_score:
            results["skills"].append({
                "id": skill_key,
                "score": score,
                **skill_info
            })
    
    for category in results:
        results[category] = sorted(results[category], key=lambda x: x["score"], reverse=True)
    
    return results


def get_all_languages() -> List[str]:
    """Get a list of all supported language names."""
    return [info["name"] for info in LANGUAGES_DATABASE.values()]


def get_all_framework_names() -> List[str]:
    """Get a list of all framework names."""
    return [info["name"] for info in FRAMEWORKS_DATABASE.values()]


def get_all_infrastructure_categories() -> List[str]:
    """Get a list of all unique infrastructure categories."""
    categories = set()
    for info in INFRASTRUCTURE_DATABASE.values():
        categories.add(info["category"])
    return sorted(list(categories))


def get_all_pattern_categories() -> List[str]:
    """Get a list of all unique pattern categories."""
    categories = set()
    for info in PATTERNS_DATABASE.values():
        categories.add(info["category"])
    return sorted(list(categories))


def get_all_skill_categories() -> List[str]:
    """Get a list of all unique skill categories."""
    categories = set()
    for info in SKILLS_DATABASE.values():
        categories.add(info["category"])
    return sorted(list(categories))


def get_statistics() -> Dict[str, Any]:
    """
    Get statistics about the knowledge base.
    
    Returns:
        Dictionary with counts and statistics
    """
    return {
        "languages_count": len(LANGUAGES_DATABASE),
        "frameworks_count": len(FRAMEWORKS_DATABASE),
        "infrastructure_count": len(INFRASTRUCTURE_DATABASE),
        "patterns_count": len(PATTERNS_DATABASE),
        "skills_count": len(SKILLS_DATABASE),
        "language_names": get_all_languages(),
        "infrastructure_categories": get_all_infrastructure_categories(),
        "pattern_categories": get_all_pattern_categories(),
        "skill_categories": get_all_skill_categories(),
        "total_entries": (
            len(LANGUAGES_DATABASE) +
            len(FRAMEWORKS_DATABASE) +
            len(INFRASTRUCTURE_DATABASE) +
            len(PATTERNS_DATABASE) +
            len(SKILLS_DATABASE)
        )
    }


if __name__ == "__main__":
    print("Platform Forge AI Knowledge Base")
    print("=" * 50)
    
    stats = get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Languages: {stats['languages_count']}")
    print(f"  Frameworks: {stats['frameworks_count']}")
    print(f"  Infrastructure Tools: {stats['infrastructure_count']}")
    print(f"  Patterns: {stats['patterns_count']}")
    print(f"  AI Skills: {stats['skills_count']}")
    print(f"  Total Entries: {stats['total_entries']}")
    
    print("\n" + "=" * 50)
    print("Example: Python Language Info")
    print("-" * 50)
    python_info = get_language_info("python")
    if python_info:
        print(f"  Name: {python_info['name']}")
        print(f"  Expertise Level: {python_info['expertise_level']}/10")
        print(f"  Paradigms: {', '.join(python_info['paradigms'])}")
    
    print("\n" + "=" * 50)
    print("Example: Python Frameworks")
    print("-" * 50)
    for fw in get_frameworks_for_language("python")[:5]:
        print(f"  - {fw['name']}: {fw['description'][:60]}...")
    
    print("\n" + "=" * 50)
    print("Example: Search for 'kubernetes'")
    print("-" * 50)
    results = search_knowledge_base("kubernetes")
    for category, items in results.items():
        if items:
            print(f"\n  {category.upper()}:")
            for item in items[:3]:
                print(f"    - {item['name']} (score: {item['score']:.2f})")
