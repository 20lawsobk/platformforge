# PlatformBuilder - Infrastructure Generation Platform

## Overview

PlatformBuilder is a web application designed to automate the creation of production-ready infrastructure configurations from user-provided code or GitHub repositories. It analyzes code, identifies architectural patterns, and generates infrastructure-as-code artifacts such as Terraform, Kubernetes manifests, and Dockerfiles. The platform aims to streamline the deployment process by providing tailored, downloadable infrastructure templates.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions

The frontend is built with React and TypeScript, leveraging Vite for tooling. It employs Tailwind CSS v4 with a custom dark cybernetic theme and `shadcn/ui` components for an accessible and customizable user experience. Design prioritizes a modern technical aesthetic with custom fonts (Space Grotesk, JetBrains Mono, Inter) and efficient server state management via TanStack Query.

### Technical Implementations

**Frontend:**
-   **Framework:** React with TypeScript
-   **Build Tooling:** Vite
-   **Routing:** Wouter
-   **State Management:** TanStack Query for server state
-   **Styling:** Tailwind CSS with custom theming, `shadcn/ui` component library.

**Backend:**
-   **Framework:** Express.js with TypeScript
-   **Build Strategy:** `esbuild` for optimized production bundles
-   **API:** RESTful endpoints under `/api`.

**Data Storage:**
-   **Database:** PostgreSQL (Neon serverless)
-   **ORM:** Drizzle ORM for type-safe queries
-   **Schema:** Shared schema (`shared/schema.ts`) with Zod validation.
-   **Identifiers:** UUID primary keys.

**Infrastructure Generation Process:**
-   Users submit code or a GitHub URL.
-   An asynchronous process analyzes the code to generate configurations.
-   Generated artifacts (Terraform, Kubernetes, Docker, scaling configs, detected dependencies) are stored and made available.

**AI Model Architecture (Platform Forge Engine):**
-   **Core Model:** Custom decoder-only Transformer with advanced features (MQA, MoE, RoPE), varying from tiny to master configurations.
-   **Knowledge Base:** Extensive knowledge base covering 30+ programming languages, 55+ frameworks, 72+ infrastructure tools, and 34 design patterns.
-   **Code Analyzer:** Performs static analysis (AST parsing), vulnerability detection, complexity metrics, code smell detection, performance analysis, and dependency analysis.
-   **Supported Languages:** Expert-level support for Python, JavaScript, TypeScript, Java, Go, Rust, and advanced/standard support for many others, including infrastructure-as-code languages like Terraform and Kubernetes.

**User Experience Systems (Platform Enhancements):**
-   **Safety Guards:** Detects and prevents destructive actions, offers confirmation prompts and safe alternatives.
-   **Cost Estimator:** Provides pre-action cost estimates, real-time tracking, budget management, and optimization suggestions.
-   **Checkpoint System:** Automated checkpoints before critical operations, with file snapshots, diff viewing, and rollback capabilities.
-   **Context Manager:** Manages a 16k token context window, project context, conversation memory, and learning memory, tracking user instructions.
-   **Instruction Validator:** Parses natural language instructions, detects negative patterns, and ensures compliance of generated code.
-   **Error Handler:** Provides user-friendly error messages, auto-fix suggestions, and tracks recurring issues.
-   **Deployment Engine:** Offers advanced deployment types (AUTOSCALE, RESERVED_VM, STATIC, SCHEDULED, EDGE, SERVERLESS) with improvements over standard systems like faster cold starts, predictive auto-scaling, multi-region deployment, zero-downtime strategies, and cost optimization.

**Replit Feature Parity Systems (with improvements):**
-   **Key-Value Store:** In-memory with persistence, atomic operations, and TTL support.
-   **Object Storage:** S3-compatible, supporting large files, presigned URLs, and versioning.
-   **Secrets Manager:** AES-256-GCM encryption, versioning, audit logging, and environment variable injection.
-   **App Testing:** Browser automation, visual regression, Core Web Vitals, issue detection, and report generation.
-   **Web Search:** Multi-backend support, caching, content extraction, and documentation fetching.
-   **Image Generation:** Multiple backend support (DALL-E 3, Stability AI), style presets, image editing, and batch generation.
-   **Auth System:** OAuth, MFA, RBAC, JWT management, and robust security features.
-   **Stripe Integration:** Comprehensive customer, subscription, and payment management.
-   **Bot/Automation Framework:** Platform bots, cron-like scheduler, workflow automation, and background job processing.

## External Dependencies

### Third-Party Services

-   **Neon PostgreSQL:** Serverless database for primary data storage.
-   **GitHub API:** (Implied) For repository analysis.
-   **Cloud Providers (AWS, GCP, Azure):** (Implied) For deploying generated infrastructure.
-   **Container Registries:** (Implied) For Docker image storage.

### Key NPM Packages

-   **UI/Frontend:**
    -   `@radix-ui/*`: Accessible UI primitives.
    -   `@tanstack/react-query`: Server state management.
    -   `wouter`: Lightweight routing.
    -   `framer-motion`: Animations.
    -   `jszip`, `file-saver`: Client-side file manipulation.
-   **Backend:**
    -   `express`: HTTP server framework.
    -   `drizzle-orm`: Database ORM.
    -   `@neondatabase/serverless`: PostgreSQL driver.
    -   `zod`: Schema validation.
-   **Development:**
    -   `vite`: Frontend build tool.
    -   `esbuild`: Backend bundler.
    -   `drizzle-kit`: Database migrations.
-   **Styling:**
    -   `tailwindcss`: CSS framework.
    -   `class-variance-authority`, `tailwind-merge`, `clsx`: Styling utilities.