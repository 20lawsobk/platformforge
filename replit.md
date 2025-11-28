# PlatformBuilder - Infrastructure Generation Platform

## Overview

PlatformBuilder is a web application that transforms scripts and GitHub repositories into production-ready infrastructure configurations. The platform analyzes code, detects architecture patterns, and automatically generates infrastructure-as-code artifacts including Terraform configurations, Kubernetes manifests, Dockerfiles, and scaling configurations. Users can input a GitHub URL or upload code, which is then processed to produce downloadable infrastructure templates tailored for production deployment.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack:**
- React with TypeScript as the core UI framework
- Vite for build tooling and development server
- Wouter for client-side routing (lightweight alternative to React Router)
- TanStack Query (React Query) for server state management and data fetching
- Tailwind CSS v4 with custom theming for styling
- shadcn/ui component library (Radix UI primitives) for UI components

**Design Decisions:**
- **Component Library Choice**: Uses shadcn/ui with the "new-york" style preset, providing accessible, customizable components built on Radix UI primitives
- **Styling Approach**: Tailwind CSS with custom CSS variables for theming, enabling a dark cybernetic aesthetic with easy theme customization
- **State Management**: React Query handles server state, eliminating need for Redux/Context for API data
- **Routing Strategy**: Wouter chosen for minimal bundle size while providing essential routing features
- **Custom Font Stack**: Uses Space Grotesk (headings), JetBrains Mono (code/mono), and Inter (body text) for a modern technical aesthetic

**Key Pages:**
- Home: Landing page with input for GitHub URLs or code
- Builder: Main interface showing file system, generated infrastructure, and build logs
- Compare: Feature comparison with competitors
- Dashboard: Project overview, deployments, logs, and settings

### Backend Architecture

**Technology Stack:**
- Express.js as the HTTP server framework
- TypeScript with ES modules
- HTTP server (Node.js native) wrapping Express

**Design Decisions:**
- **API Structure**: RESTful API endpoints under `/api` prefix
- **Development vs Production**: Vite dev server with HMR in development, static file serving in production
- **Build Strategy**: esbuild bundles server code with selective dependency bundling to optimize cold starts
- **Logging**: Custom request logging middleware with timestamp formatting and JSON response capture
- **Static Assets**: Express serves built frontend from `dist/public` directory in production

**Key API Endpoints:**
- `POST /api/projects` - Create new project and trigger infrastructure generation
- `GET /api/projects/:id` - Fetch specific project details
- `GET /api/projects` - List all projects

### Data Storage

**Database:**
- PostgreSQL via Neon serverless driver
- Drizzle ORM for type-safe database queries and migrations
- Connection pooling handled by `@neondatabase/serverless`

**Schema Design:**
- **Projects Table**: Stores project metadata (name, source URL, type, status, timestamps)
- **Infrastructure Templates Table**: Stores AI analysis results and generated configurations (Terraform, Kubernetes, Docker, scaling parameters, detected dependencies)
- **Build Logs Table**: Stores timestamped log entries with severity levels (ai, system, action, cmd, success, error)

**Design Decisions:**
- **ORM Choice**: Drizzle selected for type safety and lightweight footprint
- **Schema Location**: Shared schema (`shared/schema.ts`) accessible to both frontend and backend
- **Validation**: Zod schemas auto-generated from Drizzle schemas via `drizzle-zod`
- **UUID Primary Keys**: Uses PostgreSQL's `gen_random_uuid()` for distributed-friendly IDs

### Infrastructure Generation Process

**Workflow:**
1. User submits GitHub URL or code upload
2. Backend creates project record with 'pending' status
3. Async process (`generateInfrastructure`) analyzes code and generates configurations
4. Build logs stream progress to client
5. Generated artifacts stored in infrastructure templates table
6. Status updates to 'complete' or 'failed'

**Generated Artifacts:**
- Terraform configurations for cloud resources
- Kubernetes manifests for container orchestration
- Dockerfiles for containerization
- Docker Compose configurations
- Auto-scaling parameters (min/max instances)
- Detected dependencies (databases, caches, queues)

### Development Tooling

**Replit Integration:**
- Custom Vite plugins for Replit-specific features (cartographer, dev banner)
- Meta image plugin updates OpenGraph tags for proper social sharing
- Runtime error overlay for better DX

**Build Process:**
- Client: Vite builds React app to `dist/public`
- Server: esbuild bundles TypeScript server code with dependency allowlist
- Single build command produces production-ready artifacts

## External Dependencies

### Third-Party Services

**Database:**
- Neon PostgreSQL (serverless) - Primary data store
- Connection via `DATABASE_URL` environment variable

**Potential Integrations (based on schema):**
- Cloud providers (AWS, GCP, Azure) for infrastructure deployment
- GitHub API for repository analysis
- Container registries for Docker image storage

### Key NPM Packages

**UI/Frontend:**
- `@radix-ui/*` - Accessible component primitives (40+ packages)
- `@tanstack/react-query` - Server state management
- `framer-motion` - Animation library
- `wouter` - Lightweight routing
- `jszip` & `file-saver` - Client-side file generation/download

**Backend:**
- `express` - HTTP server framework
- `drizzle-orm` - Database ORM
- `@neondatabase/serverless` - PostgreSQL driver
- `zod` - Schema validation

**Development:**
- `vite` - Build tool and dev server
- `tsx` - TypeScript execution
- `esbuild` - Production server bundler
- `drizzle-kit` - Database migrations

**Styling:**
- `tailwindcss` - Utility-first CSS framework
- `class-variance-authority` - Component variant management
- `tailwind-merge` & `clsx` - Class name utilities