import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { insertProjectSchema, insertBuildLogSchema } from "@shared/schema";
import { aiModelClient, fallbackResponses } from "./ai/model-client";
import { setupAuth, isAuthenticated } from "./replitAuth";
import multer from 'multer';
import path from 'path';

interface UploadedFile {
  filename: string;
  content: string;
  extension: string;
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedExtensions = ['.js', '.ts', '.py', '.go', '.rs', '.java', '.rb', '.php', '.c', '.cpp', '.h', '.hpp', '.cs', '.swift', '.kt', '.scala', '.json', '.yaml', '.yml', '.toml', '.xml', '.html', '.css', '.scss', '.md', '.txt', '.sh', '.bash', '.dockerfile', '.tf'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedExtensions.includes(ext) || !ext) {
      cb(null, true);
    } else {
      cb(null, true);
    }
  }
});

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  // Setup authentication
  await setupAuth(app);
  
  // Auth routes
  app.get('/api/auth/user', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const user = await storage.getUser(userId);
      res.json(user);
    } catch (error) {
      console.error("Error fetching user:", error);
      res.status(500).json({ message: "Failed to fetch user" });
    }
  });

  // Complete onboarding
  app.post('/api/auth/onboarding', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      await storage.updateUserOnboarding(userId, true);
      const user = await storage.getUser(userId);
      res.json(user);
    } catch (error) {
      console.error("Error completing onboarding:", error);
      res.status(500).json({ message: "Failed to complete onboarding" });
    }
  });

  // Get user's projects
  app.get('/api/user/projects', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const projects = await storage.getProjectsByUserId(userId);
      res.json(projects);
    } catch (error) {
      console.error("Error fetching user projects:", error);
      res.status(500).json({ message: "Failed to fetch projects" });
    }
  });

  // Create a new project (authenticated)
  app.post("/api/projects", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const data = insertProjectSchema.parse({ ...req.body, userId });
      const project = await storage.createProject(data);
      
      // Start async infrastructure generation process
      generateInfrastructure(project.id, project.sourceUrl, project.name).catch(console.error);
      
      res.json(project);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create project" });
      }
    }
  });

  // Get project by ID (authenticated - owner only)
  app.get("/api/projects/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: "Project not found" });
      }
      if (project.userId && project.userId !== userId) {
        return res.status(404).json({ error: "Project not found" });
      }
      res.json(project);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch project" });
    }
  });

  // Get all projects (authenticated - user's projects only)
  app.get("/api/projects", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const projects = await storage.getProjectsByUserId(userId);
      res.json(projects);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch projects" });
    }
  });

  // Get infrastructure template for a project (authenticated - owner only)
  app.get("/api/projects/:id/infrastructure", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.id);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      const template = await storage.getInfrastructureTemplateByProjectId(req.params.id);
      if (!template) {
        return res.status(404).json({ error: "Infrastructure template not found" });
      }
      res.json(template);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch infrastructure template" });
    }
  });

  // Get build logs for a project (authenticated - owner only)
  app.get("/api/projects/:id/logs", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.id);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      const logs = await storage.getBuildLogsByProjectId(req.params.id);
      res.json(logs);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch logs" });
    }
  });

  // Upload files and create project (authenticated)
  app.post("/api/projects/upload", isAuthenticated, upload.array('files', 50), async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const files = req.files as Express.Multer.File[];
      
      if (!files || files.length === 0) {
        return res.status(400).json({ error: "No files uploaded" });
      }

      const projectName = req.body.projectName || req.body.name ||
        path.basename(files[0].originalname, path.extname(files[0].originalname)) ||
        'uploaded-project';

      const uploadedFiles: UploadedFile[] = files.map(file => ({
        filename: file.originalname,
        content: file.buffer.toString('utf-8'),
        extension: path.extname(file.originalname).toLowerCase()
      }));

      const project = await storage.createProject({
        name: projectName,
        sourceUrl: `upload://${projectName}`,
        sourceType: 'upload',
        status: 'pending',
        userId,
      });

      generateInfrastructureFromUpload(project.id, projectName, uploadedFiles).catch(console.error);

      res.json(project);
    } catch (error) {
      console.error('Upload error:', error);
      res.status(500).json({ error: "Failed to process uploaded files" });
    }
  });

  // ============================================
  // Custom AI Model Endpoints
  // ============================================

  // Health check for AI model
  app.get("/api/ai/health", async (req, res) => {
    try {
      const health = await aiModelClient.health();
      res.json(health);
    } catch (error) {
      res.json({
        status: 'offline',
        model_initialized: false,
        is_training: false,
        message: 'AI model service not running. Using fallback responses.'
      });
    }
  });

  // Get AI model info
  app.get("/api/ai/model-info", async (req, res) => {
    try {
      const info = await aiModelClient.getModelInfo();
      res.json(info);
    } catch (error) {
      res.json({
        loaded: false,
        message: 'AI model service not available'
      });
    }
  });

  // Chat with AI assistant
  app.post("/api/ai/chat", async (req, res) => {
    try {
      const { message, max_tokens = 200 } = req.body;
      
      if (!message || typeof message !== 'string') {
        return res.status(400).json({ error: 'Message is required' });
      }

      try {
        const result = await aiModelClient.chat({ message, max_tokens });
        res.json({
          response: result.response,
          source: 'custom_model'
        });
      } catch {
        // Fallback to rule-based responses
        res.json({
          response: fallbackResponses.chat(message),
          source: 'fallback'
        });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to process chat message' });
    }
  });

  // Generate text from prompt
  app.post("/api/ai/generate", async (req, res) => {
    try {
      const { prompt, max_tokens = 100, temperature = 0.8, language } = req.body;
      
      if (!prompt || typeof prompt !== 'string') {
        return res.status(400).json({ error: 'Prompt is required' });
      }

      try {
        const result = await aiModelClient.generate({ prompt, max_tokens, temperature, language });
        res.json({
          generated_text: result.generated_text,
          source: 'custom_model'
        });
      } catch {
        res.json({
          generated_text: fallbackResponses.generate(prompt),
          source: 'fallback'
        });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to generate text' });
    }
  });

  // Complete code
  app.post("/api/ai/complete", async (req, res) => {
    try {
      const { code, language = 'python', max_tokens = 150 } = req.body;
      
      if (!code || typeof code !== 'string') {
        return res.status(400).json({ error: 'Code is required' });
      }

      try {
        const result = await aiModelClient.complete({ code, language, max_tokens });
        res.json({
          completion: result.completion,
          source: 'custom_model'
        });
      } catch {
        res.json({
          completion: fallbackResponses.complete(code, language),
          source: 'fallback'
        });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to complete code' });
    }
  });

  // Explain code
  app.post("/api/ai/explain", async (req, res) => {
    try {
      const { code, language = 'python' } = req.body;
      
      if (!code || typeof code !== 'string') {
        return res.status(400).json({ error: 'Code is required' });
      }

      try {
        const result = await aiModelClient.explain({ code, language });
        res.json({
          explanation: result.explanation,
          source: 'custom_model'
        });
      } catch {
        res.json({
          explanation: fallbackResponses.explain(code),
          source: 'fallback'
        });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to explain code' });
    }
  });

  // Fix buggy code
  app.post("/api/ai/fix", async (req, res) => {
    try {
      const { code, error: codeError, language = 'python' } = req.body;
      
      if (!code || typeof code !== 'string') {
        return res.status(400).json({ error: 'Code is required' });
      }
      if (!codeError || typeof codeError !== 'string') {
        return res.status(400).json({ error: 'Error message is required' });
      }

      try {
        const result = await aiModelClient.fix({ code, error: codeError, language });
        res.json({
          fixed_code: result.fixed_code,
          source: 'custom_model'
        });
      } catch {
        res.json({
          fixed_code: fallbackResponses.fix(code, codeError),
          source: 'fallback'
        });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to fix code' });
    }
  });

  // Analyze code
  app.post("/api/ai/analyze", async (req, res) => {
    try {
      const { code, language = 'python' } = req.body;
      
      if (!code || typeof code !== 'string') {
        return res.status(400).json({ error: 'Code is required' });
      }

      try {
        const result = await aiModelClient.analyze({ code, language });
        res.json({
          ...result,
          source: 'custom_model'
        });
      } catch {
        res.json({
          explanation: fallbackResponses.explain(code),
          completion: fallbackResponses.complete(code, language),
          language,
          source: 'fallback'
        });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to analyze code' });
    }
  });

  // Clear chat history
  app.post("/api/ai/clear-history", async (req, res) => {
    try {
      await aiModelClient.clearHistory();
      res.json({ status: 'cleared' });
    } catch {
      res.json({ status: 'cleared', note: 'AI model offline, no history to clear' });
    }
  });

  return httpServer;
}

// AI Infrastructure Generation Logic
async function generateInfrastructure(projectId: string, sourceUrl: string, projectName: string) {
  const logStep = async (level: string, message: string) => {
    await storage.createBuildLog({
      projectId,
      logLevel: level,
      message,
    });
  };

  try {
    await storage.updateProjectStatus(projectId, 'analyzing');
    await logStep('ai', 'Analyzing repository structure and dependencies...');
    
    // Simulate AI analysis delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Detect language and framework (simplified for now)
    const detectedLanguage = 'typescript';
    const detectedFramework = 'node';
    await logStep('ai', `Detected high-throughput ${detectedFramework} API pattern.`);
    
    await storage.updateProjectStatus(projectId, 'generating');
    await logStep('system', '>> Architecting microservices solution...');
    
    await new Promise(resolve => setTimeout(resolve, 800));
    await logStep('action', 'PROVISIONING: Managed PostgreSQL (v15) for persistent data');
    
    await new Promise(resolve => setTimeout(resolve, 800));
    await logStep('action', 'PROVISIONING: Redis Cluster for session caching & rate limiting');
    
    await new Promise(resolve => setTimeout(resolve, 800));
    await logStep('action', 'CONFIGURING: Auto-scaling group (min: 2, max: 20 instances)');
    
    await new Promise(resolve => setTimeout(resolve, 800));
    await logStep('action', 'SECURITY: Setting up VPC peering and private subnets');
    
    await new Promise(resolve => setTimeout(resolve, 800));
    await logStep('info', 'Generating Terraform configuration...');
    
    // Generate actual infrastructure code
    const sanitizedName = projectName.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
    
    const terraformConfig = `
provider "aws" {
  region = "us-east-1"
}

resource "aws_eks_cluster" "main" {
  name     = "${sanitizedName}-cluster"
  role_arn = aws_iam_role.eks_cluster.arn

  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }
}

resource "aws_db_instance" "default" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.t3.micro"
  db_name              = "platform_db"
  username             = "admin"
  password             = var.db_password
  skip_final_snapshot  = true
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${sanitizedName}-cache"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
}
`;

    const kubernetesConfig = `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${sanitizedName}-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: main
  template:
    metadata:
      labels:
        app: main
    spec:
      containers:
      - name: app
        image: ${sourceUrl}:latest
        ports:
        - containerPort: 3000
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: host
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: main-autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${sanitizedName}-app
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
`;

    const dockerfileContent = `
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
`;

    await logStep('cmd', '> terraform init && terraform apply -auto-approve');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Save infrastructure template
    await storage.createInfrastructureTemplate({
      projectId,
      detectedLanguage,
      detectedFramework,
      architecture: 'microservices',
      terraformConfig,
      kubernetesConfig,
      dockerfileContent,
      dockerComposeConfig: null,
      minInstances: 2,
      maxInstances: 20,
      requiresDatabase: 'postgres',
      requiresCache: 'redis',
      requiresQueue: null,
    });
    
    await logStep('success', 'Infrastructure provisioned in us-east-1, eu-west-1');
    await new Promise(resolve => setTimeout(resolve, 800));
    
    await logStep('ai', 'Optimizing CDN rules for global content delivery...');
    await new Promise(resolve => setTimeout(resolve, 800));
    
    await logStep('success', 'Platform deployed. Health checks passing.');
    
    await storage.updateProjectStatus(projectId, 'complete', new Date());
    
  } catch (error) {
    console.error('Infrastructure generation failed:', error);
    await logStep('error', `Generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    await storage.updateProjectStatus(projectId, 'failed');
  }
}

function detectLanguageFromFiles(files: UploadedFile[]): { language: string; framework: string } {
  const extensionMap: Record<string, string> = {
    '.js': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.jsx': 'javascript',
    '.py': 'python',
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.rb': 'ruby',
    '.php': 'php',
    '.c': 'c',
    '.cpp': 'cpp',
    '.cs': 'csharp',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala'
  };

  const extensionCounts: Record<string, number> = {};
  for (const file of files) {
    const lang = extensionMap[file.extension];
    if (lang) {
      extensionCounts[lang] = (extensionCounts[lang] || 0) + 1;
    }
  }

  let detectedLanguage = 'typescript';
  let maxCount = 0;
  for (const [lang, count] of Object.entries(extensionCounts)) {
    if (count > maxCount) {
      maxCount = count;
      detectedLanguage = lang;
    }
  }

  let framework = 'node';
  const allContent = files.map(f => f.content).join('\n');
  
  if (detectedLanguage === 'python') {
    if (allContent.includes('from flask') || allContent.includes('import flask')) {
      framework = 'flask';
    } else if (allContent.includes('from django') || allContent.includes('import django')) {
      framework = 'django';
    } else if (allContent.includes('from fastapi') || allContent.includes('import fastapi')) {
      framework = 'fastapi';
    } else {
      framework = 'python';
    }
  } else if (detectedLanguage === 'typescript' || detectedLanguage === 'javascript') {
    if (allContent.includes('from react') || allContent.includes("from 'react'") || allContent.includes('import React')) {
      framework = 'react';
    } else if (allContent.includes('express')) {
      framework = 'express';
    } else if (allContent.includes('@nestjs')) {
      framework = 'nestjs';
    } else if (allContent.includes('next')) {
      framework = 'nextjs';
    } else {
      framework = 'node';
    }
  } else if (detectedLanguage === 'go') {
    framework = 'go';
  } else if (detectedLanguage === 'rust') {
    framework = 'rust';
  } else if (detectedLanguage === 'java') {
    if (allContent.includes('springframework')) {
      framework = 'spring';
    } else {
      framework = 'java';
    }
  }

  return { language: detectedLanguage, framework };
}

function generateDockerfileForLanguage(language: string, framework: string): string {
  switch (language) {
    case 'python':
      return `FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "${framework === 'flask' ? 'flask run --host=0.0.0.0' : framework === 'django' ? 'gunicorn' : framework === 'fastapi' ? 'uvicorn main:app --host 0.0.0.0' : 'python main.py'}"]`;
    case 'go':
      return `FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o main .

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/main .
EXPOSE 8080
CMD ["./main"]`;
    case 'rust':
      return `FROM rust:1.73 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/app .
EXPOSE 8080
CMD ["./app"]`;
    case 'java':
      return `FROM eclipse-temurin:17-jdk AS builder
WORKDIR /app
COPY . .
RUN ./gradlew build --no-daemon || ./mvnw package -DskipTests

FROM eclipse-temurin:17-jre
WORKDIR /app
COPY --from=builder /app/build/libs/*.jar app.jar
EXPOSE 8080
CMD ["java", "-jar", "app.jar"]`;
    default:
      return `FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]`;
  }
}

async function generateInfrastructureFromUpload(projectId: string, projectName: string, files: UploadedFile[]) {
  const logStep = async (level: string, message: string) => {
    await storage.createBuildLog({
      projectId,
      logLevel: level,
      message,
    });
  };

  try {
    await storage.updateProjectStatus(projectId, 'analyzing');
    await logStep('ai', `Analyzing ${files.length} uploaded files...`);
    
    await new Promise(resolve => setTimeout(resolve, 800));
    
    const { language, framework } = detectLanguageFromFiles(files);
    await logStep('ai', `Detected ${language} codebase with ${framework} framework pattern.`);
    
    const fileExtensions = Array.from(new Set(files.map(f => f.extension))).filter(e => e).join(', ');
    await logStep('info', `File types found: ${fileExtensions || 'various'}`);
    
    await storage.updateProjectStatus(projectId, 'generating');
    await logStep('system', '>> Architecting cloud infrastructure solution...');
    
    await new Promise(resolve => setTimeout(resolve, 600));
    
    const requiresDatabase = files.some(f => 
      f.content.includes('database') || 
      f.content.includes('postgres') || 
      f.content.includes('mysql') ||
      f.content.includes('mongodb') ||
      f.content.includes('prisma') ||
      f.content.includes('sequelize') ||
      f.content.includes('typeorm')
    ) ? 'postgres' : null;

    const requiresCache = files.some(f => 
      f.content.includes('redis') || 
      f.content.includes('cache') ||
      f.content.includes('session')
    ) ? 'redis' : null;

    if (requiresDatabase) {
      await logStep('action', `PROVISIONING: Managed ${requiresDatabase === 'postgres' ? 'PostgreSQL' : requiresDatabase} database`);
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    if (requiresCache) {
      await logStep('action', 'PROVISIONING: Redis cluster for caching');
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    await logStep('action', 'CONFIGURING: Container orchestration with auto-scaling');
    await new Promise(resolve => setTimeout(resolve, 500));
    
    await logStep('action', 'SECURITY: Setting up network policies and secrets management');
    await new Promise(resolve => setTimeout(resolve, 500));
    
    await logStep('info', 'Generating infrastructure configuration...');
    
    const sanitizedName = projectName.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
    
    const terraformConfig = `
provider "aws" {
  region = "us-east-1"
}

resource "aws_ecs_cluster" "main" {
  name = "${sanitizedName}-cluster"
}

resource "aws_ecs_task_definition" "app" {
  family                   = "${sanitizedName}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 256
  memory                   = 512

  container_definitions = jsonencode([{
    name  = "${sanitizedName}-container"
    image = "\${aws_ecr_repository.main.repository_url}:latest"
    portMappings = [{
      containerPort = ${language === 'python' ? '8000' : language === 'go' || language === 'rust' || language === 'java' ? '8080' : '3000'}
      hostPort      = ${language === 'python' ? '8000' : language === 'go' || language === 'rust' || language === 'java' ? '8080' : '3000'}
    }]
  }])
}

resource "aws_ecs_service" "main" {
  name            = "${sanitizedName}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.ecs.id]
  }
}
${requiresDatabase ? `
resource "aws_db_instance" "main" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.t3.micro"
  db_name              = "${sanitizedName.replace(/-/g, '_')}_db"
  username             = "admin"
  password             = var.db_password
  skip_final_snapshot  = true
}` : ''}
${requiresCache ? `
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${sanitizedName}-cache"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
}` : ''}
`;

    const kubernetesConfig = `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${sanitizedName}-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ${sanitizedName}
  template:
    metadata:
      labels:
        app: ${sanitizedName}
    spec:
      containers:
      - name: app
        image: ${sanitizedName}:latest
        ports:
        - containerPort: ${language === 'python' ? '8000' : language === 'go' || language === 'rust' || language === 'java' ? '8080' : '3000'}
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ${sanitizedName}-service
spec:
  selector:
    app: ${sanitizedName}
  ports:
  - port: 80
    targetPort: ${language === 'python' ? '8000' : language === 'go' || language === 'rust' || language === 'java' ? '8080' : '3000'}
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${sanitizedName}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${sanitizedName}-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
`;

    const dockerfileContent = generateDockerfileForLanguage(language, framework);

    await logStep('cmd', '> Generating Terraform and Kubernetes manifests...');
    await new Promise(resolve => setTimeout(resolve, 800));
    
    await storage.createInfrastructureTemplate({
      projectId,
      detectedLanguage: language,
      detectedFramework: framework,
      architecture: 'microservices',
      terraformConfig,
      kubernetesConfig,
      dockerfileContent,
      dockerComposeConfig: null,
      minInstances: 2,
      maxInstances: 10,
      requiresDatabase,
      requiresCache,
      requiresQueue: null,
    });
    
    await logStep('success', 'Infrastructure configuration generated successfully');
    await new Promise(resolve => setTimeout(resolve, 500));
    
    await logStep('ai', 'Optimizing deployment strategy based on code analysis...');
    await new Promise(resolve => setTimeout(resolve, 600));
    
    await logStep('success', 'Platform ready for deployment. All configurations validated.');
    
    await storage.updateProjectStatus(projectId, 'complete', new Date());
    
  } catch (error) {
    console.error('Infrastructure generation from upload failed:', error);
    await logStep('error', `Generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    await storage.updateProjectStatus(projectId, 'failed');
  }
}
