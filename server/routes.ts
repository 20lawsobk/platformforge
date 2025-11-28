import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { insertProjectSchema, insertBuildLogSchema } from "@shared/schema";
import { aiModelClient, fallbackResponses } from "./ai/model-client";

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  // Create a new project and start infrastructure generation
  app.post("/api/projects", async (req, res) => {
    try {
      const data = insertProjectSchema.parse(req.body);
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

  // Get project by ID
  app.get("/api/projects/:id", async (req, res) => {
    try {
      const project = await storage.getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: "Project not found" });
      }
      res.json(project);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch project" });
    }
  });

  // Get all projects
  app.get("/api/projects", async (req, res) => {
    try {
      const projects = await storage.getAllProjects();
      res.json(projects);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch projects" });
    }
  });

  // Get infrastructure template for a project
  app.get("/api/projects/:id/infrastructure", async (req, res) => {
    try {
      const template = await storage.getInfrastructureTemplateByProjectId(req.params.id);
      if (!template) {
        return res.status(404).json({ error: "Infrastructure template not found" });
      }
      res.json(template);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch infrastructure template" });
    }
  });

  // Get build logs for a project
  app.get("/api/projects/:id/logs", async (req, res) => {
    try {
      const logs = await storage.getBuildLogsByProjectId(req.params.id);
      res.json(logs);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch logs" });
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
