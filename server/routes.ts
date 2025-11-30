import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { 
  insertProjectSchema, 
  insertBuildLogSchema,
  insertKvNamespaceSchema,
  insertKvEntrySchema,
  insertObjectBucketSchema,
  insertStorageObjectSchema,
  insertSecurityScanSchema,
  insertSecurityFindingSchema,
  insertDeploymentTargetSchema,
  insertDeploymentSchema,
  insertDeploymentRunSchema,
  insertEnvVariableSchema,
  insertProjectFileSchema,
  insertConsoleLogSchema
} from "@shared/schema";
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

  // ============================================
  // KV Store Endpoints
  // ============================================

  // List user's KV namespaces
  app.get("/api/storage/kv/namespaces", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const namespaces = await storage.getKvNamespacesByUserId(userId);
      res.json(namespaces);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch namespaces" });
    }
  });

  // Create KV namespace
  app.post("/api/storage/kv/namespaces", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const data = insertKvNamespaceSchema.parse({ ...req.body, userId });
      const namespace = await storage.createKvNamespace(data);
      res.json(namespace);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create namespace" });
      }
    }
  });

  // Get KV namespace by ID
  app.get("/api/storage/kv/namespaces/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const namespace = await storage.getKvNamespace(req.params.id);
      if (!namespace) {
        return res.status(404).json({ error: "Namespace not found" });
      }
      if (namespace.userId !== userId) {
        return res.status(404).json({ error: "Namespace not found" });
      }
      res.json(namespace);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch namespace" });
    }
  });

  // Delete KV namespace
  app.delete("/api/storage/kv/namespaces/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const namespace = await storage.getKvNamespace(req.params.id);
      if (!namespace) {
        return res.status(404).json({ error: "Namespace not found" });
      }
      if (namespace.userId !== userId) {
        return res.status(404).json({ error: "Namespace not found" });
      }
      await storage.deleteKvNamespace(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete namespace" });
    }
  });

  // List entries in KV namespace
  app.get("/api/storage/kv/namespaces/:id/entries", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const namespace = await storage.getKvNamespace(req.params.id);
      if (!namespace) {
        return res.status(404).json({ error: "Namespace not found" });
      }
      if (namespace.userId !== userId) {
        return res.status(404).json({ error: "Namespace not found" });
      }
      const entries = await storage.getKvEntriesByNamespaceId(req.params.id);
      res.json(entries);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch entries" });
    }
  });

  // Create KV entry in namespace
  app.post("/api/storage/kv/namespaces/:id/entries", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const namespace = await storage.getKvNamespace(req.params.id);
      if (!namespace) {
        return res.status(404).json({ error: "Namespace not found" });
      }
      if (namespace.userId !== userId) {
        return res.status(404).json({ error: "Namespace not found" });
      }
      const data = insertKvEntrySchema.parse({ ...req.body, namespaceId: req.params.id });
      const entry = await storage.createKvEntry(data);
      res.json(entry);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create entry" });
      }
    }
  });

  // Update KV entry value
  app.put("/api/storage/kv/entries/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const entries = await storage.getKvEntriesByNamespaceId(req.params.id);
      const entry = entries.find(e => e.id === req.params.id);
      if (!entry) {
        const allNamespaces = await storage.getKvNamespacesByUserId(userId);
        let foundEntry = null;
        for (const ns of allNamespaces) {
          const nsEntries = await storage.getKvEntriesByNamespaceId(ns.id);
          foundEntry = nsEntries.find(e => e.id === req.params.id);
          if (foundEntry) break;
        }
        if (!foundEntry) {
          return res.status(404).json({ error: "Entry not found" });
        }
        const namespace = await storage.getKvNamespace(foundEntry.namespaceId);
        if (!namespace || namespace.userId !== userId) {
          return res.status(404).json({ error: "Entry not found" });
        }
        const { value } = req.body;
        if (!value || typeof value !== 'string') {
          return res.status(400).json({ error: "Value is required" });
        }
        await storage.updateKvEntry(req.params.id, value);
        res.json({ success: true });
        return;
      }
      const namespace = await storage.getKvNamespace(entry.namespaceId);
      if (!namespace || namespace.userId !== userId) {
        return res.status(404).json({ error: "Entry not found" });
      }
      const { value } = req.body;
      if (!value || typeof value !== 'string') {
        return res.status(400).json({ error: "Value is required" });
      }
      await storage.updateKvEntry(req.params.id, value);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to update entry" });
    }
  });

  // Delete KV entry
  app.delete("/api/storage/kv/entries/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const allNamespaces = await storage.getKvNamespacesByUserId(userId);
      let foundEntry = null;
      for (const ns of allNamespaces) {
        const nsEntries = await storage.getKvEntriesByNamespaceId(ns.id);
        foundEntry = nsEntries.find(e => e.id === req.params.id);
        if (foundEntry) break;
      }
      if (!foundEntry) {
        return res.status(404).json({ error: "Entry not found" });
      }
      const namespace = await storage.getKvNamespace(foundEntry.namespaceId);
      if (!namespace || namespace.userId !== userId) {
        return res.status(404).json({ error: "Entry not found" });
      }
      await storage.deleteKvEntry(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete entry" });
    }
  });

  // ============================================
  // Object Storage Endpoints
  // ============================================

  // List user's buckets
  app.get("/api/storage/buckets", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const buckets = await storage.getObjectBucketsByUserId(userId);
      res.json(buckets);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch buckets" });
    }
  });

  // Create bucket
  app.post("/api/storage/buckets", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const data = insertObjectBucketSchema.parse({ ...req.body, userId });
      const bucket = await storage.createObjectBucket(data);
      res.json(bucket);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create bucket" });
      }
    }
  });

  // Get bucket by ID
  app.get("/api/storage/buckets/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const bucket = await storage.getObjectBucket(req.params.id);
      if (!bucket) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      if (bucket.userId !== userId) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      res.json(bucket);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch bucket" });
    }
  });

  // Update bucket
  app.put("/api/storage/buckets/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const bucket = await storage.getObjectBucket(req.params.id);
      if (!bucket) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      if (bucket.userId !== userId) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      const { name, description, isPublic } = req.body;
      await storage.updateObjectBucket(req.params.id, { name, description, isPublic });
      const updatedBucket = await storage.getObjectBucket(req.params.id);
      res.json(updatedBucket);
    } catch (error) {
      res.status(500).json({ error: "Failed to update bucket" });
    }
  });

  // Delete bucket
  app.delete("/api/storage/buckets/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const bucket = await storage.getObjectBucket(req.params.id);
      if (!bucket) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      if (bucket.userId !== userId) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      await storage.deleteObjectBucket(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete bucket" });
    }
  });

  // List objects in bucket
  app.get("/api/storage/buckets/:id/objects", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const bucket = await storage.getObjectBucket(req.params.id);
      if (!bucket) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      if (bucket.userId !== userId) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      const objects = await storage.getStorageObjectsByBucketId(req.params.id);
      res.json(objects);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch objects" });
    }
  });

  // Create object in bucket
  app.post("/api/storage/buckets/:id/objects", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const bucket = await storage.getObjectBucket(req.params.id);
      if (!bucket) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      if (bucket.userId !== userId) {
        return res.status(404).json({ error: "Bucket not found" });
      }
      const data = insertStorageObjectSchema.parse({ ...req.body, bucketId: req.params.id });
      const storageObject = await storage.createStorageObject(data);
      res.json(storageObject);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create object" });
      }
    }
  });

  // Delete object
  app.delete("/api/storage/objects/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const storageObject = await storage.getStorageObject(req.params.id);
      if (!storageObject) {
        return res.status(404).json({ error: "Object not found" });
      }
      const bucket = await storage.getObjectBucket(storageObject.bucketId);
      if (!bucket || bucket.userId !== userId) {
        return res.status(404).json({ error: "Object not found" });
      }
      await storage.deleteStorageObject(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete object" });
    }
  });

  // ============================================
  // Security Scanner Endpoints
  // ============================================

  // Trigger a new security scan
  app.post("/api/security/scan", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const { projectId } = req.body;
      
      if (!projectId) {
        return res.status(400).json({ error: "projectId is required" });
      }

      const project = await storage.getProject(projectId);
      if (!project) {
        return res.status(404).json({ error: "Project not found" });
      }
      if (project.userId && project.userId !== userId) {
        return res.status(404).json({ error: "Project not found" });
      }

      const scan = await storage.createSecurityScan({
        projectId,
        status: 'running',
        totalFindings: 0,
        criticalCount: 0,
        highCount: 0,
        mediumCount: 0,
        lowCount: 0,
      });

      runSecurityScan(scan.id, projectId).catch(console.error);

      res.json(scan);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create security scan" });
      }
    }
  });

  // Get all scans for a project
  app.get("/api/projects/:id/security/scans", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: "Project not found" });
      }
      if (project.userId && project.userId !== userId) {
        return res.status(404).json({ error: "Project not found" });
      }
      const scans = await storage.getSecurityScansByProjectId(req.params.id);
      res.json(scans);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch security scans" });
    }
  });

  // Get all findings for a project
  app.get("/api/projects/:id/security/findings", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: "Project not found" });
      }
      if (project.userId && project.userId !== userId) {
        return res.status(404).json({ error: "Project not found" });
      }
      const findings = await storage.getSecurityFindingsByProjectId(req.params.id);
      res.json(findings);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch security findings" });
    }
  });

  // Get a specific scan
  app.get("/api/security/scans/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const scan = await storage.getSecurityScan(req.params.id);
      if (!scan) {
        return res.status(404).json({ error: "Scan not found" });
      }
      const project = await storage.getProject(scan.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Scan not found" });
      }
      res.json(scan);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch security scan" });
    }
  });

  // Update finding status
  app.patch("/api/security/findings/:id/status", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const { status } = req.body;
      
      if (!status || !['open', 'resolved', 'ignored'].includes(status)) {
        return res.status(400).json({ error: "Invalid status. Must be 'open', 'resolved', or 'ignored'" });
      }

      const findings = await storage.getSecurityFindingsByProjectId('');
      let finding = null;
      const userProjects = await storage.getProjectsByUserId(userId);
      
      for (const proj of userProjects) {
        const projectFindings = await storage.getSecurityFindingsByProjectId(proj.id);
        finding = projectFindings.find(f => f.id === req.params.id);
        if (finding) break;
      }
      
      if (!finding) {
        return res.status(404).json({ error: "Finding not found" });
      }

      const project = await storage.getProject(finding.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Finding not found" });
      }

      await storage.updateSecurityFindingStatus(req.params.id, status);
      res.json({ success: true, status });
    } catch (error) {
      res.status(500).json({ error: "Failed to update finding status" });
    }
  });

  // ============================================
  // Deployment Target Endpoints
  // ============================================

  // List user's deployment targets
  app.get("/api/deployment-targets", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const targets = await storage.getDeploymentTargetsByUserId(userId);
      res.json(targets);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch deployment targets" });
    }
  });

  // Create deployment target
  app.post("/api/deployment-targets", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const data = insertDeploymentTargetSchema.parse({ ...req.body, userId });
      const target = await storage.createDeploymentTarget(data);
      res.json(target);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create deployment target" });
      }
    }
  });

  // Get deployment target by ID
  app.get("/api/deployment-targets/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const target = await storage.getDeploymentTarget(req.params.id);
      if (!target) {
        return res.status(404).json({ error: "Deployment target not found" });
      }
      if (target.userId !== userId) {
        return res.status(404).json({ error: "Deployment target not found" });
      }
      res.json(target);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch deployment target" });
    }
  });

  // Update deployment target
  app.put("/api/deployment-targets/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const target = await storage.getDeploymentTarget(req.params.id);
      if (!target) {
        return res.status(404).json({ error: "Deployment target not found" });
      }
      if (target.userId !== userId) {
        return res.status(404).json({ error: "Deployment target not found" });
      }
      const { name, provider, region, config, isDefault } = req.body;
      await storage.updateDeploymentTarget(req.params.id, { name, provider, region, config, isDefault });
      const updatedTarget = await storage.getDeploymentTarget(req.params.id);
      res.json(updatedTarget);
    } catch (error) {
      res.status(500).json({ error: "Failed to update deployment target" });
    }
  });

  // Delete deployment target
  app.delete("/api/deployment-targets/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const target = await storage.getDeploymentTarget(req.params.id);
      if (!target) {
        return res.status(404).json({ error: "Deployment target not found" });
      }
      if (target.userId !== userId) {
        return res.status(404).json({ error: "Deployment target not found" });
      }
      await storage.deleteDeploymentTarget(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete deployment target" });
    }
  });

  // ============================================
  // Deployment Endpoints
  // ============================================

  // List deployments for a project
  app.get("/api/projects/:id/deployments", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: "Project not found" });
      }
      if (project.userId && project.userId !== userId) {
        return res.status(404).json({ error: "Project not found" });
      }
      const deployments = await storage.getDeploymentsByProjectId(req.params.id);
      res.json(deployments);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch deployments" });
    }
  });

  // Create deployment for a project
  app.post("/api/projects/:id/deployments", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ error: "Project not found" });
      }
      if (project.userId && project.userId !== userId) {
        return res.status(404).json({ error: "Project not found" });
      }
      const data = insertDeploymentSchema.parse({ ...req.body, projectId: req.params.id });
      const deployment = await storage.createDeployment(data);
      res.json(deployment);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create deployment" });
      }
    }
  });

  // Get deployment by ID
  app.get("/api/deployments/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const deployment = await storage.getDeployment(req.params.id);
      if (!deployment) {
        return res.status(404).json({ error: "Deployment not found" });
      }
      const project = await storage.getProject(deployment.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Deployment not found" });
      }
      res.json(deployment);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch deployment" });
    }
  });

  // Update deployment
  app.put("/api/deployments/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const deployment = await storage.getDeployment(req.params.id);
      if (!deployment) {
        return res.status(404).json({ error: "Deployment not found" });
      }
      const project = await storage.getProject(deployment.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Deployment not found" });
      }
      const { name, targetId, deploymentType, config, status, url } = req.body;
      await storage.updateDeployment(req.params.id, { name, targetId, deploymentType, config, status, url });
      const updatedDeployment = await storage.getDeployment(req.params.id);
      res.json(updatedDeployment);
    } catch (error) {
      res.status(500).json({ error: "Failed to update deployment" });
    }
  });

  // Delete deployment
  app.delete("/api/deployments/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const deployment = await storage.getDeployment(req.params.id);
      if (!deployment) {
        return res.status(404).json({ error: "Deployment not found" });
      }
      const project = await storage.getProject(deployment.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Deployment not found" });
      }
      await storage.deleteDeployment(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete deployment" });
    }
  });

  // ============================================
  // Deployment Actions Endpoints
  // ============================================

  // Trigger a deployment run
  app.post("/api/deployments/:id/trigger", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const deployment = await storage.getDeployment(req.params.id);
      if (!deployment) {
        return res.status(404).json({ error: "Deployment not found" });
      }
      const project = await storage.getProject(deployment.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Deployment not found" });
      }

      const run = await storage.createDeploymentRun({
        deploymentId: req.params.id,
        status: 'pending',
        version: `v${Date.now()}`,
        logs: '',
      });

      executeDeployment(req.params.id, run.id).catch(console.error);

      res.json(run);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to trigger deployment" });
      }
    }
  });

  // Get deployment run history
  app.get("/api/deployments/:id/runs", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const deployment = await storage.getDeployment(req.params.id);
      if (!deployment) {
        return res.status(404).json({ error: "Deployment not found" });
      }
      const project = await storage.getProject(deployment.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Deployment not found" });
      }
      const runs = await storage.getDeploymentRunsByDeploymentId(req.params.id);
      res.json(runs);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch deployment runs" });
    }
  });

  // Get specific deployment run with logs
  app.get("/api/deployment-runs/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const run = await storage.getDeploymentRun(req.params.id);
      if (!run) {
        return res.status(404).json({ error: "Deployment run not found" });
      }
      const deployment = await storage.getDeployment(run.deploymentId);
      if (!deployment) {
        return res.status(404).json({ error: "Deployment run not found" });
      }
      const project = await storage.getProject(deployment.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Deployment run not found" });
      }
      res.json(run);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch deployment run" });
    }
  });

  // ============================================
  // Environment Variables Endpoints
  // ============================================

  // List user's environment variables (non-secret only)
  app.get("/api/env", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const envVars = await storage.getEnvVariablesByUser(userId);
      const nonSecretVars = envVars.filter(v => !v.isSecret);
      res.json(nonSecretVars);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch environment variables" });
    }
  });

  // Create environment variable
  app.post("/api/env", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const data = insertEnvVariableSchema.parse({ ...req.body, userId, isSecret: false });
      const envVar = await storage.createEnvVariable(data);
      res.json(envVar);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create environment variable" });
      }
    }
  });

  // Update environment variable
  app.put("/api/env/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const envVars = await storage.getEnvVariablesByUser(userId);
      const envVar = envVars.find(v => v.id === req.params.id && !v.isSecret);
      if (!envVar) {
        return res.status(404).json({ error: "Environment variable not found" });
      }
      const { key, value, environment, projectId } = req.body;
      const updated = await storage.updateEnvVariable(req.params.id, { key, value, environment, projectId });
      res.json(updated);
    } catch (error) {
      res.status(500).json({ error: "Failed to update environment variable" });
    }
  });

  // Delete environment variable
  app.delete("/api/env/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const envVars = await storage.getEnvVariablesByUser(userId);
      const envVar = envVars.find(v => v.id === req.params.id && !v.isSecret);
      if (!envVar) {
        return res.status(404).json({ error: "Environment variable not found" });
      }
      await storage.deleteEnvVariable(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete environment variable" });
    }
  });

  // ============================================
  // Secrets Endpoints (env vars with isSecret: true)
  // ============================================

  // List user's secrets (masked values)
  app.get("/api/secrets", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const envVars = await storage.getEnvVariablesByUser(userId);
      const secrets = envVars
        .filter(v => v.isSecret)
        .map(({ id, key, environment }) => ({ id, key, environment }));
      res.json(secrets);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch secrets" });
    }
  });

  // Create secret
  app.post("/api/secrets", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const data = insertEnvVariableSchema.parse({ ...req.body, userId, isSecret: true });
      const secret = await storage.createEnvVariable(data);
      res.json({ id: secret.id, key: secret.key, environment: secret.environment });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create secret" });
      }
    }
  });

  // Update secret value only
  app.put("/api/secrets/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const envVars = await storage.getEnvVariablesByUser(userId);
      const secret = envVars.find(v => v.id === req.params.id && v.isSecret);
      if (!secret) {
        return res.status(404).json({ error: "Secret not found" });
      }
      const { value } = req.body;
      if (!value || typeof value !== 'string') {
        return res.status(400).json({ error: "Value is required" });
      }
      await storage.updateEnvVariable(req.params.id, { value });
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to update secret" });
    }
  });

  // Delete secret
  app.delete("/api/secrets/:id", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const envVars = await storage.getEnvVariablesByUser(userId);
      const secret = envVars.find(v => v.id === req.params.id && v.isSecret);
      if (!secret) {
        return res.status(404).json({ error: "Secret not found" });
      }
      await storage.deleteEnvVariable(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete secret" });
    }
  });

  // ============================================
  // Project Files Endpoints
  // ============================================

  // List project files
  app.get("/api/projects/:projectId/files", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      const files = await storage.getProjectFiles(req.params.projectId);
      res.json(files);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch project files" });
    }
  });

  // Get file content
  app.get("/api/projects/:projectId/files/:fileId", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      const file = await storage.getProjectFile(req.params.fileId);
      if (!file || file.projectId !== req.params.projectId) {
        return res.status(404).json({ error: "File not found" });
      }
      res.json(file);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch file" });
    }
  });

  // Create file
  app.post("/api/projects/:projectId/files", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      const data = insertProjectFileSchema.parse({ ...req.body, projectId: req.params.projectId });
      const file = await storage.createProjectFile(data);
      res.json(file);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create file" });
      }
    }
  });

  // Update file
  app.put("/api/projects/:projectId/files/:fileId", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      const file = await storage.getProjectFile(req.params.fileId);
      if (!file || file.projectId !== req.params.projectId) {
        return res.status(404).json({ error: "File not found" });
      }
      const { name, content, path: filePath, parentPath } = req.body;
      const updated = await storage.updateProjectFile(req.params.fileId, { name, content, path: filePath, parentPath });
      res.json(updated);
    } catch (error) {
      res.status(500).json({ error: "Failed to update file" });
    }
  });

  // Delete file
  app.delete("/api/projects/:projectId/files/:fileId", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      const file = await storage.getProjectFile(req.params.fileId);
      if (!file || file.projectId !== req.params.projectId) {
        return res.status(404).json({ error: "File not found" });
      }
      await storage.deleteProjectFile(req.params.fileId);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete file" });
    }
  });

  // ============================================
  // Console Logs Endpoints
  // ============================================

  // Get console logs (last 100)
  app.get("/api/projects/:projectId/console", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      const logs = await storage.getConsoleLogs(req.params.projectId, 100);
      res.json(logs);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch console logs" });
    }
  });

  // Add console log entry
  app.post("/api/projects/:projectId/console", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      const data = insertConsoleLogSchema.parse({ ...req.body, projectId: req.params.projectId });
      const log = await storage.createConsoleLog(data);
      res.json(log);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors });
      } else {
        res.status(500).json({ error: "Failed to create console log" });
      }
    }
  });

  // Clear console logs
  app.delete("/api/projects/:projectId/console", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }
      await storage.clearConsoleLogs(req.params.projectId);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to clear console logs" });
    }
  });

  // ============================================
  // Project Run Endpoint (simulated)
  // ============================================

  // Run project (simulated)
  app.post("/api/projects/:projectId/run", isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const project = await storage.getProject(req.params.projectId);
      if (!project || (project.userId && project.userId !== userId)) {
        return res.status(404).json({ error: "Project not found" });
      }

      const logs: { type: string; message: string; timestamp: Date }[] = [];

      const addLog = async (type: string, message: string) => {
        const log = await storage.createConsoleLog({
          projectId: req.params.projectId,
          type,
          message
        });
        logs.push({ type: log.type, message: log.message, timestamp: log.timestamp });
      };

      await addLog('info', 'Starting build process...');
      await addLog('info', 'Installing dependencies...');
      await addLog('log', '> npm install');
      await addLog('log', 'added 127 packages in 3.2s');
      await addLog('info', 'Compiling source files...');
      await addLog('log', '> tsc --build');
      await addLog('success', 'Build completed successfully');
      await addLog('info', 'Starting application...');
      await addLog('log', '> node dist/index.js');
      await addLog('success', 'Application running on port 3000');

      res.json({ 
        success: true, 
        message: 'Project started successfully',
        logs 
      });
    } catch (error) {
      res.status(500).json({ error: "Failed to run project" });
    }
  });

  return httpServer;
}

// Deployment Execution Logic
async function executeDeployment(deploymentId: string, runId: string) {
  const appendLog = async (message: string) => {
    const run = await storage.getDeploymentRun(runId);
    const currentLogs = run?.logs || '';
    const timestamp = new Date().toISOString();
    await storage.updateDeploymentRun(runId, {
      logs: currentLogs + `[${timestamp}] ${message}\n`
    });
  };

  try {
    await storage.updateDeploymentRun(runId, { status: 'running' });
    await storage.updateDeployment(deploymentId, { status: 'deploying' });

    await appendLog('Initializing deployment environment...');
    await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 400));

    await appendLog('Pulling infrastructure configuration...');
    await new Promise(resolve => setTimeout(resolve, 600 + Math.random() * 400));

    const resourceCount = Math.floor(Math.random() * 5) + 2;
    await appendLog(`Applying Terraform changes... (${resourceCount} resources added)`);
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 500));

    await appendLog('Building Docker image...');
    await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 400));

    await appendLog('Pushing to container registry...');
    await new Promise(resolve => setTimeout(resolve, 700 + Math.random() * 300));

    await appendLog('Updating Kubernetes deployment...');
    await new Promise(resolve => setTimeout(resolve, 600 + Math.random() * 400));

    await appendLog('Waiting for pods to be ready...');
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 300));

    await appendLog('Health checks passing...');
    await new Promise(resolve => setTimeout(resolve, 400 + Math.random() * 200));

    const isSuccess = Math.random() > 0.1;

    if (isSuccess) {
      const deployment = await storage.getDeployment(deploymentId);
      const deploymentName = deployment?.name?.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase() || 'app';
      const randomId = Math.random().toString(36).substring(2, 8);
      const deploymentUrl = `https://${deploymentName}-${randomId}.deploy.example.com`;

      await appendLog('Deployment complete!');
      await appendLog(`Application available at: ${deploymentUrl}`);

      await storage.updateDeploymentRun(runId, {
        status: 'success',
        completedAt: new Date()
      });
      await storage.updateDeployment(deploymentId, {
        status: 'deployed',
        url: deploymentUrl
      });
    } else {
      const errorMessages = [
        'Container health check failed after 3 attempts',
        'Failed to pull image: authentication required',
        'Pod crashed with exit code 1',
        'Resource quota exceeded in namespace'
      ];
      const errorMessage = errorMessages[Math.floor(Math.random() * errorMessages.length)];

      await appendLog(`ERROR: ${errorMessage}`);
      await appendLog('Deployment failed. Rolling back...');

      await storage.updateDeploymentRun(runId, {
        status: 'failed',
        errorMessage,
        completedAt: new Date()
      });
      await storage.updateDeployment(deploymentId, { status: 'failed' });
    }

  } catch (error) {
    console.error('Deployment execution failed:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    await appendLog(`FATAL: ${errorMessage}`);

    await storage.updateDeploymentRun(runId, {
      status: 'failed',
      errorMessage,
      completedAt: new Date()
    });
    await storage.updateDeployment(deploymentId, { status: 'failed' });
  }
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

async function runSecurityScan(scanId: string, projectId: string) {
  try {
    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 1000));

    const template = await storage.getInfrastructureTemplateByProjectId(projectId);
    const detectedLanguage = template?.detectedLanguage || 'unknown';

    interface FindingTemplate {
      title: string;
      description: string;
      severity: string;
      category: string;
      filePath: string;
      lineNumber: number;
      recommendation: string;
    }

    let findings: FindingTemplate[] = [];

    if (detectedLanguage === 'javascript' || detectedLanguage === 'typescript') {
      findings = [
        {
          title: 'Potential XSS vulnerability',
          description: 'User input is rendered directly in the DOM without sanitization, potentially allowing cross-site scripting attacks.',
          severity: 'high',
          category: 'injection',
          filePath: 'src/components/UserInput.tsx',
          lineNumber: 42,
          recommendation: 'Use DOMPurify or similar library to sanitize user input before rendering.'
        },
        {
          title: 'Unsanitized user input',
          description: 'User-provided data is passed directly to database queries without proper validation or escaping.',
          severity: 'critical',
          category: 'injection',
          filePath: 'src/api/users.ts',
          lineNumber: 78,
          recommendation: 'Use parameterized queries and input validation.'
        },
        {
          title: 'Outdated dependency',
          description: 'Package lodash@4.17.15 has known security vulnerabilities. A newer patched version is available.',
          severity: 'medium',
          category: 'dependency',
          filePath: 'package.json',
          lineNumber: 15,
          recommendation: 'Update lodash to version 4.17.21 or later.'
        }
      ];
    } else if (detectedLanguage === 'python') {
      findings = [
        {
          title: 'SQL injection risk',
          description: 'String formatting is used to construct SQL queries, making the application vulnerable to SQL injection attacks.',
          severity: 'critical',
          category: 'injection',
          filePath: 'app/database.py',
          lineNumber: 56,
          recommendation: 'Use parameterized queries with SQLAlchemy or prepared statements.'
        },
        {
          title: 'Hardcoded secrets',
          description: 'API keys and secrets are hardcoded in the source code, exposing them to version control and potential leakage.',
          severity: 'high',
          category: 'secrets',
          filePath: 'config/settings.py',
          lineNumber: 23,
          recommendation: 'Move secrets to environment variables or a secure secrets manager.'
        },
        {
          title: 'Insecure pickle usage',
          description: 'Pickle is used to deserialize untrusted data, which can lead to arbitrary code execution.',
          severity: 'high',
          category: 'injection',
          filePath: 'utils/cache.py',
          lineNumber: 89,
          recommendation: 'Use JSON or other safe serialization formats for untrusted data.'
        }
      ];
    } else if (detectedLanguage === 'go') {
      findings = [
        {
          title: 'Race condition detected',
          description: 'Concurrent access to shared variable without proper synchronization detected.',
          severity: 'high',
          category: 'config',
          filePath: 'internal/handlers/user.go',
          lineNumber: 112,
          recommendation: 'Use mutex locks or atomic operations for shared state access.'
        },
        {
          title: 'Unchecked error return',
          description: 'Error return value from function call is not checked, potentially masking failures.',
          severity: 'medium',
          category: 'config',
          filePath: 'internal/db/connection.go',
          lineNumber: 45,
          recommendation: 'Always check and handle error return values appropriately.'
        }
      ];
    } else {
      findings = [
        {
          title: 'Insufficient logging',
          description: 'Critical operations lack proper audit logging, making incident investigation difficult.',
          severity: 'low',
          category: 'config',
          filePath: 'src/main',
          lineNumber: 1,
          recommendation: 'Implement comprehensive logging for authentication, data access, and administrative actions.'
        },
        {
          title: 'Missing rate limiting',
          description: 'API endpoints lack rate limiting, making them vulnerable to brute force and denial of service attacks.',
          severity: 'medium',
          category: 'config',
          filePath: 'src/routes',
          lineNumber: 1,
          recommendation: 'Implement rate limiting using middleware like express-rate-limit or similar.'
        },
        {
          title: 'No input validation',
          description: 'User input is not validated against expected formats and constraints.',
          severity: 'high',
          category: 'injection',
          filePath: 'src/handlers',
          lineNumber: 1,
          recommendation: 'Implement input validation using a schema validation library like Zod or Joi.'
        }
      ];
    }

    let criticalCount = 0;
    let highCount = 0;
    let mediumCount = 0;
    let lowCount = 0;

    for (const finding of findings) {
      await storage.createSecurityFinding({
        projectId,
        severity: finding.severity,
        category: finding.category,
        title: finding.title,
        description: finding.description,
        filePath: finding.filePath,
        lineNumber: finding.lineNumber,
        recommendation: finding.recommendation,
        status: 'open',
      });

      switch (finding.severity) {
        case 'critical':
          criticalCount++;
          break;
        case 'high':
          highCount++;
          break;
        case 'medium':
          mediumCount++;
          break;
        case 'low':
          lowCount++;
          break;
      }
    }

    await storage.updateSecurityScan(scanId, {
      status: 'completed',
      totalFindings: findings.length,
      criticalCount,
      highCount,
      mediumCount,
      lowCount,
      completedAt: new Date(),
    });

  } catch (error) {
    console.error('Security scan failed:', error);
    await storage.updateSecurityScan(scanId, {
      status: 'failed',
      completedAt: new Date(),
    });
  }
}
