import { 
  type Project, 
  type InsertProject,
  type InfrastructureTemplate,
  type InsertInfrastructureTemplate,
  type BuildLog,
  type InsertBuildLog,
  type User,
  type UpsertUser,
  type KvNamespace,
  type InsertKvNamespace,
  type KvEntry,
  type InsertKvEntry,
  type ObjectBucket,
  type InsertObjectBucket,
  type StorageObject,
  type InsertStorageObject,
  type SecurityFinding,
  type InsertSecurityFinding,
  type SecurityScan,
  type InsertSecurityScan,
  type DeploymentTarget,
  type InsertDeploymentTarget,
  type Deployment,
  type InsertDeployment,
  type DeploymentRun,
  type InsertDeploymentRun,
  type EnvVariable,
  type InsertEnvVariable,
  type ProjectFile,
  type InsertProjectFile,
  type ConsoleLog,
  type InsertConsoleLog,
  projects,
  infrastructureTemplates,
  buildLogs,
  users,
  kvNamespaces,
  kvEntries,
  objectBuckets,
  storageObjects,
  securityFindings,
  securityScans,
  deploymentTargets,
  deployments,
  deploymentRuns,
  envVariables,
  projectFiles,
  consoleLogs
} from "@shared/schema";
import { db } from "../db";
import { eq, desc, and } from "drizzle-orm";

export interface IStorage {
  // User operations (IMPORTANT: mandatory for Replit Auth)
  getUser(id: string): Promise<User | undefined>;
  upsertUser(user: UpsertUser): Promise<User>;
  updateUserOnboarding(id: string, completed: boolean): Promise<void>;
  
  // Projects
  createProject(project: InsertProject): Promise<Project>;
  getProject(id: string): Promise<Project | undefined>;
  updateProjectStatus(id: string, status: string, completedAt?: Date): Promise<void>;
  getAllProjects(): Promise<Project[]>;
  getProjectsByUserId(userId: string): Promise<Project[]>;
  
  // Infrastructure Templates
  createInfrastructureTemplate(template: InsertInfrastructureTemplate): Promise<InfrastructureTemplate>;
  getInfrastructureTemplateByProjectId(projectId: string): Promise<InfrastructureTemplate | undefined>;
  
  // Build Logs
  createBuildLog(log: InsertBuildLog): Promise<BuildLog>;
  getBuildLogsByProjectId(projectId: string): Promise<BuildLog[]>;

  // KV Store Operations
  createKvNamespace(namespace: InsertKvNamespace): Promise<KvNamespace>;
  getKvNamespace(id: string): Promise<KvNamespace | undefined>;
  getKvNamespacesByUserId(userId: string): Promise<KvNamespace[]>;
  deleteKvNamespace(id: string): Promise<void>;
  createKvEntry(entry: InsertKvEntry): Promise<KvEntry>;
  getKvEntry(namespaceId: string, key: string): Promise<KvEntry | undefined>;
  getKvEntriesByNamespaceId(namespaceId: string): Promise<KvEntry[]>;
  updateKvEntry(id: string, value: string): Promise<void>;
  deleteKvEntry(id: string): Promise<void>;

  // Object Storage Operations
  createObjectBucket(bucket: InsertObjectBucket): Promise<ObjectBucket>;
  getObjectBucket(id: string): Promise<ObjectBucket | undefined>;
  getObjectBucketsByUserId(userId: string): Promise<ObjectBucket[]>;
  updateObjectBucket(id: string, updates: Partial<InsertObjectBucket>): Promise<void>;
  deleteObjectBucket(id: string): Promise<void>;
  createStorageObject(obj: InsertStorageObject): Promise<StorageObject>;
  getStorageObject(id: string): Promise<StorageObject | undefined>;
  getStorageObjectsByBucketId(bucketId: string): Promise<StorageObject[]>;
  deleteStorageObject(id: string): Promise<void>;

  // Security Scanner Operations
  createSecurityScan(scan: InsertSecurityScan): Promise<SecurityScan>;
  getSecurityScan(id: string): Promise<SecurityScan | undefined>;
  getSecurityScansByProjectId(projectId: string): Promise<SecurityScan[]>;
  updateSecurityScan(id: string, updates: Partial<SecurityScan>): Promise<void>;
  createSecurityFinding(finding: InsertSecurityFinding): Promise<SecurityFinding>;
  getSecurityFindingsByScanId(scanId: string): Promise<SecurityFinding[]>;
  getSecurityFindingsByProjectId(projectId: string): Promise<SecurityFinding[]>;
  updateSecurityFindingStatus(id: string, status: string): Promise<void>;

  // Deployment Operations
  createDeploymentTarget(target: InsertDeploymentTarget): Promise<DeploymentTarget>;
  getDeploymentTarget(id: string): Promise<DeploymentTarget | undefined>;
  getDeploymentTargetsByUserId(userId: string): Promise<DeploymentTarget[]>;
  updateDeploymentTarget(id: string, updates: Partial<InsertDeploymentTarget>): Promise<void>;
  deleteDeploymentTarget(id: string): Promise<void>;
  createDeployment(deployment: InsertDeployment): Promise<Deployment>;
  getDeployment(id: string): Promise<Deployment | undefined>;
  getDeploymentsByProjectId(projectId: string): Promise<Deployment[]>;
  updateDeployment(id: string, updates: Partial<Deployment>): Promise<void>;
  deleteDeployment(id: string): Promise<void>;
  createDeploymentRun(run: InsertDeploymentRun): Promise<DeploymentRun>;
  getDeploymentRun(id: string): Promise<DeploymentRun | undefined>;
  getDeploymentRunsByDeploymentId(deploymentId: string): Promise<DeploymentRun[]>;
  updateDeploymentRun(id: string, updates: Partial<DeploymentRun>): Promise<void>;

  // Environment Variables Operations
  getEnvVariablesByUser(userId: string): Promise<EnvVariable[]>;
  getEnvVariablesByProject(projectId: string): Promise<EnvVariable[]>;
  createEnvVariable(data: InsertEnvVariable): Promise<EnvVariable>;
  updateEnvVariable(id: string, data: Partial<InsertEnvVariable>): Promise<EnvVariable | undefined>;
  deleteEnvVariable(id: string): Promise<void>;

  // Project Files Operations
  getProjectFiles(projectId: string): Promise<ProjectFile[]>;
  getProjectFile(id: string): Promise<ProjectFile | undefined>;
  getProjectFileByPath(projectId: string, path: string): Promise<ProjectFile | undefined>;
  createProjectFile(data: InsertProjectFile): Promise<ProjectFile>;
  updateProjectFile(id: string, data: Partial<InsertProjectFile>): Promise<ProjectFile | undefined>;
  deleteProjectFile(id: string): Promise<void>;

  // Console Logs Operations
  getConsoleLogs(projectId: string, limit?: number): Promise<ConsoleLog[]>;
  createConsoleLog(data: InsertConsoleLog): Promise<ConsoleLog>;
  clearConsoleLogs(projectId: string): Promise<void>;
}

export class DatabaseStorage implements IStorage {
  // User operations (IMPORTANT: mandatory for Replit Auth)
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async upsertUser(userData: UpsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(userData)
      .onConflictDoUpdate({
        target: users.id,
        set: {
          ...userData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return user;
  }

  async updateUserOnboarding(id: string, completed: boolean): Promise<void> {
    await db.update(users)
      .set({ onboardingCompleted: completed, updatedAt: new Date() })
      .where(eq(users.id, id));
  }

  // Projects
  async createProject(insertProject: InsertProject): Promise<Project> {
    const [project] = await db.insert(projects).values(insertProject).returning();
    return project;
  }

  async getProject(id: string): Promise<Project | undefined> {
    const [project] = await db.select().from(projects).where(eq(projects.id, id));
    return project;
  }

  async updateProjectStatus(id: string, status: string, completedAt?: Date): Promise<void> {
    await db.update(projects)
      .set({ status, completedAt })
      .where(eq(projects.id, id));
  }

  async getAllProjects(): Promise<Project[]> {
    return await db.select().from(projects).orderBy(desc(projects.createdAt));
  }

  async getProjectsByUserId(userId: string): Promise<Project[]> {
    return await db.select()
      .from(projects)
      .where(eq(projects.userId, userId))
      .orderBy(desc(projects.createdAt));
  }

  // Infrastructure Templates
  async createInfrastructureTemplate(insertTemplate: InsertInfrastructureTemplate): Promise<InfrastructureTemplate> {
    const [template] = await db.insert(infrastructureTemplates).values(insertTemplate).returning();
    return template;
  }

  async getInfrastructureTemplateByProjectId(projectId: string): Promise<InfrastructureTemplate | undefined> {
    const [template] = await db.select()
      .from(infrastructureTemplates)
      .where(eq(infrastructureTemplates.projectId, projectId));
    return template;
  }

  // Build Logs
  async createBuildLog(insertLog: InsertBuildLog): Promise<BuildLog> {
    const [log] = await db.insert(buildLogs).values(insertLog).returning();
    return log;
  }

  async getBuildLogsByProjectId(projectId: string): Promise<BuildLog[]> {
    return await db.select()
      .from(buildLogs)
      .where(eq(buildLogs.projectId, projectId))
      .orderBy(buildLogs.timestamp);
  }

  // KV Store Operations
  async createKvNamespace(insertNamespace: InsertKvNamespace): Promise<KvNamespace> {
    const [namespace] = await db.insert(kvNamespaces).values(insertNamespace).returning();
    return namespace;
  }

  async getKvNamespace(id: string): Promise<KvNamespace | undefined> {
    const [namespace] = await db.select().from(kvNamespaces).where(eq(kvNamespaces.id, id));
    return namespace;
  }

  async getKvNamespacesByUserId(userId: string): Promise<KvNamespace[]> {
    return await db.select()
      .from(kvNamespaces)
      .where(eq(kvNamespaces.userId, userId))
      .orderBy(desc(kvNamespaces.createdAt));
  }

  async deleteKvNamespace(id: string): Promise<void> {
    await db.delete(kvEntries).where(eq(kvEntries.namespaceId, id));
    await db.delete(kvNamespaces).where(eq(kvNamespaces.id, id));
  }

  async createKvEntry(insertEntry: InsertKvEntry): Promise<KvEntry> {
    const [entry] = await db.insert(kvEntries).values(insertEntry).returning();
    return entry;
  }

  async getKvEntry(namespaceId: string, key: string): Promise<KvEntry | undefined> {
    const [entry] = await db.select()
      .from(kvEntries)
      .where(and(eq(kvEntries.namespaceId, namespaceId), eq(kvEntries.key, key)));
    return entry;
  }

  async getKvEntriesByNamespaceId(namespaceId: string): Promise<KvEntry[]> {
    return await db.select()
      .from(kvEntries)
      .where(eq(kvEntries.namespaceId, namespaceId))
      .orderBy(desc(kvEntries.createdAt));
  }

  async updateKvEntry(id: string, value: string): Promise<void> {
    await db.update(kvEntries)
      .set({ value, updatedAt: new Date() })
      .where(eq(kvEntries.id, id));
  }

  async deleteKvEntry(id: string): Promise<void> {
    await db.delete(kvEntries).where(eq(kvEntries.id, id));
  }

  // Object Storage Operations
  async createObjectBucket(insertBucket: InsertObjectBucket): Promise<ObjectBucket> {
    const [bucket] = await db.insert(objectBuckets).values(insertBucket).returning();
    return bucket;
  }

  async getObjectBucket(id: string): Promise<ObjectBucket | undefined> {
    const [bucket] = await db.select().from(objectBuckets).where(eq(objectBuckets.id, id));
    return bucket;
  }

  async getObjectBucketsByUserId(userId: string): Promise<ObjectBucket[]> {
    return await db.select()
      .from(objectBuckets)
      .where(eq(objectBuckets.userId, userId))
      .orderBy(desc(objectBuckets.createdAt));
  }

  async updateObjectBucket(id: string, updates: Partial<InsertObjectBucket>): Promise<void> {
    await db.update(objectBuckets)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(objectBuckets.id, id));
  }

  async deleteObjectBucket(id: string): Promise<void> {
    await db.delete(storageObjects).where(eq(storageObjects.bucketId, id));
    await db.delete(objectBuckets).where(eq(objectBuckets.id, id));
  }

  async createStorageObject(insertObj: InsertStorageObject): Promise<StorageObject> {
    const [obj] = await db.insert(storageObjects).values(insertObj).returning();
    const bucket = await this.getObjectBucket(insertObj.bucketId);
    if (bucket) {
      await db.update(objectBuckets)
        .set({ 
          objectCount: (bucket.objectCount || 0) + 1,
          totalSize: (bucket.totalSize || 0) + (insertObj.size || 0),
          updatedAt: new Date()
        })
        .where(eq(objectBuckets.id, insertObj.bucketId));
    }
    return obj;
  }

  async getStorageObject(id: string): Promise<StorageObject | undefined> {
    const [obj] = await db.select().from(storageObjects).where(eq(storageObjects.id, id));
    return obj;
  }

  async getStorageObjectsByBucketId(bucketId: string): Promise<StorageObject[]> {
    return await db.select()
      .from(storageObjects)
      .where(eq(storageObjects.bucketId, bucketId))
      .orderBy(desc(storageObjects.createdAt));
  }

  async deleteStorageObject(id: string): Promise<void> {
    const obj = await this.getStorageObject(id);
    if (obj) {
      const bucket = await this.getObjectBucket(obj.bucketId);
      if (bucket) {
        await db.update(objectBuckets)
          .set({ 
            objectCount: Math.max((bucket.objectCount || 0) - 1, 0),
            totalSize: Math.max((bucket.totalSize || 0) - (obj.size || 0), 0),
            updatedAt: new Date()
          })
          .where(eq(objectBuckets.id, obj.bucketId));
      }
    }
    await db.delete(storageObjects).where(eq(storageObjects.id, id));
  }

  // Security Scanner Operations
  async createSecurityScan(insertScan: InsertSecurityScan): Promise<SecurityScan> {
    const [scan] = await db.insert(securityScans).values(insertScan).returning();
    return scan;
  }

  async getSecurityScan(id: string): Promise<SecurityScan | undefined> {
    const [scan] = await db.select().from(securityScans).where(eq(securityScans.id, id));
    return scan;
  }

  async getSecurityScansByProjectId(projectId: string): Promise<SecurityScan[]> {
    return await db.select()
      .from(securityScans)
      .where(eq(securityScans.projectId, projectId))
      .orderBy(desc(securityScans.startedAt));
  }

  async updateSecurityScan(id: string, updates: Partial<SecurityScan>): Promise<void> {
    await db.update(securityScans)
      .set(updates)
      .where(eq(securityScans.id, id));
  }

  async createSecurityFinding(insertFinding: InsertSecurityFinding): Promise<SecurityFinding> {
    const [finding] = await db.insert(securityFindings).values(insertFinding).returning();
    return finding;
  }

  async getSecurityFindingsByScanId(scanId: string): Promise<SecurityFinding[]> {
    const scan = await this.getSecurityScan(scanId);
    if (!scan) return [];
    return await db.select()
      .from(securityFindings)
      .where(eq(securityFindings.projectId, scan.projectId))
      .orderBy(desc(securityFindings.createdAt));
  }

  async getSecurityFindingsByProjectId(projectId: string): Promise<SecurityFinding[]> {
    return await db.select()
      .from(securityFindings)
      .where(eq(securityFindings.projectId, projectId))
      .orderBy(desc(securityFindings.createdAt));
  }

  async updateSecurityFindingStatus(id: string, status: string): Promise<void> {
    await db.update(securityFindings)
      .set({ status })
      .where(eq(securityFindings.id, id));
  }

  // Deployment Operations
  async createDeploymentTarget(insertTarget: InsertDeploymentTarget): Promise<DeploymentTarget> {
    const [target] = await db.insert(deploymentTargets).values(insertTarget).returning();
    return target;
  }

  async getDeploymentTarget(id: string): Promise<DeploymentTarget | undefined> {
    const [target] = await db.select().from(deploymentTargets).where(eq(deploymentTargets.id, id));
    return target;
  }

  async getDeploymentTargetsByUserId(userId: string): Promise<DeploymentTarget[]> {
    return await db.select()
      .from(deploymentTargets)
      .where(eq(deploymentTargets.userId, userId))
      .orderBy(desc(deploymentTargets.createdAt));
  }

  async updateDeploymentTarget(id: string, updates: Partial<InsertDeploymentTarget>): Promise<void> {
    await db.update(deploymentTargets)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(deploymentTargets.id, id));
  }

  async deleteDeploymentTarget(id: string): Promise<void> {
    await db.delete(deploymentTargets).where(eq(deploymentTargets.id, id));
  }

  async createDeployment(insertDeployment: InsertDeployment): Promise<Deployment> {
    const [deployment] = await db.insert(deployments).values(insertDeployment).returning();
    return deployment;
  }

  async getDeployment(id: string): Promise<Deployment | undefined> {
    const [deployment] = await db.select().from(deployments).where(eq(deployments.id, id));
    return deployment;
  }

  async getDeploymentsByProjectId(projectId: string): Promise<Deployment[]> {
    return await db.select()
      .from(deployments)
      .where(eq(deployments.projectId, projectId))
      .orderBy(desc(deployments.createdAt));
  }

  async updateDeployment(id: string, updates: Partial<Deployment>): Promise<void> {
    await db.update(deployments)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(deployments.id, id));
  }

  async deleteDeployment(id: string): Promise<void> {
    await db.delete(deploymentRuns).where(eq(deploymentRuns.deploymentId, id));
    await db.delete(deployments).where(eq(deployments.id, id));
  }

  async createDeploymentRun(insertRun: InsertDeploymentRun): Promise<DeploymentRun> {
    const [run] = await db.insert(deploymentRuns).values(insertRun).returning();
    return run;
  }

  async getDeploymentRun(id: string): Promise<DeploymentRun | undefined> {
    const [run] = await db.select().from(deploymentRuns).where(eq(deploymentRuns.id, id));
    return run;
  }

  async getDeploymentRunsByDeploymentId(deploymentId: string): Promise<DeploymentRun[]> {
    return await db.select()
      .from(deploymentRuns)
      .where(eq(deploymentRuns.deploymentId, deploymentId))
      .orderBy(desc(deploymentRuns.startedAt));
  }

  async updateDeploymentRun(id: string, updates: Partial<DeploymentRun>): Promise<void> {
    await db.update(deploymentRuns)
      .set(updates)
      .where(eq(deploymentRuns.id, id));
  }

  // Environment Variables Operations
  async getEnvVariablesByUser(userId: string): Promise<EnvVariable[]> {
    return await db.select()
      .from(envVariables)
      .where(eq(envVariables.userId, userId))
      .orderBy(desc(envVariables.createdAt));
  }

  async getEnvVariablesByProject(projectId: string): Promise<EnvVariable[]> {
    return await db.select()
      .from(envVariables)
      .where(eq(envVariables.projectId, projectId))
      .orderBy(desc(envVariables.createdAt));
  }

  async createEnvVariable(data: InsertEnvVariable): Promise<EnvVariable> {
    const [envVar] = await db.insert(envVariables).values(data).returning();
    return envVar;
  }

  async updateEnvVariable(id: string, data: Partial<InsertEnvVariable>): Promise<EnvVariable | undefined> {
    const [updated] = await db.update(envVariables)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(envVariables.id, id))
      .returning();
    return updated;
  }

  async deleteEnvVariable(id: string): Promise<void> {
    await db.delete(envVariables).where(eq(envVariables.id, id));
  }

  // Project Files Operations
  async getProjectFiles(projectId: string): Promise<ProjectFile[]> {
    return await db.select()
      .from(projectFiles)
      .where(eq(projectFiles.projectId, projectId))
      .orderBy(projectFiles.path);
  }

  async getProjectFile(id: string): Promise<ProjectFile | undefined> {
    const [file] = await db.select().from(projectFiles).where(eq(projectFiles.id, id));
    return file;
  }

  async getProjectFileByPath(projectId: string, path: string): Promise<ProjectFile | undefined> {
    const [file] = await db.select()
      .from(projectFiles)
      .where(and(eq(projectFiles.projectId, projectId), eq(projectFiles.path, path)));
    return file;
  }

  async createProjectFile(data: InsertProjectFile): Promise<ProjectFile> {
    const [file] = await db.insert(projectFiles).values(data).returning();
    return file;
  }

  async updateProjectFile(id: string, data: Partial<InsertProjectFile>): Promise<ProjectFile | undefined> {
    const [updated] = await db.update(projectFiles)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(projectFiles.id, id))
      .returning();
    return updated;
  }

  async deleteProjectFile(id: string): Promise<void> {
    await db.delete(projectFiles).where(eq(projectFiles.id, id));
  }

  // Console Logs Operations
  async getConsoleLogs(projectId: string, limit?: number): Promise<ConsoleLog[]> {
    const query = db.select()
      .from(consoleLogs)
      .where(eq(consoleLogs.projectId, projectId))
      .orderBy(desc(consoleLogs.timestamp));
    
    if (limit) {
      return await query.limit(limit);
    }
    return await query;
  }

  async createConsoleLog(data: InsertConsoleLog): Promise<ConsoleLog> {
    const [log] = await db.insert(consoleLogs).values(data).returning();
    return log;
  }

  async clearConsoleLogs(projectId: string): Promise<void> {
    await db.delete(consoleLogs).where(eq(consoleLogs.projectId, projectId));
  }
}

export const storage = new DatabaseStorage();
