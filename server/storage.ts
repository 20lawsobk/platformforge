import { 
  type Project, 
  type InsertProject,
  type InfrastructureTemplate,
  type InsertInfrastructureTemplate,
  type BuildLog,
  type InsertBuildLog,
  projects,
  infrastructureTemplates,
  buildLogs
} from "@shared/schema";
import { db } from "../db";
import { eq, desc } from "drizzle-orm";

export interface IStorage {
  // Projects
  createProject(project: InsertProject): Promise<Project>;
  getProject(id: string): Promise<Project | undefined>;
  updateProjectStatus(id: string, status: string, completedAt?: Date): Promise<void>;
  getAllProjects(): Promise<Project[]>;
  
  // Infrastructure Templates
  createInfrastructureTemplate(template: InsertInfrastructureTemplate): Promise<InfrastructureTemplate>;
  getInfrastructureTemplateByProjectId(projectId: string): Promise<InfrastructureTemplate | undefined>;
  
  // Build Logs
  createBuildLog(log: InsertBuildLog): Promise<BuildLog>;
  getBuildLogsByProjectId(projectId: string): Promise<BuildLog[]>;
}

export class DatabaseStorage implements IStorage {
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
}

export const storage = new DatabaseStorage();
