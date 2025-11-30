import { 
  type Project, 
  type InsertProject,
  type InfrastructureTemplate,
  type InsertInfrastructureTemplate,
  type BuildLog,
  type InsertBuildLog,
  type User,
  type UpsertUser,
  projects,
  infrastructureTemplates,
  buildLogs,
  users
} from "@shared/schema";
import { db } from "../db";
import { eq, desc } from "drizzle-orm";

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
}

export const storage = new DatabaseStorage();
