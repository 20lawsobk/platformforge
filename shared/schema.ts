import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, jsonb, integer } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const projects = pgTable("projects", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  sourceUrl: text("source_url").notNull(),
  sourceType: text("source_type").notNull(), // 'github' | 'script' | 'upload'
  status: text("status").notNull().default('pending'), // 'pending' | 'analyzing' | 'generating' | 'complete' | 'failed'
  createdAt: timestamp("created_at").defaultNow().notNull(),
  completedAt: timestamp("completed_at"),
});

export const infrastructureTemplates = pgTable("infrastructure_templates", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id).notNull(),
  
  // AI Analysis Results
  detectedLanguage: text("detected_language"),
  detectedFramework: text("detected_framework"),
  architecture: text("architecture"), // 'monolith' | 'microservices' | 'serverless'
  
  // Generated Infrastructure
  terraformConfig: text("terraform_config"),
  kubernetesConfig: text("kubernetes_config"),
  dockerfileContent: text("dockerfile_content"),
  dockerComposeConfig: text("docker_compose_config"),
  
  // Scaling Configuration
  minInstances: integer("min_instances").default(2),
  maxInstances: integer("max_instances").default(20),
  
  // Dependencies
  requiresDatabase: text("requires_database"), // 'postgres' | 'mysql' | 'mongodb' | null
  requiresCache: text("requires_cache"), // 'redis' | 'memcached' | null
  requiresQueue: text("requires_queue"), // 'rabbitmq' | 'kafka' | null
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const buildLogs = pgTable("build_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id).notNull(),
  logLevel: text("log_level").notNull(), // 'ai' | 'system' | 'action' | 'cmd' | 'success' | 'error'
  message: text("message").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

// Insert Schemas
export const insertProjectSchema = createInsertSchema(projects).omit({
  id: true,
  createdAt: true,
  completedAt: true,
});

export const insertInfrastructureTemplateSchema = createInsertSchema(infrastructureTemplates).omit({
  id: true,
  createdAt: true,
});

export const insertBuildLogSchema = createInsertSchema(buildLogs).omit({
  id: true,
  timestamp: true,
});

// Types
export type InsertProject = z.infer<typeof insertProjectSchema>;
export type Project = typeof projects.$inferSelect;

export type InsertInfrastructureTemplate = z.infer<typeof insertInfrastructureTemplateSchema>;
export type InfrastructureTemplate = typeof infrastructureTemplates.$inferSelect;

export type InsertBuildLog = z.infer<typeof insertBuildLogSchema>;
export type BuildLog = typeof buildLogs.$inferSelect;
