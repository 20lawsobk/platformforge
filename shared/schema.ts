import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, jsonb, integer, index, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Session storage table.
// (IMPORTANT) This table is mandatory for Replit Auth, don't drop it.
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: jsonb("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// User storage table.
// (IMPORTANT) This table is mandatory for Replit Auth, don't drop it.
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  email: varchar("email").unique(),
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  onboardingCompleted: boolean("onboarding_completed").default(false),
  plan: varchar("plan").default("free"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export type UpsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect;

// Projects - linked to users
export const projects = pgTable("projects", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id),
  name: text("name").notNull(),
  sourceUrl: text("source_url").notNull(),
  sourceType: text("source_type").notNull(),
  status: text("status").notNull().default('pending'),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  completedAt: timestamp("completed_at"),
});

export const infrastructureTemplates = pgTable("infrastructure_templates", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id).notNull(),
  detectedLanguage: text("detected_language"),
  detectedFramework: text("detected_framework"),
  architecture: text("architecture"),
  terraformConfig: text("terraform_config"),
  kubernetesConfig: text("kubernetes_config"),
  dockerfileContent: text("dockerfile_content"),
  dockerComposeConfig: text("docker_compose_config"),
  minInstances: integer("min_instances").default(2),
  maxInstances: integer("max_instances").default(20),
  requiresDatabase: text("requires_database"),
  requiresCache: text("requires_cache"),
  requiresQueue: text("requires_queue"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const buildLogs = pgTable("build_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id).notNull(),
  logLevel: text("log_level").notNull(),
  message: text("message").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

// KV Store Namespaces
export const kvNamespaces = pgTable("kv_namespaces", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id).notNull(),
  name: varchar("name").notNull(),
  description: text("description"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// KV Store Entries
export const kvEntries = pgTable("kv_entries", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  namespaceId: varchar("namespace_id").references(() => kvNamespaces.id).notNull(),
  key: varchar("key").notNull(),
  value: text("value").notNull(),
  expiresAt: timestamp("expires_at"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Object Storage Buckets
export const objectBuckets = pgTable("object_buckets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id).notNull(),
  name: varchar("name").notNull(),
  description: text("description"),
  isPublic: boolean("is_public").default(false),
  totalSize: integer("total_size").default(0),
  objectCount: integer("object_count").default(0),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Object Storage Objects
export const storageObjects = pgTable("storage_objects", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  bucketId: varchar("bucket_id").references(() => objectBuckets.id).notNull(),
  key: varchar("key").notNull(),
  contentType: varchar("content_type"),
  size: integer("size").default(0),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Security Findings
export const securityFindings = pgTable("security_findings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id).notNull(),
  severity: varchar("severity").notNull(),
  category: varchar("category").notNull(),
  title: varchar("title").notNull(),
  description: text("description").notNull(),
  filePath: varchar("file_path"),
  lineNumber: integer("line_number"),
  recommendation: text("recommendation"),
  status: varchar("status").default("open"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Security Scans
export const securityScans = pgTable("security_scans", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id).notNull(),
  status: varchar("status").notNull().default("pending"),
  totalFindings: integer("total_findings").default(0),
  criticalCount: integer("critical_count").default(0),
  highCount: integer("high_count").default(0),
  mediumCount: integer("medium_count").default(0),
  lowCount: integer("low_count").default(0),
  startedAt: timestamp("started_at").defaultNow().notNull(),
  completedAt: timestamp("completed_at"),
});

// Deployment Targets (cloud provider configs)
export const deploymentTargets = pgTable("deployment_targets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id).notNull(),
  name: varchar("name").notNull(),
  provider: varchar("provider").notNull(),
  region: varchar("region"),
  config: jsonb("config"),
  isDefault: boolean("is_default").default(false),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Deployments
export const deployments = pgTable("deployments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id).notNull(),
  targetId: varchar("target_id").references(() => deploymentTargets.id),
  name: varchar("name").notNull(),
  status: varchar("status").notNull().default("pending"),
  deploymentType: varchar("deployment_type").notNull(),
  url: varchar("url"),
  config: jsonb("config"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Deployment Runs (history)
export const deploymentRuns = pgTable("deployment_runs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  deploymentId: varchar("deployment_id").references(() => deployments.id).notNull(),
  status: varchar("status").notNull().default("pending"),
  version: varchar("version"),
  logs: text("logs"),
  errorMessage: text("error_message"),
  startedAt: timestamp("started_at").defaultNow().notNull(),
  completedAt: timestamp("completed_at"),
});

// Environment Variables
export const envVariables = pgTable("env_variables", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id).notNull(),
  projectId: varchar("project_id").references(() => projects.id),
  key: varchar("key").notNull(),
  value: text("value").notNull(),
  environment: varchar("environment").notNull().default("shared"),
  isSecret: boolean("is_secret").default(false),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Project Files (for IDE)
export const projectFiles = pgTable("project_files", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id).notNull(),
  path: text("path").notNull(),
  name: varchar("name").notNull(),
  content: text("content"),
  isFolder: boolean("is_folder").default(false),
  parentPath: text("parent_path"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Console Logs (for IDE terminal output)
export const consoleLogs = pgTable("console_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => projects.id).notNull(),
  type: varchar("type").notNull().default("log"),
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

export const insertKvNamespaceSchema = createInsertSchema(kvNamespaces).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertKvEntrySchema = createInsertSchema(kvEntries).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertObjectBucketSchema = createInsertSchema(objectBuckets).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
  totalSize: true,
  objectCount: true,
});

export const insertStorageObjectSchema = createInsertSchema(storageObjects).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertSecurityFindingSchema = createInsertSchema(securityFindings).omit({
  id: true,
  createdAt: true,
});

export const insertSecurityScanSchema = createInsertSchema(securityScans).omit({
  id: true,
  startedAt: true,
  completedAt: true,
});

export const insertDeploymentTargetSchema = createInsertSchema(deploymentTargets).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertDeploymentSchema = createInsertSchema(deployments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertDeploymentRunSchema = createInsertSchema(deploymentRuns).omit({
  id: true,
  startedAt: true,
  completedAt: true,
});

export const insertEnvVariableSchema = createInsertSchema(envVariables).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertProjectFileSchema = createInsertSchema(projectFiles).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertConsoleLogSchema = createInsertSchema(consoleLogs).omit({
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

export type InsertKvNamespace = z.infer<typeof insertKvNamespaceSchema>;
export type KvNamespace = typeof kvNamespaces.$inferSelect;

export type InsertKvEntry = z.infer<typeof insertKvEntrySchema>;
export type KvEntry = typeof kvEntries.$inferSelect;

export type InsertObjectBucket = z.infer<typeof insertObjectBucketSchema>;
export type ObjectBucket = typeof objectBuckets.$inferSelect;

export type InsertStorageObject = z.infer<typeof insertStorageObjectSchema>;
export type StorageObject = typeof storageObjects.$inferSelect;

export type InsertSecurityFinding = z.infer<typeof insertSecurityFindingSchema>;
export type SecurityFinding = typeof securityFindings.$inferSelect;

export type InsertSecurityScan = z.infer<typeof insertSecurityScanSchema>;
export type SecurityScan = typeof securityScans.$inferSelect;

export type InsertDeploymentTarget = z.infer<typeof insertDeploymentTargetSchema>;
export type DeploymentTarget = typeof deploymentTargets.$inferSelect;

export type InsertDeployment = z.infer<typeof insertDeploymentSchema>;
export type Deployment = typeof deployments.$inferSelect;

export type InsertDeploymentRun = z.infer<typeof insertDeploymentRunSchema>;
export type DeploymentRun = typeof deploymentRuns.$inferSelect;

export type InsertEnvVariable = z.infer<typeof insertEnvVariableSchema>;
export type EnvVariable = typeof envVariables.$inferSelect;

export type InsertProjectFile = z.infer<typeof insertProjectFileSchema>;
export type ProjectFile = typeof projectFiles.$inferSelect;

export type InsertConsoleLog = z.infer<typeof insertConsoleLogSchema>;
export type ConsoleLog = typeof consoleLogs.$inferSelect;
