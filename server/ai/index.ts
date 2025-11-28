export { AIReasoningEngine, aiEngine } from './engine';
export type { 
  AIContext, 
  AnalysisResult, 
  AIDecision, 
  ReasoningStep,
  DetectedService,
  DetectedDatabase,
  DetectedCache,
  DetectedQueue,
  SecurityConcern,
  ScalabilityFactor,
  ComplexityScore 
} from './engine';

export { InfrastructureKnowledgeBase, knowledgeBase } from './knowledge-base';
export type { 
  TerraformPattern, 
  KubernetesPattern, 
  DockerPattern, 
  ArchitecturePattern 
} from './knowledge-base';

export { CodeAnalyzer, codeAnalyzer } from './code-analyzer';
export type { 
  CodeAnalysis, 
  CodeMetrics, 
  CodeIssue, 
  DetectedPattern 
} from './code-analyzer';

export { InfrastructureGenerator, generator } from './generation-engine';
export type { 
  GeneratedInfrastructure, 
  GenerationOptions 
} from './generation-engine';

import { aiEngine, AIContext, AnalysisResult, AIDecision } from './engine';
import { codeAnalyzer, CodeAnalysis } from './code-analyzer';
import { generator, GeneratedInfrastructure, GenerationOptions } from './generation-engine';
import { knowledgeBase } from './knowledge-base';

export interface AIResponse {
  success: boolean;
  message: string;
  data?: any;
  reasoning?: any[];
  confidence?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    intent?: string;
    entities?: Record<string, string[]>;
    actions?: string[];
  };
}

export interface AgentTask {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'running' | 'awaiting_approval' | 'completed' | 'failed';
  steps: AgentStep[];
  fileChanges: FileChange[];
  createdAt: Date;
  completedAt?: Date;
}

export interface AgentStep {
  id: string;
  action: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  output?: string;
  error?: string;
}

export interface FileChange {
  path: string;
  action: 'create' | 'modify' | 'delete';
  content?: string;
  diff?: string;
}

export interface ArchitectRecommendation {
  id: string;
  category: 'security' | 'performance' | 'scalability' | 'reliability' | 'cost';
  priority: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  rationale: string;
  implementation: string;
  effort: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
}

export class AIService {
  private conversationHistory: Map<string, ChatMessage[]> = new Map();
  private activeTasks: Map<string, AgentTask> = new Map();

  async chat(
    projectId: string,
    message: string,
    context?: Partial<AIContext>
  ): Promise<ChatMessage> {
    const history = this.conversationHistory.get(projectId) || [];
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date(),
    };
    history.push(userMessage);

    const fullContext: AIContext = {
      projectId,
      sourceCode: context?.sourceCode || '',
      fileStructure: context?.fileStructure || [],
      dependencies: context?.dependencies || [],
      existingInfrastructure: context?.existingInfrastructure,
      userQuery: message,
      conversationHistory: history.map(m => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
      })),
    };

    const response = await aiEngine.processQuery(message, fullContext);
    
    const assistantMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: response,
      timestamp: new Date(),
      metadata: {
        intent: this.classifyIntent(message),
        entities: this.extractEntities(message),
      },
    };
    
    history.push(assistantMessage);
    this.conversationHistory.set(projectId, history);

    return assistantMessage;
  }

  async analyzeProject(context: AIContext): Promise<{
    analysis: AnalysisResult;
    codeAnalysis?: CodeAnalysis;
    recommendations: ArchitectRecommendation[];
  }> {
    const analysis = await aiEngine.analyzeProject(context);
    
    let codeAnalysis: CodeAnalysis | undefined;
    if (context.fileStructure.length > 0) {
      const mainFile = context.fileStructure.find(f => 
        f.path?.includes('index') || f.path?.includes('main') || f.path?.includes('app')
      );
      if (mainFile?.content) {
        codeAnalysis = codeAnalyzer.analyzeCode(mainFile.content, mainFile.path || 'index.ts');
      }
    }

    const recommendations = this.generateRecommendations(analysis);

    return { analysis, codeAnalysis, recommendations };
  }

  async generateInfrastructure(
    analysis: AnalysisResult,
    options?: Partial<GenerationOptions>
  ): Promise<GeneratedInfrastructure> {
    const generatorInstance = new (generator.constructor as any)(options);
    return generatorInstance.generate(analysis);
  }

  async createAgentTask(
    projectId: string,
    title: string,
    description: string
  ): Promise<AgentTask> {
    const task: AgentTask = {
      id: Date.now().toString(),
      title,
      description,
      status: 'pending',
      steps: this.planTaskSteps(title, description),
      fileChanges: [],
      createdAt: new Date(),
    };

    this.activeTasks.set(task.id, task);
    this.executeTask(task);

    return task;
  }

  async getTaskStatus(taskId: string): Promise<AgentTask | undefined> {
    return this.activeTasks.get(taskId);
  }

  async approveTask(taskId: string): Promise<AgentTask | undefined> {
    const task = this.activeTasks.get(taskId);
    if (task && task.status === 'awaiting_approval') {
      task.status = 'completed';
      task.completedAt = new Date();
    }
    return task;
  }

  async rejectTask(taskId: string): Promise<AgentTask | undefined> {
    const task = this.activeTasks.get(taskId);
    if (task && task.status === 'awaiting_approval') {
      task.status = 'failed';
      task.completedAt = new Date();
    }
    return task;
  }

  private planTaskSteps(title: string, description: string): AgentStep[] {
    const lowerTitle = title.toLowerCase();
    const steps: AgentStep[] = [];

    steps.push({
      id: '1',
      action: 'Analyzing project structure and requirements',
      status: 'pending',
    });

    if (lowerTitle.includes('terraform') || lowerTitle.includes('infrastructure')) {
      steps.push(
        { id: '2', action: 'Generating Terraform configurations', status: 'pending' },
        { id: '3', action: 'Creating variable definitions', status: 'pending' },
        { id: '4', action: 'Setting up module structure', status: 'pending' },
      );
    } else if (lowerTitle.includes('kubernetes') || lowerTitle.includes('k8s')) {
      steps.push(
        { id: '2', action: 'Creating deployment manifests', status: 'pending' },
        { id: '3', action: 'Configuring services and ingress', status: 'pending' },
        { id: '4', action: 'Setting up HPA and network policies', status: 'pending' },
      );
    } else if (lowerTitle.includes('docker')) {
      steps.push(
        { id: '2', action: 'Generating optimized Dockerfile', status: 'pending' },
        { id: '3', action: 'Creating docker-compose configuration', status: 'pending' },
      );
    } else {
      steps.push(
        { id: '2', action: 'Processing request', status: 'pending' },
        { id: '3', action: 'Generating output', status: 'pending' },
      );
    }

    steps.push({
      id: (steps.length + 1).toString(),
      action: 'Awaiting approval to apply changes',
      status: 'pending',
    });

    return steps;
  }

  private async executeTask(task: AgentTask): Promise<void> {
    task.status = 'running';

    for (const step of task.steps) {
      step.status = 'running';
      
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));
      
      if (step.action.includes('approval')) {
        task.status = 'awaiting_approval';
        step.status = 'pending';
        break;
      }

      step.status = 'completed';
      step.output = `Completed: ${step.action}`;

      if (step.action.includes('Generating') || step.action.includes('Creating')) {
        task.fileChanges.push({
          path: this.getFilePath(step.action),
          action: 'create',
        });
      }
    }
  }

  private getFilePath(action: string): string {
    if (action.includes('Terraform')) return '/infrastructure/main.tf';
    if (action.includes('deployment')) return '/kubernetes/deployment.yaml';
    if (action.includes('Dockerfile')) return '/Dockerfile';
    if (action.includes('docker-compose')) return '/docker-compose.yaml';
    if (action.includes('variable')) return '/infrastructure/variables.tf';
    if (action.includes('ingress')) return '/kubernetes/ingress.yaml';
    if (action.includes('HPA')) return '/kubernetes/hpa.yaml';
    return '/generated/output.txt';
  }

  private generateRecommendations(analysis: AnalysisResult): ArchitectRecommendation[] {
    const recommendations: ArchitectRecommendation[] = [];

    for (const concern of analysis.securityConcerns) {
      recommendations.push({
        id: `sec-${recommendations.length}`,
        category: 'security',
        priority: concern.severity === 'critical' ? 'critical' : concern.severity as any,
        title: concern.description,
        description: concern.recommendation,
        rationale: `This is a ${concern.severity} security concern in the ${concern.category} area.`,
        implementation: concern.recommendation,
        effort: concern.autoFixable ? 'low' : 'medium',
        impact: concern.severity === 'critical' ? 'high' : 'medium',
      });
    }

    for (const factor of analysis.scalabilityFactors) {
      recommendations.push({
        id: `scale-${recommendations.length}`,
        category: 'scalability',
        priority: factor.priority as any,
        title: `Optimize ${factor.component}`,
        description: factor.recommendation,
        rationale: `Current bottleneck: ${factor.bottleneck}. Current capacity: ${factor.currentCapacity}`,
        implementation: factor.recommendation,
        effort: 'medium',
        impact: factor.priority === 'high' ? 'high' : 'medium',
      });
    }

    for (const metric of analysis.performanceMetrics) {
      if (metric.current !== metric.target) {
        recommendations.push({
          id: `perf-${recommendations.length}`,
          category: 'performance',
          priority: 'medium',
          title: `Improve ${metric.name}`,
          description: metric.optimization,
          rationale: `Current: ${metric.current}, Target: ${metric.target}`,
          implementation: metric.optimization,
          effort: 'medium',
          impact: 'medium',
        });
      }
    }

    if (analysis.databases.some(db => !db.highAvailability)) {
      recommendations.push({
        id: `rel-${recommendations.length}`,
        category: 'reliability',
        priority: 'high',
        title: 'Enable database high availability',
        description: 'Configure Multi-AZ deployment for production databases',
        rationale: 'Single-AZ databases are vulnerable to availability zone failures',
        implementation: 'Set multi_az = true in RDS configuration',
        effort: 'low',
        impact: 'high',
      });
    }

    if (analysis.complexity.overall > 7) {
      recommendations.push({
        id: `cost-${recommendations.length}`,
        category: 'cost',
        priority: 'medium',
        title: 'Review infrastructure costs',
        description: 'High complexity may lead to increased operational costs',
        rationale: `Complexity score: ${analysis.complexity.overall}/10`,
        implementation: 'Consider using Spot instances, Reserved capacity, and right-sizing',
        effort: 'medium',
        impact: 'medium',
      });
    }

    return recommendations;
  }

  private classifyIntent(message: string): string {
    const lower = message.toLowerCase();
    if (lower.includes('explain') || lower.includes('what')) return 'explanation';
    if (lower.includes('generate') || lower.includes('create')) return 'generation';
    if (lower.includes('fix') || lower.includes('debug')) return 'debugging';
    if (lower.includes('optimize') || lower.includes('improve')) return 'optimization';
    if (lower.includes('deploy')) return 'deployment';
    return 'general';
  }

  private extractEntities(message: string): Record<string, string[]> {
    const entities: Record<string, string[]> = {
      technologies: [],
      files: [],
      actions: [],
    };

    const techKeywords = ['terraform', 'kubernetes', 'docker', 'aws', 'gcp', 'azure', 'redis', 'postgres'];
    entities.technologies = techKeywords.filter(t => message.toLowerCase().includes(t));

    const filePattern = /[\w\/]+\.(ts|js|py|tf|yaml|json)/g;
    entities.files = message.match(filePattern) || [];

    return entities;
  }

  getBestPractices(component: string): string[] {
    return knowledgeBase.getBestPractices(component);
  }

  getSecurityRecommendations(component: string): string[] {
    return knowledgeBase.getSecurityRecommendations(component);
  }
}

export const aiService = new AIService();
