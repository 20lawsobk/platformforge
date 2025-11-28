/**
 * Client for communicating with the Custom AI Model Python API
 */

const AI_MODEL_URL = process.env.AI_MODEL_URL || 'http://localhost:8001';

interface GenerateRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  language?: string;
}

interface GenerateResponse {
  generated_text: string;
  prompt: string;
}

interface ChatRequest {
  message: string;
  max_tokens?: number;
}

interface ChatResponse {
  response: string;
  message: string;
}

interface CompleteRequest {
  code: string;
  language?: string;
  max_tokens?: number;
}

interface CompleteResponse {
  completion: string;
  original_code: string;
}

interface ExplainRequest {
  code: string;
  language?: string;
}

interface ExplainResponse {
  explanation: string;
  code: string;
}

interface FixRequest {
  code: string;
  error: string;
  language?: string;
}

interface FixResponse {
  fixed_code: string;
  original_code: string;
  error: string;
}

interface AnalyzeRequest {
  code: string;
  language?: string;
}

interface AnalyzeResponse {
  explanation: string;
  completion: string;
  language: string;
}

interface ModelInfo {
  loaded: boolean;
  parameters?: number;
  config?: Record<string, unknown>;
  device?: string;
  vocab_size?: number;
}

interface HealthStatus {
  status: string;
  model_initialized: boolean;
  is_training: boolean;
}

async function fetchFromModel<T>(
  endpoint: string,
  method: 'GET' | 'POST' = 'GET',
  body?: Record<string, unknown>
): Promise<T> {
  const options: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
  };

  if (body) {
    options.body = JSON.stringify(body);
  }

  try {
    const response = await fetch(`${AI_MODEL_URL}${endpoint}`, options);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(error.error || `HTTP ${response.status}`);
    }

    return response.json();
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes('ECONNREFUSED')) {
        throw new Error('AI Model service not available. Please ensure the Python AI server is running.');
      }
      throw error;
    }
    throw new Error('Failed to communicate with AI Model service');
  }
}

export const aiModelClient = {
  /**
   * Check if the AI model service is healthy
   */
  async health(): Promise<HealthStatus> {
    return fetchFromModel<HealthStatus>('/health');
  },

  /**
   * Get information about the loaded model
   */
  async getModelInfo(): Promise<ModelInfo> {
    return fetchFromModel<ModelInfo>('/model/info');
  },

  /**
   * Initialize the model (optionally from checkpoint)
   */
  async initialize(checkpointPath?: string): Promise<{ status: string }> {
    return fetchFromModel<{ status: string }>('/initialize', 'POST', {
      checkpoint_path: checkpointPath,
    });
  },

  /**
   * Generate text from a prompt
   */
  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    return fetchFromModel<GenerateResponse>('/generate', 'POST', {
      prompt: request.prompt,
      max_tokens: request.max_tokens ?? 100,
      temperature: request.temperature ?? 0.8,
      language: request.language,
    });
  },

  /**
   * Chat with the AI assistant
   */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    return fetchFromModel<ChatResponse>('/chat', 'POST', {
      message: request.message,
      max_tokens: request.max_tokens ?? 200,
    });
  },

  /**
   * Complete code snippet
   */
  async complete(request: CompleteRequest): Promise<CompleteResponse> {
    return fetchFromModel<CompleteResponse>('/complete', 'POST', {
      code: request.code,
      language: request.language ?? 'python',
      max_tokens: request.max_tokens ?? 150,
    });
  },

  /**
   * Explain code
   */
  async explain(request: ExplainRequest): Promise<ExplainResponse> {
    return fetchFromModel<ExplainResponse>('/explain', 'POST', {
      code: request.code,
      language: request.language ?? 'python',
    });
  },

  /**
   * Fix buggy code
   */
  async fix(request: FixRequest): Promise<FixResponse> {
    return fetchFromModel<FixResponse>('/fix', 'POST', {
      code: request.code,
      error: request.error,
      language: request.language ?? 'python',
    });
  },

  /**
   * Analyze code
   */
  async analyze(request: AnalyzeRequest): Promise<AnalyzeResponse> {
    return fetchFromModel<AnalyzeResponse>('/analyze', 'POST', {
      code: request.code,
      language: request.language ?? 'python',
    });
  },

  /**
   * Clear conversation history
   */
  async clearHistory(): Promise<{ status: string }> {
    return fetchFromModel<{ status: string }>('/clear_history', 'POST');
  },
};

/**
 * Fallback responses when the AI model is not available
 */
export const fallbackResponses = {
  generate(prompt: string): string {
    return `[AI Model Offline] Processing prompt: "${prompt.slice(0, 50)}..."

To use the custom AI model:
1. Start the Python AI server: python -m server.ai_model.api
2. Train the model with your data
3. The model will then provide intelligent responses

Currently using rule-based fallback processing.`;
  },

  chat(message: string): string {
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
      return "Hello! I'm the PlatformBuilder AI Assistant. I can help you with code generation, infrastructure configuration, and development questions. Note: The custom neural network model is currently offline - using rule-based responses.";
    }
    
    if (lowerMessage.includes('terraform') || lowerMessage.includes('infrastructure')) {
      return `For infrastructure generation, I can help you create:
- Terraform configurations for AWS, GCP, and Azure
- Kubernetes manifests for container orchestration
- Docker configurations for containerization
- Auto-scaling configurations

What cloud provider would you like to target?`;
    }
    
    if (lowerMessage.includes('code') || lowerMessage.includes('generate')) {
      return `I can assist with code generation for:
- API endpoints and routes
- Database models and migrations
- Frontend components
- Backend services

What type of code would you like me to generate?`;
    }
    
    if (lowerMessage.includes('help')) {
      return `I can help you with:
1. **Code Generation** - Create code snippets and complete implementations
2. **Code Explanation** - Understand what code does
3. **Bug Fixing** - Fix errors in your code
4. **Infrastructure** - Generate Terraform, Kubernetes, Docker configs
5. **Architecture** - Design system architectures

What would you like help with?`;
    }
    
    return `I received your message: "${message.slice(0, 100)}..."

I can help with code generation, infrastructure configuration, and development questions. The custom AI model is currently in rule-based mode. For more intelligent responses, please train the neural network model.`;
  },

  complete(code: string, language: string): string {
    if (language === 'python') {
      if (code.includes('def ')) {
        return code + '\n    """Implementation goes here."""\n    pass';
      }
      if (code.includes('class ')) {
        return code + '\n    def __init__(self):\n        pass';
      }
    }
    
    if (language === 'javascript' || language === 'typescript') {
      if (code.includes('function ')) {
        return code + '\n  // Implementation goes here\n  return null;\n}';
      }
      if (code.includes('const ') && code.includes('=> {')) {
        return code + '\n  // Implementation\n};';
      }
    }
    
    return code + '\n// [AI Model Offline - Unable to generate completion]';
  },

  explain(code: string): string {
    const lines = code.split('\n').length;
    const hasFunction = code.includes('function') || code.includes('def ');
    const hasClass = code.includes('class ');
    
    return `[Rule-based Analysis]
This code snippet contains ${lines} lines of code.
${hasFunction ? '- Contains function definition(s)' : ''}
${hasClass ? '- Contains class definition(s)' : ''}

For detailed AI-powered explanation, please start the neural network model.`;
  },

  fix(code: string, error: string): string {
    return `[AI Model Offline]
Original error: ${error}

Suggested debugging steps:
1. Check syntax errors
2. Verify variable names and imports
3. Check function signatures
4. Review logic flow

For AI-powered fixes, please start the neural network model.

Original code:
${code}`;
  },
};
