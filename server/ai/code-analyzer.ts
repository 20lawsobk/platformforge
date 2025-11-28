export interface CodeAnalysis {
  imports: ImportStatement[];
  exports: ExportStatement[];
  functions: FunctionInfo[];
  classes: ClassInfo[];
  variables: VariableInfo[];
  dependencies: DependencyInfo[];
  patterns: DetectedPattern[];
  metrics: CodeMetrics;
  issues: CodeIssue[];
}

export interface ImportStatement {
  module: string;
  imports: string[];
  isDefault: boolean;
  isNamespace: boolean;
  line: number;
}

export interface ExportStatement {
  name: string;
  type: 'function' | 'class' | 'variable' | 'type' | 'interface';
  isDefault: boolean;
  line: number;
}

export interface FunctionInfo {
  name: string;
  params: ParamInfo[];
  returnType?: string;
  isAsync: boolean;
  isArrow: boolean;
  complexity: number;
  lines: number;
  startLine: number;
  endLine: number;
  calls: string[];
}

export interface ParamInfo {
  name: string;
  type?: string;
  defaultValue?: string;
  isOptional: boolean;
}

export interface ClassInfo {
  name: string;
  extends?: string;
  implements: string[];
  methods: FunctionInfo[];
  properties: PropertyInfo[];
  isAbstract: boolean;
  decorators: string[];
  startLine: number;
  endLine: number;
}

export interface PropertyInfo {
  name: string;
  type?: string;
  visibility: 'public' | 'private' | 'protected';
  isStatic: boolean;
  isReadonly: boolean;
}

export interface VariableInfo {
  name: string;
  type?: string;
  kind: 'const' | 'let' | 'var';
  value?: string;
  line: number;
}

export interface DependencyInfo {
  name: string;
  version?: string;
  type: 'runtime' | 'dev' | 'peer';
  usageCount: number;
  usedIn: string[];
}

export interface DetectedPattern {
  name: string;
  type: 'design-pattern' | 'architectural' | 'anti-pattern';
  confidence: number;
  location: string;
  description: string;
  recommendation?: string;
}

export interface CodeMetrics {
  linesOfCode: number;
  linesOfComments: number;
  blankLines: number;
  functions: number;
  classes: number;
  imports: number;
  exports: number;
  cyclomaticComplexity: number;
  maintainabilityIndex: number;
  technicalDebt: string;
}

export interface CodeIssue {
  severity: 'error' | 'warning' | 'info';
  category: 'security' | 'performance' | 'maintainability' | 'reliability' | 'style';
  message: string;
  file: string;
  line: number;
  column?: number;
  suggestion?: string;
  autoFixable: boolean;
}

export class CodeAnalyzer {
  analyzeCode(code: string, filename: string): CodeAnalysis {
    const language = this.detectLanguage(filename);
    
    return {
      imports: this.extractImports(code, language),
      exports: this.extractExports(code, language),
      functions: this.extractFunctions(code, language),
      classes: this.extractClasses(code, language),
      variables: this.extractVariables(code, language),
      dependencies: this.extractDependencies(code),
      patterns: this.detectPatterns(code, language),
      metrics: this.calculateMetrics(code),
      issues: this.findIssues(code, filename, language),
    };
  }

  private detectLanguage(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    const langMap: Record<string, string> = {
      'ts': 'typescript',
      'tsx': 'typescript',
      'js': 'javascript',
      'jsx': 'javascript',
      'py': 'python',
      'go': 'go',
      'rs': 'rust',
      'java': 'java',
      'rb': 'ruby',
      'php': 'php',
    };
    return langMap[ext] || 'unknown';
  }

  private extractImports(code: string, language: string): ImportStatement[] {
    const imports: ImportStatement[] = [];
    const lines = code.split('\n');

    if (language === 'typescript' || language === 'javascript') {
      const importRegex = /import\s+(?:(\*\s+as\s+\w+)|(\{\s*[^}]+\s*\})|(\w+))?\s*(?:,\s*(\{\s*[^}]+\s*\}))?\s*from\s+['"]([^'"]+)['"]/g;
      const requireRegex = /(?:const|let|var)\s+(\{\s*[^}]+\s*\}|\w+)\s*=\s*require\(['"]([^'"]+)['"]\)/g;

      let match;
      while ((match = importRegex.exec(code)) !== null) {
        const lineNumber = code.substring(0, match.index).split('\n').length;
        imports.push({
          module: match[5],
          imports: this.parseImportNames(match[1] || match[2] || match[3] || '', match[4] || ''),
          isDefault: !!match[3] && !match[1],
          isNamespace: !!match[1],
          line: lineNumber,
        });
      }

      while ((match = requireRegex.exec(code)) !== null) {
        const lineNumber = code.substring(0, match.index).split('\n').length;
        imports.push({
          module: match[2],
          imports: this.parseImportNames(match[1], ''),
          isDefault: !match[1].startsWith('{'),
          isNamespace: false,
          line: lineNumber,
        });
      }
    }

    if (language === 'python') {
      const importRegex = /^(?:from\s+(\S+)\s+)?import\s+(.+)$/gm;
      let match;
      while ((match = importRegex.exec(code)) !== null) {
        const lineNumber = code.substring(0, match.index).split('\n').length;
        imports.push({
          module: match[1] || match[2].split(',')[0].trim(),
          imports: match[2].split(',').map(s => s.trim()),
          isDefault: !match[1],
          isNamespace: match[2].includes('*'),
          line: lineNumber,
        });
      }
    }

    return imports;
  }

  private parseImportNames(named: string, additional: string): string[] {
    const names: string[] = [];
    const combined = `${named} ${additional}`;
    const matches = combined.match(/\w+/g);
    if (matches) {
      names.push(...matches.filter(m => !['as', 'from'].includes(m)));
    }
    return names;
  }

  private extractExports(code: string, language: string): ExportStatement[] {
    const exports: ExportStatement[] = [];

    if (language === 'typescript' || language === 'javascript') {
      const exportPatterns = [
        { regex: /export\s+default\s+(function|class|const|let|var)\s+(\w+)/g, isDefault: true },
        { regex: /export\s+(function|class|const|let|var|interface|type)\s+(\w+)/g, isDefault: false },
        { regex: /export\s+\{\s*([^}]+)\s*\}/g, isDefault: false },
      ];

      for (const pattern of exportPatterns) {
        let match;
        while ((match = pattern.regex.exec(code)) !== null) {
          const lineNumber = code.substring(0, match.index).split('\n').length;
          if (pattern.isDefault || match[1]) {
            exports.push({
              name: match[2] || 'default',
              type: this.mapExportType(match[1]),
              isDefault: pattern.isDefault,
              line: lineNumber,
            });
          }
        }
      }
    }

    return exports;
  }

  private mapExportType(keyword?: string): ExportStatement['type'] {
    switch (keyword) {
      case 'function': return 'function';
      case 'class': return 'class';
      case 'interface': return 'interface';
      case 'type': return 'type';
      default: return 'variable';
    }
  }

  private extractFunctions(code: string, language: string): FunctionInfo[] {
    const functions: FunctionInfo[] = [];

    if (language === 'typescript' || language === 'javascript') {
      const funcPatterns = [
        /(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*\{/g,
        /(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)(?:\s*:\s*([^=]+))?\s*=>/g,
        /(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function\s*\(([^)]*)\)/g,
      ];

      for (const pattern of funcPatterns) {
        let match;
        while ((match = pattern.exec(code)) !== null) {
          const startLine = code.substring(0, match.index).split('\n').length;
          const isArrow = match[0].includes('=>');
          const isAsync = match[0].includes('async');

          functions.push({
            name: match[1],
            params: this.parseParams(match[2]),
            returnType: match[3]?.trim(),
            isAsync,
            isArrow,
            complexity: this.calculateFunctionComplexity(code, match.index),
            lines: this.countFunctionLines(code, match.index),
            startLine,
            endLine: startLine + this.countFunctionLines(code, match.index),
            calls: this.extractFunctionCalls(code, match.index),
          });
        }
      }
    }

    return functions;
  }

  private parseParams(paramString: string): ParamInfo[] {
    if (!paramString.trim()) return [];
    
    return paramString.split(',').map(param => {
      const parts = param.trim().split(/[=:]/);
      const name = parts[0].replace(/[?.\s]/g, '');
      return {
        name,
        type: parts[1]?.trim(),
        defaultValue: parts[2]?.trim(),
        isOptional: param.includes('?') || param.includes('='),
      };
    });
  }

  private calculateFunctionComplexity(code: string, startIndex: number): number {
    const funcBody = this.extractFunctionBody(code, startIndex);
    let complexity = 1;

    const complexityPatterns = [
      /\bif\s*\(/g,
      /\belse\s+if\s*\(/g,
      /\bfor\s*\(/g,
      /\bwhile\s*\(/g,
      /\bcase\s+/g,
      /\bcatch\s*\(/g,
      /\?\?/g,
      /\|\|/g,
      /&&/g,
      /\?[^:]+:/g,
    ];

    for (const pattern of complexityPatterns) {
      const matches = funcBody.match(pattern);
      if (matches) complexity += matches.length;
    }

    return complexity;
  }

  private extractFunctionBody(code: string, startIndex: number): string {
    let braceCount = 0;
    let started = false;
    let endIndex = startIndex;

    for (let i = startIndex; i < code.length; i++) {
      if (code[i] === '{') {
        braceCount++;
        started = true;
      } else if (code[i] === '}') {
        braceCount--;
        if (started && braceCount === 0) {
          endIndex = i;
          break;
        }
      }
    }

    return code.substring(startIndex, endIndex);
  }

  private countFunctionLines(code: string, startIndex: number): number {
    const body = this.extractFunctionBody(code, startIndex);
    return body.split('\n').length;
  }

  private extractFunctionCalls(code: string, startIndex: number): string[] {
    const body = this.extractFunctionBody(code, startIndex);
    const callPattern = /(\w+)\s*\(/g;
    const calls: string[] = [];
    let match;

    const keywords = ['if', 'for', 'while', 'switch', 'catch', 'function', 'return', 'new'];
    while ((match = callPattern.exec(body)) !== null) {
      if (!keywords.includes(match[1])) {
        calls.push(match[1]);
      }
    }

    return Array.from(new Set(calls));
  }

  private extractClasses(code: string, language: string): ClassInfo[] {
    const classes: ClassInfo[] = [];

    if (language === 'typescript' || language === 'javascript') {
      const classPattern = /(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{/g;
      let match;

      while ((match = classPattern.exec(code)) !== null) {
        const startLine = code.substring(0, match.index).split('\n').length;
        const classBody = this.extractFunctionBody(code, match.index);

        classes.push({
          name: match[1],
          extends: match[2],
          implements: match[3]?.split(',').map(s => s.trim()) || [],
          methods: this.extractMethods(classBody),
          properties: this.extractProperties(classBody),
          isAbstract: match[0].includes('abstract'),
          decorators: this.extractDecorators(code, match.index),
          startLine,
          endLine: startLine + classBody.split('\n').length,
        });
      }
    }

    return classes;
  }

  private extractMethods(classBody: string): FunctionInfo[] {
    const methods: FunctionInfo[] = [];
    const methodPattern = /(?:async\s+)?(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*\{/g;
    let match;

    while ((match = methodPattern.exec(classBody)) !== null) {
      if (match[1] !== 'constructor') {
        methods.push({
          name: match[1],
          params: this.parseParams(match[2]),
          returnType: match[3]?.trim(),
          isAsync: match[0].includes('async'),
          isArrow: false,
          complexity: this.calculateFunctionComplexity(classBody, match.index),
          lines: this.countFunctionLines(classBody, match.index),
          startLine: 0,
          endLine: 0,
          calls: [],
        });
      }
    }

    return methods;
  }

  private extractProperties(classBody: string): PropertyInfo[] {
    const properties: PropertyInfo[] = [];
    const propPattern = /(private|public|protected)?\s*(static)?\s*(readonly)?\s*(\w+)(?:\s*:\s*([^=;]+))?(?:\s*=)?/g;
    let match;

    while ((match = propPattern.exec(classBody)) !== null) {
      if (!match[0].includes('(') && !match[0].includes('function')) {
        properties.push({
          name: match[4],
          type: match[5]?.trim(),
          visibility: (match[1] as any) || 'public',
          isStatic: !!match[2],
          isReadonly: !!match[3],
        });
      }
    }

    return properties;
  }

  private extractDecorators(code: string, classIndex: number): string[] {
    const decorators: string[] = [];
    const beforeClass = code.substring(0, classIndex);
    const lines = beforeClass.split('\n');
    
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line.startsWith('@')) {
        decorators.push(line.match(/@(\w+)/)?.[1] || '');
      } else if (line && !line.startsWith('//')) {
        break;
      }
    }

    return decorators.reverse();
  }

  private extractVariables(code: string, language: string): VariableInfo[] {
    const variables: VariableInfo[] = [];

    if (language === 'typescript' || language === 'javascript') {
      const varPattern = /(?:export\s+)?(const|let|var)\s+(\w+)(?:\s*:\s*([^=]+))?\s*=\s*([^;]+)/g;
      let match;

      while ((match = varPattern.exec(code)) !== null) {
        const lineNumber = code.substring(0, match.index).split('\n').length;
        if (!match[0].includes('function') && !match[0].includes('=>')) {
          variables.push({
            name: match[2],
            type: match[3]?.trim(),
            kind: match[1] as 'const' | 'let' | 'var',
            value: match[4].trim().substring(0, 50),
            line: lineNumber,
          });
        }
      }
    }

    return variables;
  }

  private extractDependencies(code: string): DependencyInfo[] {
    const deps = new Map<string, DependencyInfo>();
    const importPattern = /(?:import|require)\s*\(?['"]([^'"]+)['"]\)?/g;
    let match;

    while ((match = importPattern.exec(code)) !== null) {
      const module = match[1];
      if (!module.startsWith('.') && !module.startsWith('/')) {
        const name = module.startsWith('@') 
          ? module.split('/').slice(0, 2).join('/')
          : module.split('/')[0];
        
        const existing = deps.get(name);
        if (existing) {
          existing.usageCount++;
        } else {
          deps.set(name, {
            name,
            type: 'runtime',
            usageCount: 1,
            usedIn: [],
          });
        }
      }
    }

    return Array.from(deps.values());
  }

  private detectPatterns(code: string, language: string): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];

    const patternChecks = [
      {
        name: 'Singleton',
        type: 'design-pattern' as const,
        check: /private\s+static\s+instance|getInstance\s*\(\)/,
        description: 'Singleton pattern detected - ensures single instance',
      },
      {
        name: 'Factory',
        type: 'design-pattern' as const,
        check: /create\w+\s*\(|factory|Factory/,
        description: 'Factory pattern - creates objects without exposing instantiation logic',
      },
      {
        name: 'Observer',
        type: 'design-pattern' as const,
        check: /subscribe|unsubscribe|notify|addEventListener|removeEventListener/,
        description: 'Observer pattern - pub/sub notification system',
      },
      {
        name: 'Repository',
        type: 'architectural' as const,
        check: /Repository|findById|findAll|save|delete/,
        description: 'Repository pattern - abstracts data access layer',
      },
      {
        name: 'Dependency Injection',
        type: 'architectural' as const,
        check: /@Inject|@Injectable|constructor\s*\([^)]*private/,
        description: 'Dependency injection for loose coupling',
      },
      {
        name: 'Callback Hell',
        type: 'anti-pattern' as const,
        check: /\)\s*=>\s*\{[^}]*\)\s*=>\s*\{[^}]*\)\s*=>\s*\{/,
        description: 'Deeply nested callbacks - consider async/await',
        recommendation: 'Refactor to use async/await or Promise chains',
      },
      {
        name: 'God Class',
        type: 'anti-pattern' as const,
        check: /class\s+\w+[^}]{5000,}/,
        description: 'Very large class with too many responsibilities',
        recommendation: 'Break down into smaller, focused classes',
      },
    ];

    for (const check of patternChecks) {
      if (check.check.test(code)) {
        const match = code.match(check.check);
        patterns.push({
          name: check.name,
          type: check.type,
          confidence: 0.8,
          location: match ? `Line ${code.substring(0, match.index).split('\n').length}` : 'Unknown',
          description: check.description,
          recommendation: check.recommendation,
        });
      }
    }

    return patterns;
  }

  private calculateMetrics(code: string): CodeMetrics {
    const lines = code.split('\n');
    const codeLines = lines.filter(l => l.trim() && !l.trim().startsWith('//') && !l.trim().startsWith('/*'));
    const commentLines = lines.filter(l => l.trim().startsWith('//') || l.trim().startsWith('/*'));
    const blankLines = lines.filter(l => !l.trim());

    const functionCount = (code.match(/function\s+\w+|=>\s*\{|=>\s*[^{]/g) || []).length;
    const classCount = (code.match(/class\s+\w+/g) || []).length;
    const importCount = (code.match(/import\s+|require\s*\(/g) || []).length;
    const exportCount = (code.match(/export\s+/g) || []).length;

    const complexity = this.calculateOverallComplexity(code);
    const maintainability = this.calculateMaintainabilityIndex(codeLines.length, complexity, commentLines.length);

    return {
      linesOfCode: codeLines.length,
      linesOfComments: commentLines.length,
      blankLines: blankLines.length,
      functions: functionCount,
      classes: classCount,
      imports: importCount,
      exports: exportCount,
      cyclomaticComplexity: complexity,
      maintainabilityIndex: maintainability,
      technicalDebt: this.estimateTechnicalDebt(complexity, maintainability),
    };
  }

  private calculateOverallComplexity(code: string): number {
    let complexity = 1;
    const complexityIndicators = [
      /\bif\b/g, /\belse\b/g, /\bfor\b/g, /\bwhile\b/g,
      /\bcase\b/g, /\bcatch\b/g, /\?\?/g, /\|\|/g, /&&/g,
    ];

    for (const pattern of complexityIndicators) {
      const matches = code.match(pattern);
      if (matches) complexity += matches.length;
    }

    return complexity;
  }

  private calculateMaintainabilityIndex(loc: number, complexity: number, comments: number): number {
    const halsteadVolume = loc * Math.log2(loc + 1);
    const mi = 171 - 5.2 * Math.log(halsteadVolume) - 0.23 * complexity - 16.2 * Math.log(loc);
    const normalizedMi = Math.max(0, Math.min(100, mi * 100 / 171));
    return Math.round(normalizedMi);
  }

  private estimateTechnicalDebt(complexity: number, maintainability: number): string {
    const debtScore = (complexity * 2) + (100 - maintainability);
    
    if (debtScore < 50) return 'Low (< 1 hour)';
    if (debtScore < 100) return 'Medium (1-4 hours)';
    if (debtScore < 200) return 'High (4-16 hours)';
    return 'Critical (> 16 hours)';
  }

  private findIssues(code: string, filename: string, language: string): CodeIssue[] {
    const issues: CodeIssue[] = [];
    const lines = code.split('\n');

    const checks = [
      {
        pattern: /console\.(log|debug|info|warn|error)/g,
        severity: 'warning' as const,
        category: 'maintainability' as const,
        message: 'Console statement should be removed in production',
        autoFixable: true,
      },
      {
        pattern: /TODO|FIXME|HACK|XXX/gi,
        severity: 'info' as const,
        category: 'maintainability' as const,
        message: 'Technical debt marker found',
        autoFixable: false,
      },
      {
        pattern: /password\s*=\s*['"][^'"]+['"]/gi,
        severity: 'error' as const,
        category: 'security' as const,
        message: 'Hardcoded password detected',
        suggestion: 'Use environment variables for secrets',
        autoFixable: false,
      },
      {
        pattern: /api[_-]?key\s*=\s*['"][^'"]+['"]/gi,
        severity: 'error' as const,
        category: 'security' as const,
        message: 'Hardcoded API key detected',
        suggestion: 'Use environment variables for API keys',
        autoFixable: false,
      },
      {
        pattern: /eval\s*\(/g,
        severity: 'error' as const,
        category: 'security' as const,
        message: 'eval() is a security risk',
        suggestion: 'Avoid eval() - use safer alternatives',
        autoFixable: false,
      },
      {
        pattern: /==(?!=)/g,
        severity: 'warning' as const,
        category: 'reliability' as const,
        message: 'Use === instead of == for strict equality',
        autoFixable: true,
      },
      {
        pattern: /var\s+\w+/g,
        severity: 'info' as const,
        category: 'style' as const,
        message: 'Use const or let instead of var',
        autoFixable: true,
      },
      {
        pattern: /catch\s*\(\s*\w+\s*\)\s*\{\s*\}/g,
        severity: 'warning' as const,
        category: 'reliability' as const,
        message: 'Empty catch block swallows errors',
        suggestion: 'Handle or log the error',
        autoFixable: false,
      },
    ];

    for (const check of checks) {
      let match;
      while ((match = check.pattern.exec(code)) !== null) {
        const lineNumber = code.substring(0, match.index).split('\n').length;
        issues.push({
          severity: check.severity,
          category: check.category,
          message: check.message,
          file: filename,
          line: lineNumber,
          suggestion: check.suggestion,
          autoFixable: check.autoFixable,
        });
      }
    }

    return issues;
  }

  analyzeArchitecture(files: { path: string; content: string }[]): {
    layers: string[];
    components: string[];
    dependencies: { from: string; to: string }[];
    suggestions: string[];
  } {
    const layers = new Set<string>();
    const components = new Set<string>();
    const dependencies: { from: string; to: string }[] = [];
    const suggestions: string[] = [];

    for (const file of files) {
      const path = file.path.toLowerCase();
      
      if (path.includes('/api/') || path.includes('/routes/')) {
        layers.add('API Layer');
        components.add('Routes');
      }
      if (path.includes('/service') || path.includes('/services/')) {
        layers.add('Service Layer');
        components.add('Business Logic');
      }
      if (path.includes('/repository') || path.includes('/db/') || path.includes('/database/')) {
        layers.add('Data Layer');
        components.add('Data Access');
      }
      if (path.includes('/model') || path.includes('/entity') || path.includes('/schema')) {
        layers.add('Domain Layer');
        components.add('Domain Models');
      }
      if (path.includes('/util') || path.includes('/helper') || path.includes('/lib/')) {
        layers.add('Utility Layer');
        components.add('Utilities');
      }

      const analysis = this.analyzeCode(file.content, file.path);
      for (const imp of analysis.imports) {
        if (!imp.module.startsWith('.')) continue;
        dependencies.push({
          from: file.path,
          to: imp.module,
        });
      }
    }

    if (!layers.has('Service Layer')) {
      suggestions.push('Consider adding a service layer to separate business logic from API handlers');
    }
    if (!layers.has('Data Layer')) {
      suggestions.push('Consider using a repository pattern for data access abstraction');
    }
    if (dependencies.length > files.length * 5) {
      suggestions.push('High coupling detected - consider reducing dependencies between modules');
    }

    return {
      layers: Array.from(layers),
      components: Array.from(components),
      dependencies,
      suggestions,
    };
  }
}

export const codeAnalyzer = new CodeAnalyzer();
