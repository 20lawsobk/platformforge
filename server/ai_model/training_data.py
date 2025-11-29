"""
Comprehensive Training Dataset for Code Generation Model
Contains diverse code samples across multiple languages and patterns
"""

PYTHON_SAMPLES = [
    '''def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[-1] + dp[-2])
    return dp[n]''',

    '''def binary_search(arr: list, target: int) -> int:
    """Perform binary search on a sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',

    '''class LinkedList:
    """Implementation of a singly linked list."""
    
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, value):
        new_node = Node(value)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def remove(self, value):
        if not self.head:
            return False
        if self.head.value == value:
            self.head = self.head.next
            self.size -= 1
            return True
        current = self.head
        while current.next:
            if current.next.value == value:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        return False''',

    '''async def fetch_data(url: str, session: aiohttp.ClientSession) -> dict:
    """Fetch JSON data from a URL asynchronously."""
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        raise HTTPError(f"Request failed with status {response.status}")''',

    '''def quicksort(arr: list) -> list:
    """Sort array using quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)''',

    '''class DatabaseConnection:
    """Context manager for database connections."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        self.connection = create_connection(self.connection_string)
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
        return False''',

    '''def merge_sort(arr: list) -> list:
    """Sort array using merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left: list, right: list) -> list:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result''',

    '''@dataclass
class User:
    """User model with validation."""
    id: int
    username: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.validate_email(self.email):
            raise ValueError(f"Invalid email: {self.email}")
    
    @staticmethod
    def validate_email(email: str) -> bool:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))''',

    '''def lru_cache(maxsize: int = 128):
    """Decorator implementing LRU cache."""
    def decorator(func):
        cache = OrderedDict()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result
        return wrapper
    return decorator''',

    '''class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root: TreeNode) -> list:
    """Perform inorder traversal of binary tree."""
    result = []
    stack = []
    current = root
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    
    return result''',
]

JAVASCRIPT_SAMPLES = [
    '''async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch user:', error);
        throw error;
    }
}''',

    '''class EventEmitter {
    constructor() {
        this.events = new Map();
    }
    
    on(event, callback) {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        this.events.get(event).push(callback);
        return this;
    }
    
    emit(event, ...args) {
        if (this.events.has(event)) {
            this.events.get(event).forEach(cb => cb(...args));
        }
        return this;
    }
    
    off(event, callback) {
        if (this.events.has(event)) {
            const callbacks = this.events.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
        return this;
    }
}''',

    '''const debounce = (fn, delay) => {
    let timeoutId;
    return (...args) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn(...args), delay);
    };
};

const throttle = (fn, limit) => {
    let inThrottle;
    return (...args) => {
        if (!inThrottle) {
            fn(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
};''',

    '''function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    
    if (Array.isArray(obj)) {
        return obj.map(item => deepClone(item));
    }
    
    const cloned = {};
    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            cloned[key] = deepClone(obj[key]);
        }
    }
    return cloned;
}''',

    '''class Observable {
    constructor(subscribe) {
        this._subscribe = subscribe;
    }
    
    subscribe(observer) {
        return this._subscribe(observer);
    }
    
    map(fn) {
        return new Observable(observer => {
            return this.subscribe({
                next: value => observer.next(fn(value)),
                error: err => observer.error(err),
                complete: () => observer.complete()
            });
        });
    }
    
    filter(predicate) {
        return new Observable(observer => {
            return this.subscribe({
                next: value => predicate(value) && observer.next(value),
                error: err => observer.error(err),
                complete: () => observer.complete()
            });
        });
    }
}''',
]

TYPESCRIPT_SAMPLES = [
    '''interface User {
    id: number;
    name: string;
    email: string;
    role: 'admin' | 'user' | 'guest';
}

class UserService {
    private users: Map<number, User> = new Map();
    
    async getUser(id: number): Promise<User | undefined> {
        return this.users.get(id);
    }
    
    async createUser(data: Omit<User, 'id'>): Promise<User> {
        const id = Date.now();
        const user: User = { id, ...data };
        this.users.set(id, user);
        return user;
    }
    
    async updateUser(id: number, data: Partial<User>): Promise<User | undefined> {
        const user = this.users.get(id);
        if (!user) return undefined;
        const updated = { ...user, ...data };
        this.users.set(id, updated);
        return updated;
    }
}''',

    '''type Result<T, E = Error> = 
    | { success: true; data: T }
    | { success: false; error: E };

function tryAsync<T>(fn: () => Promise<T>): Promise<Result<T>> {
    return fn()
        .then(data => ({ success: true as const, data }))
        .catch(error => ({ success: false as const, error }));
}

async function safeFetch<T>(url: string): Promise<Result<T>> {
    return tryAsync(async () => {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    });
}''',

    '''function createStore<T>(initialState: T) {
    let state = initialState;
    const listeners: Set<(state: T) => void> = new Set();
    
    return {
        getState: () => state,
        setState: (updater: T | ((prev: T) => T)) => {
            state = typeof updater === 'function' 
                ? (updater as (prev: T) => T)(state) 
                : updater;
            listeners.forEach(listener => listener(state));
        },
        subscribe: (listener: (state: T) => void) => {
            listeners.add(listener);
            return () => listeners.delete(listener);
        }
    };
}''',

    '''class Queue<T> {
    private items: T[] = [];
    
    enqueue(item: T): void {
        this.items.push(item);
    }
    
    dequeue(): T | undefined {
        return this.items.shift();
    }
    
    peek(): T | undefined {
        return this.items[0];
    }
    
    get size(): number {
        return this.items.length;
    }
    
    isEmpty(): boolean {
        return this.items.length === 0;
    }
}''',
]

REACT_SAMPLES = [
    '''import React, { useState, useEffect, useCallback } from 'react';

function useDebounce<T>(value: T, delay: number): T {
    const [debouncedValue, setDebouncedValue] = useState(value);
    
    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);
        
        return () => clearTimeout(handler);
    }, [value, delay]);
    
    return debouncedValue;
}

function SearchInput({ onSearch }) {
    const [query, setQuery] = useState('');
    const debouncedQuery = useDebounce(query, 300);
    
    useEffect(() => {
        if (debouncedQuery) {
            onSearch(debouncedQuery);
        }
    }, [debouncedQuery, onSearch]);
    
    return (
        <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search..."
        />
    );
}''',

    '''function useFetch<T>(url: string) {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<Error | null>(null);
    
    useEffect(() => {
        const controller = new AbortController();
        
        async function fetchData() {
            try {
                setLoading(true);
                const response = await fetch(url, { signal: controller.signal });
                if (!response.ok) throw new Error('Network error');
                const json = await response.json();
                setData(json);
            } catch (err) {
                if (err.name !== 'AbortError') {
                    setError(err as Error);
                }
            } finally {
                setLoading(false);
            }
        }
        
        fetchData();
        return () => controller.abort();
    }, [url]);
    
    return { data, loading, error };
}''',

    '''const TodoList = () => {
    const [todos, setTodos] = useState([]);
    const [input, setInput] = useState('');
    
    const addTodo = useCallback(() => {
        if (input.trim()) {
            setTodos(prev => [...prev, { id: Date.now(), text: input, done: false }]);
            setInput('');
        }
    }, [input]);
    
    const toggleTodo = useCallback((id) => {
        setTodos(prev => prev.map(todo =>
            todo.id === id ? { ...todo, done: !todo.done } : todo
        ));
    }, []);
    
    const deleteTodo = useCallback((id) => {
        setTodos(prev => prev.filter(todo => todo.id !== id));
    }, []);
    
    return (
        <div>
            <input value={input} onChange={e => setInput(e.target.value)} />
            <button onClick={addTodo}>Add</button>
            <ul>
                {todos.map(todo => (
                    <li key={todo.id}>
                        <span style={{ textDecoration: todo.done ? 'line-through' : 'none' }}>
                            {todo.text}
                        </span>
                        <button onClick={() => toggleTodo(todo.id)}>Toggle</button>
                        <button onClick={() => deleteTodo(todo.id)}>Delete</button>
                    </li>
                ))}
            </ul>
        </div>
    );
};''',
]

SQL_SAMPLES = [
    '''CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);''',

    '''SELECT 
    u.username,
    COUNT(o.id) as order_count,
    SUM(o.total_amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY u.id, u.username
HAVING COUNT(o.id) > 0
ORDER BY total_spent DESC
LIMIT 10;''',

    '''WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        product_id,
        SUM(quantity) as total_quantity,
        SUM(amount) as total_revenue
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY DATE_TRUNC('month', order_date), product_id
)
SELECT 
    p.name as product_name,
    ms.month,
    ms.total_quantity,
    ms.total_revenue,
    LAG(ms.total_revenue) OVER (PARTITION BY ms.product_id ORDER BY ms.month) as prev_month_revenue
FROM monthly_sales ms
JOIN products p ON ms.product_id = p.id
ORDER BY ms.month, ms.total_revenue DESC;''',
]

TERRAFORM_SAMPLES = [
    '''provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.project_name}-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-${count.index + 1}"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}''',

    '''resource "aws_eks_cluster" "main" {
  name     = "${var.project_name}-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = aws_subnet.private[*].id
    endpoint_private_access = true
    endpoint_public_access  = true
    security_group_ids      = [aws_security_group.eks_cluster.id]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy
  ]
}

resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.project_name}-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = var.node_instance_types

  scaling_config {
    desired_size = var.node_desired_size
    max_size     = var.node_max_size
    min_size     = var.node_min_size
  }
}''',

    '''resource "aws_db_instance" "main" {
  identifier           = "${var.project_name}-db"
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = var.db_instance_class
  allocated_storage    = var.db_allocated_storage
  storage_encrypted    = true
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  multi_az               = var.environment == "production"
  skip_final_snapshot    = var.environment != "production"

  tags = {
    Name        = "${var.project_name}-db"
    Environment = var.environment
  }
}''',
]

KUBERNETES_SAMPLES = [
    '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  labels:
    app: api-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
      - name: api
        image: myregistry/api:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5''',

    '''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60''',

    '''apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-server
            port:
              number: 80''',
]

DOCKER_SAMPLES = [
    '''FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
USER nodejs
EXPOSE 3000
CMD ["node", "dist/index.js"]''',

    '''FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:create_app()"]''',
]

def get_all_training_data():
    """Get all training samples as a single list."""
    all_samples = []
    all_samples.extend(PYTHON_SAMPLES)
    all_samples.extend(JAVASCRIPT_SAMPLES)
    all_samples.extend(TYPESCRIPT_SAMPLES)
    all_samples.extend(REACT_SAMPLES)
    all_samples.extend(SQL_SAMPLES)
    all_samples.extend(TERRAFORM_SAMPLES)
    all_samples.extend(KUBERNETES_SAMPLES)
    all_samples.extend(DOCKER_SAMPLES)
    return all_samples

def get_training_data_by_category():
    """Get training data organized by category."""
    return {
        'python': PYTHON_SAMPLES,
        'javascript': JAVASCRIPT_SAMPLES,
        'typescript': TYPESCRIPT_SAMPLES,
        'react': REACT_SAMPLES,
        'sql': SQL_SAMPLES,
        'terraform': TERRAFORM_SAMPLES,
        'kubernetes': KUBERNETES_SAMPLES,
        'docker': DOCKER_SAMPLES,
    }
