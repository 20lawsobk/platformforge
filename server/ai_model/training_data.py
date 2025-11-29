"""
Comprehensive Training Dataset for Code Generation Model
Contains diverse code samples across multiple languages and patterns
Organized into batches for efficient training
"""

import random
from typing import List, Dict

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

    '''def dijkstra(graph: dict, start: str) -> dict:
    """Find shortest paths using Dijkstra's algorithm."""
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances''',

    '''class Stack:
    """Implementation of a stack data structure."""
    
    def __init__(self):
        self._items = []
    
    def push(self, item):
        self._items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items[-1]
    
    def is_empty(self):
        return len(self._items) == 0
    
    def size(self):
        return len(self._items)''',

    '''def read_csv(filepath: str) -> list:
    """Read CSV file and return list of dictionaries."""
    import csv
    with open(filepath, 'r', newline='') as file:
        reader = csv.DictReader(file)
        return list(reader)

def write_csv(filepath: str, data: list, fieldnames: list):
    """Write list of dictionaries to CSV file."""
    import csv
    with open(filepath, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)''',

    '''class APIClient:
    """HTTP API client with retry logic."""
    
    def __init__(self, base_url: str, max_retries: int = 3):
        self.base_url = base_url
        self.max_retries = max_retries
        self.session = requests.Session()
    
    def get(self, endpoint: str, **kwargs) -> dict:
        return self._request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, data: dict = None, **kwargs) -> dict:
        return self._request('POST', endpoint, json=data, **kwargs)
    
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)''',

    '''def validate_json_schema(data: dict, schema: dict) -> bool:
    """Validate data against JSON schema."""
    def check_type(value, expected_type):
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        return isinstance(value, type_map.get(expected_type, object))
    
    for key, rules in schema.get('properties', {}).items():
        if key in schema.get('required', []) and key not in data:
            return False
        if key in data:
            if not check_type(data[key], rules.get('type')):
                return False
    return True''',

    '''class EventEmitter:
    """Simple event emitter pattern implementation."""
    
    def __init__(self):
        self._listeners = {}
    
    def on(self, event: str, callback):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs):
        if event in self._listeners:
            for callback in self._listeners[event]:
                callback(*args, **kwargs)
    
    def off(self, event: str, callback=None):
        if callback is None:
            self._listeners.pop(event, None)
        elif event in self._listeners:
            self._listeners[event] = [
                cb for cb in self._listeners[event] if cb != callback
            ]''',

    '''def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('input', type=str, help='Input file path')
    parser.add_argument('--output', '-o', type=str, default='output.txt',
                        help='Output file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--count', '-c', type=int, default=10,
                        help='Number of items to process')
    return parser.parse_args()''',

    '''class Singleton:
    """Singleton metaclass implementation."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=Singleton):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    def connect(self):
        if not self.connection:
            self.connection = create_connection(self.connection_string)
        return self.connection''',

    '''def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator''',

    '''class PriorityQueue:
    """Priority queue implementation using heapq."""
    
    def __init__(self):
        self._queue = []
        self._index = 0
    
    def push(self, item, priority: int):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return heapq.heappop(self._queue)[2]
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return self._queue[0][2]
    
    def is_empty(self):
        return len(self._queue) == 0''',
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

    '''const memoize = (fn) => {
    const cache = new Map();
    return (...args) => {
        const key = JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key);
        }
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
};''',

    '''class Router {
    constructor() {
        this.routes = new Map();
        window.addEventListener('popstate', () => this.handleRoute());
    }
    
    addRoute(path, handler) {
        this.routes.set(path, handler);
        return this;
    }
    
    navigate(path) {
        window.history.pushState({}, '', path);
        this.handleRoute();
    }
    
    handleRoute() {
        const path = window.location.pathname;
        const handler = this.routes.get(path);
        if (handler) {
            handler();
        }
    }
}''',

    '''async function* asyncGenerator(items) {
    for (const item of items) {
        await new Promise(resolve => setTimeout(resolve, 100));
        yield item;
    }
}

async function processItems(items) {
    const results = [];
    for await (const item of asyncGenerator(items)) {
        results.push(await processItem(item));
    }
    return results;
}''',

    '''class LocalStorageCache {
    constructor(prefix = 'cache_') {
        this.prefix = prefix;
    }
    
    set(key, value, ttl = 3600000) {
        const item = {
            value,
            expiry: Date.now() + ttl
        };
        localStorage.setItem(this.prefix + key, JSON.stringify(item));
    }
    
    get(key) {
        const item = localStorage.getItem(this.prefix + key);
        if (!item) return null;
        
        const parsed = JSON.parse(item);
        if (Date.now() > parsed.expiry) {
            localStorage.removeItem(this.prefix + key);
            return null;
        }
        return parsed.value;
    }
    
    clear() {
        Object.keys(localStorage)
            .filter(key => key.startsWith(this.prefix))
            .forEach(key => localStorage.removeItem(key));
    }
}''',

    '''function createStore(reducer, initialState) {
    let state = initialState;
    const listeners = [];
    
    return {
        getState: () => state,
        dispatch: (action) => {
            state = reducer(state, action);
            listeners.forEach(listener => listener());
        },
        subscribe: (listener) => {
            listeners.push(listener);
            return () => {
                const index = listeners.indexOf(listener);
                listeners.splice(index, 1);
            };
        }
    };
}''',

    '''class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('Connected');
            this.reconnectAttempts = 0;
        };
        
        this.ws.onclose = () => {
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                setTimeout(() => {
                    this.reconnectAttempts++;
                    this.connect();
                }, 1000 * Math.pow(2, this.reconnectAttempts));
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    send(data) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
}''',

    '''const pipe = (...fns) => (value) => 
    fns.reduce((acc, fn) => fn(acc), value);

const compose = (...fns) => (value) => 
    fns.reduceRight((acc, fn) => fn(acc), value);

const curry = (fn) => {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        }
        return (...nextArgs) => curried(...args, ...nextArgs);
    };
};''',
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

    '''interface HttpClient {
    get<T>(url: string): Promise<T>;
    post<T>(url: string, data: unknown): Promise<T>;
    put<T>(url: string, data: unknown): Promise<T>;
    delete<T>(url: string): Promise<T>;
}

class FetchHttpClient implements HttpClient {
    constructor(private baseUrl: string) {}
    
    async get<T>(url: string): Promise<T> {
        const response = await fetch(`${this.baseUrl}${url}`);
        return response.json();
    }
    
    async post<T>(url: string, data: unknown): Promise<T> {
        const response = await fetch(`${this.baseUrl}${url}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    }
    
    async put<T>(url: string, data: unknown): Promise<T> {
        const response = await fetch(`${this.baseUrl}${url}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    }
    
    async delete<T>(url: string): Promise<T> {
        const response = await fetch(`${this.baseUrl}${url}`, { method: 'DELETE' });
        return response.json();
    }
}''',

    '''type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type DeepReadonly<T> = {
    readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

function deepMerge<T extends object>(target: T, source: DeepPartial<T>): T {
    const result = { ...target };
    for (const key in source) {
        const sourceValue = source[key];
        const targetValue = target[key];
        if (sourceValue && typeof sourceValue === 'object' && !Array.isArray(sourceValue)) {
            result[key] = deepMerge(targetValue as object, sourceValue) as T[typeof key];
        } else if (sourceValue !== undefined) {
            result[key] = sourceValue as T[typeof key];
        }
    }
    return result;
}''',

    '''class EventBus<Events extends Record<string, unknown>> {
    private listeners = new Map<keyof Events, Set<(data: unknown) => void>>();
    
    on<K extends keyof Events>(event: K, callback: (data: Events[K]) => void): () => void {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(callback as (data: unknown) => void);
        return () => this.off(event, callback);
    }
    
    off<K extends keyof Events>(event: K, callback: (data: Events[K]) => void): void {
        this.listeners.get(event)?.delete(callback as (data: unknown) => void);
    }
    
    emit<K extends keyof Events>(event: K, data: Events[K]): void {
        this.listeners.get(event)?.forEach(cb => cb(data));
    }
}''',

    '''interface Validator<T> {
    validate(value: unknown): value is T;
    parse(value: unknown): T;
}

const string = (): Validator<string> => ({
    validate: (value): value is string => typeof value === 'string',
    parse: (value) => {
        if (typeof value !== 'string') throw new Error('Expected string');
        return value;
    }
});

const number = (): Validator<number> => ({
    validate: (value): value is number => typeof value === 'number',
    parse: (value) => {
        if (typeof value !== 'number') throw new Error('Expected number');
        return value;
    }
});

const object = <T extends Record<string, Validator<unknown>>>(schema: T): Validator<{[K in keyof T]: T[K] extends Validator<infer U> ? U : never}> => ({
    validate: (value): value is {[K in keyof T]: T[K] extends Validator<infer U> ? U : never} => {
        if (typeof value !== 'object' || value === null) return false;
        return Object.entries(schema).every(([key, validator]) => validator.validate((value as Record<string, unknown>)[key]));
    },
    parse: (value) => {
        if (typeof value !== 'object' || value === null) throw new Error('Expected object');
        const result: Record<string, unknown> = {};
        for (const [key, validator] of Object.entries(schema)) {
            result[key] = validator.parse((value as Record<string, unknown>)[key]);
        }
        return result as {[K in keyof T]: T[K] extends Validator<infer U> ? U : never};
    }
});''',
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

    '''function useLocalStorage<T>(key: string, initialValue: T) {
    const [storedValue, setStoredValue] = useState<T>(() => {
        try {
            const item = window.localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch (error) {
            return initialValue;
        }
    });
    
    const setValue = useCallback((value: T | ((val: T) => T)) => {
        try {
            const valueToStore = value instanceof Function ? value(storedValue) : value;
            setStoredValue(valueToStore);
            window.localStorage.setItem(key, JSON.stringify(valueToStore));
        } catch (error) {
            console.error(error);
        }
    }, [key, storedValue]);
    
    return [storedValue, setValue] as const;
}''',

    '''const Modal = ({ isOpen, onClose, title, children }) => {
    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape') onClose();
        };
        
        if (isOpen) {
            document.addEventListener('keydown', handleEscape);
            document.body.style.overflow = 'hidden';
        }
        
        return () => {
            document.removeEventListener('keydown', handleEscape);
            document.body.style.overflow = 'unset';
        };
    }, [isOpen, onClose]);
    
    if (!isOpen) return null;
    
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2>{title}</h2>
                    <button onClick={onClose}>&times;</button>
                </div>
                <div className="modal-body">{children}</div>
            </div>
        </div>
    );
};''',

    '''function useMediaQuery(query: string): boolean {
    const [matches, setMatches] = useState(false);
    
    useEffect(() => {
        const media = window.matchMedia(query);
        if (media.matches !== matches) {
            setMatches(media.matches);
        }
        
        const listener = () => setMatches(media.matches);
        media.addEventListener('change', listener);
        return () => media.removeEventListener('change', listener);
    }, [matches, query]);
    
    return matches;
}

const ResponsiveComponent = () => {
    const isMobile = useMediaQuery('(max-width: 768px)');
    const isTablet = useMediaQuery('(min-width: 769px) and (max-width: 1024px)');
    
    return (
        <div>
            {isMobile && <MobileLayout />}
            {isTablet && <TabletLayout />}
            {!isMobile && !isTablet && <DesktopLayout />}
        </div>
    );
};''',

    '''const InfiniteScroll = ({ loadMore, hasMore, children }) => {
    const observerRef = useRef(null);
    const loadingRef = useRef(null);
    
    useEffect(() => {
        const options = {
            root: null,
            rootMargin: '100px',
            threshold: 0.1
        };
        
        observerRef.current = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && hasMore) {
                loadMore();
            }
        }, options);
        
        if (loadingRef.current) {
            observerRef.current.observe(loadingRef.current);
        }
        
        return () => {
            if (observerRef.current) {
                observerRef.current.disconnect();
            }
        };
    }, [loadMore, hasMore]);
    
    return (
        <div>
            {children}
            <div ref={loadingRef}>
                {hasMore && <div>Loading more...</div>}
            </div>
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

    '''CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10, 2) NOT NULL,
    shipping_address JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL
);

CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);''',

    '''WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id, 0 as depth
    FROM categories
    WHERE parent_id IS NULL
    UNION ALL
    SELECT c.id, c.name, c.parent_id, ct.depth + 1
    FROM categories c
    INNER JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree ORDER BY depth, name;''',

    '''INSERT INTO products (name, price, category_id, stock_quantity)
VALUES 
    ('Product A', 29.99, 1, 100),
    ('Product B', 49.99, 1, 50),
    ('Product C', 19.99, 2, 200)
ON CONFLICT (name) 
DO UPDATE SET 
    price = EXCLUDED.price,
    stock_quantity = products.stock_quantity + EXCLUDED.stock_quantity;''',

    '''CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_timestamp
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();''',
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

    '''resource "aws_lambda_function" "api" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.project_name}-api"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "index.handler"
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  runtime          = "nodejs18.x"
  timeout          = 30
  memory_size      = 256

  environment {
    variables = {
      DATABASE_URL = aws_db_instance.main.endpoint
      NODE_ENV     = var.environment
    }
  }

  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }
}

resource "aws_api_gateway_rest_api" "main" {
  name        = "${var.project_name}-api"
  description = "API Gateway for ${var.project_name}"
}''',

    '''resource "aws_s3_bucket" "static" {
  bucket = "${var.project_name}-static-${var.environment}"

  tags = {
    Name        = "${var.project_name}-static"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_public_access_block" "static" {
  bucket = aws_s3_bucket.static.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudfront_distribution" "static" {
  enabled             = true
  default_root_object = "index.html"

  origin {
    domain_name = aws_s3_bucket.static.bucket_regional_domain_name
    origin_id   = "S3Origin"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.static.cloudfront_access_identity_path
    }
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3Origin"
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}''',

    '''resource "google_compute_instance" "web" {
  name         = "${var.project_name}-web"
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 50
    }
  }

  network_interface {
    network    = google_compute_network.main.id
    subnetwork = google_compute_subnetwork.public.id

    access_config {
      // Ephemeral public IP
    }
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io
    systemctl start docker
    docker pull ${var.docker_image}
    docker run -d -p 80:8080 ${var.docker_image}
  EOF

  service_account {
    scopes = ["cloud-platform"]
  }

  tags = ["http-server", "https-server"]
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

    '''apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  APP_ENV: "production"
  LOG_LEVEL: "info"
  CACHE_TTL: "3600"
---
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData:
  url: "postgresql://user:password@db:5432/mydb"
  username: "user"
  password: "password"''',

    '''apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-job
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup-image:latest
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
            - name: S3_BUCKET
              value: "backups-bucket"
          restartPolicy: OnFailure
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1''',

    '''apiVersion: v1
kind: Service
metadata:
  name: api-service
spec:
  type: ClusterIP
  selector:
    app: api-server
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: api-external
spec:
  type: LoadBalancer
  selector:
    app: api-server
  ports:
  - port: 443
    targetPort: 8080''',
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

    '''version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:''',

    '''FROM rust:1.70 AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

COPY src ./src
RUN touch src/main.rs
RUN cargo build --release

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/myapp /usr/local/bin/
EXPOSE 8080
CMD ["myapp"]''',
]


def get_all_training_data(shuffle: bool = True, seed: int = 42) -> List[str]:
    """Get all training samples as a single list with optional shuffling."""
    all_samples = []
    all_samples.extend(PYTHON_SAMPLES)
    all_samples.extend(JAVASCRIPT_SAMPLES)
    all_samples.extend(TYPESCRIPT_SAMPLES)
    all_samples.extend(REACT_SAMPLES)
    all_samples.extend(SQL_SAMPLES)
    all_samples.extend(TERRAFORM_SAMPLES)
    all_samples.extend(KUBERNETES_SAMPLES)
    all_samples.extend(DOCKER_SAMPLES)
    
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_samples)
    
    return all_samples


def get_training_data_by_category() -> Dict[str, List[str]]:
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


def create_training_batches(batch_size: int = 8, shuffle: bool = True, seed: int = 42) -> List[List[str]]:
    """Create organized training batches from all samples."""
    all_data = get_all_training_data(shuffle=shuffle, seed=seed)
    batches = []
    
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]
        if batch:
            batches.append(batch)
    
    return batches


def get_training_stats() -> Dict:
    """Get statistics about the training data."""
    categories = get_training_data_by_category()
    total_samples = sum(len(samples) for samples in categories.values())
    total_chars = sum(sum(len(s) for s in samples) for samples in categories.values())
    
    return {
        'total_samples': total_samples,
        'total_characters': total_chars,
        'avg_sample_length': total_chars // total_samples if total_samples > 0 else 0,
        'categories': {name: len(samples) for name, samples in categories.items()},
        'batch_count_8': (total_samples + 7) // 8,
        'batch_count_4': (total_samples + 3) // 4,
    }
