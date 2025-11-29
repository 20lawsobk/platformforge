"""
Comprehensive Training Dataset for Code Generation Model
Maximum batches with diverse code samples across multiple languages
"""

import random
from typing import List, Dict

PYTHON_SAMPLES = [
    # Algorithms
    '''def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[-1] + dp[-2])
    return dp[n]''',

    '''def binary_search(arr: list, target: int) -> int:
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

    '''def quicksort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)''',

    '''def merge_sort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)''',

    '''def dijkstra(graph: dict, start: str) -> dict:
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

    '''def bfs(graph: dict, start: str) -> list:
    visited = set([start])
    queue = deque([start])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result''',

    '''def dfs(graph: dict, start: str, visited=None) -> list:
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    return result''',

    '''def kadane_max_subarray(arr: list) -> int:
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum''',

    '''def longest_common_subsequence(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]''',

    '''def knapsack(weights: list, values: list, capacity: int) -> int:
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]''',

    # Data Structures
    '''class LinkedList:
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
        self.size += 1''',

    '''class Stack:
    def __init__(self):
        self._items = []
    
    def push(self, item):
        self._items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items.pop()
    
    def peek(self):
        return self._items[-1] if self._items else None
    
    def is_empty(self):
        return len(self._items) == 0''',

    '''class Queue:
    def __init__(self):
        self._items = deque()
    
    def enqueue(self, item):
        self._items.append(item)
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items.popleft()
    
    def is_empty(self):
        return len(self._items) == 0''',

    '''class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.val:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)''',

    '''class MinHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)
    
    def pop(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return root''',

    '''class HashTable:
    def __init__(self, size=100):
        self.size = size
        self.buckets = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def set(self, key, value):
        index = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[index]):
            if k == key:
                self.buckets[index][i] = (key, value)
                return
        self.buckets[index].append((key, value))
    
    def get(self, key):
        index = self._hash(key)
        for k, v in self.buckets[index]:
            if k == key:
                return v
        return None''',

    '''class Trie:
    def __init__(self):
        self.root = {}
    
    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = True
    
    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return '$' in node''',

    '''class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)''',

    # Async and IO
    '''async def fetch_data(url: str, session: aiohttp.ClientSession) -> dict:
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        raise HTTPError(f"Request failed: {response.status}")''',

    '''async def fetch_all(urls: list) -> list:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(url, session) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)''',

    '''def read_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath: str, content: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)''',

    '''def read_json(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def write_json(filepath: str, data: dict):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)''',

    '''def read_csv(filepath: str) -> list:
    import csv
    with open(filepath, 'r', newline='') as file:
        reader = csv.DictReader(file)
        return list(reader)''',

    # Classes and OOP
    '''class DatabaseConnection:
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

    '''@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.validate_email(self.email):
            raise ValueError(f"Invalid email: {self.email}")
    
    @staticmethod
    def validate_email(email: str) -> bool:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))''',

    '''class APIClient:
    def __init__(self, base_url: str, max_retries: int = 3):
        self.base_url = base_url
        self.max_retries = max_retries
        self.session = requests.Session()
    
    def get(self, endpoint: str, **kwargs) -> dict:
        return self._request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, data: dict = None, **kwargs) -> dict:
        return self._request('POST', endpoint, json=data, **kwargs)''',

    '''class EventEmitter:
    def __init__(self):
        self._listeners = {}
    
    def on(self, event: str, callback):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs):
        if event in self._listeners:
            for callback in self._listeners[event]:
                callback(*args, **kwargs)''',

    '''class Singleton:
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]''',

    # Decorators and Functional
    '''def lru_cache(maxsize: int = 128):
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

    '''def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
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

    '''def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper''',

    '''def validate_types(*types):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for arg, expected_type in zip(args, types):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Expected {expected_type}, got {type(arg)}")
            return func(*args, **kwargs)
        return wrapper
    return decorator''',

    '''def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper''',

    # Utilities
    '''def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('input', type=str, help='Input file path')
    parser.add_argument('--output', '-o', type=str, default='output.txt')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()''',

    '''def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)''',

    '''def validate_json_schema(data: dict, schema: dict) -> bool:
    def check_type(value, expected_type):
        type_map = {
            'string': str, 'number': (int, float), 'integer': int,
            'boolean': bool, 'array': list, 'object': dict
        }
        return isinstance(value, type_map.get(expected_type, object))
    
    for key, rules in schema.get('properties', {}).items():
        if key in schema.get('required', []) and key not in data:
            return False
        if key in data and not check_type(data[key], rules.get('type')):
            return False
    return True''',

    '''def chunk_list(lst: list, chunk_size: int) -> list:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]''',

    '''def flatten(nested_list: list) -> list:
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result''',

    '''def deep_merge(dict1: dict, dict2: dict) -> dict:
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result''',

    '''def generate_uuid() -> str:
    import uuid
    return str(uuid.uuid4())''',

    '''def hash_password(password: str) -> str:
    import hashlib
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt.hex() + key.hex()''',

    '''def verify_password(password: str, hashed: str) -> bool:
    import hashlib
    salt = bytes.fromhex(hashed[:64])
    stored_key = hashed[64:]
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return key.hex() == stored_key''',

    '''def paginate(items: list, page: int, per_page: int) -> dict:
    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    return {
        'items': items[start:end],
        'page': page,
        'per_page': per_page,
        'total': total,
        'pages': (total + per_page - 1) // per_page
    }''',

    '''class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        self.requests[key] = [t for t in self.requests[key] if now - t < self.window]
        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return True
        return False''',

    '''def create_jwt(payload: dict, secret: str, exp_hours: int = 24) -> str:
    import jwt
    payload['exp'] = datetime.utcnow() + timedelta(hours=exp_hours)
    return jwt.encode(payload, secret, algorithm='HS256')

def verify_jwt(token: str, secret: str) -> dict:
    import jwt
    return jwt.decode(token, secret, algorithms=['HS256'])''',

    '''class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = None
        self.state = 'closed'
    
    def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure > self.reset_timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        try:
            result = func(*args, **kwargs)
            self.failures = 0
            self.state = 'closed'
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.failure_threshold:
                self.state = 'open'
            raise''',
]

JAVASCRIPT_SAMPLES = [
    '''async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
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
        if (!this.events.has(event)) this.events.set(event, []);
        this.events.get(event).push(callback);
        return this;
    }
    
    emit(event, ...args) {
        if (this.events.has(event)) {
            this.events.get(event).forEach(cb => cb(...args));
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
};''',

    '''const throttle = (fn, limit) => {
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
    if (obj === null || typeof obj !== 'object') return obj;
    if (Array.isArray(obj)) return obj.map(item => deepClone(item));
    const cloned = {};
    for (const key in obj) {
        if (obj.hasOwnProperty(key)) cloned[key] = deepClone(obj[key]);
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
}''',

    '''const memoize = (fn) => {
    const cache = new Map();
    return (...args) => {
        const key = JSON.stringify(args);
        if (cache.has(key)) return cache.get(key);
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
        const handler = this.routes.get(window.location.pathname);
        if (handler) handler();
    }
}''',

    '''async function* asyncGenerator(items) {
    for (const item of items) {
        await new Promise(resolve => setTimeout(resolve, 100));
        yield item;
    }
}''',

    '''class LocalStorageCache {
    constructor(prefix = 'cache_') {
        this.prefix = prefix;
    }
    
    set(key, value, ttl = 3600000) {
        const item = { value, expiry: Date.now() + ttl };
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
            return () => listeners.splice(listeners.indexOf(listener), 1);
        }
    };
}''',

    '''class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
    }
    
    connect() {
        this.ws = new WebSocket(this.url);
        this.ws.onopen = () => { this.reconnectAttempts = 0; };
        this.ws.onclose = () => this.reconnect();
    }
    
    reconnect() {
        if (this.reconnectAttempts < 5) {
            setTimeout(() => {
                this.reconnectAttempts++;
                this.connect();
            }, 1000 * Math.pow(2, this.reconnectAttempts));
        }
    }
    
    send(data) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
}''',

    '''const pipe = (...fns) => (value) => fns.reduce((acc, fn) => fn(acc), value);
const compose = (...fns) => (value) => fns.reduceRight((acc, fn) => fn(acc), value);''',

    '''const curry = (fn) => {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        }
        return (...nextArgs) => curried(...args, ...nextArgs);
    };
};''',

    '''function promiseAll(promises) {
    return new Promise((resolve, reject) => {
        const results = [];
        let completed = 0;
        promises.forEach((promise, index) => {
            Promise.resolve(promise)
                .then(value => {
                    results[index] = value;
                    completed++;
                    if (completed === promises.length) resolve(results);
                })
                .catch(reject);
        });
    });
}''',

    '''function retry(fn, maxRetries = 3, delay = 1000) {
    return new Promise((resolve, reject) => {
        const attempt = (retries) => {
            fn()
                .then(resolve)
                .catch(err => {
                    if (retries > 0) {
                        setTimeout(() => attempt(retries - 1), delay);
                    } else {
                        reject(err);
                    }
                });
        };
        attempt(maxRetries);
    });
}''',

    '''class PubSub {
    constructor() {
        this.subscribers = {};
    }
    
    subscribe(topic, callback) {
        if (!this.subscribers[topic]) this.subscribers[topic] = [];
        this.subscribers[topic].push(callback);
        return () => this.unsubscribe(topic, callback);
    }
    
    publish(topic, data) {
        if (this.subscribers[topic]) {
            this.subscribers[topic].forEach(cb => cb(data));
        }
    }
    
    unsubscribe(topic, callback) {
        if (this.subscribers[topic]) {
            this.subscribers[topic] = this.subscribers[topic].filter(cb => cb !== callback);
        }
    }
}''',

    '''const flattenObject = (obj, prefix = '') => {
    return Object.keys(obj).reduce((acc, key) => {
        const newKey = prefix ? `${prefix}.${key}` : key;
        if (typeof obj[key] === 'object' && obj[key] !== null && !Array.isArray(obj[key])) {
            Object.assign(acc, flattenObject(obj[key], newKey));
        } else {
            acc[newKey] = obj[key];
        }
        return acc;
    }, {});
};''',

    '''function createAsyncQueue(concurrency = 1) {
    const queue = [];
    let running = 0;
    
    const process = async () => {
        if (running >= concurrency || queue.length === 0) return;
        running++;
        const { task, resolve, reject } = queue.shift();
        try {
            resolve(await task());
        } catch (e) {
            reject(e);
        } finally {
            running--;
            process();
        }
    };
    
    return (task) => new Promise((resolve, reject) => {
        queue.push({ task, resolve, reject });
        process();
    });
}''',

    '''class LazyLoader {
    constructor() {
        this.observer = new IntersectionObserver(this.handleIntersect.bind(this));
    }
    
    observe(element) {
        this.observer.observe(element);
    }
    
    handleIntersect(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                this.observer.unobserve(img);
            }
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
}''',

    '''type Result<T, E = Error> = 
    | { success: true; data: T }
    | { success: false; error: E };

async function tryAsync<T>(fn: () => Promise<T>): Promise<Result<T>> {
    try {
        return { success: true, data: await fn() };
    } catch (error) {
        return { success: false, error: error as Error };
    }
}''',

    '''function createStore<T>(initialState: T) {
    let state = initialState;
    const listeners: Set<(state: T) => void> = new Set();
    
    return {
        getState: () => state,
        setState: (updater: T | ((prev: T) => T)) => {
            state = typeof updater === 'function' 
                ? (updater as (prev: T) => T)(state) : updater;
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
    
    enqueue(item: T): void { this.items.push(item); }
    dequeue(): T | undefined { return this.items.shift(); }
    peek(): T | undefined { return this.items[0]; }
    get size(): number { return this.items.length; }
    isEmpty(): boolean { return this.items.length === 0; }
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
}''',

    '''type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type DeepReadonly<T> = {
    readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};''',

    '''class EventBus<Events extends Record<string, unknown>> {
    private listeners = new Map<keyof Events, Set<(data: unknown) => void>>();
    
    on<K extends keyof Events>(event: K, callback: (data: Events[K]) => void): () => void {
        if (!this.listeners.has(event)) this.listeners.set(event, new Set());
        this.listeners.get(event)!.add(callback as (data: unknown) => void);
        return () => this.off(event, callback);
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
});''',

    '''type Prettify<T> = { [K in keyof T]: T[K] } & {};

type Merge<T, U> = Prettify<Omit<T, keyof U> & U>;

type Optional<T, K extends keyof T> = Prettify<Omit<T, K> & Partial<Pick<T, K>>>;

type Required<T, K extends keyof T> = Prettify<Omit<T, K> & Required<Pick<T, K>>>;''',

    '''class AsyncQueue<T> {
    private queue: T[] = [];
    private resolvers: ((value: T) => void)[] = [];
    
    async dequeue(): Promise<T> {
        if (this.queue.length > 0) {
            return this.queue.shift()!;
        }
        return new Promise(resolve => this.resolvers.push(resolve));
    }
    
    enqueue(item: T): void {
        if (this.resolvers.length > 0) {
            const resolve = this.resolvers.shift()!;
            resolve(item);
        } else {
            this.queue.push(item);
        }
    }
}''',

    '''interface Plugin<T = unknown> {
    name: string;
    version: string;
    install(app: Application, options?: T): void;
}

class PluginManager {
    private plugins: Map<string, Plugin> = new Map();
    
    register<T>(plugin: Plugin<T>, options?: T): void {
        if (this.plugins.has(plugin.name)) {
            throw new Error(`Plugin ${plugin.name} already registered`);
        }
        this.plugins.set(plugin.name, plugin);
        plugin.install(this.app, options);
    }
}''',

    '''type Action<T extends string, P = undefined> = P extends undefined
    ? { type: T }
    : { type: T; payload: P };

type ActionCreator<T extends string, P = undefined> = P extends undefined
    ? () => Action<T>
    : (payload: P) => Action<T, P>;

function createAction<T extends string>(type: T): ActionCreator<T>;
function createAction<T extends string, P>(type: T): ActionCreator<T, P>;
function createAction<T extends string, P>(type: T): ActionCreator<T, P> {
    return ((payload?: P) => ({ type, payload })) as ActionCreator<T, P>;
}''',

    '''class StateMachine<S extends string, E extends string> {
    private state: S;
    private transitions: Map<S, Map<E, S>> = new Map();
    
    constructor(initialState: S) {
        this.state = initialState;
    }
    
    addTransition(from: S, event: E, to: S): this {
        if (!this.transitions.has(from)) {
            this.transitions.set(from, new Map());
        }
        this.transitions.get(from)!.set(event, to);
        return this;
    }
    
    send(event: E): S {
        const nextState = this.transitions.get(this.state)?.get(event);
        if (nextState) this.state = nextState;
        return this.state;
    }
}''',

    '''interface Middleware<T> {
    (context: T, next: () => Promise<void>): Promise<void>;
}

class MiddlewareChain<T> {
    private middlewares: Middleware<T>[] = [];
    
    use(middleware: Middleware<T>): this {
        this.middlewares.push(middleware);
        return this;
    }
    
    async execute(context: T): Promise<void> {
        let index = 0;
        const next = async (): Promise<void> => {
            if (index < this.middlewares.length) {
                const middleware = this.middlewares[index++];
                await middleware(context, next);
            }
        };
        await next();
    }
}''',

    '''type Awaited<T> = T extends Promise<infer U> ? Awaited<U> : T;

type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;

type PromiseValue<T extends Promise<unknown>> = T extends Promise<infer U> ? U : never;

async function parallel<T extends readonly unknown[] | []>(
    promises: T
): Promise<{ -readonly [P in keyof T]: Awaited<T[P]> }> {
    return Promise.all(promises) as Promise<{ -readonly [P in keyof T]: Awaited<T[P]> }>;
}''',
]

REACT_SAMPLES = [
    '''function useDebounce<T>(value: T, delay: number): T {
    const [debouncedValue, setDebouncedValue] = useState(value);
    
    useEffect(() => {
        const handler = setTimeout(() => setDebouncedValue(value), delay);
        return () => clearTimeout(handler);
    }, [value, delay]);
    
    return debouncedValue;
}''',

    '''function useFetch<T>(url: string) {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<Error | null>(null);
    
    useEffect(() => {
        const controller = new AbortController();
        fetch(url, { signal: controller.signal })
            .then(res => res.json())
            .then(setData)
            .catch(err => { if (err.name !== 'AbortError') setError(err); })
            .finally(() => setLoading(false));
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
    
    return (
        <div>
            <input value={input} onChange={e => setInput(e.target.value)} />
            <button onClick={addTodo}>Add</button>
            <ul>
                {todos.map(todo => (
                    <li key={todo.id} onClick={() => toggleTodo(todo.id)}>
                        {todo.text}
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
        } catch { return initialValue; }
    });
    
    const setValue = useCallback((value: T | ((val: T) => T)) => {
        const valueToStore = value instanceof Function ? value(storedValue) : value;
        setStoredValue(valueToStore);
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
    }, [key, storedValue]);
    
    return [storedValue, setValue] as const;
}''',

    '''const Modal = ({ isOpen, onClose, title, children }) => {
    useEffect(() => {
        const handleEscape = (e) => { if (e.key === 'Escape') onClose(); };
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
                <h2>{title}</h2>
                {children}
            </div>
        </div>
    );
};''',

    '''function useMediaQuery(query: string): boolean {
    const [matches, setMatches] = useState(false);
    
    useEffect(() => {
        const media = window.matchMedia(query);
        setMatches(media.matches);
        const listener = () => setMatches(media.matches);
        media.addEventListener('change', listener);
        return () => media.removeEventListener('change', listener);
    }, [query]);
    
    return matches;
}''',

    '''const InfiniteScroll = ({ loadMore, hasMore, children }) => {
    const loadingRef = useRef(null);
    
    useEffect(() => {
        const observer = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && hasMore) loadMore();
        }, { rootMargin: '100px' });
        
        if (loadingRef.current) observer.observe(loadingRef.current);
        return () => observer.disconnect();
    }, [loadMore, hasMore]);
    
    return (
        <div>
            {children}
            <div ref={loadingRef}>{hasMore && 'Loading...'}</div>
        </div>
    );
};''',

    '''function useOnClickOutside<T extends HTMLElement>(
    ref: React.RefObject<T>,
    handler: (event: MouseEvent | TouchEvent) => void
) {
    useEffect(() => {
        const listener = (event: MouseEvent | TouchEvent) => {
            if (!ref.current || ref.current.contains(event.target as Node)) return;
            handler(event);
        };
        document.addEventListener('mousedown', listener);
        document.addEventListener('touchstart', listener);
        return () => {
            document.removeEventListener('mousedown', listener);
            document.removeEventListener('touchstart', listener);
        };
    }, [ref, handler]);
}''',

    '''function usePrevious<T>(value: T): T | undefined {
    const ref = useRef<T>();
    useEffect(() => { ref.current = value; }, [value]);
    return ref.current;
}''',

    '''function useInterval(callback: () => void, delay: number | null) {
    const savedCallback = useRef(callback);
    
    useEffect(() => { savedCallback.current = callback; }, [callback]);
    
    useEffect(() => {
        if (delay === null) return;
        const id = setInterval(() => savedCallback.current(), delay);
        return () => clearInterval(id);
    }, [delay]);
}''',

    '''function useAsync<T>(asyncFn: () => Promise<T>, deps: unknown[] = []) {
    const [state, setState] = useState<{
        loading: boolean;
        error: Error | null;
        data: T | null;
    }>({ loading: true, error: null, data: null });
    
    useEffect(() => {
        setState(s => ({ ...s, loading: true }));
        asyncFn()
            .then(data => setState({ loading: false, error: null, data }))
            .catch(error => setState({ loading: false, error, data: null }));
    }, deps);
    
    return state;
}''',

    '''function useForm<T extends Record<string, unknown>>(initialValues: T) {
    const [values, setValues] = useState(initialValues);
    const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
    
    const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setValues(prev => ({ ...prev, [name]: value }));
    }, []);
    
    const reset = useCallback(() => {
        setValues(initialValues);
        setErrors({});
    }, [initialValues]);
    
    return { values, errors, handleChange, reset, setErrors };
}''',

    '''const VirtualList = ({ items, itemHeight, containerHeight, renderItem }) => {
    const [scrollTop, setScrollTop] = useState(0);
    
    const startIndex = Math.floor(scrollTop / itemHeight);
    const endIndex = Math.min(
        startIndex + Math.ceil(containerHeight / itemHeight) + 1,
        items.length
    );
    
    const visibleItems = items.slice(startIndex, endIndex);
    const offsetY = startIndex * itemHeight;
    
    return (
        <div
            style={{ height: containerHeight, overflow: 'auto' }}
            onScroll={e => setScrollTop(e.currentTarget.scrollTop)}
        >
            <div style={{ height: items.length * itemHeight, position: 'relative' }}>
                <div style={{ transform: `translateY(${offsetY}px)` }}>
                    {visibleItems.map((item, i) => renderItem(item, startIndex + i))}
                </div>
            </div>
        </div>
    );
};''',

    '''function useThrottle<T>(value: T, limit: number): T {
    const [throttledValue, setThrottledValue] = useState(value);
    const lastRan = useRef(Date.now());
    
    useEffect(() => {
        const handler = setTimeout(() => {
            if (Date.now() - lastRan.current >= limit) {
                setThrottledValue(value);
                lastRan.current = Date.now();
            }
        }, limit - (Date.now() - lastRan.current));
        
        return () => clearTimeout(handler);
    }, [value, limit]);
    
    return throttledValue;
}''',

    '''const ErrorBoundary = ({ children, fallback }) => {
    const [hasError, setHasError] = useState(false);
    
    useEffect(() => {
        const errorHandler = () => setHasError(true);
        window.addEventListener('error', errorHandler);
        return () => window.removeEventListener('error', errorHandler);
    }, []);
    
    if (hasError) return fallback;
    return children;
};''',
]

SQL_SAMPLES = [
    '''CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_users_email ON users(email);''',

    '''SELECT u.username, COUNT(o.id) as order_count, SUM(o.total_amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY u.id
HAVING COUNT(o.id) > 0
ORDER BY total_spent DESC
LIMIT 10;''',

    '''WITH monthly_sales AS (
    SELECT DATE_TRUNC('month', order_date) as month, product_id,
           SUM(quantity) as total_qty, SUM(amount) as revenue
    FROM orders WHERE order_date >= '2024-01-01'
    GROUP BY 1, 2
)
SELECT p.name, ms.month, ms.revenue,
       LAG(ms.revenue) OVER (PARTITION BY ms.product_id ORDER BY ms.month) as prev_revenue
FROM monthly_sales ms JOIN products p ON ms.product_id = p.id;''',

    '''CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);''',

    '''WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id, 0 as depth FROM categories WHERE parent_id IS NULL
    UNION ALL
    SELECT c.id, c.name, c.parent_id, ct.depth + 1
    FROM categories c INNER JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree ORDER BY depth, name;''',

    '''INSERT INTO products (name, price, category_id, stock)
VALUES ('Product A', 29.99, 1, 100), ('Product B', 49.99, 1, 50)
ON CONFLICT (name) DO UPDATE SET price = EXCLUDED.price, stock = products.stock + EXCLUDED.stock;''',

    '''CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = CURRENT_TIMESTAMP; RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_timestamp
BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at();''',

    '''SELECT p.name, p.price,
    RANK() OVER (ORDER BY p.price DESC) as price_rank,
    DENSE_RANK() OVER (PARTITION BY p.category_id ORDER BY p.price DESC) as category_rank
FROM products p;''',

    '''CREATE MATERIALIZED VIEW daily_sales AS
SELECT DATE(order_date) as sale_date, COUNT(*) as orders, SUM(total_amount) as revenue
FROM orders WHERE status = 'completed'
GROUP BY DATE(order_date);''',

    '''SELECT customer_id, order_date, total_amount,
    SUM(total_amount) OVER (PARTITION BY customer_id ORDER BY order_date) as running_total,
    AVG(total_amount) OVER (PARTITION BY customer_id) as avg_order
FROM orders;''',

    '''CREATE INDEX CONCURRENTLY idx_orders_status_date ON orders(status, created_at DESC);
CREATE INDEX idx_products_category ON products(category_id) WHERE active = true;''',

    '''SELECT COALESCE(c.name, 'Uncategorized') as category,
    COUNT(p.id) as product_count, AVG(p.price) as avg_price
FROM products p LEFT JOIN categories c ON p.category_id = c.id
GROUP BY ROLLUP(c.name);''',

    '''WITH order_stats AS (
    SELECT user_id, COUNT(*) as order_count, SUM(total_amount) as total_spent
    FROM orders GROUP BY user_id
)
UPDATE users u SET loyalty_points = os.total_spent * 10
FROM order_stats os WHERE u.id = os.user_id AND os.order_count >= 5;''',

    '''SELECT DATE_TRUNC('week', created_at) as week,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled
FROM orders GROUP BY 1 ORDER BY 1;''',

    '''CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    record_id INTEGER NOT NULL,
    action VARCHAR(10) NOT NULL,
    old_data JSONB,
    new_data JSONB,
    changed_by INTEGER REFERENCES users(id),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);''',
]

TERRAFORM_SAMPLES = [
    '''provider "aws" { region = var.aws_region }

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  tags = { Name = "${var.project_name}-vpc" }
}''',

    '''resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "${var.project_name}-public-${count.index + 1}" }
}''',

    '''resource "aws_eks_cluster" "main" {
  name     = "${var.project_name}-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version
  vpc_config {
    subnet_ids              = aws_subnet.private[*].id
    endpoint_private_access = true
    endpoint_public_access  = true
  }
}''',

    '''resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.project_name}-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  scaling_config {
    desired_size = var.node_desired_size
    max_size     = var.node_max_size
    min_size     = var.node_min_size
  }
}''',

    '''resource "aws_db_instance" "main" {
  identifier        = "${var.project_name}-db"
  engine            = "postgres"
  engine_version    = "15.3"
  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  storage_encrypted = true
  db_name           = var.db_name
  username          = var.db_username
  password          = var.db_password
  multi_az          = var.environment == "production"
}''',

    '''resource "aws_lambda_function" "api" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.project_name}-api"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "index.handler"
  runtime          = "nodejs18.x"
  timeout          = 30
  memory_size      = 256
  environment { variables = { DATABASE_URL = aws_db_instance.main.endpoint } }
}''',

    '''resource "aws_s3_bucket" "static" {
  bucket = "${var.project_name}-static-${var.environment}"
  tags = { Name = "${var.project_name}-static", Environment = var.environment }
}

resource "aws_s3_bucket_public_access_block" "static" {
  bucket                  = aws_s3_bucket.static.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}''',

    '''resource "aws_cloudfront_distribution" "static" {
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
  }
}''',

    '''resource "google_compute_instance" "web" {
  name         = "${var.project_name}-web"
  machine_type = var.machine_type
  zone         = var.zone
  boot_disk {
    initialize_params { image = "debian-cloud/debian-11" size = 50 }
  }
  network_interface {
    network = google_compute_network.main.id
    access_config {}
  }
}''',

    '''resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project_name}-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  security_group_ids   = [aws_security_group.redis.id]
  subnet_group_name    = aws_elasticache_subnet_group.main.name
}''',

    '''resource "aws_sqs_queue" "main" {
  name                      = "${var.project_name}-queue"
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 345600
  receive_wait_time_seconds = 10
  visibility_timeout_seconds = 30
}''',

    '''resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}''',

    '''resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}''',

    '''resource "aws_security_group" "web" {
  name   = "${var.project_name}-web-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}''',

    '''resource "aws_autoscaling_group" "web" {
  name                = "${var.project_name}-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.web.arn]
  min_size            = var.asg_min_size
  max_size            = var.asg_max_size
  desired_capacity    = var.asg_desired_size
  launch_template {
    id      = aws_launch_template.web.id
    version = "$Latest"
  }
}''',
]

KUBERNETES_SAMPLES = [
    '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
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
        resources:
          requests: { memory: "256Mi", cpu: "250m" }
          limits: { memory: "512Mi", cpu: "500m" }''',

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
      target: { type: Utilization, averageUtilization: 70 }''',

    '''apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts: [api.example.com]
    secretName: api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service: { name: api-server, port: { number: 80 } }''',

    '''apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  APP_ENV: "production"
  LOG_LEVEL: "info"
---
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData:
  url: "postgresql://user:password@db:5432/mydb"''',

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
          restartPolicy: OnFailure''',

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
    targetPort: 8080''',

    '''apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard''',

    '''apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: database
spec:
  serviceName: database
  replicas: 3
  selector:
    matchLabels:
      app: database
  template:
    metadata:
      labels:
        app: database
    spec:
      containers:
      - name: postgres
        image: postgres:15
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ReadWriteOnce]
      resources:
        requests:
          storage: 10Gi''',

    '''apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: api-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: api-server''',

    '''apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
spec:
  podSelector:
    matchLabels:
      app: api-server
  policyTypes: [Ingress, Egress]
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080''',

    '''apiVersion: v1
kind: ServiceAccount
metadata:
  name: api-service-account
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: api-role
rules:
- apiGroups: [""]
  resources: [pods, services]
  verbs: [get, list, watch]''',

    '''apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: log-collector
spec:
  selector:
    matchLabels:
      name: log-collector
  template:
    metadata:
      labels:
        name: log-collector
    spec:
      containers:
      - name: fluentd
        image: fluent/fluentd:v1.14
        volumeMounts:
        - name: varlog
          mountPath: /var/log
      volumes:
      - name: varlog
        hostPath:
          path: /var/log''',

    '''apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    pods: "20"''',

    '''apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
spec:
  limits:
  - default:
      cpu: 500m
      memory: 512Mi
    defaultRequest:
      cpu: 100m
      memory: 128Mi
    type: Container''',

    '''apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: api-cert
spec:
  secretName: api-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.example.com
  - www.api.example.com''',
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
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
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
    build: .
    ports: ["3000:3000"]
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on: [db, redis]
  db:
    image: postgres:15-alpine
    volumes: [postgres_data:/var/lib/postgresql/data]
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
  redis:
    image: redis:7-alpine
volumes:
  postgres_data:''',

    '''FROM rust:1.70 AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release
COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/myapp /usr/local/bin/
EXPOSE 8080
CMD ["myapp"]''',

    '''FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

FROM alpine:3.18
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/main /main
EXPOSE 8080
CMD ["/main"]''',

    '''FROM nginx:alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]''',

    '''version: '3.8'
services:
  web:
    build:
      context: .
      args:
        NODE_ENV: production
    deploy:
      replicas: 3
      resources:
        limits: { cpus: '0.5', memory: 512M }
        reservations: { cpus: '0.25', memory: 256M }
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3''',

    '''FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY *.csproj ./
RUN dotnet restore
COPY . .
RUN dotnet publish -c Release -o /app

FROM mcr.microsoft.com/dotnet/aspnet:7.0
WORKDIR /app
COPY --from=build /app .
EXPOSE 80
ENTRYPOINT ["dotnet", "MyApp.dll"]''',

    '''FROM php:8.2-fpm-alpine
RUN apk add --no-cache nginx supervisor
RUN docker-php-ext-install pdo pdo_mysql
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer
WORKDIR /var/www
COPY . .
RUN composer install --no-dev --optimize-autoloader
EXPOSE 80
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]''',

    '''FROM ruby:3.2-alpine
RUN apk add --no-cache build-base postgresql-dev nodejs yarn
WORKDIR /app
COPY Gemfile Gemfile.lock ./
RUN bundle install --without development test
COPY . .
RUN rails assets:precompile
EXPOSE 3000
CMD ["rails", "server", "-b", "0.0.0.0"]''',
]

GO_SAMPLES = [
    '''package main

import (
    "encoding/json"
    "log"
    "net/http"
)

type User struct {
    ID       int    `json:"id"`
    Name     string `json:"name"`
    Email    string `json:"email"`
}

func main() {
    http.HandleFunc("/users", handleUsers)
    log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleUsers(w http.ResponseWriter, r *http.Request) {
    users := []User{{ID: 1, Name: "Alice", Email: "alice@example.com"}}
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(users)
}''',

    '''package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

func worker(ctx context.Context, id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {
    defer wg.Done()
    for {
        select {
        case <-ctx.Done():
            return
        case job, ok := <-jobs:
            if !ok {
                return
            }
            results <- job * 2
        }
    }
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    jobs := make(chan int, 100)
    results := make(chan int, 100)
    var wg sync.WaitGroup
    
    for w := 1; w <= 3; w++ {
        wg.Add(1)
        go worker(ctx, w, jobs, results, &wg)
    }
    
    for j := 1; j <= 10; j++ {
        jobs <- j
    }
    close(jobs)
    wg.Wait()
    close(results)
}''',

    '''package main

import "fmt"

type Shape interface {
    Area() float64
    Perimeter() float64
}

type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return 3.14159 * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * 3.14159 * c.Radius
}

func PrintShapeInfo(s Shape) {
    fmt.Printf("Area: %.2f, Perimeter: %.2f\\n", s.Area(), s.Perimeter())
}''',

    '''package main

import (
    "errors"
    "fmt"
)

var (
    ErrNotFound     = errors.New("resource not found")
    ErrUnauthorized = errors.New("unauthorized access")
    ErrValidation   = errors.New("validation failed")
)

type AppError struct {
    Code    int
    Message string
    Err     error
}

func (e *AppError) Error() string {
    return fmt.Sprintf("code %d: %s: %v", e.Code, e.Message, e.Err)
}

func (e *AppError) Unwrap() error {
    return e.Err
}

func FindUser(id int) (*User, error) {
    if id <= 0 {
        return nil, &AppError{Code: 400, Message: "invalid user id", Err: ErrValidation}
    }
    if id > 1000 {
        return nil, &AppError{Code: 404, Message: "user not found", Err: ErrNotFound}
    }
    return &User{ID: id, Name: "John"}, nil
}''',

    '''package middleware

import (
    "log"
    "net/http"
    "time"
)

type Middleware func(http.Handler) http.Handler

func Chain(middlewares ...Middleware) Middleware {
    return func(final http.Handler) http.Handler {
        for i := len(middlewares) - 1; i >= 0; i-- {
            final = middlewares[i](final)
        }
        return final
    }
}

func Logger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
    })
}

func Auth(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }
        next.ServeHTTP(w, r)
    })
}''',

    '''package main

import (
    "encoding/json"
    "net/http"
)

type APIResponse struct {
    Success bool        `json:"success"`
    Data    interface{} `json:"data,omitempty"`
    Error   string      `json:"error,omitempty"`
}

func respondJSON(w http.ResponseWriter, status int, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(APIResponse{Success: status < 400, Data: data})
}

func respondError(w http.ResponseWriter, status int, message string) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(APIResponse{Success: false, Error: message})
}

func decodeJSON(r *http.Request, v interface{}) error {
    defer r.Body.Close()
    return json.NewDecoder(r.Body).Decode(v)
}''',

    '''package main

import (
    "sync"
)

type Cache struct {
    mu    sync.RWMutex
    items map[string]interface{}
}

func NewCache() *Cache {
    return &Cache{items: make(map[string]interface{})}
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    val, ok := c.items[key]
    return val, ok
}

func (c *Cache) Set(key string, value interface{}) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.items[key] = value
}

func (c *Cache) Delete(key string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    delete(c.items, key)
}''',

    '''package main

func Map[T, U any](slice []T, fn func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}

func Filter[T any](slice []T, fn func(T) bool) []T {
    var result []T
    for _, v := range slice {
        if fn(v) {
            result = append(result, v)
        }
    }
    return result
}

func Reduce[T, U any](slice []T, initial U, fn func(U, T) U) U {
    result := initial
    for _, v := range slice {
        result = fn(result, v)
    }
    return result
}''',

    '''package main

import (
    "context"
    "database/sql"
    _ "github.com/lib/pq"
)

type UserRepository struct {
    db *sql.DB
}

func (r *UserRepository) FindByID(ctx context.Context, id int) (*User, error) {
    var user User
    err := r.db.QueryRowContext(ctx,
        "SELECT id, name, email FROM users WHERE id = $1", id,
    ).Scan(&user.ID, &user.Name, &user.Email)
    if err == sql.ErrNoRows {
        return nil, ErrNotFound
    }
    return &user, err
}

func (r *UserRepository) Create(ctx context.Context, user *User) error {
    return r.db.QueryRowContext(ctx,
        "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
        user.Name, user.Email,
    ).Scan(&user.ID)
}''',

    '''package main

import (
    "fmt"
    "time"
)

func fanIn(channels ...<-chan int) <-chan int {
    out := make(chan int)
    var wg sync.WaitGroup
    wg.Add(len(channels))
    for _, ch := range channels {
        go func(c <-chan int) {
            defer wg.Done()
            for v := range c {
                out <- v
            }
        }(ch)
    }
    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

func generator(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for _, n := range nums {
            out <- n
        }
    }()
    return out
}''',

    '''package main

import (
    "net/http"
    "github.com/gorilla/mux"
)

type Server struct {
    router *mux.Router
    addr   string
}

func NewServer(addr string) *Server {
    s := &Server{
        router: mux.NewRouter(),
        addr:   addr,
    }
    s.routes()
    return s
}

func (s *Server) routes() {
    s.router.HandleFunc("/api/users", s.handleGetUsers()).Methods("GET")
    s.router.HandleFunc("/api/users", s.handleCreateUser()).Methods("POST")
    s.router.HandleFunc("/api/users/{id}", s.handleGetUser()).Methods("GET")
}

func (s *Server) handleGetUsers() http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        respondJSON(w, http.StatusOK, []User{})
    }
}

func (s *Server) Run() error {
    return http.ListenAndServe(s.addr, s.router)
}''',

    '''package main

import (
    "testing"
)

func TestFibonacci(t *testing.T) {
    tests := []struct {
        name     string
        input    int
        expected int
    }{
        {"zero", 0, 0},
        {"one", 1, 1},
        {"ten", 10, 55},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := Fibonacci(tt.input)
            if result != tt.expected {
                t.Errorf("Fibonacci(%d) = %d; want %d", tt.input, result, tt.expected)
            }
        })
    }
}

func BenchmarkFibonacci(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Fibonacci(20)
    }
}''',

    '''package main

import (
    "os"
    "os/signal"
    "syscall"
)

type App struct {
    done chan struct{}
}

func (a *App) Start() {
    a.done = make(chan struct{})
    go a.run()
}

func (a *App) run() {
    for {
        select {
        case <-a.done:
            return
        default:
            // Main application logic
        }
    }
}

func (a *App) Stop() {
    close(a.done)
}

func main() {
    app := &App{}
    app.Start()
    
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    <-sigChan
    
    app.Stop()
}''',

    '''package main

import (
    "time"
)

type RateLimiter struct {
    tokens   chan struct{}
    interval time.Duration
}

func NewRateLimiter(rate int, interval time.Duration) *RateLimiter {
    rl := &RateLimiter{
        tokens:   make(chan struct{}, rate),
        interval: interval,
    }
    for i := 0; i < rate; i++ {
        rl.tokens <- struct{}{}
    }
    go rl.refill(rate)
    return rl
}

func (rl *RateLimiter) refill(rate int) {
    ticker := time.NewTicker(rl.interval)
    for range ticker.C {
        for i := 0; i < rate; i++ {
            select {
            case rl.tokens <- struct{}{}:
            default:
            }
        }
    }
}

func (rl *RateLimiter) Allow() bool {
    select {
    case <-rl.tokens:
        return true
    default:
        return false
    }
}''',

    '''package main

import (
    "context"
    "golang.org/x/sync/errgroup"
)

func ProcessItems(ctx context.Context, items []Item) error {
    g, ctx := errgroup.WithContext(ctx)
    results := make(chan Result, len(items))
    
    for _, item := range items {
        item := item
        g.Go(func() error {
            result, err := process(ctx, item)
            if err != nil {
                return err
            }
            results <- result
            return nil
        })
    }
    
    go func() {
        g.Wait()
        close(results)
    }()
    
    if err := g.Wait(); err != nil {
        return err
    }
    return nil
}''',

    '''package config

import (
    "os"
    "strconv"
    "time"
)

type Config struct {
    Port        int
    DatabaseURL string
    RedisURL    string
    JWTSecret   string
    Timeout     time.Duration
}

func Load() *Config {
    return &Config{
        Port:        getEnvInt("PORT", 8080),
        DatabaseURL: getEnv("DATABASE_URL", "postgres://localhost/db"),
        RedisURL:    getEnv("REDIS_URL", "redis://localhost:6379"),
        JWTSecret:   getEnv("JWT_SECRET", "secret"),
        Timeout:     getEnvDuration("TIMEOUT", 30*time.Second),
    }
}

func getEnv(key, fallback string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return fallback
}

func getEnvInt(key string, fallback int) int {
    if value := os.Getenv(key); value != "" {
        if i, err := strconv.Atoi(value); err == nil {
            return i
        }
    }
    return fallback
}''',

    '''package main

import (
    "html/template"
    "net/http"
)

type PageData struct {
    Title   string
    Message string
    Items   []string
}

var templates = template.Must(template.ParseGlob("templates/*.html"))

func renderTemplate(w http.ResponseWriter, tmpl string, data interface{}) {
    err := templates.ExecuteTemplate(w, tmpl+".html", data)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
    }
}

func homeHandler(w http.ResponseWriter, r *http.Request) {
    data := PageData{
        Title:   "Home",
        Message: "Welcome!",
        Items:   []string{"Item 1", "Item 2", "Item 3"},
    }
    renderTemplate(w, "home", data)
}''',

    '''package main

type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) {
        s.port = port
    }
}

func WithTimeout(timeout time.Duration) Option {
    return func(s *Server) {
        s.timeout = timeout
    }
}

func WithLogger(logger *log.Logger) Option {
    return func(s *Server) {
        s.logger = logger
    }
}

func NewServerWithOptions(opts ...Option) *Server {
    s := &Server{
        port:    8080,
        timeout: 30 * time.Second,
        logger:  log.Default(),
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}''',

    '''package main

type LinkedList[T any] struct {
    head *Node[T]
    tail *Node[T]
    size int
}

type Node[T any] struct {
    value T
    next  *Node[T]
}

func NewLinkedList[T any]() *LinkedList[T] {
    return &LinkedList[T]{}
}

func (l *LinkedList[T]) Append(value T) {
    node := &Node[T]{value: value}
    if l.tail == nil {
        l.head = node
        l.tail = node
    } else {
        l.tail.next = node
        l.tail = node
    }
    l.size++
}

func (l *LinkedList[T]) ToSlice() []T {
    result := make([]T, 0, l.size)
    for node := l.head; node != nil; node = node.next {
        result = append(result, node.value)
    }
    return result
}''',
]

RUST_SAMPLES = [
    '''use std::collections::HashMap;

fn main() {
    let mut scores: HashMap<String, i32> = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Red"), 50);
    
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }
    
    let team = String::from("Blue");
    if let Some(score) = scores.get(&team) {
        println!("Score for {}: {}", team, score);
    }
}''',

    '''#[derive(Debug, Clone)]
struct User {
    id: u64,
    name: String,
    email: String,
    active: bool,
}

impl User {
    fn new(id: u64, name: impl Into<String>, email: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            email: email.into(),
            active: true,
        }
    }
    
    fn deactivate(&mut self) {
        self.active = false;
    }
    
    fn is_active(&self) -> bool {
        self.active
    }
}

impl Default for User {
    fn default() -> Self {
        Self::new(0, "Unknown", "unknown@example.com")
    }
}''',

    '''use std::error::Error;
use std::fmt;

#[derive(Debug)]
enum AppError {
    NotFound(String),
    Unauthorized,
    ValidationError(String),
    DatabaseError(Box<dyn Error>),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AppError::NotFound(msg) => write!(f, "Not found: {}", msg),
            AppError::Unauthorized => write!(f, "Unauthorized access"),
            AppError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            AppError::DatabaseError(e) => write!(f, "Database error: {}", e),
        }
    }
}

impl Error for AppError {}

fn find_user(id: u64) -> Result<User, AppError> {
    if id == 0 {
        return Err(AppError::ValidationError("Invalid user ID".into()));
    }
    Err(AppError::NotFound(format!("User {} not found", id)))
}''',

    '''trait Animal {
    fn name(&self) -> &str;
    fn speak(&self) -> String;
    
    fn describe(&self) -> String {
        format!("{} says: {}", self.name(), self.speak())
    }
}

struct Dog {
    name: String,
}

impl Animal for Dog {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn speak(&self) -> String {
        String::from("Woof!")
    }
}

struct Cat {
    name: String,
}

impl Animal for Cat {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn speak(&self) -> String {
        String::from("Meow!")
    }
}''',

    '''fn process_data<T>(items: &[T]) -> Vec<T>
where
    T: Clone + std::fmt::Debug,
{
    items.iter()
        .filter(|item| format!("{:?}", item).len() > 2)
        .cloned()
        .collect()
}

fn find_max<T: Ord>(items: &[T]) -> Option<&T> {
    items.iter().max()
}

fn apply_twice<F, T>(f: F, x: T) -> T
where
    F: Fn(T) -> T,
{
    f(f(x))
}''',

    '''use std::sync::{Arc, Mutex};
use std::thread;

fn parallel_sum(numbers: Vec<i32>) -> i32 {
    let chunk_size = (numbers.len() + 3) / 4;
    let numbers = Arc::new(numbers);
    let mut handles = vec![];
    
    for i in 0..4 {
        let nums = Arc::clone(&numbers);
        let handle = thread::spawn(move || {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(nums.len());
            if start < nums.len() {
                nums[start..end].iter().sum::<i32>()
            } else {
                0
            }
        });
        handles.push(handle);
    }
    
    handles.into_iter()
        .map(|h| h.join().unwrap())
        .sum()
}''',

    '''use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

async fn producer(tx: mpsc::Sender<i32>) {
    for i in 0..10 {
        if tx.send(i).await.is_err() {
            break;
        }
        sleep(Duration::from_millis(100)).await;
    }
}

async fn consumer(mut rx: mpsc::Receiver<i32>) {
    while let Some(value) = rx.recv().await {
        println!("Received: {}", value);
    }
}

#[tokio::main]
async fn main() {
    let (tx, rx) = mpsc::channel(32);
    
    tokio::spawn(producer(tx));
    consumer(rx).await;
}''',

    '''struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    fn new() -> Self {
        Stack { items: Vec::new() }
    }
    
    fn push(&mut self, item: T) {
        self.items.push(item);
    }
    
    fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }
    
    fn peek(&self) -> Option<&T> {
        self.items.last()
    }
    
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    fn len(&self) -> usize {
        self.items.len()
    }
}

impl<T> Default for Stack<T> {
    fn default() -> Self {
        Self::new()
    }
}''',

    '''fn fibonacci() -> impl Iterator<Item = u64> {
    let mut state = (0, 1);
    std::iter::from_fn(move || {
        let current = state.0;
        state = (state.1, state.0 + state.1);
        Some(current)
    })
}

fn main() {
    let sum: u64 = fibonacci()
        .take(20)
        .filter(|n| n % 2 == 0)
        .sum();
    
    println!("Sum of first 20 even Fibonacci numbers: {}", sum);
    
    let squares: Vec<u64> = (1..=10)
        .map(|x| x * x)
        .collect();
    println!("Squares: {:?}", squares);
}''',

    '''use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    host: String,
    port: u16,
    database_url: String,
    #[serde(default)]
    debug: bool,
}

impl Config {
    fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}''',

    '''use std::cell::RefCell;
use std::rc::Rc;

type Link<T> = Option<Rc<RefCell<Node<T>>>>;

struct Node<T> {
    value: T,
    next: Link<T>,
}

struct LinkedList<T> {
    head: Link<T>,
    tail: Link<T>,
    len: usize,
}

impl<T> LinkedList<T> {
    fn new() -> Self {
        LinkedList { head: None, tail: None, len: 0 }
    }
    
    fn push_back(&mut self, value: T) {
        let new_node = Rc::new(RefCell::new(Node { value, next: None }));
        match self.tail.take() {
            Some(old_tail) => {
                old_tail.borrow_mut().next = Some(Rc::clone(&new_node));
            }
            None => {
                self.head = Some(Rc::clone(&new_node));
            }
        }
        self.tail = Some(new_node);
        self.len += 1;
    }
}''',

    '''#[derive(Debug)]
enum State {
    Idle,
    Running { progress: f32 },
    Paused,
    Completed { result: String },
    Failed { error: String },
}

struct Task {
    id: u64,
    name: String,
    state: State,
}

impl Task {
    fn new(id: u64, name: impl Into<String>) -> Self {
        Task {
            id,
            name: name.into(),
            state: State::Idle,
        }
    }
    
    fn start(&mut self) {
        if matches!(self.state, State::Idle | State::Paused) {
            self.state = State::Running { progress: 0.0 };
        }
    }
    
    fn complete(&mut self, result: String) {
        if matches!(self.state, State::Running { .. }) {
            self.state = State::Completed { result };
        }
    }
}''',

    '''use std::ops::{Add, Mul};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Vector2D {
    x: f64,
    y: f64,
}

impl Vector2D {
    fn new(x: f64, y: f64) -> Self {
        Vector2D { x, y }
    }
    
    fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    
    fn normalize(&self) -> Self {
        let mag = self.magnitude();
        Vector2D::new(self.x / mag, self.y / mag)
    }
}

impl Add for Vector2D {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Vector2D::new(self.x + other.x, self.y + other.y)
    }
}

impl Mul<f64> for Vector2D {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Vector2D::new(self.x * scalar, self.y * scalar)
    }
}''',

    '''use std::collections::HashMap;
use std::hash::Hash;

struct LRUCache<K, V> {
    capacity: usize,
    map: HashMap<K, V>,
    order: Vec<K>,
}

impl<K: Eq + Hash + Clone, V> LRUCache<K, V> {
    fn new(capacity: usize) -> Self {
        LRUCache {
            capacity,
            map: HashMap::new(),
            order: Vec::new(),
        }
    }
    
    fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            self.order.retain(|k| k != key);
            self.order.push(key.clone());
            self.map.get(key)
        } else {
            None
        }
    }
    
    fn put(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            if let Some(oldest) = self.order.first().cloned() {
                self.map.remove(&oldest);
                self.order.remove(0);
            }
        }
        self.order.retain(|k| k != &key);
        self.order.push(key.clone());
        self.map.insert(key, value);
    }
}''',

    '''use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

struct Delay {
    when: std::time::Instant,
}

impl Delay {
    fn new(duration: std::time::Duration) -> Self {
        Delay {
            when: std::time::Instant::now() + duration,
        }
    }
}

impl Future for Delay {
    type Output = ();
    
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if std::time::Instant::now() >= self.when {
            Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}''',

    '''macro_rules! vec_of_strings {
    ($($x:expr),*) => {
        vec![$(String::from($x)),*]
    };
}

macro_rules! hashmap {
    ($($key:expr => $value:expr),* $(,)?) => {{
        let mut map = std::collections::HashMap::new();
        $(map.insert($key, $value);)*
        map
    }};
}

fn main() {
    let names = vec_of_strings!["Alice", "Bob", "Charlie"];
    println!("{:?}", names);
    
    let scores = hashmap! {
        "Alice" => 100,
        "Bob" => 85,
        "Charlie" => 92,
    };
    println!("{:?}", scores);
}''',

    '''use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

type SharedState = Arc<RwLock<AppState>>;

struct AppState {
    users: Vec<User>,
}

async fn get_users(State(state): State<SharedState>) -> Json<Vec<User>> {
    let state = state.read().await;
    Json(state.users.clone())
}

async fn create_user(
    State(state): State<SharedState>,
    Json(payload): Json<CreateUser>,
) -> (StatusCode, Json<User>) {
    let mut state = state.write().await;
    let user = User { id: state.users.len() as u64, name: payload.name };
    state.users.push(user.clone());
    (StatusCode::CREATED, Json(user))
}''',

    '''fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let mut left = 0;
    let mut right = arr.len();
    
    while left < right {
        let mid = left + (right - left) / 2;
        match arr[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater => right = mid,
        }
    }
    None
}

fn merge_sort<T: Ord + Clone>(arr: &mut [T]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }
    let mid = len / 2;
    merge_sort(&mut arr[..mid]);
    merge_sort(&mut arr[mid..]);
    
    let left: Vec<T> = arr[..mid].to_vec();
    let right: Vec<T> = arr[mid..].to_vec();
    merge(arr, &left, &right);
}''',

    '''struct Builder {
    name: Option<String>,
    port: Option<u16>,
    timeout: Option<u64>,
}

impl Builder {
    fn new() -> Self {
        Builder {
            name: None,
            port: None,
            timeout: None,
        }
    }
    
    fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }
    
    fn timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }
    
    fn build(self) -> Result<Server, &'static str> {
        Ok(Server {
            name: self.name.ok_or("name is required")?,
            port: self.port.unwrap_or(8080),
            timeout: self.timeout.unwrap_or(30),
        })
    }
}''',
]

JAVA_SAMPLES = [
    '''public class User {
    private final Long id;
    private String name;
    private String email;
    private boolean active;
    
    public User(Long id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.active = true;
    }
    
    public Long getId() { return id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public boolean isActive() { return active; }
    public void deactivate() { this.active = false; }
    
    @Override
    public String toString() {
        return String.format("User{id=%d, name='%s', email='%s'}", id, name, email);
    }
}''',

    '''import java.util.*;
import java.util.stream.*;

public class StreamExamples {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        int sum = numbers.stream()
            .filter(n -> n % 2 == 0)
            .mapToInt(Integer::intValue)
            .sum();
        
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");
        Map<Integer, List<String>> byLength = names.stream()
            .collect(Collectors.groupingBy(String::length));
        
        Optional<String> longest = names.stream()
            .max(Comparator.comparingInt(String::length));
        
        String joined = names.stream()
            .collect(Collectors.joining(", ", "[", "]"));
    }
}''',

    '''import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class ThreadPoolExample {
    private final ExecutorService executor;
    private final AtomicInteger taskCount = new AtomicInteger(0);
    
    public ThreadPoolExample(int poolSize) {
        this.executor = Executors.newFixedThreadPool(poolSize);
    }
    
    public CompletableFuture<String> submitTask(String input) {
        return CompletableFuture.supplyAsync(() -> {
            int id = taskCount.incrementAndGet();
            try {
                Thread.sleep(100);
                return String.format("Task %d processed: %s", id, input.toUpperCase());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }
        }, executor);
    }
    
    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}''',

    '''import org.springframework.web.bind.annotation.*;
import org.springframework.http.*;

@RestController
@RequestMapping("/api/users")
public class UserController {
    private final UserService userService;
    
    public UserController(UserService userService) {
        this.userService = userService;
    }
    
    @GetMapping
    public ResponseEntity<List<User>> getAllUsers() {
        return ResponseEntity.ok(userService.findAll());
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        return userService.findById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody @Valid UserDto dto) {
        User user = userService.create(dto);
        return ResponseEntity.status(HttpStatus.CREATED).body(user);
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable Long id) {
        userService.delete(id);
        return ResponseEntity.noContent().build();
    }
}''',

    '''import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class UserService {
    private final UserRepository userRepository;
    private final EmailService emailService;
    
    public UserService(UserRepository userRepository, EmailService emailService) {
        this.userRepository = userRepository;
        this.emailService = emailService;
    }
    
    @Transactional(readOnly = true)
    public List<User> findAll() {
        return userRepository.findAll();
    }
    
    @Transactional(readOnly = true)
    public Optional<User> findById(Long id) {
        return userRepository.findById(id);
    }
    
    @Transactional
    public User create(UserDto dto) {
        User user = new User(null, dto.getName(), dto.getEmail());
        user = userRepository.save(user);
        emailService.sendWelcomeEmail(user);
        return user;
    }
    
    @Transactional
    public void delete(Long id) {
        userRepository.deleteById(id);
    }
}''',

    '''public interface Repository<T, ID> {
    Optional<T> findById(ID id);
    List<T> findAll();
    T save(T entity);
    void deleteById(ID id);
    boolean existsById(ID id);
}

public interface UserRepository extends Repository<User, Long> {
    Optional<User> findByEmail(String email);
    List<User> findByActiveTrue();
    List<User> findByNameContainingIgnoreCase(String name);
}

public class InMemoryUserRepository implements UserRepository {
    private final Map<Long, User> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(0);
    
    @Override
    public Optional<User> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }
    
    @Override
    public User save(User user) {
        if (user.getId() == null) {
            user = new User(idGenerator.incrementAndGet(), user.getName(), user.getEmail());
        }
        storage.put(user.getId(), user);
        return user;
    }
}''',

    '''public class GenericCache<K, V> {
    private final Map<K, CacheEntry<V>> cache = new ConcurrentHashMap<>();
    private final long ttlMillis;
    
    public GenericCache(long ttlMillis) {
        this.ttlMillis = ttlMillis;
    }
    
    public void put(K key, V value) {
        cache.put(key, new CacheEntry<>(value, System.currentTimeMillis()));
    }
    
    public Optional<V> get(K key) {
        CacheEntry<V> entry = cache.get(key);
        if (entry == null) {
            return Optional.empty();
        }
        if (System.currentTimeMillis() - entry.timestamp > ttlMillis) {
            cache.remove(key);
            return Optional.empty();
        }
        return Optional.of(entry.value);
    }
    
    private static class CacheEntry<V> {
        final V value;
        final long timestamp;
        
        CacheEntry(V value, long timestamp) {
            this.value = value;
            this.timestamp = timestamp;
        }
    }
}''',

    '''public sealed interface Result<T> permits Success, Failure {
    static <T> Result<T> success(T value) {
        return new Success<>(value);
    }
    
    static <T> Result<T> failure(String error) {
        return new Failure<>(error);
    }
    
    <U> Result<U> map(Function<T, U> mapper);
    <U> Result<U> flatMap(Function<T, Result<U>> mapper);
    T orElse(T defaultValue);
}

public record Success<T>(T value) implements Result<T> {
    @Override
    public <U> Result<U> map(Function<T, U> mapper) {
        return Result.success(mapper.apply(value));
    }
    
    @Override
    public <U> Result<U> flatMap(Function<T, Result<U>> mapper) {
        return mapper.apply(value);
    }
    
    @Override
    public T orElse(T defaultValue) {
        return value;
    }
}

public record Failure<T>(String error) implements Result<T> {
    @Override
    public <U> Result<U> map(Function<T, U> mapper) {
        return Result.failure(error);
    }
    
    @Override
    public <U> Result<U> flatMap(Function<T, Result<U>> mapper) {
        return Result.failure(error);
    }
    
    @Override
    public T orElse(T defaultValue) {
        return defaultValue;
    }
}''',

    '''import java.util.*;

public class Graph<T> {
    private final Map<T, Set<T>> adjacencyList = new HashMap<>();
    
    public void addVertex(T vertex) {
        adjacencyList.putIfAbsent(vertex, new HashSet<>());
    }
    
    public void addEdge(T from, T to) {
        addVertex(from);
        addVertex(to);
        adjacencyList.get(from).add(to);
    }
    
    public List<T> bfs(T start) {
        List<T> result = new ArrayList<>();
        Set<T> visited = new HashSet<>();
        Queue<T> queue = new LinkedList<>();
        
        queue.offer(start);
        visited.add(start);
        
        while (!queue.isEmpty()) {
            T current = queue.poll();
            result.add(current);
            
            for (T neighbor : adjacencyList.getOrDefault(current, Set.of())) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        return result;
    }
}''',

    '''public class Builder {
    public static class UserBuilder {
        private String name;
        private String email;
        private Integer age;
        private String address;
        
        public UserBuilder name(String name) {
            this.name = name;
            return this;
        }
        
        public UserBuilder email(String email) {
            this.email = email;
            return this;
        }
        
        public UserBuilder age(Integer age) {
            this.age = age;
            return this;
        }
        
        public UserBuilder address(String address) {
            this.address = address;
            return this;
        }
        
        public User build() {
            if (name == null || email == null) {
                throw new IllegalStateException("Name and email are required");
            }
            return new User(name, email, age, address);
        }
    }
    
    public static UserBuilder builder() {
        return new UserBuilder();
    }
}''',

    '''import java.lang.reflect.*;

public class SimpleInjector {
    private final Map<Class<?>, Object> instances = new HashMap<>();
    
    public <T> void register(Class<T> type, T instance) {
        instances.put(type, instance);
    }
    
    @SuppressWarnings("unchecked")
    public <T> T getInstance(Class<T> type) {
        if (instances.containsKey(type)) {
            return (T) instances.get(type);
        }
        
        Constructor<?>[] constructors = type.getConstructors();
        if (constructors.length == 0) {
            throw new RuntimeException("No public constructor found for " + type);
        }
        
        Constructor<?> constructor = constructors[0];
        Class<?>[] paramTypes = constructor.getParameterTypes();
        Object[] params = new Object[paramTypes.length];
        
        for (int i = 0; i < paramTypes.length; i++) {
            params[i] = getInstance(paramTypes[i]);
        }
        
        try {
            T instance = (T) constructor.newInstance(params);
            instances.put(type, instance);
            return instance;
        } catch (Exception e) {
            throw new RuntimeException("Failed to create instance", e);
        }
    }
}''',

    '''import java.util.function.*;

public class FunctionalUtils {
    public static <T> Predicate<T> not(Predicate<T> predicate) {
        return predicate.negate();
    }
    
    public static <T, R> Function<T, R> memoize(Function<T, R> function) {
        Map<T, R> cache = new ConcurrentHashMap<>();
        return input -> cache.computeIfAbsent(input, function);
    }
    
    public static <T> Function<T, T> compose(List<Function<T, T>> functions) {
        return functions.stream()
            .reduce(Function.identity(), Function::andThen);
    }
    
    public static <T, U, R> BiFunction<T, U, R> curry(
            Function<T, Function<U, R>> curried) {
        return (t, u) -> curried.apply(t).apply(u);
    }
    
    public static <T> Supplier<T> lazy(Supplier<T> supplier) {
        return new Supplier<>() {
            private T value;
            private boolean computed = false;
            
            @Override
            public synchronized T get() {
                if (!computed) {
                    value = supplier.get();
                    computed = true;
                }
                return value;
            }
        };
    }
}''',

    '''import javax.persistence.*;
import java.time.*;

@Entity
@Table(name = "orders")
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;
    
    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<OrderItem> items = new ArrayList<>();
    
    @Enumerated(EnumType.STRING)
    private OrderStatus status;
    
    @Column(name = "created_at")
    private LocalDateTime createdAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        status = OrderStatus.PENDING;
    }
    
    public void addItem(OrderItem item) {
        items.add(item);
        item.setOrder(this);
    }
    
    public BigDecimal getTotal() {
        return items.stream()
            .map(OrderItem::getSubtotal)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
    }
}''',

    '''public class EventBus {
    private final Map<Class<?>, List<Consumer<?>>> handlers = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newCachedThreadPool();
    
    public <T> void subscribe(Class<T> eventType, Consumer<T> handler) {
        handlers.computeIfAbsent(eventType, k -> new CopyOnWriteArrayList<>())
            .add(handler);
    }
    
    public <T> void unsubscribe(Class<T> eventType, Consumer<T> handler) {
        List<Consumer<?>> eventHandlers = handlers.get(eventType);
        if (eventHandlers != null) {
            eventHandlers.remove(handler);
        }
    }
    
    @SuppressWarnings("unchecked")
    public <T> void publish(T event) {
        List<Consumer<?>> eventHandlers = handlers.get(event.getClass());
        if (eventHandlers != null) {
            for (Consumer<?> handler : eventHandlers) {
                executor.submit(() -> ((Consumer<T>) handler).accept(event));
            }
        }
    }
    
    public void shutdown() {
        executor.shutdown();
    }
}''',

    '''public class RateLimiter {
    private final int maxRequests;
    private final long windowMillis;
    private final Map<String, Deque<Long>> requests = new ConcurrentHashMap<>();
    
    public RateLimiter(int maxRequests, long windowMillis) {
        this.maxRequests = maxRequests;
        this.windowMillis = windowMillis;
    }
    
    public synchronized boolean tryAcquire(String key) {
        long now = System.currentTimeMillis();
        long windowStart = now - windowMillis;
        
        Deque<Long> timestamps = requests.computeIfAbsent(key, k -> new LinkedList<>());
        
        while (!timestamps.isEmpty() && timestamps.peekFirst() < windowStart) {
            timestamps.pollFirst();
        }
        
        if (timestamps.size() < maxRequests) {
            timestamps.addLast(now);
            return true;
        }
        return false;
    }
    
    public void cleanup() {
        long windowStart = System.currentTimeMillis() - windowMillis;
        requests.forEach((key, timestamps) -> {
            timestamps.removeIf(t -> t < windowStart);
        });
        requests.entrySet().removeIf(e -> e.getValue().isEmpty());
    }
}''',

    '''import io.reactivex.rxjava3.core.*;
import io.reactivex.rxjava3.schedulers.*;

public class ReactiveExample {
    public Observable<String> fetchData(List<String> urls) {
        return Observable.fromIterable(urls)
            .flatMap(url -> fetchUrl(url)
                .subscribeOn(Schedulers.io())
                .onErrorResumeNext(e -> Observable.just("Error: " + e.getMessage())))
            .observeOn(Schedulers.computation())
            .map(String::toUpperCase)
            .buffer(5)
            .flatMap(batch -> Observable.fromIterable(batch));
    }
    
    private Observable<String> fetchUrl(String url) {
        return Observable.fromCallable(() -> {
            Thread.sleep(100);
            return "Response from " + url;
        });
    }
    
    public static void main(String[] args) {
        new ReactiveExample()
            .fetchData(List.of("url1", "url2", "url3"))
            .subscribe(
                System.out::println,
                Throwable::printStackTrace,
                () -> System.out.println("Complete")
            );
    }
}''',

    '''public class CircuitBreaker {
    private final int failureThreshold;
    private final long resetTimeoutMs;
    private int failureCount = 0;
    private State state = State.CLOSED;
    private long lastFailureTime = 0;
    
    public enum State { CLOSED, OPEN, HALF_OPEN }
    
    public CircuitBreaker(int failureThreshold, long resetTimeoutMs) {
        this.failureThreshold = failureThreshold;
        this.resetTimeoutMs = resetTimeoutMs;
    }
    
    public synchronized <T> T execute(Supplier<T> action) throws Exception {
        if (state == State.OPEN) {
            if (System.currentTimeMillis() - lastFailureTime > resetTimeoutMs) {
                state = State.HALF_OPEN;
            } else {
                throw new RuntimeException("Circuit breaker is OPEN");
            }
        }
        
        try {
            T result = action.get();
            reset();
            return result;
        } catch (Exception e) {
            recordFailure();
            throw e;
        }
    }
    
    private void recordFailure() {
        failureCount++;
        lastFailureTime = System.currentTimeMillis();
        if (failureCount >= failureThreshold) {
            state = State.OPEN;
        }
    }
    
    private void reset() {
        failureCount = 0;
        state = State.CLOSED;
    }
}''',
]

CSHARP_SAMPLES = [
    '''public class User
{
    public int Id { get; init; }
    public string Name { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
    public bool IsActive { get; private set; } = true;
    
    public void Deactivate() => IsActive = false;
    
    public override string ToString() => $"User {{ Id = {Id}, Name = {Name} }}";
}

public record UserDto(string Name, string Email);

public record UserResponse(int Id, string Name, string Email, DateTime CreatedAt);''',

    '''using System.Linq;

public class LinqExamples
{
    public void Examples()
    {
        var numbers = Enumerable.Range(1, 100);
        
        var evenSquares = numbers
            .Where(n => n % 2 == 0)
            .Select(n => n * n)
            .Take(10)
            .ToList();
        
        var grouped = numbers
            .GroupBy(n => n % 10)
            .Select(g => new { Key = g.Key, Sum = g.Sum() });
        
        var users = new List<User>();
        var activeUsers = users
            .Where(u => u.IsActive)
            .OrderByDescending(u => u.CreatedAt)
            .ThenBy(u => u.Name)
            .ToList();
        
        var usersByEmail = users.ToDictionary(u => u.Email, u => u);
    }
}''',

    '''public async Task<IEnumerable<User>> GetUsersAsync(CancellationToken cancellationToken = default)
{
    await using var context = await _contextFactory.CreateDbContextAsync(cancellationToken);
    return await context.Users
        .Where(u => u.IsActive)
        .OrderBy(u => u.Name)
        .ToListAsync(cancellationToken);
}

public async Task<User?> GetUserByIdAsync(int id, CancellationToken cancellationToken = default)
{
    await using var context = await _contextFactory.CreateDbContextAsync(cancellationToken);
    return await context.Users.FindAsync(new object[] { id }, cancellationToken);
}

public async Task<User> CreateUserAsync(UserDto dto, CancellationToken cancellationToken = default)
{
    await using var context = await _contextFactory.CreateDbContextAsync(cancellationToken);
    var user = new User { Name = dto.Name, Email = dto.Email };
    context.Users.Add(user);
    await context.SaveChangesAsync(cancellationToken);
    return user;
}''',

    '''[ApiController]
[Route("api/[controller]")]
public class UsersController : ControllerBase
{
    private readonly IUserService _userService;
    private readonly ILogger<UsersController> _logger;
    
    public UsersController(IUserService userService, ILogger<UsersController> logger)
    {
        _userService = userService;
        _logger = logger;
    }
    
    [HttpGet]
    public async Task<ActionResult<IEnumerable<UserResponse>>> GetUsers()
    {
        var users = await _userService.GetAllAsync();
        return Ok(users.Select(u => new UserResponse(u.Id, u.Name, u.Email, u.CreatedAt)));
    }
    
    [HttpGet("{id}")]
    public async Task<ActionResult<UserResponse>> GetUser(int id)
    {
        var user = await _userService.GetByIdAsync(id);
        if (user is null) return NotFound();
        return Ok(new UserResponse(user.Id, user.Name, user.Email, user.CreatedAt));
    }
    
    [HttpPost]
    public async Task<ActionResult<UserResponse>> CreateUser([FromBody] UserDto dto)
    {
        var user = await _userService.CreateAsync(dto);
        return CreatedAtAction(nameof(GetUser), new { id = user.Id }, 
            new UserResponse(user.Id, user.Name, user.Email, user.CreatedAt));
    }
}''',

    '''public class Result<T>
{
    public bool IsSuccess { get; }
    public T? Value { get; }
    public string? Error { get; }
    
    private Result(bool isSuccess, T? value, string? error)
    {
        IsSuccess = isSuccess;
        Value = value;
        Error = error;
    }
    
    public static Result<T> Success(T value) => new(true, value, null);
    public static Result<T> Failure(string error) => new(false, default, error);
    
    public Result<U> Map<U>(Func<T, U> mapper)
    {
        return IsSuccess 
            ? Result<U>.Success(mapper(Value!))
            : Result<U>.Failure(Error!);
    }
    
    public async Task<Result<U>> MapAsync<U>(Func<T, Task<U>> mapper)
    {
        return IsSuccess
            ? Result<U>.Success(await mapper(Value!))
            : Result<U>.Failure(Error!);
    }
    
    public T GetValueOrDefault(T defaultValue) => IsSuccess ? Value! : defaultValue;
}''',

    '''public interface IRepository<T> where T : class
{
    Task<T?> GetByIdAsync(int id);
    Task<IEnumerable<T>> GetAllAsync();
    Task<T> AddAsync(T entity);
    Task UpdateAsync(T entity);
    Task DeleteAsync(int id);
}

public class Repository<T> : IRepository<T> where T : class
{
    protected readonly DbContext _context;
    protected readonly DbSet<T> _dbSet;
    
    public Repository(DbContext context)
    {
        _context = context;
        _dbSet = context.Set<T>();
    }
    
    public async Task<T?> GetByIdAsync(int id) => await _dbSet.FindAsync(id);
    
    public async Task<IEnumerable<T>> GetAllAsync() => await _dbSet.ToListAsync();
    
    public async Task<T> AddAsync(T entity)
    {
        await _dbSet.AddAsync(entity);
        await _context.SaveChangesAsync();
        return entity;
    }
    
    public async Task UpdateAsync(T entity)
    {
        _dbSet.Update(entity);
        await _context.SaveChangesAsync();
    }
    
    public async Task DeleteAsync(int id)
    {
        var entity = await GetByIdAsync(id);
        if (entity != null)
        {
            _dbSet.Remove(entity);
            await _context.SaveChangesAsync();
        }
    }
}''',

    '''public class EventAggregator
{
    private readonly ConcurrentDictionary<Type, List<Delegate>> _handlers = new();
    
    public void Subscribe<TEvent>(Action<TEvent> handler)
    {
        var handlers = _handlers.GetOrAdd(typeof(TEvent), _ => new List<Delegate>());
        lock (handlers)
        {
            handlers.Add(handler);
        }
    }
    
    public void Unsubscribe<TEvent>(Action<TEvent> handler)
    {
        if (_handlers.TryGetValue(typeof(TEvent), out var handlers))
        {
            lock (handlers)
            {
                handlers.Remove(handler);
            }
        }
    }
    
    public void Publish<TEvent>(TEvent eventData)
    {
        if (_handlers.TryGetValue(typeof(TEvent), out var handlers))
        {
            List<Delegate> handlersCopy;
            lock (handlers)
            {
                handlersCopy = handlers.ToList();
            }
            foreach (var handler in handlersCopy)
            {
                ((Action<TEvent>)handler)(eventData);
            }
        }
    }
}''',

    '''public class CacheService<TKey, TValue> where TKey : notnull
{
    private readonly ConcurrentDictionary<TKey, CacheEntry> _cache = new();
    private readonly TimeSpan _defaultTtl;
    
    public CacheService(TimeSpan defaultTtl)
    {
        _defaultTtl = defaultTtl;
    }
    
    public void Set(TKey key, TValue value, TimeSpan? ttl = null)
    {
        var expiry = DateTime.UtcNow.Add(ttl ?? _defaultTtl);
        _cache[key] = new CacheEntry(value, expiry);
    }
    
    public TValue? Get(TKey key)
    {
        if (_cache.TryGetValue(key, out var entry))
        {
            if (entry.Expiry > DateTime.UtcNow)
            {
                return entry.Value;
            }
            _cache.TryRemove(key, out _);
        }
        return default;
    }
    
    public async Task<TValue> GetOrCreateAsync(TKey key, Func<Task<TValue>> factory, TimeSpan? ttl = null)
    {
        var cached = Get(key);
        if (cached is not null) return cached;
        
        var value = await factory();
        Set(key, value, ttl);
        return value;
    }
    
    private record CacheEntry(TValue Value, DateTime Expiry);
}''',

    '''public static class Extensions
{
    public static string ToSnakeCase(this string str)
    {
        return string.Concat(str.Select((c, i) =>
            i > 0 && char.IsUpper(c) ? "_" + c : c.ToString())).ToLower();
    }
    
    public static IEnumerable<IEnumerable<T>> Batch<T>(this IEnumerable<T> source, int size)
    {
        var batch = new List<T>(size);
        foreach (var item in source)
        {
            batch.Add(item);
            if (batch.Count == size)
            {
                yield return batch;
                batch = new List<T>(size);
            }
        }
        if (batch.Count > 0)
        {
            yield return batch;
        }
    }
    
    public static async Task<T[]> WhenAll<T>(this IEnumerable<Task<T>> tasks)
    {
        return await Task.WhenAll(tasks);
    }
    
    public static T? FirstOrNull<T>(this IEnumerable<T> source) where T : struct
    {
        foreach (var item in source)
        {
            return item;
        }
        return null;
    }
}''',

    '''public class Middleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<Middleware> _logger;
    
    public Middleware(RequestDelegate next, ILogger<Middleware> logger)
    {
        _next = next;
        _logger = logger;
    }
    
    public async Task InvokeAsync(HttpContext context)
    {
        var stopwatch = Stopwatch.StartNew();
        var requestId = Guid.NewGuid().ToString("N")[..8];
        
        context.Items["RequestId"] = requestId;
        _logger.LogInformation("Request {RequestId} started: {Method} {Path}",
            requestId, context.Request.Method, context.Request.Path);
        
        try
        {
            await _next(context);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Request {RequestId} failed", requestId);
            context.Response.StatusCode = 500;
            await context.Response.WriteAsJsonAsync(new { error = "Internal server error" });
        }
        finally
        {
            stopwatch.Stop();
            _logger.LogInformation("Request {RequestId} completed in {ElapsedMs}ms with status {StatusCode}",
                requestId, stopwatch.ElapsedMilliseconds, context.Response.StatusCode);
        }
    }
}''',

    '''public class BackgroundTaskQueue : IHostedService
{
    private readonly Channel<Func<CancellationToken, Task>> _queue;
    private readonly ILogger<BackgroundTaskQueue> _logger;
    private Task? _processingTask;
    private CancellationTokenSource? _cts;
    
    public BackgroundTaskQueue(ILogger<BackgroundTaskQueue> logger)
    {
        _queue = Channel.CreateUnbounded<Func<CancellationToken, Task>>();
        _logger = logger;
    }
    
    public async ValueTask EnqueueAsync(Func<CancellationToken, Task> workItem)
    {
        await _queue.Writer.WriteAsync(workItem);
    }
    
    public Task StartAsync(CancellationToken cancellationToken)
    {
        _cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _processingTask = ProcessQueueAsync(_cts.Token);
        return Task.CompletedTask;
    }
    
    private async Task ProcessQueueAsync(CancellationToken cancellationToken)
    {
        await foreach (var workItem in _queue.Reader.ReadAllAsync(cancellationToken))
        {
            try
            {
                await workItem(cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing background task");
            }
        }
    }
    
    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _queue.Writer.Complete();
        _cts?.Cancel();
        if (_processingTask != null)
        {
            await _processingTask;
        }
    }
}''',

    '''public class Specification<T>
{
    private readonly Func<T, bool> _predicate;
    
    public Specification(Func<T, bool> predicate)
    {
        _predicate = predicate;
    }
    
    public bool IsSatisfiedBy(T entity) => _predicate(entity);
    
    public Specification<T> And(Specification<T> other)
    {
        return new Specification<T>(x => IsSatisfiedBy(x) && other.IsSatisfiedBy(x));
    }
    
    public Specification<T> Or(Specification<T> other)
    {
        return new Specification<T>(x => IsSatisfiedBy(x) || other.IsSatisfiedBy(x));
    }
    
    public Specification<T> Not()
    {
        return new Specification<T>(x => !IsSatisfiedBy(x));
    }
}

public static class UserSpecifications
{
    public static Specification<User> IsActive() => 
        new(u => u.IsActive);
    
    public static Specification<User> HasEmail(string domain) =>
        new(u => u.Email.EndsWith(domain));
    
    public static Specification<User> CreatedAfter(DateTime date) =>
        new(u => u.CreatedAt > date);
}''',

    '''public class RateLimiter
{
    private readonly SemaphoreSlim _semaphore;
    private readonly int _maxRequests;
    private readonly TimeSpan _window;
    private readonly Queue<DateTime> _requestTimes = new();
    private readonly object _lock = new();
    
    public RateLimiter(int maxRequests, TimeSpan window)
    {
        _maxRequests = maxRequests;
        _window = window;
        _semaphore = new SemaphoreSlim(1, 1);
    }
    
    public async Task<bool> TryAcquireAsync(CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            var now = DateTime.UtcNow;
            var windowStart = now - _window;
            
            lock (_lock)
            {
                while (_requestTimes.Count > 0 && _requestTimes.Peek() < windowStart)
                {
                    _requestTimes.Dequeue();
                }
                
                if (_requestTimes.Count >= _maxRequests)
                {
                    return false;
                }
                
                _requestTimes.Enqueue(now);
                return true;
            }
        }
        finally
        {
            _semaphore.Release();
        }
    }
}''',

    '''public class RetryPolicy
{
    private readonly int _maxRetries;
    private readonly TimeSpan _initialDelay;
    private readonly double _multiplier;
    
    public RetryPolicy(int maxRetries = 3, TimeSpan? initialDelay = null, double multiplier = 2)
    {
        _maxRetries = maxRetries;
        _initialDelay = initialDelay ?? TimeSpan.FromSeconds(1);
        _multiplier = multiplier;
    }
    
    public async Task<T> ExecuteAsync<T>(Func<Task<T>> operation, CancellationToken cancellationToken = default)
    {
        Exception? lastException = null;
        var delay = _initialDelay;
        
        for (var attempt = 0; attempt <= _maxRetries; attempt++)
        {
            try
            {
                return await operation();
            }
            catch (Exception ex) when (attempt < _maxRetries)
            {
                lastException = ex;
                await Task.Delay(delay, cancellationToken);
                delay = TimeSpan.FromMilliseconds(delay.TotalMilliseconds * _multiplier);
            }
        }
        
        throw lastException!;
    }
}''',

    '''public class Pipeline<TIn, TOut>
{
    private readonly Func<TIn, Task<TOut>> _process;
    
    private Pipeline(Func<TIn, Task<TOut>> process)
    {
        _process = process;
    }
    
    public static Pipeline<T, T> Create<T>()
    {
        return new Pipeline<T, T>(x => Task.FromResult(x));
    }
    
    public Pipeline<TIn, TNext> Then<TNext>(Func<TOut, Task<TNext>> step)
    {
        return new Pipeline<TIn, TNext>(async input =>
        {
            var result = await _process(input);
            return await step(result);
        });
    }
    
    public Pipeline<TIn, TOut> When(Func<TOut, bool> condition, Func<TOut, Task<TOut>> step)
    {
        return new Pipeline<TIn, TOut>(async input =>
        {
            var result = await _process(input);
            return condition(result) ? await step(result) : result;
        });
    }
    
    public Task<TOut> ExecuteAsync(TIn input)
    {
        return _process(input);
    }
}''',

    '''public class HealthCheck : IHealthCheck
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<HealthCheck> _logger;
    
    public HealthCheck(IServiceProvider serviceProvider, ILogger<HealthCheck> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
    }
    
    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, 
        CancellationToken cancellationToken = default)
    {
        var checks = new Dictionary<string, object>();
        
        try
        {
            using var scope = _serviceProvider.CreateScope();
            var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            
            var canConnect = await dbContext.Database.CanConnectAsync(cancellationToken);
            checks["database"] = canConnect ? "healthy" : "unhealthy";
            
            if (!canConnect)
            {
                return HealthCheckResult.Unhealthy("Database connection failed", data: checks);
            }
            
            return HealthCheckResult.Healthy("All systems operational", checks);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Health check failed");
            return HealthCheckResult.Unhealthy(ex.Message, ex, checks);
        }
    }
}''',
]

CPP_SAMPLES = [
    '''#include <memory>
#include <string>
#include <vector>

class User {
public:
    User(int id, std::string name, std::string email)
        : id_(id), name_(std::move(name)), email_(std::move(email)) {}
    
    int getId() const { return id_; }
    const std::string& getName() const { return name_; }
    const std::string& getEmail() const { return email_; }
    
    void setName(std::string name) { name_ = std::move(name); }
    void setEmail(std::string email) { email_ = std::move(email); }

private:
    int id_;
    std::string name_;
    std::string email_;
};

class UserRepository {
public:
    void add(std::unique_ptr<User> user) {
        users_.push_back(std::move(user));
    }
    
    User* findById(int id) {
        for (const auto& user : users_) {
            if (user->getId() == id) return user.get();
        }
        return nullptr;
    }

private:
    std::vector<std::unique_ptr<User>> users_;
};''',

    '''#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

template<typename T>
class Statistics {
public:
    static T sum(const std::vector<T>& data) {
        return std::accumulate(data.begin(), data.end(), T{});
    }
    
    static double mean(const std::vector<T>& data) {
        if (data.empty()) return 0.0;
        return static_cast<double>(sum(data)) / data.size();
    }
    
    static T max(const std::vector<T>& data) {
        if (data.empty()) throw std::runtime_error("Empty vector");
        return *std::max_element(data.begin(), data.end());
    }
    
    static T min(const std::vector<T>& data) {
        if (data.empty()) throw std::runtime_error("Empty vector");
        return *std::min_element(data.begin(), data.end());
    }
    
    static std::vector<T> filter(const std::vector<T>& data, std::function<bool(T)> predicate) {
        std::vector<T> result;
        std::copy_if(data.begin(), data.end(), std::back_inserter(result), predicate);
        return result;
    }
};''',

    '''#include <memory>

template<typename T>
class LinkedList {
    struct Node {
        T data;
        std::unique_ptr<Node> next;
        Node(T value) : data(std::move(value)), next(nullptr) {}
    };

public:
    void push_front(T value) {
        auto new_node = std::make_unique<Node>(std::move(value));
        new_node->next = std::move(head_);
        head_ = std::move(new_node);
        ++size_;
    }
    
    void push_back(T value) {
        auto new_node = std::make_unique<Node>(std::move(value));
        if (!head_) {
            head_ = std::move(new_node);
        } else {
            Node* current = head_.get();
            while (current->next) {
                current = current->next.get();
            }
            current->next = std::move(new_node);
        }
        ++size_;
    }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

private:
    std::unique_ptr<Node> head_;
    size_t size_ = 0;
};''',

    '''#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    
    template<typename F>
    void enqueue(F&& task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace(std::forward<F>(task));
        }
        condition_.notify_one();
    }
    
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stop_;
};''',

    '''#include <optional>
#include <variant>
#include <string>

template<typename T, typename E = std::string>
class Result {
    std::variant<T, E> value_;

public:
    static Result ok(T value) {
        Result r;
        r.value_ = std::move(value);
        return r;
    }
    
    static Result err(E error) {
        Result r;
        r.value_ = std::move(error);
        return r;
    }
    
    bool isOk() const { return std::holds_alternative<T>(value_); }
    bool isErr() const { return std::holds_alternative<E>(value_); }
    
    const T& value() const { return std::get<T>(value_); }
    const E& error() const { return std::get<E>(value_); }
    
    T valueOr(T default_value) const {
        return isOk() ? value() : std::move(default_value);
    }
    
    template<typename F>
    auto map(F&& f) const -> Result<decltype(f(std::declval<T>())), E> {
        if (isOk()) return Result<decltype(f(value())), E>::ok(f(value()));
        return Result<decltype(f(std::declval<T>())), E>::err(error());
    }
};''',

    '''#include <map>
#include <chrono>
#include <mutex>

template<typename K, typename V>
class Cache {
    struct Entry {
        V value;
        std::chrono::steady_clock::time_point expiry;
    };

public:
    explicit Cache(std::chrono::seconds ttl) : ttl_(ttl) {}
    
    void put(const K& key, V value) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto expiry = std::chrono::steady_clock::now() + ttl_;
        cache_[key] = {std::move(value), expiry};
    }
    
    std::optional<V> get(const K& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it == cache_.end()) return std::nullopt;
        
        if (std::chrono::steady_clock::now() > it->second.expiry) {
            cache_.erase(it);
            return std::nullopt;
        }
        return it->second.value;
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        for (auto it = cache_.begin(); it != cache_.end();) {
            if (now > it->second.expiry) {
                it = cache_.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    std::map<K, Entry> cache_;
    std::chrono::seconds ttl_;
    std::mutex mutex_;
};''',

    '''#include <vector>
#include <functional>

template<typename T>
class Observable {
public:
    using Observer = std::function<void(const T&)>;
    
    void subscribe(Observer observer) {
        observers_.push_back(std::move(observer));
    }
    
    void notify(const T& value) {
        for (const auto& observer : observers_) {
            observer(value);
        }
    }
    
    void unsubscribeAll() {
        observers_.clear();
    }

private:
    std::vector<Observer> observers_;
};

class EventEmitter {
public:
    template<typename T>
    void on(const std::string& event, std::function<void(const T&)> handler) {
        handlers_[event].push_back([handler](const void* data) {
            handler(*static_cast<const T*>(data));
        });
    }
    
    template<typename T>
    void emit(const std::string& event, const T& data) {
        if (auto it = handlers_.find(event); it != handlers_.end()) {
            for (const auto& handler : it->second) {
                handler(&data);
            }
        }
    }

private:
    std::map<std::string, std::vector<std::function<void(const void*)>>> handlers_;
};''',

    '''#include <array>
#include <cmath>

template<size_t N>
class Vector {
public:
    std::array<double, N> data{};
    
    Vector() = default;
    Vector(std::initializer_list<double> init) {
        size_t i = 0;
        for (auto v : init) {
            if (i >= N) break;
            data[i++] = v;
        }
    }
    
    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }
    
    Vector operator+(const Vector& other) const {
        Vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] + other[i];
        }
        return result;
    }
    
    Vector operator*(double scalar) const {
        Vector result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] * scalar;
        }
        return result;
    }
    
    double magnitude() const {
        double sum = 0;
        for (size_t i = 0; i < N; ++i) {
            sum += data[i] * data[i];
        }
        return std::sqrt(sum);
    }
    
    Vector normalize() const {
        double mag = magnitude();
        return *this * (1.0 / mag);
    }
};

using Vec2 = Vector<2>;
using Vec3 = Vector<3>;''',

    '''#include <concepts>
#include <type_traits>

template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
};

template<typename T>
concept Comparable = requires(T a, T b) {
    { a < b } -> std::convertible_to<bool>;
    { a > b } -> std::convertible_to<bool>;
    { a == b } -> std::convertible_to<bool>;
};

template<Numeric T>
T square(T x) {
    return x * x;
}

template<Comparable T>
T clamp(T value, T min_val, T max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

template<Addable T>
T sum(const std::vector<T>& items) {
    T result{};
    for (const auto& item : items) {
        result = result + item;
    }
    return result;
}''',

    '''#include <string>
#include <stdexcept>
#include <source_location>

class AppException : public std::runtime_error {
public:
    AppException(const std::string& message, 
                 std::source_location loc = std::source_location::current())
        : std::runtime_error(message), location_(loc) {}
    
    const std::source_location& location() const { return location_; }
    
    std::string fullMessage() const {
        return std::string(what()) + " at " + 
               std::string(location_.file_name()) + ":" + 
               std::to_string(location_.line());
    }

private:
    std::source_location location_;
};

class NotFoundError : public AppException {
public:
    explicit NotFoundError(const std::string& resource)
        : AppException("Resource not found: " + resource) {}
};

class ValidationError : public AppException {
public:
    explicit ValidationError(const std::string& field, const std::string& message)
        : AppException("Validation failed for '" + field + "': " + message) {}
};''',

    '''#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

class ServiceContainer {
public:
    template<typename T>
    void registerType() {
        factories_[typeid(T).name()] = []() -> std::shared_ptr<void> {
            return std::make_shared<T>();
        };
    }
    
    template<typename T, typename... Args>
    void registerFactory(std::function<std::shared_ptr<T>(Args...)> factory) {
        factories_[typeid(T).name()] = [factory]() -> std::shared_ptr<void> {
            return factory();
        };
    }
    
    template<typename T>
    std::shared_ptr<T> resolve() {
        auto it = factories_.find(typeid(T).name());
        if (it == factories_.end()) {
            throw std::runtime_error("Type not registered");
        }
        return std::static_pointer_cast<T>(it->second());
    }
    
    template<typename T>
    void registerSingleton(std::shared_ptr<T> instance) {
        singletons_[typeid(T).name()] = instance;
    }
    
    template<typename T>
    std::shared_ptr<T> getSingleton() {
        auto it = singletons_.find(typeid(T).name());
        if (it == singletons_.end()) return nullptr;
        return std::static_pointer_cast<T>(it->second);
    }

private:
    std::unordered_map<std::string, std::function<std::shared_ptr<void>()>> factories_;
    std::unordered_map<std::string, std::shared_ptr<void>> singletons_;
};''',

    '''#include <string>
#include <sstream>

class StringBuilder {
public:
    StringBuilder& append(const std::string& str) {
        buffer_ << str;
        return *this;
    }
    
    StringBuilder& append(char c) {
        buffer_ << c;
        return *this;
    }
    
    template<typename T>
    StringBuilder& append(const T& value) {
        buffer_ << value;
        return *this;
    }
    
    StringBuilder& appendLine(const std::string& str = "") {
        buffer_ << str << '\\n';
        return *this;
    }
    
    StringBuilder& clear() {
        buffer_.str("");
        buffer_.clear();
        return *this;
    }
    
    std::string toString() const {
        return buffer_.str();
    }
    
    size_t length() const {
        return buffer_.str().length();
    }

private:
    std::ostringstream buffer_;
};''',

    '''#include <fstream>
#include <string>
#include <optional>

class FileReader {
public:
    explicit FileReader(const std::string& path) : path_(path) {}
    
    std::optional<std::string> readAll() const {
        std::ifstream file(path_);
        if (!file.is_open()) return std::nullopt;
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        return content;
    }
    
    std::optional<std::vector<std::string>> readLines() const {
        std::ifstream file(path_);
        if (!file.is_open()) return std::nullopt;
        
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        return lines;
    }

private:
    std::string path_;
};

class FileWriter {
public:
    explicit FileWriter(const std::string& path, bool append = false)
        : file_(path, append ? std::ios::app : std::ios::trunc) {}
    
    bool write(const std::string& content) {
        if (!file_.is_open()) return false;
        file_ << content;
        return true;
    }
    
    bool writeLine(const std::string& line) {
        return write(line + "\\n");
    }

private:
    std::ofstream file_;
};''',

    '''#include <chrono>
#include <functional>
#include <string>

class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }
    
    void stop() {
        end_ = std::chrono::high_resolution_clock::now();
        running_ = false;
    }
    
    template<typename Duration = std::chrono::milliseconds>
    long long elapsed() const {
        auto end = running_ ? std::chrono::high_resolution_clock::now() : end_;
        return std::chrono::duration_cast<Duration>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
    bool running_ = false;
};

template<typename F>
auto measure(F&& func) {
    Timer timer;
    timer.start();
    if constexpr (std::is_void_v<std::invoke_result_t<F>>) {
        func();
        timer.stop();
        return timer.elapsed();
    } else {
        auto result = func();
        timer.stop();
        return std::make_pair(result, timer.elapsed());
    }
}''',

    '''#include <vector>
#include <algorithm>

template<typename T>
class BinaryHeap {
public:
    void push(T value) {
        data_.push_back(std::move(value));
        siftUp(data_.size() - 1);
    }
    
    T pop() {
        if (data_.empty()) throw std::runtime_error("Heap is empty");
        T result = std::move(data_[0]);
        data_[0] = std::move(data_.back());
        data_.pop_back();
        if (!data_.empty()) siftDown(0);
        return result;
    }
    
    const T& top() const {
        if (data_.empty()) throw std::runtime_error("Heap is empty");
        return data_[0];
    }
    
    bool empty() const { return data_.empty(); }
    size_t size() const { return data_.size(); }

private:
    std::vector<T> data_;
    
    void siftUp(size_t idx) {
        while (idx > 0) {
            size_t parent = (idx - 1) / 2;
            if (data_[parent] <= data_[idx]) break;
            std::swap(data_[parent], data_[idx]);
            idx = parent;
        }
    }
    
    void siftDown(size_t idx) {
        while (true) {
            size_t smallest = idx;
            size_t left = 2 * idx + 1;
            size_t right = 2 * idx + 2;
            
            if (left < data_.size() && data_[left] < data_[smallest]) smallest = left;
            if (right < data_.size() && data_[right] < data_[smallest]) smallest = right;
            
            if (smallest == idx) break;
            std::swap(data_[idx], data_[smallest]);
            idx = smallest;
        }
    }
};''',
]

C_SAMPLES = [
    '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char name[64];
    char email[128];
} User;

User* user_create(int id, const char* name, const char* email) {
    User* user = (User*)malloc(sizeof(User));
    if (!user) return NULL;
    
    user->id = id;
    strncpy(user->name, name, sizeof(user->name) - 1);
    user->name[sizeof(user->name) - 1] = '\\0';
    strncpy(user->email, email, sizeof(user->email) - 1);
    user->email[sizeof(user->email) - 1] = '\\0';
    
    return user;
}

void user_free(User* user) {
    free(user);
}

void user_print(const User* user) {
    printf("User { id: %d, name: %s, email: %s }\\n", user->id, user->name, user->email);
}''',

    '''#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node* next;
} Node;

typedef struct {
    Node* head;
    Node* tail;
    size_t size;
} LinkedList;

LinkedList* list_create(void) {
    LinkedList* list = (LinkedList*)malloc(sizeof(LinkedList));
    if (!list) return NULL;
    list->head = NULL;
    list->tail = NULL;
    list->size = 0;
    return list;
}

int list_append(LinkedList* list, int data) {
    Node* node = (Node*)malloc(sizeof(Node));
    if (!node) return -1;
    
    node->data = data;
    node->next = NULL;
    
    if (list->tail) {
        list->tail->next = node;
    } else {
        list->head = node;
    }
    list->tail = node;
    list->size++;
    return 0;
}

void list_free(LinkedList* list) {
    Node* current = list->head;
    while (current) {
        Node* next = current->next;
        free(current);
        current = next;
    }
    free(list);
}''',

    '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HASH_SIZE 256

typedef struct HashEntry {
    char* key;
    void* value;
    struct HashEntry* next;
} HashEntry;

typedef struct {
    HashEntry* buckets[HASH_SIZE];
    size_t size;
} HashMap;

static unsigned int hash(const char* key) {
    unsigned int h = 0;
    while (*key) {
        h = h * 31 + *key++;
    }
    return h % HASH_SIZE;
}

HashMap* hashmap_create(void) {
    HashMap* map = (HashMap*)calloc(1, sizeof(HashMap));
    return map;
}

int hashmap_put(HashMap* map, const char* key, void* value) {
    unsigned int index = hash(key);
    
    HashEntry* entry = map->buckets[index];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            entry->value = value;
            return 0;
        }
        entry = entry->next;
    }
    
    HashEntry* new_entry = (HashEntry*)malloc(sizeof(HashEntry));
    if (!new_entry) return -1;
    
    new_entry->key = strdup(key);
    new_entry->value = value;
    new_entry->next = map->buckets[index];
    map->buckets[index] = new_entry;
    map->size++;
    return 0;
}

void* hashmap_get(HashMap* map, const char* key) {
    unsigned int index = hash(key);
    HashEntry* entry = map->buckets[index];
    
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}''',

    '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

typedef struct {
    char* data;
    size_t length;
    size_t capacity;
} String;

String* string_create(const char* initial) {
    String* str = (String*)malloc(sizeof(String));
    if (!str) return NULL;
    
    str->length = initial ? strlen(initial) : 0;
    str->capacity = str->length + 16;
    str->data = (char*)malloc(str->capacity);
    
    if (!str->data) {
        free(str);
        return NULL;
    }
    
    if (initial) {
        strcpy(str->data, initial);
    } else {
        str->data[0] = '\\0';
    }
    return str;
}

int string_append(String* str, const char* suffix) {
    size_t suffix_len = strlen(suffix);
    size_t new_length = str->length + suffix_len;
    
    if (new_length >= str->capacity) {
        size_t new_capacity = (new_length + 1) * 2;
        char* new_data = (char*)realloc(str->data, new_capacity);
        if (!new_data) return -1;
        str->data = new_data;
        str->capacity = new_capacity;
    }
    
    strcpy(str->data + str->length, suffix);
    str->length = new_length;
    return 0;
}

void string_free(String* str) {
    if (str) {
        free(str->data);
        free(str);
    }
}''',

    '''#include <stdio.h>
#include <stdlib.h>

typedef int (*Comparator)(const void*, const void*);

void swap(void* a, void* b, size_t size) {
    char* temp = (char*)malloc(size);
    memcpy(temp, a, size);
    memcpy(a, b, size);
    memcpy(b, temp, size);
    free(temp);
}

void quicksort(void* base, size_t nmemb, size_t size, Comparator cmp) {
    if (nmemb <= 1) return;
    
    char* arr = (char*)base;
    char* pivot = arr + (nmemb - 1) * size;
    size_t i = 0;
    
    for (size_t j = 0; j < nmemb - 1; j++) {
        if (cmp(arr + j * size, pivot) < 0) {
            swap(arr + i * size, arr + j * size, size);
            i++;
        }
    }
    swap(arr + i * size, pivot, size);
    
    quicksort(arr, i, size, cmp);
    quicksort(arr + (i + 1) * size, nmemb - i - 1, size, cmp);
}

int compare_int(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}''',

    '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 4096

char* read_file(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) return NULL;
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    
    size_t read = fread(buffer, 1, size, file);
    buffer[read] = '\\0';
    
    fclose(file);
    return buffer;
}

int write_file(const char* path, const char* content) {
    FILE* file = fopen(path, "w");
    if (!file) return -1;
    
    size_t len = strlen(content);
    size_t written = fwrite(content, 1, len, file);
    
    fclose(file);
    return (written == len) ? 0 : -1;
}

int append_file(const char* path, const char* content) {
    FILE* file = fopen(path, "a");
    if (!file) return -1;
    
    int result = fputs(content, file);
    fclose(file);
    return (result >= 0) ? 0 : -1;
}''',

    '''#include <stdlib.h>

typedef struct {
    int* data;
    size_t size;
    size_t capacity;
} Vector;

Vector* vector_create(size_t initial_capacity) {
    Vector* vec = (Vector*)malloc(sizeof(Vector));
    if (!vec) return NULL;
    
    vec->data = (int*)malloc(initial_capacity * sizeof(int));
    if (!vec->data) {
        free(vec);
        return NULL;
    }
    
    vec->size = 0;
    vec->capacity = initial_capacity;
    return vec;
}

int vector_push(Vector* vec, int value) {
    if (vec->size >= vec->capacity) {
        size_t new_capacity = vec->capacity * 2;
        int* new_data = (int*)realloc(vec->data, new_capacity * sizeof(int));
        if (!new_data) return -1;
        vec->data = new_data;
        vec->capacity = new_capacity;
    }
    vec->data[vec->size++] = value;
    return 0;
}

int vector_pop(Vector* vec) {
    if (vec->size == 0) return -1;
    return vec->data[--vec->size];
}

int vector_get(Vector* vec, size_t index) {
    if (index >= vec->size) return -1;
    return vec->data[index];
}

void vector_free(Vector* vec) {
    if (vec) {
        free(vec->data);
        free(vec);
    }
}''',

    '''#include <stdio.h>
#include <stdlib.h>

typedef struct TreeNode {
    int value;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

TreeNode* tree_create_node(int value) {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    if (!node) return NULL;
    node->value = value;
    node->left = NULL;
    node->right = NULL;
    return node;
}

TreeNode* tree_insert(TreeNode* root, int value) {
    if (!root) return tree_create_node(value);
    
    if (value < root->value) {
        root->left = tree_insert(root->left, value);
    } else {
        root->right = tree_insert(root->right, value);
    }
    return root;
}

TreeNode* tree_search(TreeNode* root, int value) {
    if (!root || root->value == value) return root;
    
    if (value < root->value) {
        return tree_search(root->left, value);
    }
    return tree_search(root->right, value);
}

void tree_inorder(TreeNode* root, void (*visit)(int)) {
    if (!root) return;
    tree_inorder(root->left, visit);
    visit(root->value);
    tree_inorder(root->right, visit);
}

void tree_free(TreeNode* root) {
    if (!root) return;
    tree_free(root->left);
    tree_free(root->right);
    free(root);
}''',

    '''#include <stdlib.h>

typedef struct {
    void** items;
    size_t front;
    size_t rear;
    size_t size;
    size_t capacity;
} Queue;

Queue* queue_create(size_t capacity) {
    Queue* q = (Queue*)malloc(sizeof(Queue));
    if (!q) return NULL;
    
    q->items = (void**)malloc(capacity * sizeof(void*));
    if (!q->items) {
        free(q);
        return NULL;
    }
    
    q->front = 0;
    q->rear = 0;
    q->size = 0;
    q->capacity = capacity;
    return q;
}

int queue_enqueue(Queue* q, void* item) {
    if (q->size >= q->capacity) return -1;
    
    q->items[q->rear] = item;
    q->rear = (q->rear + 1) % q->capacity;
    q->size++;
    return 0;
}

void* queue_dequeue(Queue* q) {
    if (q->size == 0) return NULL;
    
    void* item = q->items[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->size--;
    return item;
}

int queue_is_empty(Queue* q) {
    return q->size == 0;
}

void queue_free(Queue* q) {
    if (q) {
        free(q->items);
        free(q);
    }
}''',

    '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*ErrorHandler)(const char* message, int code);

typedef struct {
    int code;
    char message[256];
} Error;

static Error g_last_error = {0, ""};
static ErrorHandler g_error_handler = NULL;

void set_error_handler(ErrorHandler handler) {
    g_error_handler = handler;
}

void set_error(int code, const char* message) {
    g_last_error.code = code;
    strncpy(g_last_error.message, message, sizeof(g_last_error.message) - 1);
    g_last_error.message[sizeof(g_last_error.message) - 1] = '\\0';
    
    if (g_error_handler) {
        g_error_handler(message, code);
    }
}

int get_last_error(Error* error) {
    if (g_last_error.code == 0) return 0;
    
    if (error) {
        *error = g_last_error;
    }
    return 1;
}

void clear_error(void) {
    g_last_error.code = 0;
    g_last_error.message[0] = '\\0';
}''',

    '''#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    int* buffer;
    size_t capacity;
    size_t size;
    size_t front;
    size_t rear;
} BlockingQueue;

BlockingQueue* blocking_queue_create(size_t capacity) {
    BlockingQueue* q = (BlockingQueue*)malloc(sizeof(BlockingQueue));
    if (!q) return NULL;
    
    q->buffer = (int*)malloc(capacity * sizeof(int));
    if (!q->buffer) {
        free(q);
        return NULL;
    }
    
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
    q->capacity = capacity;
    q->size = 0;
    q->front = 0;
    q->rear = 0;
    return q;
}

void blocking_queue_put(BlockingQueue* q, int item) {
    pthread_mutex_lock(&q->mutex);
    
    while (q->size >= q->capacity) {
        pthread_cond_wait(&q->not_full, &q->mutex);
    }
    
    q->buffer[q->rear] = item;
    q->rear = (q->rear + 1) % q->capacity;
    q->size++;
    
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
}

int blocking_queue_take(BlockingQueue* q) {
    pthread_mutex_lock(&q->mutex);
    
    while (q->size == 0) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }
    
    int item = q->buffer[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->size--;
    
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return item;
}''',

    '''#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t block_size;
    size_t num_blocks;
    char* memory;
    int* free_list;
    size_t free_count;
} MemoryPool;

MemoryPool* pool_create(size_t block_size, size_t num_blocks) {
    MemoryPool* pool = (MemoryPool*)malloc(sizeof(MemoryPool));
    if (!pool) return NULL;
    
    pool->block_size = block_size;
    pool->num_blocks = num_blocks;
    pool->memory = (char*)malloc(block_size * num_blocks);
    pool->free_list = (int*)malloc(num_blocks * sizeof(int));
    
    if (!pool->memory || !pool->free_list) {
        free(pool->memory);
        free(pool->free_list);
        free(pool);
        return NULL;
    }
    
    for (size_t i = 0; i < num_blocks; i++) {
        pool->free_list[i] = i;
    }
    pool->free_count = num_blocks;
    return pool;
}

void* pool_alloc(MemoryPool* pool) {
    if (pool->free_count == 0) return NULL;
    
    int index = pool->free_list[--pool->free_count];
    return pool->memory + (index * pool->block_size);
}

void pool_free(MemoryPool* pool, void* ptr) {
    if (!ptr) return;
    
    size_t index = ((char*)ptr - pool->memory) / pool->block_size;
    if (index < pool->num_blocks) {
        pool->free_list[pool->free_count++] = index;
    }
}

void pool_destroy(MemoryPool* pool) {
    if (pool) {
        free(pool->memory);
        free(pool->free_list);
        free(pool);
    }
}''',

    '''#include <stdio.h>
#include <string.h>

typedef struct {
    char* buffer;
    size_t size;
    size_t pos;
} Buffer;

int buffer_init(Buffer* buf, char* data, size_t size) {
    buf->buffer = data;
    buf->size = size;
    buf->pos = 0;
    return 0;
}

int buffer_write_byte(Buffer* buf, unsigned char byte) {
    if (buf->pos >= buf->size) return -1;
    buf->buffer[buf->pos++] = byte;
    return 0;
}

int buffer_write_int(Buffer* buf, int value) {
    if (buf->pos + sizeof(int) > buf->size) return -1;
    memcpy(buf->buffer + buf->pos, &value, sizeof(int));
    buf->pos += sizeof(int);
    return 0;
}

int buffer_write_string(Buffer* buf, const char* str) {
    size_t len = strlen(str) + 1;
    if (buf->pos + len > buf->size) return -1;
    memcpy(buf->buffer + buf->pos, str, len);
    buf->pos += len;
    return 0;
}

int buffer_read_byte(Buffer* buf, unsigned char* byte) {
    if (buf->pos >= buf->size) return -1;
    *byte = buf->buffer[buf->pos++];
    return 0;
}

int buffer_read_int(Buffer* buf, int* value) {
    if (buf->pos + sizeof(int) > buf->size) return -1;
    memcpy(value, buf->buffer + buf->pos, sizeof(int));
    buf->pos += sizeof(int);
    return 0;
}''',

    '''#include <stdio.h>
#include <string.h>
#include <ctype.h>

typedef enum {
    TOKEN_NUMBER,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_END,
    TOKEN_ERROR
} TokenType;

typedef struct {
    TokenType type;
    double value;
} Token;

typedef struct {
    const char* input;
    size_t pos;
    Token current;
} Lexer;

void lexer_init(Lexer* lex, const char* input) {
    lex->input = input;
    lex->pos = 0;
}

Token lexer_next(Lexer* lex) {
    Token token = {TOKEN_ERROR, 0};
    
    while (isspace(lex->input[lex->pos])) lex->pos++;
    
    char c = lex->input[lex->pos];
    if (c == '\\0') { token.type = TOKEN_END; return token; }
    
    if (isdigit(c) || c == '.') {
        char* end;
        token.value = strtod(lex->input + lex->pos, &end);
        token.type = TOKEN_NUMBER;
        lex->pos = end - lex->input;
        return token;
    }
    
    lex->pos++;
    switch (c) {
        case '+': token.type = TOKEN_PLUS; break;
        case '-': token.type = TOKEN_MINUS; break;
        case '*': token.type = TOKEN_MULTIPLY; break;
        case '/': token.type = TOKEN_DIVIDE; break;
        case '(': token.type = TOKEN_LPAREN; break;
        case ')': token.type = TOKEN_RPAREN; break;
    }
    return token;
}''',

    '''#include <stdlib.h>
#include <string.h>

typedef struct {
    char** items;
    size_t count;
    size_t capacity;
} StringArray;

StringArray* string_array_create(void) {
    StringArray* arr = (StringArray*)malloc(sizeof(StringArray));
    if (!arr) return NULL;
    
    arr->items = NULL;
    arr->count = 0;
    arr->capacity = 0;
    return arr;
}

int string_array_add(StringArray* arr, const char* str) {
    if (arr->count >= arr->capacity) {
        size_t new_capacity = arr->capacity == 0 ? 8 : arr->capacity * 2;
        char** new_items = (char**)realloc(arr->items, new_capacity * sizeof(char*));
        if (!new_items) return -1;
        arr->items = new_items;
        arr->capacity = new_capacity;
    }
    
    arr->items[arr->count] = strdup(str);
    if (!arr->items[arr->count]) return -1;
    arr->count++;
    return 0;
}

const char* string_array_get(StringArray* arr, size_t index) {
    if (index >= arr->count) return NULL;
    return arr->items[index];
}

void string_array_free(StringArray* arr) {
    if (!arr) return;
    for (size_t i = 0; i < arr->count; i++) {
        free(arr->items[i]);
    }
    free(arr->items);
    free(arr);
}''',

    '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ENTRIES 100

typedef struct {
    char key[64];
    char value[256];
} ConfigEntry;

typedef struct {
    ConfigEntry entries[MAX_ENTRIES];
    size_t count;
} Config;

Config* config_load(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) return NULL;
    
    Config* config = (Config*)calloc(1, sizeof(Config));
    if (!config) {
        fclose(file);
        return NULL;
    }
    
    char line[320];
    while (fgets(line, sizeof(line), file) && config->count < MAX_ENTRIES) {
        char* eq = strchr(line, '=');
        if (!eq) continue;
        
        *eq = '\\0';
        char* key = line;
        char* value = eq + 1;
        
        while (*key == ' ') key++;
        char* end = eq - 1;
        while (end > key && *end == ' ') *end-- = '\\0';
        
        while (*value == ' ') value++;
        size_t len = strlen(value);
        if (len > 0 && value[len-1] == '\\n') value[len-1] = '\\0';
        
        strncpy(config->entries[config->count].key, key, 63);
        strncpy(config->entries[config->count].value, value, 255);
        config->count++;
    }
    
    fclose(file);
    return config;
}

const char* config_get(Config* config, const char* key) {
    for (size_t i = 0; i < config->count; i++) {
        if (strcmp(config->entries[i].key, key) == 0) {
            return config->entries[i].value;
        }
    }
    return NULL;
}''',
]

RUBY_SAMPLES = [
    # Classes and Modules
    '''class User
  attr_accessor :name, :email
  attr_reader :id
  
  def initialize(name, email)
    @id = SecureRandom.uuid
    @name = name
    @email = email
    @created_at = Time.now
  end
  
  def to_s
    "User(#{@id}): #{@name} <#{@email}>"
  end
  
  def valid?
    @name.present? && @email =~ /\A[\w+\-.]+@[a-z\d\-]+(\.[a-z\d\-]+)*\.[a-z]+\z/i
  end
end''',

    # Modules and Mixins
    '''module Searchable
  extend ActiveSupport::Concern
  
  included do
    scope :search, ->(query) { where("name ILIKE ?", "%#{query}%") }
  end
  
  class_methods do
    def find_by_query(query, limit: 10)
      search(query).limit(limit)
    end
  end
  
  def matches?(term)
    name.downcase.include?(term.downcase)
  end
end

module Cacheable
  def cache_key
    "#{self.class.name.underscore}/#{id}-#{updated_at.to_i}"
  end
  
  def cached_data
    Rails.cache.fetch(cache_key, expires_in: 1.hour) do
      to_json
    end
  end
end''',

    # Blocks, Procs and Lambdas
    '''# Blocks
[1, 2, 3].map { |x| x * 2 }

[1, 2, 3].each do |num|
  puts num
end

# Procs
square = Proc.new { |x| x ** 2 }
square.call(5)  # => 25

multiply = proc { |a, b| a * b }
multiply.call(3, 4)  # => 12

# Lambdas
greet = lambda { |name| "Hello, #{name}!" }
greet.call("World")  # => "Hello, World!"

add = ->(a, b) { a + b }
add.call(2, 3)  # => 5

# Differences
proc_example = Proc.new { |x, y| [x, y] }
lambda_example = ->(x, y) { [x, y] }
proc_example.call(1)  # => [1, nil]
# lambda_example.call(1)  # ArgumentError''',

    # Rails Controller
    '''class ArticlesController < ApplicationController
  before_action :set_article, only: [:show, :edit, :update, :destroy]
  before_action :authenticate_user!, except: [:index, :show]
  
  def index
    @articles = Article.published.includes(:author, :tags)
                       .page(params[:page]).per(20)
    
    respond_to do |format|
      format.html
      format.json { render json: @articles }
    end
  end
  
  def create
    @article = current_user.articles.build(article_params)
    
    if @article.save
      redirect_to @article, notice: 'Article was successfully created.'
    else
      render :new, status: :unprocessable_entity
    end
  end
  
  private
  
  def set_article
    @article = Article.find(params[:id])
  end
  
  def article_params
    params.require(:article).permit(:title, :body, :status, tag_ids: [])
  end
end''',

    # Rails Model with Associations
    '''class Article < ApplicationRecord
  belongs_to :author, class_name: 'User', foreign_key: 'user_id'
  has_many :comments, dependent: :destroy
  has_many :taggings, dependent: :destroy
  has_many :tags, through: :taggings
  has_one_attached :cover_image
  
  validates :title, presence: true, length: { minimum: 5, maximum: 200 }
  validates :body, presence: true
  validates :slug, uniqueness: true
  
  scope :published, -> { where(status: 'published') }
  scope :recent, -> { order(created_at: :desc) }
  scope :by_tag, ->(tag) { joins(:tags).where(tags: { name: tag }) }
  
  before_validation :generate_slug, on: :create
  after_create :notify_subscribers
  
  def excerpt(length = 200)
    ActionController::Base.helpers.truncate(body, length: length)
  end
  
  private
  
  def generate_slug
    self.slug = title.parameterize
  end
  
  def notify_subscribers
    ArticleNotificationJob.perform_later(self)
  end
end''',

    # Metaprogramming
    '''class DynamicFinder
  def method_missing(method_name, *args, &block)
    if method_name.to_s.start_with?('find_by_')
      attribute = method_name.to_s.sub('find_by_', '')
      define_finder_method(attribute)
      send(method_name, *args)
    else
      super
    end
  end
  
  def respond_to_missing?(method_name, include_private = false)
    method_name.to_s.start_with?('find_by_') || super
  end
  
  private
  
  def define_finder_method(attribute)
    self.class.define_method("find_by_#{attribute}") do |value|
      @records.find { |r| r.send(attribute) == value }
    end
  end
end

# Class macro pattern
module Validatable
  def self.included(base)
    base.extend(ClassMethods)
  end
  
  module ClassMethods
    def validates_presence_of(*attrs)
      attrs.each do |attr|
        define_method("#{attr}_valid?") do
          !send(attr).nil? && !send(attr).empty?
        end
      end
    end
  end
end''',

    # ActiveRecord Query Interface
    '''class UserQuery
  def initialize(relation = User.all)
    @relation = relation
  end
  
  def active
    @relation = @relation.where(active: true)
    self
  end
  
  def with_role(role)
    @relation = @relation.where(role: role)
    self
  end
  
  def created_after(date)
    @relation = @relation.where('created_at > ?', date)
    self
  end
  
  def order_by_recent
    @relation = @relation.order(created_at: :desc)
    self
  end
  
  def with_posts_count
    @relation = @relation.left_joins(:posts)
                         .select('users.*, COUNT(posts.id) as posts_count')
                         .group('users.id')
    self
  end
  
  def to_a
    @relation.to_a
  end
  
  def each(&block)
    @relation.each(&block)
  end
end

# Usage
UserQuery.new.active.with_role('admin').order_by_recent.to_a''',

    # Background Jobs (Sidekiq)
    '''class ImportUsersJob
  include Sidekiq::Job
  sidekiq_options queue: :default, retry: 3
  
  def perform(file_path, options = {})
    batch_size = options.fetch('batch_size', 1000)
    
    CSV.foreach(file_path, headers: true).each_slice(batch_size) do |batch|
      users = batch.map do |row|
        {
          name: row['name'],
          email: row['email'],
          created_at: Time.current,
          updated_at: Time.current
        }
      end
      
      User.insert_all(users)
    end
    
    Rails.logger.info "Imported users from #{file_path}"
  rescue StandardError => e
    Rails.logger.error "Import failed: #{e.message}"
    raise
  end
end''',

    # Service Object Pattern
    '''class CreateOrder
  Result = Struct.new(:success?, :order, :errors, keyword_init: true)
  
  def initialize(user:, items:, coupon_code: nil)
    @user = user
    @items = items
    @coupon_code = coupon_code
    @errors = []
  end
  
  def call
    ActiveRecord::Base.transaction do
      validate_items!
      order = build_order
      apply_discount(order) if @coupon_code
      order.save!
      process_payment(order)
      send_confirmation(order)
      
      Result.new(success?: true, order: order)
    end
  rescue ActiveRecord::RecordInvalid => e
    Result.new(success?: false, errors: e.record.errors.full_messages)
  rescue ValidationError => e
    Result.new(success?: false, errors: [e.message])
  end
  
  private
  
  def validate_items!
    @items.each do |item|
      unless item[:product].in_stock?(item[:quantity])
        raise ValidationError, "#{item[:product].name} is out of stock"
      end
    end
  end
  
  def build_order
    @user.orders.build(
      items: @items.map { |i| OrderItem.new(product: i[:product], quantity: i[:quantity]) },
      total: calculate_total
    )
  end
end''',

    # RSpec Testing
    '''RSpec.describe User, type: :model do
  describe 'validations' do
    it { is_expected.to validate_presence_of(:email) }
    it { is_expected.to validate_uniqueness_of(:email).case_insensitive }
    it { is_expected.to have_many(:posts).dependent(:destroy) }
  end
  
  describe '#full_name' do
    subject(:user) { build(:user, first_name: 'John', last_name: 'Doe') }
    
    it 'returns the full name' do
      expect(user.full_name).to eq('John Doe')
    end
  end
  
  describe '.active' do
    let!(:active_user) { create(:user, active: true) }
    let!(:inactive_user) { create(:user, active: false) }
    
    it 'returns only active users' do
      expect(User.active).to contain_exactly(active_user)
    end
  end
  
  describe '#premium?' do
    context 'when subscription is active' do
      before { allow(user).to receive(:subscription).and_return(double(active?: true)) }
      
      it 'returns true' do
        expect(user.premium?).to be true
      end
    end
  end
end''',

    # Custom DSL
    '''class QueryBuilder
  class << self
    def build(&block)
      builder = new
      builder.instance_eval(&block)
      builder.to_sql
    end
  end
  
  def initialize
    @selects = []
    @conditions = []
    @table = nil
  end
  
  def from(table)
    @table = table
    self
  end
  
  def select(*columns)
    @selects.concat(columns)
    self
  end
  
  def where(condition)
    @conditions << condition
    self
  end
  
  def to_sql
    sql = "SELECT #{@selects.join(', ')} FROM #{@table}"
    sql += " WHERE #{@conditions.join(' AND ')}" if @conditions.any?
    sql
  end
end

# Usage
QueryBuilder.build do
  from :users
  select :id, :name, :email
  where "active = true"
  where "role = 'admin'"
end''',

    # Enumerable Methods
    '''numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Map and collect
squares = numbers.map { |n| n ** 2 }
doubles = numbers.collect(&:to_s)

# Select and reject
evens = numbers.select(&:even?)
odds = numbers.reject(&:even?)

# Reduce and inject
sum = numbers.reduce(0, :+)
product = numbers.inject(1) { |acc, n| acc * n }

# Find and detect
first_even = numbers.find(&:even?)
first_over_5 = numbers.detect { |n| n > 5 }

# Group and partition
by_parity = numbers.group_by { |n| n.even? ? :even : :odd }
evens_odds = numbers.partition(&:even?)

# Chain methods
result = numbers.select { |n| n > 3 }
                .map { |n| n * 2 }
                .take(3)
                .reduce(:+)''',

    # Error Handling
    '''class ApiClient
  class ApiError < StandardError; end
  class RateLimitError < ApiError; end
  class AuthenticationError < ApiError; end
  
  def fetch(endpoint)
    retries = 0
    
    begin
      response = make_request(endpoint)
      parse_response(response)
    rescue Net::OpenTimeout, Net::ReadTimeout => e
      retries += 1
      retry if retries < 3
      raise ApiError, "Request timed out after #{retries} attempts"
    rescue JSON::ParserError => e
      raise ApiError, "Invalid JSON response: #{e.message}"
    rescue => e
      handle_error(e)
    ensure
      log_request(endpoint, response)
    end
  end
  
  private
  
  def handle_error(error)
    case error
    when Net::HTTPUnauthorized
      raise AuthenticationError, "Invalid credentials"
    when Net::HTTPTooManyRequests
      raise RateLimitError, "Rate limit exceeded"
    else
      raise ApiError, "Request failed: #{error.message}"
    end
  end
end''',

    # Concern Pattern
    '''module Trackable
  extend ActiveSupport::Concern
  
  included do
    has_many :activity_logs, as: :trackable, dependent: :destroy
    
    after_create :log_creation
    after_update :log_update
    after_destroy :log_destruction
  end
  
  class_methods do
    def track_fields(*fields)
      @tracked_fields = fields
    end
    
    def tracked_fields
      @tracked_fields || []
    end
  end
  
  def activity_summary
    activity_logs.order(created_at: :desc).limit(10)
  end
  
  private
  
  def log_creation
    activity_logs.create!(action: 'created', data: attributes)
  end
  
  def log_update
    return unless saved_changes.any?
    
    changes = saved_changes.slice(*self.class.tracked_fields)
    activity_logs.create!(action: 'updated', data: changes) if changes.any?
  end
  
  def log_destruction
    activity_logs.create!(action: 'destroyed', data: { id: id })
  end
end''',

    # Struct and OpenStruct
    '''# Struct
Point = Struct.new(:x, :y) do
  def distance_from_origin
    Math.sqrt(x ** 2 + y ** 2)
  end
  
  def to_s
    "(#{x}, #{y})"
  end
end

# OpenStruct for dynamic attributes
require 'ostruct'

config = OpenStruct.new(
  database: 'myapp',
  host: 'localhost',
  port: 5432
)
config.username = 'admin'

# Data class (Ruby 3.2+)
Person = Data.define(:name, :age) do
  def adult?
    age >= 18
  end
end

john = Person.new(name: 'John', age: 30)''',
]

PHP_SAMPLES = [
    # Classes and Interfaces
    '''<?php
declare(strict_types=1);

interface PaymentGatewayInterface
{
    public function charge(float $amount, string $currency): PaymentResult;
    public function refund(string $transactionId, float $amount): bool;
    public function getTransactionStatus(string $transactionId): string;
}

class StripeGateway implements PaymentGatewayInterface
{
    private string $apiKey;
    private HttpClientInterface $httpClient;
    
    public function __construct(string $apiKey, HttpClientInterface $httpClient)
    {
        $this->apiKey = $apiKey;
        $this->httpClient = $httpClient;
    }
    
    public function charge(float $amount, string $currency): PaymentResult
    {
        $response = $this->httpClient->post('/charges', [
            'amount' => (int) ($amount * 100),
            'currency' => $currency,
        ]);
        
        return new PaymentResult(
            success: $response['status'] === 'succeeded',
            transactionId: $response['id'],
            message: $response['message'] ?? null
        );
    }
    
    public function refund(string $transactionId, float $amount): bool
    {
        return $this->httpClient->post("/refunds", [
            'charge' => $transactionId,
            'amount' => (int) ($amount * 100),
        ])['status'] === 'succeeded';
    }
    
    public function getTransactionStatus(string $transactionId): string
    {
        return $this->httpClient->get("/charges/{$transactionId}")['status'];
    }
}''',

    # Traits
    '''<?php
trait Timestampable
{
    protected ?DateTime $createdAt = null;
    protected ?DateTime $updatedAt = null;
    
    public function setCreatedAt(DateTime $createdAt): self
    {
        $this->createdAt = $createdAt;
        return $this;
    }
    
    public function getCreatedAt(): ?DateTime
    {
        return $this->createdAt;
    }
    
    public function setUpdatedAt(DateTime $updatedAt): self
    {
        $this->updatedAt = $updatedAt;
        return $this;
    }
    
    public function touch(): self
    {
        $this->updatedAt = new DateTime();
        return $this;
    }
}

trait SoftDeletes
{
    protected ?DateTime $deletedAt = null;
    
    public function delete(): void
    {
        $this->deletedAt = new DateTime();
    }
    
    public function restore(): void
    {
        $this->deletedAt = null;
    }
    
    public function isDeleted(): bool
    {
        return $this->deletedAt !== null;
    }
    
    public function forceDelete(): void
    {
        // Actual deletion logic
    }
}

class Article
{
    use Timestampable, SoftDeletes;
    
    private int $id;
    private string $title;
    private string $content;
}''',

    # Laravel Controller
    '''<?php

namespace App\\Http\\Controllers;

use App\\Models\\Article;
use App\\Http\\Requests\\StoreArticleRequest;
use App\\Http\\Resources\\ArticleResource;
use Illuminate\\Http\\JsonResponse;
use Illuminate\\Http\\Request;

class ArticleController extends Controller
{
    public function __construct()
    {
        $this->middleware('auth:api')->except(['index', 'show']);
    }
    
    public function index(Request $request): JsonResponse
    {
        $articles = Article::query()
            ->when($request->has('category'), fn($q) => $q->where('category_id', $request->category))
            ->when($request->has('search'), fn($q) => $q->where('title', 'like', "%{$request->search}%"))
            ->with(['author', 'tags'])
            ->published()
            ->latest()
            ->paginate($request->get('per_page', 15));
        
        return response()->json([
            'data' => ArticleResource::collection($articles),
            'meta' => [
                'total' => $articles->total(),
                'per_page' => $articles->perPage(),
                'current_page' => $articles->currentPage(),
            ]
        ]);
    }
    
    public function store(StoreArticleRequest $request): JsonResponse
    {
        $article = $request->user()->articles()->create($request->validated());
        $article->tags()->attach($request->input('tags', []));
        
        return response()->json(new ArticleResource($article), 201);
    }
    
    public function update(StoreArticleRequest $request, Article $article): JsonResponse
    {
        $this->authorize('update', $article);
        
        $article->update($request->validated());
        $article->tags()->sync($request->input('tags', []));
        
        return response()->json(new ArticleResource($article->fresh()));
    }
}''',

    # Laravel Model with Eloquent
    '''<?php

namespace App\\Models;

use Illuminate\\Database\\Eloquent\\Model;
use Illuminate\\Database\\Eloquent\\Relations\\BelongsTo;
use Illuminate\\Database\\Eloquent\\Relations\\HasMany;
use Illuminate\\Database\\Eloquent\\Relations\\BelongsToMany;
use Illuminate\\Database\\Eloquent\\SoftDeletes;
use Illuminate\\Database\\Eloquent\\Factories\\HasFactory;
use Illuminate\\Database\\Eloquent\\Builder;

class Article extends Model
{
    use HasFactory, SoftDeletes;
    
    protected $fillable = ['title', 'slug', 'content', 'status', 'published_at'];
    
    protected $casts = [
        'published_at' => 'datetime',
        'is_featured' => 'boolean',
        'meta' => 'array',
    ];
    
    protected $appends = ['excerpt', 'reading_time'];
    
    public function author(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id');
    }
    
    public function comments(): HasMany
    {
        return $this->hasMany(Comment::class)->latest();
    }
    
    public function tags(): BelongsToMany
    {
        return $this->belongsToMany(Tag::class)->withTimestamps();
    }
    
    public function scopePublished(Builder $query): Builder
    {
        return $query->where('status', 'published')
                     ->whereNotNull('published_at')
                     ->where('published_at', '<=', now());
    }
    
    public function scopeFeatured(Builder $query): Builder
    {
        return $query->where('is_featured', true);
    }
    
    public function getExcerptAttribute(): string
    {
        return Str::limit(strip_tags($this->content), 200);
    }
    
    public function getReadingTimeAttribute(): int
    {
        return (int) ceil(str_word_count($this->content) / 200);
    }
}''',

    # PDO Database Access
    '''<?php

class DatabaseConnection
{
    private static ?PDO $instance = null;
    
    public static function getInstance(): PDO
    {
        if (self::$instance === null) {
            $dsn = sprintf(
                "mysql:host=%s;dbname=%s;charset=utf8mb4",
                $_ENV['DB_HOST'],
                $_ENV['DB_NAME']
            );
            
            self::$instance = new PDO($dsn, $_ENV['DB_USER'], $_ENV['DB_PASS'], [
                PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                PDO::ATTR_EMULATE_PREPARES => false,
            ]);
        }
        
        return self::$instance;
    }
}

class UserRepository
{
    private PDO $db;
    
    public function __construct()
    {
        $this->db = DatabaseConnection::getInstance();
    }
    
    public function findById(int $id): ?array
    {
        $stmt = $this->db->prepare("SELECT * FROM users WHERE id = :id");
        $stmt->execute(['id' => $id]);
        return $stmt->fetch() ?: null;
    }
    
    public function create(array $data): int
    {
        $stmt = $this->db->prepare(
            "INSERT INTO users (name, email, password, created_at) VALUES (:name, :email, :password, NOW())"
        );
        $stmt->execute([
            'name' => $data['name'],
            'email' => $data['email'],
            'password' => password_hash($data['password'], PASSWORD_BCRYPT),
        ]);
        return (int) $this->db->lastInsertId();
    }
    
    public function update(int $id, array $data): bool
    {
        $sets = [];
        $params = ['id' => $id];
        
        foreach ($data as $key => $value) {
            $sets[] = "{$key} = :{$key}";
            $params[$key] = $value;
        }
        
        $stmt = $this->db->prepare(
            "UPDATE users SET " . implode(', ', $sets) . " WHERE id = :id"
        );
        return $stmt->execute($params);
    }
}''',

    # Dependency Injection Container
    '''<?php

class Container
{
    private array $bindings = [];
    private array $instances = [];
    
    public function bind(string $abstract, callable|string $concrete): void
    {
        $this->bindings[$abstract] = $concrete;
    }
    
    public function singleton(string $abstract, callable|string $concrete): void
    {
        $this->bind($abstract, function($container) use ($abstract, $concrete) {
            if (!isset($this->instances[$abstract])) {
                $this->instances[$abstract] = is_callable($concrete)
                    ? $concrete($container)
                    : $container->resolve($concrete);
            }
            return $this->instances[$abstract];
        });
    }
    
    public function resolve(string $abstract): object
    {
        if (isset($this->bindings[$abstract])) {
            $concrete = $this->bindings[$abstract];
            return is_callable($concrete) ? $concrete($this) : $this->build($concrete);
        }
        
        return $this->build($abstract);
    }
    
    private function build(string $concrete): object
    {
        $reflector = new ReflectionClass($concrete);
        $constructor = $reflector->getConstructor();
        
        if ($constructor === null) {
            return new $concrete();
        }
        
        $dependencies = array_map(function($param) {
            $type = $param->getType();
            if ($type instanceof ReflectionNamedType && !$type->isBuiltin()) {
                return $this->resolve($type->getName());
            }
            return $param->isDefaultValueAvailable() ? $param->getDefaultValue() : null;
        }, $constructor->getParameters());
        
        return $reflector->newInstanceArgs($dependencies);
    }
}''',

    # Request Validation
    '''<?php

namespace App\\Http\\Requests;

use Illuminate\\Foundation\\Http\\FormRequest;
use Illuminate\\Validation\\Rule;

class StoreUserRequest extends FormRequest
{
    public function authorize(): bool
    {
        return $this->user()->can('create', User::class);
    }
    
    public function rules(): array
    {
        return [
            'name' => ['required', 'string', 'min:2', 'max:255'],
            'email' => [
                'required',
                'email',
                Rule::unique('users')->ignore($this->user),
            ],
            'password' => ['required', 'string', 'min:8', 'confirmed'],
            'role' => ['required', Rule::in(['admin', 'editor', 'user'])],
            'profile.bio' => ['nullable', 'string', 'max:1000'],
            'profile.avatar' => ['nullable', 'image', 'max:2048'],
            'permissions' => ['array'],
            'permissions.*' => ['exists:permissions,id'],
        ];
    }
    
    public function messages(): array
    {
        return [
            'email.unique' => 'This email is already registered.',
            'password.min' => 'Password must be at least 8 characters.',
        ];
    }
    
    protected function prepareForValidation(): void
    {
        $this->merge([
            'email' => strtolower($this->email),
            'name' => trim($this->name),
        ]);
    }
}''',

    # Middleware
    '''<?php

namespace App\\Http\\Middleware;

use Closure;
use Illuminate\\Http\\Request;
use Illuminate\\Support\\Facades\\RateLimiter;
use Symfony\\Component\\HttpFoundation\\Response;

class ApiRateLimiter
{
    public function handle(Request $request, Closure $next, string $limiterName = 'api'): Response
    {
        $key = $this->resolveRequestSignature($request);
        
        if (RateLimiter::tooManyAttempts($key, $this->maxAttempts($limiterName))) {
            return response()->json([
                'message' => 'Too many requests',
                'retry_after' => RateLimiter::availableIn($key),
            ], 429)->withHeaders([
                'X-RateLimit-Limit' => $this->maxAttempts($limiterName),
                'X-RateLimit-Remaining' => RateLimiter::remaining($key, $this->maxAttempts($limiterName)),
                'Retry-After' => RateLimiter::availableIn($key),
            ]);
        }
        
        RateLimiter::hit($key);
        
        $response = $next($request);
        
        return $response->withHeaders([
            'X-RateLimit-Limit' => $this->maxAttempts($limiterName),
            'X-RateLimit-Remaining' => RateLimiter::remaining($key, $this->maxAttempts($limiterName)),
        ]);
    }
    
    protected function resolveRequestSignature(Request $request): string
    {
        return sha1(implode('|', [
            $request->method(),
            $request->path(),
            $request->ip(),
            $request->user()?->id,
        ]));
    }
    
    protected function maxAttempts(string $limiterName): int
    {
        return config("rate_limits.{$limiterName}", 60);
    }
}''',

    # Event and Listener
    '''<?php

namespace App\\Events;

use App\\Models\\Order;
use Illuminate\\Broadcasting\\Channel;
use Illuminate\\Broadcasting\\InteractsWithSockets;
use Illuminate\\Contracts\\Broadcasting\\ShouldBroadcast;
use Illuminate\\Foundation\\Events\\Dispatchable;
use Illuminate\\Queue\\SerializesModels;

class OrderPlaced implements ShouldBroadcast
{
    use Dispatchable, InteractsWithSockets, SerializesModels;
    
    public function __construct(public Order $order)
    {
    }
    
    public function broadcastOn(): array
    {
        return [
            new Channel('orders'),
            new Channel("user.{$this->order->user_id}"),
        ];
    }
    
    public function broadcastWith(): array
    {
        return [
            'id' => $this->order->id,
            'total' => $this->order->total,
            'status' => $this->order->status,
        ];
    }
}

namespace App\\Listeners;

use App\\Events\\OrderPlaced;
use App\\Mail\\OrderConfirmation;
use Illuminate\\Contracts\\Queue\\ShouldQueue;
use Illuminate\\Support\\Facades\\Mail;

class SendOrderConfirmation implements ShouldQueue
{
    public $queue = 'emails';
    public $delay = 10;
    
    public function handle(OrderPlaced $event): void
    {
        Mail::to($event->order->user->email)
            ->send(new OrderConfirmation($event->order));
    }
    
    public function failed(OrderPlaced $event, \\Throwable $exception): void
    {
        Log::error('Failed to send order confirmation', [
            'order_id' => $event->order->id,
            'error' => $exception->getMessage(),
        ]);
    }
}''',

    # Repository Pattern
    '''<?php

namespace App\\Repositories;

use App\\Models\\Product;
use Illuminate\\Contracts\\Pagination\\LengthAwarePaginator;
use Illuminate\\Database\\Eloquent\\Collection;

interface ProductRepositoryInterface
{
    public function all(): Collection;
    public function find(int $id): ?Product;
    public function create(array $data): Product;
    public function update(int $id, array $data): bool;
    public function delete(int $id): bool;
    public function paginate(int $perPage = 15): LengthAwarePaginator;
}

class ProductRepository implements ProductRepositoryInterface
{
    public function __construct(private Product $model)
    {
    }
    
    public function all(): Collection
    {
        return $this->model->with(['category', 'tags'])->get();
    }
    
    public function find(int $id): ?Product
    {
        return $this->model->with(['category', 'tags', 'reviews'])->find($id);
    }
    
    public function create(array $data): Product
    {
        $product = $this->model->create($data);
        
        if (isset($data['tags'])) {
            $product->tags()->attach($data['tags']);
        }
        
        return $product->fresh(['category', 'tags']);
    }
    
    public function update(int $id, array $data): bool
    {
        $product = $this->model->findOrFail($id);
        
        if (isset($data['tags'])) {
            $product->tags()->sync($data['tags']);
            unset($data['tags']);
        }
        
        return $product->update($data);
    }
    
    public function delete(int $id): bool
    {
        return $this->model->findOrFail($id)->delete();
    }
    
    public function paginate(int $perPage = 15): LengthAwarePaginator
    {
        return $this->model->with(['category'])->latest()->paginate($perPage);
    }
}''',

    # Queue Jobs
    '''<?php

namespace App\\Jobs;

use App\\Models\\Order;
use App\\Services\\PaymentService;
use Illuminate\\Bus\\Queueable;
use Illuminate\\Contracts\\Queue\\ShouldQueue;
use Illuminate\\Foundation\\Bus\\Dispatchable;
use Illuminate\\Queue\\InteractsWithQueue;
use Illuminate\\Queue\\SerializesModels;
use Illuminate\\Support\\Facades\\DB;

class ProcessOrder implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;
    
    public $tries = 3;
    public $backoff = [10, 60, 300];
    public $timeout = 120;
    
    public function __construct(private Order $order)
    {
    }
    
    public function handle(PaymentService $paymentService): void
    {
        DB::transaction(function () use ($paymentService) {
            $this->order->update(['status' => 'processing']);
            
            foreach ($this->order->items as $item) {
                $item->product->decrement('stock', $item->quantity);
            }
            
            $result = $paymentService->charge(
                $this->order->total,
                $this->order->payment_method_id
            );
            
            if ($result->success) {
                $this->order->update([
                    'status' => 'completed',
                    'transaction_id' => $result->transactionId,
                ]);
            } else {
                throw new PaymentException($result->message);
            }
        });
    }
    
    public function failed(\\Throwable $exception): void
    {
        $this->order->update(['status' => 'failed']);
        
        Notification::send(
            $this->order->user,
            new OrderFailedNotification($this->order, $exception->getMessage())
        );
    }
}''',

    # API Resource
    '''<?php

namespace App\\Http\\Resources;

use Illuminate\\Http\\Request;
use Illuminate\\Http\\Resources\\Json\\JsonResource;

class ProductResource extends JsonResource
{
    public function toArray(Request $request): array
    {
        return [
            'id' => $this->id,
            'name' => $this->name,
            'slug' => $this->slug,
            'description' => $this->description,
            'price' => [
                'amount' => $this->price,
                'formatted' => number_format($this->price, 2),
                'currency' => 'USD',
            ],
            'stock' => $this->when($request->user()?->isAdmin(), $this->stock),
            'in_stock' => $this->stock > 0,
            'category' => new CategoryResource($this->whenLoaded('category')),
            'tags' => TagResource::collection($this->whenLoaded('tags')),
            'images' => $this->images->map(fn($img) => [
                'url' => $img->url,
                'thumbnail' => $img->thumbnail_url,
            ]),
            'rating' => [
                'average' => round($this->reviews_avg_rating, 1),
                'count' => $this->reviews_count,
            ],
            'created_at' => $this->created_at->toIso8601String(),
            'links' => [
                'self' => route('products.show', $this),
                'category' => route('categories.show', $this->category),
            ],
        ];
    }
    
    public function with(Request $request): array
    {
        return [
            'meta' => [
                'version' => '1.0',
                'timestamp' => now()->toIso8601String(),
            ],
        ];
    }
}''',

    # Service Provider
    '''<?php

namespace App\\Providers;

use App\\Repositories\\ProductRepository;
use App\\Repositories\\ProductRepositoryInterface;
use App\\Services\\PaymentService;
use App\\Services\\StripePaymentService;
use Illuminate\\Support\\ServiceProvider;

class AppServiceProvider extends ServiceProvider
{
    public array $bindings = [
        ProductRepositoryInterface::class => ProductRepository::class,
    ];
    
    public function register(): void
    {
        $this->app->singleton(PaymentService::class, function ($app) {
            return new StripePaymentService(
                config('services.stripe.key'),
                config('services.stripe.secret')
            );
        });
        
        $this->app->when(OrderController::class)
            ->needs(PaymentService::class)
            ->give(StripePaymentService::class);
    }
    
    public function boot(): void
    {
        Model::preventLazyLoading(!$this->app->isProduction());
        
        DB::whenQueryingForLongerThan(500, function ($connection, $event) {
            Log::warning('Slow query detected', [
                'sql' => $event->sql,
                'time' => $event->time,
            ]);
        });
        
        View::composer('partials.navigation', function ($view) {
            $view->with('categories', Category::cached());
        });
    }
}''',

    # Testing
    '''<?php

namespace Tests\\Feature;

use App\\Models\\User;
use App\\Models\\Article;
use Illuminate\\Foundation\\Testing\\RefreshDatabase;
use Tests\\TestCase;

class ArticleApiTest extends TestCase
{
    use RefreshDatabase;
    
    public function test_can_list_articles(): void
    {
        Article::factory()->count(5)->create();
        
        $response = $this->getJson('/api/articles');
        
        $response->assertOk()
            ->assertJsonCount(5, 'data')
            ->assertJsonStructure([
                'data' => [['id', 'title', 'excerpt', 'author']],
                'meta' => ['total', 'per_page', 'current_page'],
            ]);
    }
    
    public function test_authenticated_user_can_create_article(): void
    {
        $user = User::factory()->create();
        
        $response = $this->actingAs($user)
            ->postJson('/api/articles', [
                'title' => 'Test Article',
                'content' => 'Article content here...',
                'status' => 'published',
            ]);
        
        $response->assertCreated()
            ->assertJsonPath('data.title', 'Test Article');
        
        $this->assertDatabaseHas('articles', [
            'title' => 'Test Article',
            'user_id' => $user->id,
        ]);
    }
    
    public function test_guest_cannot_create_article(): void
    {
        $response = $this->postJson('/api/articles', [
            'title' => 'Test Article',
        ]);
        
        $response->assertUnauthorized();
    }
}''',

    # Facade Pattern
    '''<?php

namespace App\\Services;

use Illuminate\\Support\\Facades\\Facade;

class CacheService
{
    private array $store = [];
    private $driver;
    
    public function __construct($driver = null)
    {
        $this->driver = $driver ?? config('cache.default');
    }
    
    public function get(string $key, mixed $default = null): mixed
    {
        return $this->store[$key] ?? $default;
    }
    
    public function put(string $key, mixed $value, int $ttl = 3600): bool
    {
        $this->store[$key] = $value;
        return true;
    }
    
    public function remember(string $key, int $ttl, callable $callback): mixed
    {
        if (isset($this->store[$key])) {
            return $this->store[$key];
        }
        
        return $this->store[$key] = $callback();
    }
    
    public function forget(string $key): bool
    {
        unset($this->store[$key]);
        return true;
    }
    
    public function flush(): bool
    {
        $this->store = [];
        return true;
    }
}

class Cache extends Facade
{
    protected static function getFacadeAccessor(): string
    {
        return CacheService::class;
    }
}''',
]

SWIFT_SAMPLES = [
    # Classes and Structs
    '''import Foundation

struct User: Codable, Identifiable {
    let id: UUID
    var name: String
    var email: String
    var createdAt: Date
    
    init(name: String, email: String) {
        self.id = UUID()
        self.name = name
        self.email = email
        self.createdAt = Date()
    }
}

class UserService {
    private var users: [UUID: User] = [:]
    
    func createUser(name: String, email: String) -> User {
        let user = User(name: name, email: email)
        users[user.id] = user
        return user
    }
    
    func getUser(id: UUID) -> User? {
        return users[id]
    }
    
    func updateUser(id: UUID, name: String?, email: String?) -> User? {
        guard var user = users[id] else { return nil }
        
        if let name = name { user.name = name }
        if let email = email { user.email = email }
        
        users[id] = user
        return user
    }
    
    func deleteUser(id: UUID) -> Bool {
        return users.removeValue(forKey: id) != nil
    }
}''',

    # Enums with Associated Values
    '''enum NetworkError: Error, LocalizedError {
    case invalidURL
    case noConnection
    case timeout
    case serverError(statusCode: Int)
    case decodingError(underlying: Error)
    case unknown(Error)
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "The URL is invalid"
        case .noConnection:
            return "No internet connection"
        case .timeout:
            return "Request timed out"
        case .serverError(let code):
            return "Server error: \\(code)"
        case .decodingError(let error):
            return "Failed to decode: \\(error.localizedDescription)"
        case .unknown(let error):
            return "Unknown error: \\(error.localizedDescription)"
        }
    }
}

enum Result<Success, Failure: Error> {
    case success(Success)
    case failure(Failure)
    
    func map<NewSuccess>(_ transform: (Success) -> NewSuccess) -> Result<NewSuccess, Failure> {
        switch self {
        case .success(let value):
            return .success(transform(value))
        case .failure(let error):
            return .failure(error)
        }
    }
    
    func flatMap<NewSuccess>(_ transform: (Success) -> Result<NewSuccess, Failure>) -> Result<NewSuccess, Failure> {
        switch self {
        case .success(let value):
            return transform(value)
        case .failure(let error):
            return .failure(error)
        }
    }
}''',

    # Optionals and Guard
    '''func processUserData(json: [String: Any]?) throws -> User {
    guard let json = json else {
        throw ValidationError.missingData
    }
    
    guard let name = json["name"] as? String, !name.isEmpty else {
        throw ValidationError.invalidField("name")
    }
    
    guard let email = json["email"] as? String,
          email.contains("@") else {
        throw ValidationError.invalidField("email")
    }
    
    let age = json["age"] as? Int
    
    return User(name: name, email: email, age: age)
}

extension Optional {
    func unwrap(orThrow error: Error) throws -> Wrapped {
        guard let value = self else {
            throw error
        }
        return value
    }
    
    func or(_ defaultValue: @autoclosure () -> Wrapped) -> Wrapped {
        return self ?? defaultValue()
    }
}

let user: User? = fetchUser()
let name = user?.name ?? "Guest"
let email = user?.email.lowercased()

if let user = user, user.isActive {
    print("Active user: \\(user.name)")
}''',

    # Closures and Higher-Order Functions
    '''let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

let doubled = numbers.map { $0 * 2 }
let evens = numbers.filter { $0 % 2 == 0 }
let sum = numbers.reduce(0, +)

let sortedByAbsolute = numbers.sorted { abs($0) < abs($1) }

typealias Completion<T> = (Result<T, Error>) -> Void

func fetchData<T: Decodable>(from url: URL, completion: @escaping Completion<T>) {
    URLSession.shared.dataTask(with: url) { data, response, error in
        if let error = error {
            completion(.failure(error))
            return
        }
        
        guard let data = data else {
            completion(.failure(NetworkError.noData))
            return
        }
        
        do {
            let decoded = try JSONDecoder().decode(T.self, from: data)
            completion(.success(decoded))
        } catch {
            completion(.failure(error))
        }
    }.resume()
}

func compose<A, B, C>(_ f: @escaping (B) -> C, _ g: @escaping (A) -> B) -> (A) -> C {
    return { x in f(g(x)) }
}''',

    # Protocols and Extensions
    '''protocol Identifiable {
    associatedtype ID: Hashable
    var id: ID { get }
}

protocol Persistable: Codable {
    static var entityName: String { get }
    func save() throws
    static func fetch(id: String) throws -> Self?
}

extension Persistable {
    static var entityName: String {
        return String(describing: Self.self)
    }
}

protocol NetworkService {
    func fetch<T: Decodable>(endpoint: Endpoint) async throws -> T
    func post<T: Encodable, R: Decodable>(endpoint: Endpoint, body: T) async throws -> R
}

extension Array where Element: Numeric {
    var sum: Element {
        return reduce(0, +)
    }
    
    var average: Double where Element: BinaryInteger {
        return isEmpty ? 0 : Double(sum) / Double(count)
    }
}

extension String {
    var isValidEmail: Bool {
        let pattern = "[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Za-z]{2,64}"
        return range(of: pattern, options: .regularExpression) != nil
    }
    
    func truncated(to length: Int, trailing: String = "...") -> String {
        if count <= length { return self }
        return String(prefix(length)) + trailing
    }
}''',

    # SwiftUI Views
    '''import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ContentViewModel()
    @State private var searchText = ""
    @State private var isShowingSheet = false
    
    var body: some View {
        NavigationStack {
            VStack {
                SearchBar(text: $searchText)
                
                if viewModel.isLoading {
                    ProgressView()
                        .scaleEffect(1.5)
                } else if let error = viewModel.error {
                    ErrorView(error: error) {
                        Task { await viewModel.refresh() }
                    }
                } else {
                    List(viewModel.filteredItems(searchText)) { item in
                        NavigationLink(value: item) {
                            ItemRow(item: item)
                        }
                    }
                    .refreshable {
                        await viewModel.refresh()
                    }
                }
            }
            .navigationTitle("Items")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: { isShowingSheet = true }) {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $isShowingSheet) {
                AddItemView(onSave: { item in
                    viewModel.addItem(item)
                })
            }
            .navigationDestination(for: Item.self) { item in
                ItemDetailView(item: item)
            }
        }
        .task {
            await viewModel.loadItems()
        }
    }
}''',

    # SwiftUI View Model
    '''import Combine
import SwiftUI

@MainActor
class ContentViewModel: ObservableObject {
    @Published private(set) var items: [Item] = []
    @Published private(set) var isLoading = false
    @Published private(set) var error: Error?
    
    private let repository: ItemRepository
    private var cancellables = Set<AnyCancellable>()
    
    init(repository: ItemRepository = .shared) {
        self.repository = repository
    }
    
    func loadItems() async {
        isLoading = true
        error = nil
        
        do {
            items = try await repository.fetchAll()
        } catch {
            self.error = error
        }
        
        isLoading = false
    }
    
    func refresh() async {
        await loadItems()
    }
    
    func addItem(_ item: Item) {
        items.append(item)
        
        Task {
            do {
                try await repository.save(item)
            } catch {
                items.removeAll { $0.id == item.id }
                self.error = error
            }
        }
    }
    
    func deleteItem(at offsets: IndexSet) {
        let itemsToDelete = offsets.map { items[$0] }
        items.remove(atOffsets: offsets)
        
        Task {
            for item in itemsToDelete {
                try? await repository.delete(item)
            }
        }
    }
    
    func filteredItems(_ searchText: String) -> [Item] {
        guard !searchText.isEmpty else { return items }
        return items.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
    }
}''',

    # Async/Await
    '''actor DataStore {
    private var cache: [String: Data] = [:]
    
    func get(_ key: String) -> Data? {
        return cache[key]
    }
    
    func set(_ key: String, value: Data) {
        cache[key] = value
    }
    
    func clear() {
        cache.removeAll()
    }
}

class APIClient {
    private let session: URLSession
    private let decoder: JSONDecoder
    
    init(session: URLSession = .shared) {
        self.session = session
        self.decoder = JSONDecoder()
        self.decoder.keyDecodingStrategy = .convertFromSnakeCase
    }
    
    func fetch<T: Decodable>(_ endpoint: Endpoint) async throws -> T {
        let request = try endpoint.urlRequest()
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }
        
        guard (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.serverError(statusCode: httpResponse.statusCode)
        }
        
        return try decoder.decode(T.self, from: data)
    }
    
    func fetchAll<T: Decodable>(_ endpoints: [Endpoint]) async throws -> [T] {
        try await withThrowingTaskGroup(of: T.self) { group in
            for endpoint in endpoints {
                group.addTask { try await self.fetch(endpoint) }
            }
            
            var results: [T] = []
            for try await result in group {
                results.append(result)
            }
            return results
        }
    }
}''',

    # Property Wrappers
    '''@propertyWrapper
struct UserDefault<Value> {
    let key: String
    let defaultValue: Value
    let container: UserDefaults
    
    init(wrappedValue: Value, key: String, container: UserDefaults = .standard) {
        self.key = key
        self.defaultValue = wrappedValue
        self.container = container
    }
    
    var wrappedValue: Value {
        get { container.object(forKey: key) as? Value ?? defaultValue }
        set { container.set(newValue, forKey: key) }
    }
}

@propertyWrapper
struct Clamped<Value: Comparable> {
    private var value: Value
    private let range: ClosedRange<Value>
    
    init(wrappedValue: Value, _ range: ClosedRange<Value>) {
        self.range = range
        self.value = min(max(wrappedValue, range.lowerBound), range.upperBound)
    }
    
    var wrappedValue: Value {
        get { value }
        set { value = min(max(newValue, range.lowerBound), range.upperBound) }
    }
}

class Settings {
    @UserDefault(key: "username", container: .standard)
    var username: String = "Guest"
    
    @UserDefault(key: "notificationsEnabled", container: .standard)
    var notificationsEnabled: Bool = true
    
    @Clamped(0...100)
    var volume: Int = 50
}''',

    # Generics and Constraints
    '''struct Stack<Element> {
    private var items: [Element] = []
    
    var isEmpty: Bool { items.isEmpty }
    var count: Int { items.count }
    var top: Element? { items.last }
    
    mutating func push(_ item: Element) {
        items.append(item)
    }
    
    mutating func pop() -> Element? {
        return items.popLast()
    }
}

extension Stack where Element: Equatable {
    func contains(_ item: Element) -> Bool {
        return items.contains(item)
    }
}

extension Stack where Element: Numeric {
    var sum: Element {
        return items.reduce(0, +)
    }
}

protocol Container {
    associatedtype Item
    var count: Int { get }
    mutating func append(_ item: Item)
    subscript(i: Int) -> Item { get }
}

extension Stack: Container {
    subscript(i: Int) -> Element {
        return items[i]
    }
    
    mutating func append(_ item: Element) {
        push(item)
    }
}

func compare<T: Comparable>(_ a: T, _ b: T) -> Bool {
    return a < b
}''',

    # Combine Framework
    '''import Combine

class SearchViewModel: ObservableObject {
    @Published var searchText = ""
    @Published private(set) var results: [SearchResult] = []
    @Published private(set) var isSearching = false
    
    private let searchService: SearchService
    private var cancellables = Set<AnyCancellable>()
    
    init(searchService: SearchService) {
        self.searchService = searchService
        setupBindings()
    }
    
    private func setupBindings() {
        $searchText
            .debounce(for: .milliseconds(300), scheduler: RunLoop.main)
            .removeDuplicates()
            .filter { !$0.isEmpty }
            .handleEvents(receiveOutput: { [weak self] _ in
                self?.isSearching = true
            })
            .flatMap { [searchService] query in
                searchService.search(query: query)
                    .catch { _ in Just([]) }
            }
            .receive(on: DispatchQueue.main)
            .sink { [weak self] results in
                self?.results = results
                self?.isSearching = false
            }
            .store(in: &cancellables)
    }
}

extension Publisher {
    func retryWithDelay<S: Scheduler>(
        retries: Int,
        delay: S.SchedulerTimeType.Stride,
        scheduler: S
    ) -> AnyPublisher<Output, Failure> {
        self.catch { _ in
            Just(())
                .delay(for: delay, scheduler: scheduler)
                .flatMap { _ in self }
        }
        .retry(retries)
        .eraseToAnyPublisher()
    }
}''',

    # Core Data
    '''import CoreData

class CoreDataStack {
    static let shared = CoreDataStack()
    
    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "Model")
        container.loadPersistentStores { _, error in
            if let error = error {
                fatalError("Core Data failed: \\(error)")
            }
        }
        container.viewContext.automaticallyMergesChangesFromParent = true
        return container
    }()
    
    var viewContext: NSManagedObjectContext {
        persistentContainer.viewContext
    }
    
    func newBackgroundContext() -> NSManagedObjectContext {
        persistentContainer.newBackgroundContext()
    }
    
    func saveContext() {
        let context = viewContext
        if context.hasChanges {
            do {
                try context.save()
            } catch {
                print("Save error: \\(error)")
            }
        }
    }
    
    func performBackgroundTask(_ block: @escaping (NSManagedObjectContext) -> Void) {
        persistentContainer.performBackgroundTask(block)
    }
}

@objc(TaskEntity)
class TaskEntity: NSManagedObject {
    @NSManaged var id: UUID
    @NSManaged var title: String
    @NSManaged var isCompleted: Bool
    @NSManaged var createdAt: Date
    
    static func fetchRequest() -> NSFetchRequest<TaskEntity> {
        return NSFetchRequest<TaskEntity>(entityName: "TaskEntity")
    }
}''',

    # Testing
    '''import XCTest
@testable import MyApp

final class UserServiceTests: XCTestCase {
    var sut: UserService!
    var mockRepository: MockUserRepository!
    
    override func setUp() {
        super.setUp()
        mockRepository = MockUserRepository()
        sut = UserService(repository: mockRepository)
    }
    
    override func tearDown() {
        sut = nil
        mockRepository = nil
        super.tearDown()
    }
    
    func testCreateUser_WithValidData_ReturnsUser() async throws {
        let user = try await sut.createUser(name: "John", email: "john@example.com")
        
        XCTAssertEqual(user.name, "John")
        XCTAssertEqual(user.email, "john@example.com")
        XCTAssertTrue(mockRepository.saveCalled)
    }
    
    func testCreateUser_WithInvalidEmail_ThrowsError() async {
        do {
            _ = try await sut.createUser(name: "John", email: "invalid")
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertEqual(error as? ValidationError, .invalidEmail)
        }
    }
    
    func testFetchUser_WhenUserExists_ReturnsUser() async throws {
        let expectedUser = User(name: "John", email: "john@example.com")
        mockRepository.users[expectedUser.id] = expectedUser
        
        let user = try await sut.fetchUser(id: expectedUser.id)
        
        XCTAssertEqual(user?.id, expectedUser.id)
    }
}''',

    # Dependency Injection
    '''protocol DependencyContainer {
    func resolve<T>() -> T
    func register<T>(_ type: T.Type, factory: @escaping () -> T)
}

final class Container: DependencyContainer {
    static let shared = Container()
    
    private var factories: [String: () -> Any] = [:]
    private var singletons: [String: Any] = [:]
    
    func register<T>(_ type: T.Type, factory: @escaping () -> T) {
        let key = String(describing: type)
        factories[key] = factory
    }
    
    func registerSingleton<T>(_ type: T.Type, factory: @escaping () -> T) {
        let key = String(describing: type)
        factories[key] = { [weak self] in
            if let existing = self?.singletons[key] as? T {
                return existing
            }
            let instance = factory()
            self?.singletons[key] = instance
            return instance
        }
    }
    
    func resolve<T>() -> T {
        let key = String(describing: T.self)
        guard let factory = factories[key] else {
            fatalError("No factory registered for \\(key)")
        }
        guard let instance = factory() as? T else {
            fatalError("Factory returned wrong type for \\(key)")
        }
        return instance
    }
}

@propertyWrapper
struct Injected<T> {
    private var value: T
    
    init() {
        self.value = Container.shared.resolve()
    }
    
    var wrappedValue: T {
        get { value }
        mutating set { value = newValue }
    }
}''',
]

KOTLIN_SAMPLES = [
    # Data Classes and Sealed Classes
    '''data class User(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val email: String,
    val createdAt: Instant = Instant.now()
) {
    fun toDTO() = UserDTO(id, name, email)
}

sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error(val exception: Throwable) : Result<Nothing>()
    data object Loading : Result<Nothing>()
    
    inline fun <R> map(transform: (T) -> R): Result<R> = when (this) {
        is Success -> Success(transform(data))
        is Error -> this
        is Loading -> this
    }
    
    inline fun <R> flatMap(transform: (T) -> Result<R>): Result<R> = when (this) {
        is Success -> transform(data)
        is Error -> this
        is Loading -> this
    }
    
    inline fun onSuccess(action: (T) -> Unit): Result<T> {
        if (this is Success) action(data)
        return this
    }
    
    inline fun onError(action: (Throwable) -> Unit): Result<T> {
        if (this is Error) action(exception)
        return this
    }
}

sealed interface NetworkState {
    data object Idle : NetworkState
    data object Loading : NetworkState
    data class Success<T>(val data: T) : NetworkState
    data class Error(val message: String) : NetworkState
}''',

    # Extension Functions
    '''fun String.isValidEmail(): Boolean {
    val emailPattern = Regex("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}")
    return emailPattern.matches(this)
}

fun String.truncate(maxLength: Int, suffix: String = "..."): String {
    return if (length <= maxLength) this
    else take(maxLength - suffix.length) + suffix
}

fun <T> List<T>.chunkedBy(predicate: (T, T) -> Boolean): List<List<T>> {
    if (isEmpty()) return emptyList()
    
    val result = mutableListOf<MutableList<T>>()
    var currentChunk = mutableListOf(first())
    
    for (i in 1 until size) {
        if (predicate(this[i - 1], this[i])) {
            currentChunk.add(this[i])
        } else {
            result.add(currentChunk)
            currentChunk = mutableListOf(this[i])
        }
    }
    result.add(currentChunk)
    return result
}

inline fun <T, R> T.letIf(condition: Boolean, block: (T) -> R): R? {
    return if (condition) block(this) else null
}

inline fun <T> T.also(condition: Boolean, block: T.() -> Unit): T {
    if (condition) block()
    return this
}

fun <K, V> Map<K, V>.getOrThrow(key: K, message: () -> String = { "Key not found: $key" }): V {
    return this[key] ?: throw NoSuchElementException(message())
}''',

    # Coroutines
    '''import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

class UserRepository(
    private val api: UserApi,
    private val db: UserDao,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {
    fun getUsers(): Flow<List<User>> = flow {
        emit(db.getAllUsers())
        
        try {
            val remoteUsers = api.fetchUsers()
            db.insertAll(remoteUsers)
            emit(remoteUsers)
        } catch (e: Exception) {
            emit(db.getAllUsers())
        }
    }.flowOn(dispatcher)
    
    suspend fun refreshUsers(): Result<List<User>> = withContext(dispatcher) {
        try {
            val users = api.fetchUsers()
            db.deleteAll()
            db.insertAll(users)
            Result.Success(users)
        } catch (e: Exception) {
            Result.Error(e)
        }
    }
    
    fun searchUsers(query: String): Flow<List<User>> = 
        db.searchUsers("%$query%")
            .debounce(300)
            .distinctUntilChanged()
            .flowOn(dispatcher)
}

suspend fun <T> retry(
    times: Int = 3,
    initialDelay: Long = 100,
    maxDelay: Long = 1000,
    factor: Double = 2.0,
    block: suspend () -> T
): T {
    var currentDelay = initialDelay
    repeat(times - 1) {
        try {
            return block()
        } catch (e: Exception) {
            delay(currentDelay)
            currentDelay = (currentDelay * factor).toLong().coerceAtMost(maxDelay)
        }
    }
    return block()
}''',

    # Spring Boot Controller
    '''import org.springframework.web.bind.annotation.*
import org.springframework.http.ResponseEntity
import jakarta.validation.Valid

@RestController
@RequestMapping("/api/v1/articles")
class ArticleController(
    private val articleService: ArticleService
) {
    @GetMapping
    suspend fun getArticles(
        @RequestParam(defaultValue = "0") page: Int,
        @RequestParam(defaultValue = "20") size: Int,
        @RequestParam(required = false) category: String?
    ): ResponseEntity<PageResponse<ArticleDTO>> {
        val articles = articleService.findAll(page, size, category)
        return ResponseEntity.ok(articles)
    }
    
    @GetMapping("/{id}")
    suspend fun getArticle(@PathVariable id: String): ResponseEntity<ArticleDTO> {
        return articleService.findById(id)?.let {
            ResponseEntity.ok(it.toDTO())
        } ?: ResponseEntity.notFound().build()
    }
    
    @PostMapping
    suspend fun createArticle(
        @Valid @RequestBody request: CreateArticleRequest,
        @AuthenticationPrincipal user: UserPrincipal
    ): ResponseEntity<ArticleDTO> {
        val article = articleService.create(request, user.id)
        return ResponseEntity.status(HttpStatus.CREATED).body(article.toDTO())
    }
    
    @PutMapping("/{id}")
    suspend fun updateArticle(
        @PathVariable id: String,
        @Valid @RequestBody request: UpdateArticleRequest,
        @AuthenticationPrincipal user: UserPrincipal
    ): ResponseEntity<ArticleDTO> {
        val article = articleService.update(id, request, user.id)
        return ResponseEntity.ok(article.toDTO())
    }
    
    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    suspend fun deleteArticle(
        @PathVariable id: String,
        @AuthenticationPrincipal user: UserPrincipal
    ) {
        articleService.delete(id, user.id)
    }
}''',

    # Spring Boot Service
    '''import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import kotlinx.coroutines.flow.Flow

@Service
class ArticleService(
    private val articleRepository: ArticleRepository,
    private val tagRepository: TagRepository,
    private val eventPublisher: ApplicationEventPublisher
) {
    suspend fun findAll(
        page: Int,
        size: Int,
        category: String?
    ): PageResponse<ArticleDTO> {
        val pageable = PageRequest.of(page, size, Sort.by("createdAt").descending())
        
        val articles = category?.let {
            articleRepository.findByCategory(it, pageable)
        } ?: articleRepository.findAll(pageable)
        
        return PageResponse(
            content = articles.content.map { it.toDTO() },
            page = page,
            size = size,
            totalElements = articles.totalElements,
            totalPages = articles.totalPages
        )
    }
    
    @Transactional
    suspend fun create(request: CreateArticleRequest, authorId: String): Article {
        val tags = request.tagIds?.let { tagRepository.findAllById(it) } ?: emptyList()
        
        val article = Article(
            title = request.title,
            content = request.content,
            authorId = authorId,
            tags = tags.toMutableSet(),
            status = ArticleStatus.DRAFT
        )
        
        val saved = articleRepository.save(article)
        eventPublisher.publishEvent(ArticleCreatedEvent(saved))
        
        return saved
    }
    
    @Transactional
    suspend fun update(id: String, request: UpdateArticleRequest, userId: String): Article {
        val article = articleRepository.findById(id)
            ?: throw NotFoundException("Article not found: $id")
        
        require(article.authorId == userId) { "Not authorized to update this article" }
        
        article.apply {
            request.title?.let { title = it }
            request.content?.let { content = it }
            request.status?.let { status = it }
            updatedAt = Instant.now()
        }
        
        return articleRepository.save(article)
    }
}''',

    # Kotlin DSL
    '''class HtmlBuilder {
    private val elements = mutableListOf<String>()
    
    fun head(block: HeadBuilder.() -> Unit) {
        elements += HeadBuilder().apply(block).build()
    }
    
    fun body(block: BodyBuilder.() -> Unit) {
        elements += BodyBuilder().apply(block).build()
    }
    
    fun build(): String = "<html>${elements.joinToString("")}</html>"
}

class BodyBuilder {
    private val elements = mutableListOf<String>()
    
    fun div(classes: String = "", block: BodyBuilder.() -> Unit = {}) {
        val content = BodyBuilder().apply(block).build()
        val classAttr = if (classes.isNotEmpty()) " class=\"$classes\"" else ""
        elements += "<div$classAttr>$content</div>"
    }
    
    fun p(text: String) {
        elements += "<p>$text</p>"
    }
    
    fun a(href: String, text: String) {
        elements += "<a href=\"$href\">$text</a>"
    }
    
    operator fun String.unaryPlus() {
        elements += this
    }
    
    fun build(): String = elements.joinToString("")
}

fun html(block: HtmlBuilder.() -> Unit): String = HtmlBuilder().apply(block).build()

val page = html {
    head {
        title("My Page")
        css("styles.css")
    }
    body {
        div("container") {
            p("Hello, World!")
            a("https://example.com", "Visit")
        }
    }
}''',

    # Jetpack Compose
    '''@Composable
fun UserListScreen(
    viewModel: UserListViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Users") },
                actions = {
                    IconButton(onClick = { viewModel.refresh() }) {
                        Icon(Icons.Default.Refresh, "Refresh")
                    }
                }
            )
        },
        floatingActionButton = {
            FloatingActionButton(onClick = { viewModel.showAddDialog() }) {
                Icon(Icons.Default.Add, "Add user")
            }
        }
    ) { padding ->
        Box(modifier = Modifier.padding(padding)) {
            when (val state = uiState) {
                is UiState.Loading -> {
                    CircularProgressIndicator(
                        modifier = Modifier.align(Alignment.Center)
                    )
                }
                is UiState.Success -> {
                    LazyColumn {
                        items(state.users, key = { it.id }) { user ->
                            UserItem(
                                user = user,
                                onEdit = { viewModel.editUser(it) },
                                onDelete = { viewModel.deleteUser(it) },
                                modifier = Modifier.animateItemPlacement()
                            )
                        }
                    }
                }
                is UiState.Error -> {
                    ErrorContent(
                        message = state.message,
                        onRetry = { viewModel.refresh() }
                    )
                }
            }
        }
    }
}

@Composable
fun UserItem(
    user: User,
    onEdit: (User) -> Unit,
    onDelete: (User) -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(8.dp),
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            AsyncImage(
                model = user.avatarUrl,
                contentDescription = null,
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
            )
            
            Column(
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 16.dp)
            ) {
                Text(user.name, style = MaterialTheme.typography.titleMedium)
                Text(user.email, style = MaterialTheme.typography.bodySmall)
            }
            
            IconButton(onClick = { onEdit(user) }) {
                Icon(Icons.Default.Edit, "Edit")
            }
            IconButton(onClick = { onDelete(user) }) {
                Icon(Icons.Default.Delete, "Delete")
            }
        }
    }
}''',

    # ViewModel with Flow
    '''import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class UserListViewModel @Inject constructor(
    private val repository: UserRepository,
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {
    
    private val _uiState = MutableStateFlow<UiState<List<User>>>(UiState.Loading)
    val uiState: StateFlow<UiState<List<User>>> = _uiState.asStateFlow()
    
    private val _events = MutableSharedFlow<Event>()
    val events: SharedFlow<Event> = _events.asSharedFlow()
    
    val searchQuery = savedStateHandle.getStateFlow("search", "")
    
    init {
        loadUsers()
    }
    
    private fun loadUsers() {
        viewModelScope.launch {
            repository.getUsers()
                .combine(searchQuery) { users, query ->
                    if (query.isBlank()) users
                    else users.filter { it.name.contains(query, ignoreCase = true) }
                }
                .catch { e ->
                    _uiState.value = UiState.Error(e.message ?: "Unknown error")
                }
                .collect { users ->
                    _uiState.value = UiState.Success(users)
                }
        }
    }
    
    fun refresh() {
        viewModelScope.launch {
            _uiState.value = UiState.Loading
            repository.refreshUsers()
                .onSuccess { loadUsers() }
                .onError { e ->
                    _events.emit(Event.ShowError(e.message ?: "Failed to refresh"))
                    loadUsers()
                }
        }
    }
    
    fun deleteUser(user: User) {
        viewModelScope.launch {
            repository.deleteUser(user.id)
                .onSuccess {
                    _events.emit(Event.ShowMessage("User deleted"))
                }
                .onError { e ->
                    _events.emit(Event.ShowError(e.message ?: "Failed to delete"))
                }
        }
    }
    
    fun updateSearchQuery(query: String) {
        savedStateHandle["search"] = query
    }
}''',

    # Repository Pattern with Room
    '''import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey val id: String,
    val name: String,
    val email: String,
    @ColumnInfo(name = "avatar_url") val avatarUrl: String?,
    @ColumnInfo(name = "created_at") val createdAt: Long = System.currentTimeMillis()
)

@Dao
interface UserDao {
    @Query("SELECT * FROM users ORDER BY created_at DESC")
    fun getAllUsers(): Flow<List<UserEntity>>
    
    @Query("SELECT * FROM users WHERE id = :id")
    suspend fun getUserById(id: String): UserEntity?
    
    @Query("SELECT * FROM users WHERE name LIKE :query OR email LIKE :query")
    fun searchUsers(query: String): Flow<List<UserEntity>>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(users: List<UserEntity>)
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(user: UserEntity)
    
    @Delete
    suspend fun delete(user: UserEntity)
    
    @Query("DELETE FROM users")
    suspend fun deleteAll()
}

@Database(entities = [UserEntity::class], version = 1)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
}

class UserRepositoryImpl(
    private val api: UserApi,
    private val dao: UserDao,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) : UserRepository {
    
    override fun getUsers(): Flow<List<User>> =
        dao.getAllUsers()
            .map { entities -> entities.map { it.toDomain() } }
            .flowOn(dispatcher)
    
    override suspend fun refreshUsers(): Result<Unit> = withContext(dispatcher) {
        try {
            val users = api.fetchUsers()
            dao.deleteAll()
            dao.insertAll(users.map { it.toEntity() })
            Result.Success(Unit)
        } catch (e: Exception) {
            Result.Error(e)
        }
    }
}''',

    # Dependency Injection with Hilt
    '''import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {
    
    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        return OkHttpClient.Builder()
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            })
            .addInterceptor { chain ->
                val request = chain.request().newBuilder()
                    .addHeader("Authorization", "Bearer ${TokenManager.token}")
                    .build()
                chain.proceed(request)
            }
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build()
    }
    
    @Provides
    @Singleton
    fun provideRetrofit(okHttpClient: OkHttpClient): Retrofit {
        return Retrofit.Builder()
            .baseUrl(BuildConfig.API_BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(MoshiConverterFactory.create())
            .build()
    }
    
    @Provides
    @Singleton
    fun provideUserApi(retrofit: Retrofit): UserApi {
        return retrofit.create(UserApi::class.java)
    }
}

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {
    
    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AppDatabase {
        return Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "app_database"
        )
            .fallbackToDestructiveMigration()
            .build()
    }
    
    @Provides
    fun provideUserDao(database: AppDatabase): UserDao {
        return database.userDao()
    }
}

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {
    
    @Binds
    @Singleton
    abstract fun bindUserRepository(impl: UserRepositoryImpl): UserRepository
}''',

    # Testing with Coroutines
    '''import kotlinx.coroutines.test.*
import org.junit.Before
import org.junit.Test
import org.junit.Rule
import io.mockk.*

class UserViewModelTest {
    
    @get:Rule
    val mainDispatcherRule = MainDispatcherRule()
    
    private lateinit var viewModel: UserListViewModel
    private val repository: UserRepository = mockk()
    private val savedStateHandle = SavedStateHandle()
    
    @Before
    fun setup() {
        coEvery { repository.getUsers() } returns flowOf(testUsers)
        viewModel = UserListViewModel(repository, savedStateHandle)
    }
    
    @Test
    fun `initial state is loading then success`() = runTest {
        viewModel.uiState.test {
            assertEquals(UiState.Loading, awaitItem())
            
            val success = awaitItem()
            assertTrue(success is UiState.Success)
            assertEquals(testUsers, (success as UiState.Success).data)
        }
    }
    
    @Test
    fun `refresh updates users from repository`() = runTest {
        coEvery { repository.refreshUsers() } returns Result.Success(Unit)
        
        viewModel.refresh()
        
        viewModel.uiState.test {
            skipItems(1)
            assertTrue(awaitItem() is UiState.Loading)
            assertTrue(awaitItem() is UiState.Success)
        }
        
        coVerify { repository.refreshUsers() }
    }
    
    @Test
    fun `delete user emits success event`() = runTest {
        coEvery { repository.deleteUser(any()) } returns Result.Success(Unit)
        
        viewModel.events.test {
            viewModel.deleteUser(testUsers.first())
            
            val event = awaitItem()
            assertTrue(event is Event.ShowMessage)
            assertEquals("User deleted", (event as Event.ShowMessage).message)
        }
    }
    
    companion object {
        private val testUsers = listOf(
            User("1", "John", "john@example.com"),
            User("2", "Jane", "jane@example.com")
        )
    }
}''',

    # Functional Programming
    '''inline fun <T, R> T?.mapOrNull(transform: (T) -> R): R? = this?.let(transform)

inline fun <T> T?.orElse(default: () -> T): T = this ?: default()

inline fun <T, R> Iterable<T>.mapNotNullTo(
    destination: MutableCollection<in R>,
    transform: (T) -> R?
): MutableCollection<in R> {
    for (element in this) {
        transform(element)?.let { destination.add(it) }
    }
    return destination
}

sealed class Either<out L, out R> {
    data class Left<L>(val value: L) : Either<L, Nothing>()
    data class Right<R>(val value: R) : Either<Nothing, R>()
    
    inline fun <T> fold(ifLeft: (L) -> T, ifRight: (R) -> T): T = when (this) {
        is Left -> ifLeft(value)
        is Right -> ifRight(value)
    }
    
    inline fun <T> map(transform: (R) -> T): Either<L, T> = when (this) {
        is Left -> this
        is Right -> Right(transform(value))
    }
    
    inline fun <T> flatMap(transform: (R) -> Either<L, T>): Either<L, T> = when (this) {
        is Left -> this
        is Right -> transform(value)
    }
}

suspend fun <T> Either<Throwable, T>.getOrThrow(): T = fold(
    ifLeft = { throw it },
    ifRight = { it }
)

inline fun <T> catching(block: () -> T): Either<Throwable, T> = try {
    Either.Right(block())
} catch (e: Throwable) {
    Either.Left(e)
}''',

    # Ktor Client
    '''import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.plugins.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.serialization.kotlinx.json.*

class ApiClient(
    private val baseUrl: String,
    private val tokenProvider: () -> String?
) {
    private val client = HttpClient {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                prettyPrint = true
            })
        }
        
        install(HttpTimeout) {
            requestTimeoutMillis = 30_000
            connectTimeoutMillis = 10_000
        }
        
        defaultRequest {
            url(baseUrl)
            tokenProvider()?.let { token ->
                header("Authorization", "Bearer $token")
            }
        }
        
        HttpResponseValidator {
            validateResponse { response ->
                when (response.status.value) {
                    in 400..499 -> throw ClientException(response.status.description)
                    in 500..599 -> throw ServerException(response.status.description)
                }
            }
        }
    }
    
    suspend inline fun <reified T> get(path: String): T {
        return client.get(path).body()
    }
    
    suspend inline fun <reified T, reified R> post(path: String, body: T): R {
        return client.post(path) {
            setBody(body)
        }.body()
    }
    
    suspend inline fun <reified T, reified R> put(path: String, body: T): R {
        return client.put(path) {
            setBody(body)
        }.body()
    }
    
    suspend fun delete(path: String) {
        client.delete(path)
    }
}''',
]

SCALA_SAMPLES = [
    # Case Classes and Pattern Matching
    '''case class User(
  id: String = java.util.UUID.randomUUID().toString,
  name: String,
  email: String,
  createdAt: java.time.Instant = java.time.Instant.now()
)

sealed trait Result[+A]
case class Success[A](value: A) extends Result[A]
case class Failure(error: Throwable) extends Result[Nothing]

object Result {
  def apply[A](block: => A): Result[A] = 
    try Success(block)
    catch { case e: Throwable => Failure(e) }
}

def processResult[A](result: Result[A]): String = result match {
  case Success(value) => s"Success: $value"
  case Failure(e: IllegalArgumentException) => s"Invalid argument: ${e.getMessage}"
  case Failure(e: NullPointerException) => "Null pointer encountered"
  case Failure(e) => s"Error: ${e.getMessage}"
}

sealed trait Tree[+A]
case class Leaf[A](value: A) extends Tree[A]
case class Branch[A](left: Tree[A], right: Tree[A]) extends Tree[A]

def size[A](tree: Tree[A]): Int = tree match {
  case Leaf(_) => 1
  case Branch(left, right) => 1 + size(left) + size(right)
}

def depth[A](tree: Tree[A]): Int = tree match {
  case Leaf(_) => 0
  case Branch(left, right) => 1 + (depth(left) max depth(right))
}''',

    # Higher-Order Functions
    '''object FunctionalUtils {
  def map[A, B](list: List[A])(f: A => B): List[B] = 
    list.foldRight(List.empty[B])((a, acc) => f(a) :: acc)
  
  def filter[A](list: List[A])(predicate: A => Boolean): List[A] =
    list.foldRight(List.empty[A])((a, acc) => if (predicate(a)) a :: acc else acc)
  
  def flatMap[A, B](list: List[A])(f: A => List[B]): List[B] =
    list.foldRight(List.empty[B])((a, acc) => f(a) ++ acc)
  
  def compose[A, B, C](f: B => C, g: A => B): A => C = a => f(g(a))
  
  def andThen[A, B, C](f: A => B, g: B => C): A => C = a => g(f(a))
  
  def curry[A, B, C](f: (A, B) => C): A => B => C = a => b => f(a, b)
  
  def uncurry[A, B, C](f: A => B => C): (A, B) => C = (a, b) => f(a)(b)
  
  def memoize[A, B](f: A => B): A => B = {
    val cache = scala.collection.mutable.Map.empty[A, B]
    a => cache.getOrElseUpdate(a, f(a))
  }
}

val addOne: Int => Int = _ + 1
val double: Int => Int = _ * 2
val addOneThenDouble: Int => Int = FunctionalUtils.compose(double, addOne)

val numbers = List(1, 2, 3, 4, 5)
val result = numbers
  .map(_ * 2)
  .filter(_ > 4)
  .foldLeft(0)(_ + _)''',

    # Akka Actors
    '''import akka.actor.typed.scaladsl.{ActorContext, Behaviors}
import akka.actor.typed.{ActorRef, Behavior}

object UserActor {
  sealed trait Command
  case class CreateUser(name: String, email: String, replyTo: ActorRef[Response]) extends Command
  case class GetUser(id: String, replyTo: ActorRef[Response]) extends Command
  case class UpdateUser(id: String, name: Option[String], email: Option[String], replyTo: ActorRef[Response]) extends Command
  case class DeleteUser(id: String, replyTo: ActorRef[Response]) extends Command
  
  sealed trait Response
  case class UserCreated(user: User) extends Response
  case class UserFound(user: User) extends Response
  case class UserUpdated(user: User) extends Response
  case object UserDeleted extends Response
  case class UserNotFound(id: String) extends Response
  case class Error(message: String) extends Response
  
  def apply(): Behavior[Command] = behavior(Map.empty)
  
  private def behavior(users: Map[String, User]): Behavior[Command] = 
    Behaviors.receive { (context, message) =>
      message match {
        case CreateUser(name, email, replyTo) =>
          val user = User(name = name, email = email)
          context.log.info(s"Creating user: ${user.id}")
          replyTo ! UserCreated(user)
          behavior(users + (user.id -> user))
          
        case GetUser(id, replyTo) =>
          users.get(id) match {
            case Some(user) => replyTo ! UserFound(user)
            case None => replyTo ! UserNotFound(id)
          }
          Behaviors.same
          
        case UpdateUser(id, name, email, replyTo) =>
          users.get(id) match {
            case Some(user) =>
              val updated = user.copy(
                name = name.getOrElse(user.name),
                email = email.getOrElse(user.email)
              )
              replyTo ! UserUpdated(updated)
              behavior(users + (id -> updated))
            case None =>
              replyTo ! UserNotFound(id)
              Behaviors.same
          }
          
        case DeleteUser(id, replyTo) =>
          if (users.contains(id)) {
            replyTo ! UserDeleted
            behavior(users - id)
          } else {
            replyTo ! UserNotFound(id)
            Behaviors.same
          }
      }
    }
}''',

    # Futures and Promises
    '''import scala.concurrent.{Future, Promise, ExecutionContext}
import scala.util.{Success, Failure, Try}

class AsyncService(implicit ec: ExecutionContext) {
  def fetchUser(id: String): Future[User] = Future {
    Thread.sleep(100)
    User(id = id, name = "John", email = "john@example.com")
  }
  
  def fetchUserPosts(userId: String): Future[List[Post]] = Future {
    Thread.sleep(100)
    List(Post(id = "1", userId = userId, title = "Hello"))
  }
  
  def fetchUserWithPosts(userId: String): Future[(User, List[Post])] = {
    for {
      user <- fetchUser(userId)
      posts <- fetchUserPosts(userId)
    } yield (user, posts)
  }
  
  def fetchAllUsers(ids: List[String]): Future[List[User]] = {
    Future.traverse(ids)(fetchUser)
  }
  
  def fetchFirstSuccess(ids: List[String]): Future[User] = {
    Future.firstCompletedOf(ids.map(fetchUser))
  }
  
  def retry[A](times: Int)(f: => Future[A]): Future[A] = {
    f.recoverWith {
      case _ if times > 0 => retry(times - 1)(f)
    }
  }
}

def promiseExample(): Future[Int] = {
  val promise = Promise[Int]()
  
  Future {
    Thread.sleep(100)
    if (scala.util.Random.nextBoolean()) {
      promise.success(42)
    } else {
      promise.failure(new Exception("Random failure"))
    }
  }
  
  promise.future
}

def withTimeout[A](future: Future[A], timeout: Long)(implicit ec: ExecutionContext): Future[A] = {
  val promise = Promise[A]()
  
  val timer = new java.util.Timer()
  timer.schedule(new java.util.TimerTask {
    def run(): Unit = {
      promise.tryFailure(new java.util.concurrent.TimeoutException())
    }
  }, timeout)
  
  future.onComplete { result =>
    timer.cancel()
    promise.tryComplete(result)
  }
  
  promise.future
}''',

    # Type Classes
    '''trait Monoid[A] {
  def empty: A
  def combine(x: A, y: A): A
}

object Monoid {
  def apply[A](implicit m: Monoid[A]): Monoid[A] = m
  
  implicit val intMonoid: Monoid[Int] = new Monoid[Int] {
    def empty: Int = 0
    def combine(x: Int, y: Int): Int = x + y
  }
  
  implicit val stringMonoid: Monoid[String] = new Monoid[String] {
    def empty: String = ""
    def combine(x: String, y: String): String = x + y
  }
  
  implicit def listMonoid[A]: Monoid[List[A]] = new Monoid[List[A]] {
    def empty: List[A] = Nil
    def combine(x: List[A], y: List[A]): List[A] = x ++ y
  }
  
  implicit def optionMonoid[A](implicit m: Monoid[A]): Monoid[Option[A]] = 
    new Monoid[Option[A]] {
      def empty: Option[A] = None
      def combine(x: Option[A], y: Option[A]): Option[A] = (x, y) match {
        case (Some(a), Some(b)) => Some(m.combine(a, b))
        case (Some(a), None) => Some(a)
        case (None, Some(b)) => Some(b)
        case (None, None) => None
      }
    }
}

def combineAll[A: Monoid](list: List[A]): A = 
  list.foldLeft(Monoid[A].empty)(Monoid[A].combine)

implicit class MonoidOps[A](a: A)(implicit m: Monoid[A]) {
  def |+|(b: A): A = m.combine(a, b)
}

val result = List(1, 2, 3) |+| List(4, 5, 6)
val sum = combineAll(List(1, 2, 3, 4, 5))''',

    # Cats Effect and IO
    '''import cats.effect.{IO, Resource, Sync}
import cats.implicits._

trait UserRepository[F[_]] {
  def findById(id: String): F[Option[User]]
  def findAll: F[List[User]]
  def create(user: User): F[User]
  def update(user: User): F[User]
  def delete(id: String): F[Unit]
}

class UserService[F[_]: Sync](repo: UserRepository[F]) {
  def getUser(id: String): F[User] = 
    repo.findById(id).flatMap {
      case Some(user) => Sync[F].pure(user)
      case None => Sync[F].raiseError(new NoSuchElementException(s"User not found: $id"))
    }
  
  def createUser(name: String, email: String): F[User] = {
    val user = User(name = name, email = email)
    repo.create(user)
  }
  
  def updateEmail(id: String, newEmail: String): F[User] = 
    for {
      user <- getUser(id)
      updated = user.copy(email = newEmail)
      result <- repo.update(updated)
    } yield result
}

def program: IO[Unit] = {
  val result = for {
    _ <- IO.println("Starting program")
    user <- IO.pure(User(name = "John", email = "john@example.com"))
    _ <- IO.println(s"Created user: $user")
    _ <- IO.sleep(1.second)
    _ <- IO.println("Done")
  } yield ()
  
  result.handleErrorWith { e =>
    IO.println(s"Error: ${e.getMessage}")
  }
}

def resourceExample: Resource[IO, java.sql.Connection] = 
  Resource.make(
    IO(java.sql.DriverManager.getConnection("jdbc:h2:mem:test"))
  )(conn => IO(conn.close()))''',

    # ZIO
    '''import zio._
import zio.stream._

trait UserService {
  def getUser(id: String): Task[User]
  def createUser(user: User): Task[User]
}

object UserService {
  def live: ZLayer[UserRepository, Nothing, UserService] = 
    ZLayer.fromFunction { repo =>
      new UserService {
        def getUser(id: String): Task[User] = 
          repo.findById(id).flatMap {
            case Some(user) => ZIO.succeed(user)
            case None => ZIO.fail(new NoSuchElementException(s"User not found: $id"))
          }
        
        def createUser(user: User): Task[User] = 
          repo.create(user)
      }
    }
}

def program: ZIO[UserService, Throwable, Unit] = {
  for {
    service <- ZIO.service[UserService]
    user <- service.createUser(User(name = "John", email = "john@example.com"))
    _ <- Console.printLine(s"Created: $user")
    fetched <- service.getUser(user.id)
    _ <- Console.printLine(s"Fetched: $fetched")
  } yield ()
}

def streamExample: ZStream[Any, Nothing, Int] = 
  ZStream
    .iterate(1)(_ + 1)
    .take(100)
    .filter(_ % 2 == 0)
    .mapZIO(n => ZIO.succeed(n * 2))

def retryWithBackoff[R, E, A](zio: ZIO[R, E, A]): ZIO[R, E, A] = 
  zio.retry(
    Schedule.exponential(100.millis) && Schedule.recurs(5)
  )''',

    # Http4s Server
    '''import org.http4s._
import org.http4s.dsl.io._
import org.http4s.circe.CirceEntityCodec._
import cats.effect._
import io.circe.generic.auto._

case class CreateUserRequest(name: String, email: String)
case class UserResponse(id: String, name: String, email: String)

class UserRoutes(userService: UserService[IO]) {
  val routes: HttpRoutes[IO] = HttpRoutes.of[IO] {
    case GET -> Root / "users" =>
      userService.findAll.flatMap { users =>
        Ok(users.map(u => UserResponse(u.id, u.name, u.email)))
      }
    
    case GET -> Root / "users" / id =>
      userService.findById(id).flatMap {
        case Some(user) => Ok(UserResponse(user.id, user.name, user.email))
        case None => NotFound(s"User not found: $id")
      }
    
    case req @ POST -> Root / "users" =>
      for {
        request <- req.as[CreateUserRequest]
        user = User(name = request.name, email = request.email)
        created <- userService.create(user)
        response <- Created(UserResponse(created.id, created.name, created.email))
      } yield response
    
    case DELETE -> Root / "users" / id =>
      userService.delete(id).flatMap(_ => NoContent())
  }
}

object Server extends IOApp.Simple {
  def run: IO[Unit] = {
    val userService = new InMemoryUserService[IO]
    val routes = new UserRoutes(userService).routes
    
    EmberServerBuilder
      .default[IO]
      .withHost(ipv4"0.0.0.0")
      .withPort(port"8080")
      .withHttpApp(routes.orNotFound)
      .build
      .useForever
  }
}''',

    # Slick Database
    '''import slick.jdbc.PostgresProfile.api._
import scala.concurrent.{ExecutionContext, Future}

case class UserRow(id: String, name: String, email: String, createdAt: java.sql.Timestamp)

class UsersTable(tag: Tag) extends Table[UserRow](tag, "users") {
  def id = column[String]("id", O.PrimaryKey)
  def name = column[String]("name")
  def email = column[String]("email")
  def createdAt = column[java.sql.Timestamp]("created_at")
  
  def * = (id, name, email, createdAt).mapTo[UserRow]
  
  def emailIndex = index("idx_users_email", email, unique = true)
}

class UserRepository(db: Database)(implicit ec: ExecutionContext) {
  private val users = TableQuery[UsersTable]
  
  def findById(id: String): Future[Option[User]] = 
    db.run(users.filter(_.id === id).result.headOption)
      .map(_.map(toUser))
  
  def findAll: Future[Seq[User]] = 
    db.run(users.result).map(_.map(toUser))
  
  def findByEmail(email: String): Future[Option[User]] = 
    db.run(users.filter(_.email === email).result.headOption)
      .map(_.map(toUser))
  
  def create(user: User): Future[User] = {
    val row = UserRow(user.id, user.name, user.email, java.sql.Timestamp.from(user.createdAt))
    db.run(users += row).map(_ => user)
  }
  
  def update(user: User): Future[User] = {
    val query = users.filter(_.id === user.id)
      .map(u => (u.name, u.email))
      .update((user.name, user.email))
    db.run(query).map(_ => user)
  }
  
  def delete(id: String): Future[Int] = 
    db.run(users.filter(_.id === id).delete)
  
  def search(query: String): Future[Seq[User]] = 
    db.run(users.filter(u => u.name.like(s"%$query%") || u.email.like(s"%$query%")).result)
      .map(_.map(toUser))
  
  private def toUser(row: UserRow): User = 
    User(row.id, row.name, row.email, row.createdAt.toInstant)
}''',

    # Circe JSON
    '''import io.circe._
import io.circe.generic.semiauto._
import io.circe.syntax._
import io.circe.parser._

case class Address(street: String, city: String, country: String)
case class Person(name: String, age: Int, email: Option[String], addresses: List[Address])

object Codecs {
  implicit val addressEncoder: Encoder[Address] = deriveEncoder[Address]
  implicit val addressDecoder: Decoder[Address] = deriveDecoder[Address]
  
  implicit val personEncoder: Encoder[Person] = Encoder.instance { person =>
    Json.obj(
      "name" -> person.name.asJson,
      "age" -> person.age.asJson,
      "email" -> person.email.asJson,
      "addresses" -> person.addresses.asJson
    )
  }
  
  implicit val personDecoder: Decoder[Person] = Decoder.instance { cursor =>
    for {
      name <- cursor.get[String]("name")
      age <- cursor.get[Int]("age")
      email <- cursor.get[Option[String]]("email")
      addresses <- cursor.getOrElse[List[Address]]("addresses")(Nil)
    } yield Person(name, age, email, addresses)
  }
}

import Codecs._

val person = Person("John", 30, Some("john@example.com"), List(Address("123 Main St", "NYC", "USA")))
val json = person.asJson.noSpaces

val parsed = parse(json).flatMap(_.as[Person])

def transformJson(json: Json): Json = json.mapObject { obj =>
  obj
    .add("timestamp", Json.fromLong(System.currentTimeMillis()))
    .remove("password")
}''',

    # Akka Streams
    '''import akka.stream.scaladsl._
import akka.stream._
import akka.actor.ActorSystem
import scala.concurrent.Future

implicit val system: ActorSystem = ActorSystem("streams")
implicit val ec = system.dispatcher

val source: Source[Int, NotUsed] = Source(1 to 100)

val flow: Flow[Int, Int, NotUsed] = Flow[Int]
  .filter(_ % 2 == 0)
  .map(_ * 2)
  .buffer(10, OverflowStrategy.backpressure)

val sink: Sink[Int, Future[Int]] = Sink.fold(0)(_ + _)

val result: Future[Int] = source.via(flow).runWith(sink)

val throttledSource: Source[Int, NotUsed] = 
  Source(1 to 100)
    .throttle(10, 1.second)
    .map { n =>
      println(s"Processing: $n")
      n
    }

val parallelFlow: Flow[Int, Int, NotUsed] = 
  Flow[Int].mapAsync(4) { n =>
    Future {
      Thread.sleep(100)
      n * 2
    }
  }

val graph = GraphDSL.create() { implicit builder =>
  import GraphDSL.Implicits._
  
  val broadcast = builder.add(Broadcast[Int](2))
  val merge = builder.add(Merge[Int](2))
  
  val doubler = Flow[Int].map(_ * 2)
  val tripler = Flow[Int].map(_ * 3)
  
  broadcast.out(0) ~> doubler ~> merge.in(0)
  broadcast.out(1) ~> tripler ~> merge.in(1)
  
  FlowShape(broadcast.in, merge.out)
}''',

    # Play Framework Controller
    '''import javax.inject._
import play.api.mvc._
import play.api.libs.json._
import scala.concurrent.{ExecutionContext, Future}

case class CreateArticleRequest(title: String, content: String, tags: List[String])
object CreateArticleRequest {
  implicit val reads: Reads[CreateArticleRequest] = Json.reads[CreateArticleRequest]
}

case class ArticleResponse(id: String, title: String, content: String, createdAt: String)
object ArticleResponse {
  implicit val writes: Writes[ArticleResponse] = Json.writes[ArticleResponse]
}

@Singleton
class ArticleController @Inject()(
  cc: ControllerComponents,
  articleService: ArticleService
)(implicit ec: ExecutionContext) extends AbstractController(cc) {
  
  def list: Action[AnyContent] = Action.async {
    articleService.findAll.map { articles =>
      Ok(Json.toJson(articles.map(toResponse)))
    }
  }
  
  def get(id: String): Action[AnyContent] = Action.async {
    articleService.findById(id).map {
      case Some(article) => Ok(Json.toJson(toResponse(article)))
      case None => NotFound(Json.obj("error" -> s"Article not found: $id"))
    }
  }
  
  def create: Action[JsValue] = Action.async(parse.json) { request =>
    request.body.validate[CreateArticleRequest].fold(
      errors => Future.successful(BadRequest(JsError.toJson(errors))),
      req => {
        val article = Article(title = req.title, content = req.content)
        articleService.create(article).map { created =>
          Created(Json.toJson(toResponse(created)))
        }
      }
    )
  }
  
  def delete(id: String): Action[AnyContent] = Action.async {
    articleService.delete(id).map(_ => NoContent)
  }
  
  private def toResponse(article: Article): ArticleResponse = 
    ArticleResponse(article.id, article.title, article.content, article.createdAt.toString)
}''',

    # Spark DataFrame Operations
    '''import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object SparkAnalytics {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Analytics")
      .master("local[*]")
      .getOrCreate()
    
    import spark.implicits._
    
    val salesDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("sales.csv")
    
    val monthlySales = salesDF
      .withColumn("month", month($"date"))
      .withColumn("year", year($"date"))
      .groupBy("year", "month", "product")
      .agg(
        sum("amount").as("total_amount"),
        count("*").as("transaction_count"),
        avg("amount").as("avg_amount")
      )
      .orderBy($"year".desc, $"month".desc)
    
    val topProducts = salesDF
      .groupBy("product")
      .agg(sum("amount").as("total"))
      .orderBy($"total".desc)
      .limit(10)
    
    val windowSpec = Window.partitionBy("product").orderBy("date")
    
    val runningTotal = salesDF
      .withColumn("running_total", sum("amount").over(windowSpec))
      .withColumn("row_num", row_number().over(windowSpec))
    
    monthlySales.write
      .mode("overwrite")
      .partitionBy("year", "month")
      .parquet("output/monthly_sales")
    
    spark.stop()
  }
}''',
]

SHELL_SAMPLES = [
    # Script Functions
    '''#!/bin/bash

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_warning() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

die() {
    log_error "$1"
    exit "${2:-1}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        die "Required command not found: $1"
    fi
}

confirm() {
    local prompt="${1:-Are you sure?}"
    read -r -p "$prompt [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY]) return 0 ;;
        *) return 1 ;;
    esac
}

retry() {
    local max_attempts=${1:-3}
    local delay=${2:-1}
    local attempt=1
    shift 2
    
    while [ $attempt -le $max_attempts ]; do
        if "$@"; then
            return 0
        fi
        log_warning "Attempt $attempt/$max_attempts failed. Retrying in ${delay}s..."
        sleep "$delay"
        ((attempt++))
    done
    
    log_error "All $max_attempts attempts failed"
    return 1
}''',

    # Loops and Conditionals
    '''#!/bin/bash

process_files() {
    local dir="${1:-.}"
    local pattern="${2:-*}"
    local count=0
    
    while IFS= read -r -d '' file; do
        echo "Processing: $file"
        ((count++))
    done < <(find "$dir" -name "$pattern" -type f -print0)
    
    echo "Processed $count files"
}

for file in *.txt; do
    [ -e "$file" ] || continue
    echo "Found: $file"
    
    case "$file" in
        test_*.txt)
            echo "  -> Test file"
            ;;
        config*.txt)
            echo "  -> Config file"
            ;;
        *)
            echo "  -> Regular file"
            ;;
    esac
done

numbers=(1 2 3 4 5)
for num in "${numbers[@]}"; do
    if ((num % 2 == 0)); then
        echo "$num is even"
    else
        echo "$num is odd"
    fi
done

counter=0
while [ $counter -lt 10 ]; do
    echo "Count: $counter"
    ((counter++))
done

until [ -f /tmp/ready ]; do
    echo "Waiting for ready file..."
    sleep 1
done''',

    # File Operations
    '''#!/bin/bash
set -euo pipefail

backup_dir() {
    local source="${1:?Source directory required}"
    local dest="${2:?Destination required}"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="backup_${timestamp}.tar.gz"
    
    if [ ! -d "$source" ]; then
        echo "Source directory does not exist: $source" >&2
        return 1
    fi
    
    mkdir -p "$dest"
    
    tar -czf "${dest}/${backup_name}" -C "$(dirname "$source")" "$(basename "$source")"
    
    echo "Backup created: ${dest}/${backup_name}"
}

cleanup_old_files() {
    local dir="${1:?Directory required}"
    local days="${2:-30}"
    local pattern="${3:-*}"
    
    find "$dir" -name "$pattern" -type f -mtime +"$days" -exec rm -v {} \;
}

safe_move() {
    local source="$1"
    local dest="$2"
    
    if [ ! -e "$source" ]; then
        echo "Source does not exist: $source" >&2
        return 1
    fi
    
    if [ -e "$dest" ]; then
        local backup="${dest}.bak.$(date +%s)"
        mv "$dest" "$backup"
        echo "Existing file backed up to: $backup"
    fi
    
    mv "$source" "$dest"
}

sync_directories() {
    local source="$1"
    local dest="$2"
    
    rsync -avz --delete \
        --exclude='.git/' \
        --exclude='node_modules/' \
        --exclude='*.log' \
        "$source/" "$dest/"
}''',

    # CI/CD Pipeline Script
    '''#!/bin/bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

setup_environment() {
    echo "Setting up environment..."
    
    export PATH="$PROJECT_ROOT/bin:$PATH"
    export NODE_ENV="${NODE_ENV:-production}"
    
    if [ -f "$PROJECT_ROOT/.env" ]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    fi
}

install_dependencies() {
    echo "Installing dependencies..."
    
    if [ -f "package.json" ]; then
        npm ci --production=false
    fi
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
}

run_tests() {
    echo "Running tests..."
    
    local exit_code=0
    
    if [ -f "package.json" ]; then
        npm test || exit_code=$?
    fi
    
    if [ -f "pytest.ini" ] || [ -d "tests" ]; then
        pytest --cov=src --cov-report=xml || exit_code=$?
    fi
    
    return $exit_code
}

build_application() {
    echo "Building application..."
    
    if [ -f "package.json" ]; then
        npm run build
    fi
    
    if [ -f "Dockerfile" ]; then
        docker build -t "${IMAGE_NAME:-app}:${VERSION:-latest}" .
    fi
}

deploy() {
    local environment="${1:-staging}"
    
    echo "Deploying to $environment..."
    
    case "$environment" in
        staging)
            kubectl apply -f k8s/staging/
            ;;
        production)
            kubectl apply -f k8s/production/
            ;;
        *)
            echo "Unknown environment: $environment" >&2
            return 1
            ;;
    esac
}

main() {
    setup_environment
    install_dependencies
    run_tests
    build_application
    
    if [ "${DEPLOY:-false}" = "true" ]; then
        deploy "${ENVIRONMENT:-staging}"
    fi
    
    echo "Pipeline completed successfully!"
}

main "$@"''',

    # Docker Management Script
    '''#!/bin/bash
set -euo pipefail

readonly IMAGE_NAME="${IMAGE_NAME:-myapp}"
readonly REGISTRY="${REGISTRY:-docker.io}"
readonly COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

docker_build() {
    local tag="${1:-latest}"
    local dockerfile="${2:-Dockerfile}"
    
    echo "Building image: ${IMAGE_NAME}:${tag}"
    
    docker build \
        --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        --build-arg VERSION="$tag" \
        --tag "${IMAGE_NAME}:${tag}" \
        --file "$dockerfile" \
        .
}

docker_push() {
    local tag="${1:-latest}"
    local full_name="${REGISTRY}/${IMAGE_NAME}:${tag}"
    
    echo "Pushing image: $full_name"
    
    docker tag "${IMAGE_NAME}:${tag}" "$full_name"
    docker push "$full_name"
}

docker_cleanup() {
    echo "Cleaning up Docker resources..."
    
    docker container prune -f
    docker image prune -f
    docker volume prune -f
    
    docker images "${IMAGE_NAME}" -q | xargs -r docker rmi -f || true
}

compose_up() {
    local profile="${1:-default}"
    
    docker-compose -f "$COMPOSE_FILE" \
        --profile "$profile" \
        up -d --build --remove-orphans
}

compose_down() {
    docker-compose -f "$COMPOSE_FILE" down --volumes --remove-orphans
}

compose_logs() {
    local service="${1:-}"
    
    docker-compose -f "$COMPOSE_FILE" logs -f $service
}

health_check() {
    local container="${1:?Container name required}"
    local timeout="${2:-30}"
    local interval="${3:-2}"
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if docker exec "$container" curl -sf http://localhost:8080/health > /dev/null 2>&1; then
            echo "Container $container is healthy"
            return 0
        fi
        sleep "$interval"
        ((elapsed += interval))
    done
    
    echo "Health check failed for $container" >&2
    return 1
}''',

    # Database Operations
    '''#!/bin/bash
set -euo pipefail

readonly DB_HOST="${DB_HOST:-localhost}"
readonly DB_PORT="${DB_PORT:-5432}"
readonly DB_NAME="${DB_NAME:-myapp}"
readonly DB_USER="${DB_USER:-postgres}"
readonly BACKUP_DIR="${BACKUP_DIR:-/var/backups/postgres}"

pg_dump_db() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${BACKUP_DIR}/${DB_NAME}_${timestamp}.sql.gz"
    
    mkdir -p "$BACKUP_DIR"
    
    echo "Creating backup: $backup_file"
    
    PGPASSWORD="${DB_PASSWORD:-}" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --no-owner \
        --no-privileges \
        | gzip > "$backup_file"
    
    echo "Backup completed: $backup_file"
    
    find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +7 -delete
}

pg_restore_db() {
    local backup_file="${1:?Backup file required}"
    
    if [ ! -f "$backup_file" ]; then
        echo "Backup file not found: $backup_file" >&2
        return 1
    fi
    
    echo "Restoring from: $backup_file"
    
    PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
    
    gunzip -c "$backup_file" | PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME"
    
    echo "Restore completed"
}

run_migrations() {
    local migrations_dir="${1:-./migrations}"
    
    PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "CREATE TABLE IF NOT EXISTS schema_migrations (version VARCHAR(255) PRIMARY KEY, applied_at TIMESTAMP DEFAULT NOW());"
    
    for migration in "$migrations_dir"/*.sql; do
        [ -e "$migration" ] || continue
        
        local version=$(basename "$migration" .sql)
        
        local applied=$(PGPASSWORD="${DB_PASSWORD:-}" psql \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            -tAc "SELECT 1 FROM schema_migrations WHERE version = '$version'")
        
        if [ -z "$applied" ]; then
            echo "Applying migration: $version"
            PGPASSWORD="${DB_PASSWORD:-}" psql \
                -h "$DB_HOST" \
                -p "$DB_PORT" \
                -U "$DB_USER" \
                -d "$DB_NAME" \
                -f "$migration"
            
            PGPASSWORD="${DB_PASSWORD:-}" psql \
                -h "$DB_HOST" \
                -p "$DB_PORT" \
                -U "$DB_USER" \
                -d "$DB_NAME" \
                -c "INSERT INTO schema_migrations (version) VALUES ('$version');"
        fi
    done
}''',

    # Monitoring and Alerting Script
    '''#!/bin/bash
set -euo pipefail

readonly ALERT_THRESHOLD_CPU=${ALERT_THRESHOLD_CPU:-80}
readonly ALERT_THRESHOLD_MEMORY=${ALERT_THRESHOLD_MEMORY:-80}
readonly ALERT_THRESHOLD_DISK=${ALERT_THRESHOLD_DISK:-90}
readonly SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

send_alert() {
    local message="$1"
    local severity="${2:-warning}"
    
    echo "[ALERT] $message"
    
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \":${severity}: $message\"}"
    fi
}

check_cpu() {
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print int($2)}')
    
    if [ "$cpu_usage" -gt "$ALERT_THRESHOLD_CPU" ]; then
        send_alert "High CPU usage: ${cpu_usage}% (threshold: ${ALERT_THRESHOLD_CPU}%)" "fire"
        return 1
    fi
    
    echo "CPU usage: ${cpu_usage}%"
    return 0
}

check_memory() {
    local memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
    
    if [ "$memory_usage" -gt "$ALERT_THRESHOLD_MEMORY" ]; then
        send_alert "High memory usage: ${memory_usage}% (threshold: ${ALERT_THRESHOLD_MEMORY}%)" "warning"
        return 1
    fi
    
    echo "Memory usage: ${memory_usage}%"
    return 0
}

check_disk() {
    local issues=0
    
    while read -r line; do
        local usage=$(echo "$line" | awk '{print int($5)}')
        local mount=$(echo "$line" | awk '{print $6}')
        
        if [ "$usage" -gt "$ALERT_THRESHOLD_DISK" ]; then
            send_alert "High disk usage on $mount: ${usage}% (threshold: ${ALERT_THRESHOLD_DISK}%)" "rotating_light"
            ((issues++))
        fi
    done < <(df -h | grep -E '^/dev/' | awk '{print $5, $6}')
    
    return $issues
}

check_service() {
    local service="$1"
    
    if systemctl is-active --quiet "$service"; then
        echo "Service $service is running"
        return 0
    else
        send_alert "Service $service is not running!" "x"
        return 1
    fi
}

main() {
    local exit_code=0
    
    check_cpu || exit_code=1
    check_memory || exit_code=1
    check_disk || exit_code=1
    
    for service in nginx postgresql redis; do
        if systemctl list-units --type=service | grep -q "$service"; then
            check_service "$service" || exit_code=1
        fi
    done
    
    return $exit_code
}

main''',

    # AWS CLI Automation
    '''#!/bin/bash
set -euo pipefail

readonly AWS_REGION="${AWS_REGION:-us-east-1}"
readonly CLUSTER_NAME="${CLUSTER_NAME:-my-cluster}"
readonly SERVICE_NAME="${SERVICE_NAME:-my-service}"

get_latest_image() {
    local repository="$1"
    
    aws ecr describe-images \
        --repository-name "$repository" \
        --query 'sort_by(imageDetails,& imagePushedAt)[-1].imageTags[0]' \
        --output text
}

deploy_ecs_service() {
    local image_tag="${1:-latest}"
    local task_definition="$2"
    
    echo "Deploying ECS service with image tag: $image_tag"
    
    local task_def=$(aws ecs describe-task-definition \
        --task-definition "$task_definition" \
        --query 'taskDefinition')
    
    local new_task_def=$(echo "$task_def" | jq --arg TAG "$image_tag" \
        '.containerDefinitions[0].image = (.containerDefinitions[0].image | split(":")[0] + ":" + $TAG)')
    
    local new_task_def_arn=$(echo "$new_task_def" | jq -r \
        'del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .compatibilities, .registeredAt, .registeredBy)' \
        | aws ecs register-task-definition --cli-input-json file:///dev/stdin \
        --query 'taskDefinition.taskDefinitionArn' --output text)
    
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "$SERVICE_NAME" \
        --task-definition "$new_task_def_arn" \
        --force-new-deployment
    
    echo "Deployment initiated. Waiting for service stability..."
    
    aws ecs wait services-stable \
        --cluster "$CLUSTER_NAME" \
        --services "$SERVICE_NAME"
    
    echo "Deployment completed successfully!"
}

scale_service() {
    local desired_count="${1:?Desired count required}"
    
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "$SERVICE_NAME" \
        --desired-count "$desired_count"
    
    echo "Service scaled to $desired_count tasks"
}

invalidate_cloudfront() {
    local distribution_id="${1:?Distribution ID required}"
    local paths="${2:-/*}"
    
    aws cloudfront create-invalidation \
        --distribution-id "$distribution_id" \
        --paths "$paths"
    
    echo "CloudFront invalidation created for: $paths"
}''',

    # Git Automation Script
    '''#!/bin/bash
set -euo pipefail

readonly MAIN_BRANCH="${MAIN_BRANCH:-main}"
readonly DEVELOP_BRANCH="${DEVELOP_BRANCH:-develop}"

create_release() {
    local version="${1:?Version required (e.g., 1.0.0)}"
    local branch="release/$version"
    
    git checkout "$DEVELOP_BRANCH"
    git pull origin "$DEVELOP_BRANCH"
    git checkout -b "$branch"
    
    if [ -f "package.json" ]; then
        npm version "$version" --no-git-tag-version
        git add package.json package-lock.json
    fi
    
    if [ -f "VERSION" ]; then
        echo "$version" > VERSION
        git add VERSION
    fi
    
    git commit -m "chore: bump version to $version"
    git push origin "$branch"
    
    echo "Release branch created: $branch"
}

finish_release() {
    local version="${1:?Version required}"
    local branch="release/$version"
    
    git checkout "$MAIN_BRANCH"
    git pull origin "$MAIN_BRANCH"
    git merge --no-ff "$branch" -m "Merge release $version"
    git tag -a "v$version" -m "Release $version"
    git push origin "$MAIN_BRANCH" --tags
    
    git checkout "$DEVELOP_BRANCH"
    git pull origin "$DEVELOP_BRANCH"
    git merge --no-ff "$branch" -m "Merge release $version into develop"
    git push origin "$DEVELOP_BRANCH"
    
    git branch -d "$branch"
    git push origin --delete "$branch"
    
    echo "Release $version completed"
}

create_hotfix() {
    local version="${1:?Version required}"
    local branch="hotfix/$version"
    
    git checkout "$MAIN_BRANCH"
    git pull origin "$MAIN_BRANCH"
    git checkout -b "$branch"
    
    echo "Hotfix branch created: $branch"
    echo "Make your changes and run: finish_hotfix $version"
}

cleanup_branches() {
    echo "Cleaning up merged branches..."
    
    git fetch --prune
    
    git branch --merged "$MAIN_BRANCH" | \
        grep -v "^\\*\\|$MAIN_BRANCH\\|$DEVELOP_BRANCH" | \
        xargs -r git branch -d
    
    git remote prune origin
    
    echo "Cleanup completed"
}

generate_changelog() {
    local from_tag="${1:-$(git describe --tags --abbrev=0 2>/dev/null || echo "")}"
    local to_ref="${2:-HEAD}"
    
    echo "## Changelog"
    echo ""
    
    if [ -n "$from_tag" ]; then
        echo "### Changes since $from_tag"
        git log --pretty=format:"- %s (%h)" "$from_tag..$to_ref"
    else
        echo "### All Changes"
        git log --pretty=format:"- %s (%h)" "$to_ref"
    fi
}''',

    # System Setup Script
    '''#!/bin/bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ "$(uname)" = "Darwin" ]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

install_packages_ubuntu() {
    sudo apt-get update
    sudo apt-get install -y \
        curl \
        wget \
        git \
        vim \
        htop \
        jq \
        build-essential
}

install_packages_macos() {
    if ! command -v brew &> /dev/null; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew install \
        curl \
        wget \
        git \
        vim \
        htop \
        jq
}

setup_node() {
    local version="${1:-20}"
    
    if command -v nvm &> /dev/null; then
        nvm install "$version"
        nvm use "$version"
    else
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
        nvm install "$version"
    fi
}

setup_docker() {
    local os=$(detect_os)
    
    case "$os" in
        ubuntu|debian)
            curl -fsSL https://get.docker.com | sudo sh
            sudo usermod -aG docker "$USER"
            ;;
        macos)
            brew install --cask docker
            ;;
    esac
}

setup_dotfiles() {
    local dotfiles_repo="${1:-https://github.com/user/dotfiles.git}"
    local dotfiles_dir="$HOME/.dotfiles"
    
    if [ -d "$dotfiles_dir" ]; then
        git -C "$dotfiles_dir" pull
    else
        git clone "$dotfiles_repo" "$dotfiles_dir"
    fi
    
    for file in "$dotfiles_dir"/.*; do
        [ -f "$file" ] || continue
        local basename=$(basename "$file")
        [ "$basename" = ".git" ] && continue
        
        ln -sf "$file" "$HOME/$basename"
    done
}

main() {
    local os=$(detect_os)
    echo "Detected OS: $os"
    
    case "$os" in
        ubuntu|debian)
            install_packages_ubuntu
            ;;
        macos)
            install_packages_macos
            ;;
        *)
            echo "Unsupported OS: $os" >&2
            exit 1
            ;;
    esac
    
    setup_node
    setup_docker
    setup_dotfiles
    
    echo "Setup completed successfully!"
}

main "$@"''',

    # Log Analysis Script
    '''#!/bin/bash
set -euo pipefail

readonly LOG_DIR="${LOG_DIR:-/var/log}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-./reports}"

analyze_access_log() {
    local log_file="${1:-$LOG_DIR/nginx/access.log}"
    local output_file="$OUTPUT_DIR/access_analysis.txt"
    
    mkdir -p "$OUTPUT_DIR"
    
    {
        echo "=== Access Log Analysis ==="
        echo "Generated: $(date)"
        echo ""
        
        echo "=== Top 10 IP Addresses ==="
        awk '{print $1}' "$log_file" | sort | uniq -c | sort -rn | head -10
        echo ""
        
        echo "=== Top 10 Requested URLs ==="
        awk '{print $7}' "$log_file" | sort | uniq -c | sort -rn | head -10
        echo ""
        
        echo "=== HTTP Status Code Distribution ==="
        awk '{print $9}' "$log_file" | sort | uniq -c | sort -rn
        echo ""
        
        echo "=== Requests per Hour ==="
        awk '{print $4}' "$log_file" | cut -d: -f2 | sort | uniq -c
        echo ""
        
        echo "=== Top User Agents ==="
        awk -F'"' '{print $6}' "$log_file" | sort | uniq -c | sort -rn | head -10
        
    } > "$output_file"
    
    echo "Analysis saved to: $output_file"
}

analyze_error_log() {
    local log_file="${1:-$LOG_DIR/nginx/error.log}"
    local output_file="$OUTPUT_DIR/error_analysis.txt"
    
    mkdir -p "$OUTPUT_DIR"
    
    {
        echo "=== Error Log Analysis ==="
        echo "Generated: $(date)"
        echo ""
        
        echo "=== Error Count by Level ==="
        grep -oE '\\[(emerg|alert|crit|error|warn|notice|info|debug)\\]' "$log_file" | \
            sort | uniq -c | sort -rn
        echo ""
        
        echo "=== Recent Errors (last 50) ==="
        tail -50 "$log_file"
        
    } > "$output_file"
    
    echo "Analysis saved to: $output_file"
}

search_logs() {
    local pattern="${1:?Search pattern required}"
    local log_pattern="${2:-*.log}"
    
    echo "Searching for: $pattern"
    echo "In files matching: $log_pattern"
    echo ""
    
    find "$LOG_DIR" -name "$log_pattern" -type f -exec \
        grep -l "$pattern" {} \; 2>/dev/null | while read -r file; do
        echo "=== $file ==="
        grep --color=auto -n "$pattern" "$file" | head -20
        echo ""
    done
}

tail_logs() {
    local pattern="${1:-*.log}"
    local lines="${2:-100}"
    
    find "$LOG_DIR" -name "$pattern" -type f -mmin -60 | \
        xargs -I{} sh -c "echo '=== {} ===' && tail -$lines {}"
}

compress_old_logs() {
    local days="${1:-7}"
    local log_pattern="${2:-*.log}"
    
    find "$LOG_DIR" -name "$log_pattern" -type f -mtime +"$days" ! -name "*.gz" | \
        while read -r file; do
            echo "Compressing: $file"
            gzip "$file"
        done
    
    echo "Compression completed"
}''',

    # Kubernetes Operations
    '''#!/bin/bash
set -euo pipefail

readonly NAMESPACE="${NAMESPACE:-default}"
readonly CONTEXT="${CONTEXT:-}"

kubectl_cmd() {
    local cmd="kubectl"
    [ -n "$CONTEXT" ] && cmd="$cmd --context=$CONTEXT"
    cmd="$cmd --namespace=$NAMESPACE"
    $cmd "$@"
}

get_pods() {
    local selector="${1:-}"
    
    if [ -n "$selector" ]; then
        kubectl_cmd get pods -l "$selector" -o wide
    else
        kubectl_cmd get pods -o wide
    fi
}

pod_logs() {
    local pod="${1:?Pod name required}"
    local container="${2:-}"
    local follow="${3:-false}"
    
    local args="logs $pod"
    [ -n "$container" ] && args="$args -c $container"
    [ "$follow" = "true" ] && args="$args -f"
    
    kubectl_cmd $args
}

pod_exec() {
    local pod="${1:?Pod name required}"
    local cmd="${2:-/bin/sh}"
    
    kubectl_cmd exec -it "$pod" -- $cmd
}

restart_deployment() {
    local deployment="${1:?Deployment name required}"
    
    kubectl_cmd rollout restart deployment/"$deployment"
    kubectl_cmd rollout status deployment/"$deployment" --timeout=300s
}

scale_deployment() {
    local deployment="${1:?Deployment name required}"
    local replicas="${2:?Replica count required}"
    
    kubectl_cmd scale deployment/"$deployment" --replicas="$replicas"
}

get_resources() {
    echo "=== Pods ==="
    kubectl_cmd get pods -o wide
    echo ""
    
    echo "=== Deployments ==="
    kubectl_cmd get deployments
    echo ""
    
    echo "=== Services ==="
    kubectl_cmd get services
    echo ""
    
    echo "=== Ingresses ==="
    kubectl_cmd get ingresses
    echo ""
    
    echo "=== ConfigMaps ==="
    kubectl_cmd get configmaps
    echo ""
    
    echo "=== Secrets ==="
    kubectl_cmd get secrets
}

port_forward() {
    local resource="${1:?Resource required (e.g., pod/my-pod or svc/my-service)}"
    local ports="${2:?Ports required (e.g., 8080:80)}"
    
    kubectl_cmd port-forward "$resource" "$ports"
}

apply_manifests() {
    local dir="${1:-.}"
    local dry_run="${2:-false}"
    
    local args="apply -f $dir --recursive"
    [ "$dry_run" = "true" ] && args="$args --dry-run=client"
    
    kubectl_cmd $args
}''',

    # Testing Script
    '''#!/bin/bash
set -euo pipefail

readonly TEST_DIR="${TEST_DIR:-./tests}"
readonly COVERAGE_DIR="${COVERAGE_DIR:-./coverage}"
readonly REPORT_FILE="${REPORT_FILE:-test_report.txt}"

run_unit_tests() {
    echo "Running unit tests..."
    
    if [ -f "package.json" ]; then
        npm test -- --coverage --coverageDirectory="$COVERAGE_DIR"
    elif [ -f "pytest.ini" ] || [ -d "tests" ]; then
        pytest "$TEST_DIR" \
            --cov=src \
            --cov-report=html:"$COVERAGE_DIR" \
            --cov-report=term \
            -v
    elif [ -f "go.mod" ]; then
        go test -v -cover -coverprofile="$COVERAGE_DIR/coverage.out" ./...
        go tool cover -html="$COVERAGE_DIR/coverage.out" -o "$COVERAGE_DIR/coverage.html"
    fi
}

run_integration_tests() {
    echo "Running integration tests..."
    
    export TEST_ENV="integration"
    
    docker-compose -f docker-compose.test.yml up -d
    
    trap 'docker-compose -f docker-compose.test.yml down' EXIT
    
    sleep 10
    
    if [ -f "package.json" ]; then
        npm run test:integration
    elif [ -f "pytest.ini" ]; then
        pytest tests/integration -v
    fi
}

run_e2e_tests() {
    echo "Running E2E tests..."
    
    if command -v cypress &> /dev/null; then
        cypress run --config-file cypress.config.js
    elif command -v playwright &> /dev/null; then
        playwright test
    fi
}

lint_code() {
    echo "Running linters..."
    
    local exit_code=0
    
    if [ -f "package.json" ]; then
        npm run lint || exit_code=$?
    fi
    
    if [ -f ".eslintrc.js" ] || [ -f ".eslintrc.json" ]; then
        npx eslint . --ext .js,.jsx,.ts,.tsx || exit_code=$?
    fi
    
    if command -v shellcheck &> /dev/null; then
        find . -name "*.sh" -type f -exec shellcheck {} \; || exit_code=$?
    fi
    
    return $exit_code
}

generate_report() {
    {
        echo "=== Test Report ==="
        echo "Generated: $(date)"
        echo ""
        echo "=== Summary ==="
        
        if [ -f "$COVERAGE_DIR/coverage-summary.json" ]; then
            jq '.total' "$COVERAGE_DIR/coverage-summary.json"
        fi
        
    } > "$REPORT_FILE"
    
    echo "Report generated: $REPORT_FILE"
}

main() {
    local test_type="${1:-all}"
    
    case "$test_type" in
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        e2e)
            run_e2e_tests
            ;;
        lint)
            lint_code
            ;;
        all)
            lint_code
            run_unit_tests
            run_integration_tests
            generate_report
            ;;
        *)
            echo "Unknown test type: $test_type"
            echo "Usage: $0 [unit|integration|e2e|lint|all]"
            exit 1
            ;;
    esac
}

main "$@"''',
]

HASKELL_SAMPLES = [
    '''-- Type signatures and data types
data Tree a = Empty | Node a (Tree a) (Tree a)
    deriving (Show, Eq)

data Maybe a = Nothing | Just a
    deriving (Show, Eq, Ord)

data Either a b = Left a | Right b
    deriving (Show, Eq, Ord)

class Functor f where
    fmap :: (a -> b) -> f a -> f b''',

    '''-- Pattern matching and guards
fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)

factorial :: Integer -> Integer
factorial n
    | n < 0     = error "Negative factorial"
    | n == 0    = 1
    | otherwise = n * factorial (n - 1)''',

    '''-- List comprehensions
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = 
    quicksort [y | y <- xs, y < x] 
    ++ [x] 
    ++ quicksort [y | y <- xs, y >= x]

primes :: [Int]
primes = sieve [2..]
  where sieve (p:xs) = p : sieve [x | x <- xs, x `mod` p /= 0]''',

    '''-- Higher-order functions
map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs

filter' :: (a -> Bool) -> [a] -> [a]
filter' _ [] = []
filter' p (x:xs)
    | p x       = x : filter' p xs
    | otherwise = filter' p xs

foldr' :: (a -> b -> b) -> b -> [a] -> b
foldr' _ z [] = z
foldr' f z (x:xs) = f x (foldr' f z xs)''',

    '''-- Maybe Monad
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x

chainOperations :: Maybe Int -> Maybe Int
chainOperations mx = do
    x <- mx
    y <- Just (x * 2)
    z <- Just (y + 1)
    return z''',

    '''-- Either Monad for error handling
parseAge :: String -> Either String Int
parseAge s = case reads s of
    [(age, "")] | age >= 0 && age <= 150 -> Right age
                | otherwise -> Left "Age out of range"
    _ -> Left "Invalid number format"

validateUser :: String -> Int -> Either String User
validateUser name age
    | null name = Left "Name cannot be empty"
    | age < 0   = Left "Age cannot be negative"
    | otherwise = Right (User name age)''',

    '''-- IO Monad
main :: IO ()
main = do
    putStrLn "Enter your name:"
    name <- getLine
    putStrLn $ "Hello, " ++ name ++ "!"

readFileLines :: FilePath -> IO [String]
readFileLines path = do
    contents <- readFile path
    return (lines contents)

writeToFile :: FilePath -> String -> IO ()
writeToFile path content = writeFile path content''',

    '''-- Applicative and Functor
instance Functor Maybe where
    fmap _ Nothing  = Nothing
    fmap f (Just x) = Just (f x)

instance Applicative Maybe where
    pure = Just
    Nothing <*> _ = Nothing
    (Just f) <*> x = fmap f x

liftA2' :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
liftA2' f x y = f <$> x <*> y''',

    '''-- Binary Search Tree operations
insert :: Ord a => a -> Tree a -> Tree a
insert x Empty = Node x Empty Empty
insert x (Node y left right)
    | x < y     = Node y (insert x left) right
    | x > y     = Node y left (insert x right)
    | otherwise = Node y left right

search :: Ord a => a -> Tree a -> Bool
search _ Empty = False
search x (Node y left right)
    | x == y    = True
    | x < y     = search x left
    | otherwise = search x right''',

    '''-- State Monad
newtype State s a = State { runState :: s -> (a, s) }

instance Monad (State s) where
    return x = State $ \s -> (x, s)
    (State h) >>= f = State $ \s ->
        let (a, newState) = h s
            (State g) = f a
        in g newState

get :: State s s
get = State $ \s -> (s, s)

put :: s -> State s ()
put s = State $ \_ -> ((), s)''',

    '''-- Reader Monad for configuration
newtype Reader r a = Reader { runReader :: r -> a }

instance Monad (Reader r) where
    return x = Reader $ \_ -> x
    (Reader ra) >>= f = Reader $ \r ->
        let a = ra r
            (Reader rb) = f a
        in rb r

ask :: Reader r r
ask = Reader id

local :: (r -> r) -> Reader r a -> Reader r a
local f m = Reader $ \r -> runReader m (f r)''',

    '''-- Lens basics
data Person = Person { _name :: String, _age :: Int }

name :: Lens' Person String
name = lens _name (\person n -> person { _name = n })

age :: Lens' Person Int
age = lens _age (\person a -> person { _age = a })

view :: Lens' s a -> s -> a
view l = getConst . l Const

over :: Lens' s a -> (a -> a) -> s -> s
over l f s = runIdentity (l (Identity . f) s)''',
]

ELIXIR_SAMPLES = [
    '''# Module definition with functions
defmodule Calculator do
  def add(a, b), do: a + b
  def subtract(a, b), do: a - b
  def multiply(a, b), do: a * b
  
  def divide(_a, 0), do: {:error, :division_by_zero}
  def divide(a, b), do: {:ok, a / b}
  
  defp validate_number(n) when is_number(n), do: :ok
  defp validate_number(_), do: {:error, :not_a_number}
end''',

    '''# Pattern matching
defmodule PatternMatching do
  def describe([]), do: "empty list"
  def describe([head | tail]), do: "head: #{head}, tail length: #{length(tail)}"
  
  def handle_result({:ok, value}), do: "Success: #{value}"
  def handle_result({:error, reason}), do: "Error: #{reason}"
  
  def greet(%{name: name, age: age}) when age >= 18 do
    "Hello, adult #{name}!"
  end
  def greet(%{name: name}), do: "Hello, #{name}!"
end''',

    '''# GenServer implementation
defmodule Counter do
  use GenServer

  def start_link(initial_value \\\\ 0) do
    GenServer.start_link(__MODULE__, initial_value, name: __MODULE__)
  end

  def increment, do: GenServer.call(__MODULE__, :increment)
  def decrement, do: GenServer.call(__MODULE__, :decrement)
  def get_value, do: GenServer.call(__MODULE__, :get)

  @impl true
  def init(initial_value), do: {:ok, initial_value}

  @impl true
  def handle_call(:increment, _from, state), do: {:reply, state + 1, state + 1}
  def handle_call(:decrement, _from, state), do: {:reply, state - 1, state - 1}
  def handle_call(:get, _from, state), do: {:reply, state, state}
end''',

    '''# Supervisor setup
defmodule MyApp.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, :ok, opts)
  end

  @impl true
  def init(:ok) do
    children = [
      {MyApp.Cache, []},
      {MyApp.Worker, []},
      {Task.Supervisor, name: MyApp.TaskSupervisor}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end''',

    '''# Phoenix Controller
defmodule MyAppWeb.UserController do
  use MyAppWeb, :controller

  alias MyApp.Accounts
  alias MyApp.Accounts.User

  def index(conn, _params) do
    users = Accounts.list_users()
    render(conn, :index, users: users)
  end

  def create(conn, %{"user" => user_params}) do
    case Accounts.create_user(user_params) do
      {:ok, user} ->
        conn
        |> put_flash(:info, "User created successfully.")
        |> redirect(to: ~p"/users/#{user}")

      {:error, %Ecto.Changeset{} = changeset} ->
        render(conn, :new, changeset: changeset)
    end
  end

  def show(conn, %{"id" => id}) do
    user = Accounts.get_user!(id)
    render(conn, :show, user: user)
  end
end''',

    '''# Pipe operator and Enum
defmodule DataProcessor do
  def process_users(users) do
    users
    |> Enum.filter(&(&1.active))
    |> Enum.map(&normalize_user/1)
    |> Enum.sort_by(& &1.name)
    |> Enum.take(10)
  end

  defp normalize_user(user) do
    %{user | 
      name: String.trim(user.name) |> String.capitalize(),
      email: String.downcase(user.email)
    }
  end

  def aggregate_stats(data) do
    data
    |> Enum.reduce(%{sum: 0, count: 0}, fn x, acc ->
      %{sum: acc.sum + x, count: acc.count + 1}
    end)
    |> then(fn %{sum: sum, count: count} -> sum / count end)
  end
end''',

    '''# Ecto Schema and Changeset
defmodule MyApp.Accounts.User do
  use Ecto.Schema
  import Ecto.Changeset

  schema "users" do
    field :name, :string
    field :email, :string
    field :password_hash, :string
    field :password, :string, virtual: true
    has_many :posts, MyApp.Blog.Post

    timestamps()
  end

  def changeset(user, attrs) do
    user
    |> cast(attrs, [:name, :email, :password])
    |> validate_required([:name, :email, :password])
    |> validate_format(:email, ~r/@/)
    |> validate_length(:password, min: 8)
    |> unique_constraint(:email)
    |> put_password_hash()
  end

  defp put_password_hash(%Ecto.Changeset{valid?: true, changes: %{password: password}} = changeset) do
    put_change(changeset, :password_hash, Bcrypt.hash_pwd_salt(password))
  end
  defp put_password_hash(changeset), do: changeset
end''',

    '''# Ecto Query
defmodule MyApp.Accounts do
  import Ecto.Query
  alias MyApp.Repo
  alias MyApp.Accounts.User

  def list_active_users do
    from(u in User,
      where: u.active == true,
      order_by: [desc: u.inserted_at],
      preload: [:posts]
    )
    |> Repo.all()
  end

  def get_user_with_posts(id) do
    from(u in User,
      where: u.id == ^id,
      left_join: p in assoc(u, :posts),
      preload: [posts: p]
    )
    |> Repo.one()
  end

  def search_users(query) do
    from(u in User,
      where: ilike(u.name, ^"%#{query}%") or ilike(u.email, ^"%#{query}%")
    )
    |> Repo.all()
  end
end''',

    '''# Task and async operations
defmodule AsyncProcessor do
  def fetch_all(urls) do
    urls
    |> Enum.map(&Task.async(fn -> fetch_url(&1) end))
    |> Enum.map(&Task.await(&1, 5000))
  end

  def process_with_timeout(items, timeout \\\\ 10_000) do
    Task.Supervisor.async_stream_nolink(
      MyApp.TaskSupervisor,
      items,
      &process_item/1,
      max_concurrency: 10,
      timeout: timeout
    )
    |> Enum.reduce([], fn
      {:ok, result}, acc -> [result | acc]
      {:exit, _reason}, acc -> acc
    end)
    |> Enum.reverse()
  end

  defp fetch_url(url), do: HTTPoison.get!(url)
  defp process_item(item), do: {:processed, item}
end''',

    '''# LiveView component
defmodule MyAppWeb.CounterLive do
  use MyAppWeb, :live_view

  def mount(_params, _session, socket) do
    {:ok, assign(socket, count: 0)}
  end

  def handle_event("increment", _params, socket) do
    {:noreply, update(socket, :count, &(&1 + 1))}
  end

  def handle_event("decrement", _params, socket) do
    {:noreply, update(socket, :count, &(&1 - 1))}
  end

  def render(assigns) do
    ~H"""
    <div class="counter">
      <h1>Count: <%= @count %></h1>
      <button phx-click="decrement">-</button>
      <button phx-click="increment">+</button>
    </div>
    """
  end
end''',

    '''# Protocol definition and implementation
defprotocol Stringify do
  @doc "Converts data to string representation"
  def to_string(data)
end

defimpl Stringify, for: Map do
  def to_string(map) do
    map
    |> Enum.map(fn {k, v} -> "#{k}: #{v}" end)
    |> Enum.join(", ")
    |> then(&"{#{&1}}")
  end
end

defimpl Stringify, for: List do
  def to_string(list) do
    list
    |> Enum.map(&Kernel.to_string/1)
    |> Enum.join(", ")
    |> then(&"[#{&1}]")
  end
end''',

    '''# Agent for state management
defmodule Cache do
  use Agent

  def start_link(_opts) do
    Agent.start_link(fn -> %{} end, name: __MODULE__)
  end

  def get(key) do
    Agent.get(__MODULE__, &Map.get(&1, key))
  end

  def put(key, value) do
    Agent.update(__MODULE__, &Map.put(&1, key, value))
  end

  def get_or_put(key, fun) do
    Agent.get_and_update(__MODULE__, fn state ->
      case Map.get(state, key) do
        nil ->
          value = fun.()
          {value, Map.put(state, key, value)}
        value ->
          {value, state}
      end
    end)
  end

  def delete(key) do
    Agent.update(__MODULE__, &Map.delete(&1, key))
  end
end''',
]

CLOJURE_SAMPLES = [
    '''; Basic functions and immutable data
(defn fibonacci [n]
  (loop [a 0 b 1 cnt n]
    (if (zero? cnt)
      a
      (recur b (+ a b) (dec cnt)))))

(defn factorial [n]
  (reduce * (range 1 (inc n))))

(defn prime? [n]
  (and (> n 1)
       (not-any? #(zero? (mod n %)) (range 2 (inc (Math/sqrt n))))))''',

    '''; Higher-order functions
(defn map-indexed-fn [f coll]
  (map-indexed (fn [idx item] (f idx item)) coll))

(defn filter-map [pred f coll]
  (->> coll
       (filter pred)
       (map f)))

(defn group-by-key [k coll]
  (reduce (fn [acc item]
            (let [key (get item k)]
              (update acc key (fnil conj []) item)))
          {}
          coll))''',

    '''; Threading macros
(defn process-data [data]
  (->> data
       (filter :active)
       (map :value)
       (remove nil?)
       (map #(* % 2))
       (reduce +)))

(defn transform-user [user]
  (-> user
      (assoc :full-name (str (:first-name user) " " (:last-name user)))
      (update :age inc)
      (dissoc :password)
      (assoc :updated-at (java.time.Instant/now))))''',

    '''; Atoms for state management
(def app-state (atom {:users [] :count 0}))

(defn add-user! [user]
  (swap! app-state update :users conj user)
  (swap! app-state update :count inc))

(defn get-users []
  (:users @app-state))

(defn reset-state! []
  (reset! app-state {:users [] :count 0}))

(defn update-user! [id update-fn]
  (swap! app-state update :users
         (fn [users]
           (mapv #(if (= (:id %) id) (update-fn %) %) users))))''',

    '''; Refs and transactions
(def account-a (ref {:balance 1000}))
(def account-b (ref {:balance 500}))

(defn transfer! [from to amount]
  (dosync
    (when (>= (:balance @from) amount)
      (alter from update :balance - amount)
      (alter to update :balance + amount)
      true)))

(defn get-balances []
  (dosync
    {:a (:balance @account-a)
     :b (:balance @account-b)
     :total (+ (:balance @account-a) (:balance @account-b))}))''',

    '''; core.async channels
(require '[clojure.core.async :as async :refer [go go-loop chan <! >! <!! >!! close! timeout]])

(defn producer [out-ch items]
  (go-loop [items items]
    (when-let [item (first items)]
      (>! out-ch item)
      (recur (rest items)))
    (close! out-ch)))

(defn consumer [in-ch handler]
  (go-loop []
    (when-let [item (<! in-ch)]
      (handler item)
      (recur))))

(defn pipeline [n in-ch out-ch xf]
  (dotimes [_ n]
    (go-loop []
      (when-let [item (<! in-ch)]
        (>! out-ch (xf item))
        (recur)))))''',

    '''; Protocols and records
(defprotocol Entity
  (save [this])
  (delete [this])
  (validate [this]))

(defrecord User [id name email]
  Entity
  (save [this]
    (println "Saving user:" (:name this))
    this)
  (delete [this]
    (println "Deleting user:" (:id this))
    nil)
  (validate [this]
    (and (string? (:name this))
         (re-matches #".+@.+\\..+" (:email this)))))

(defrecord Product [id name price]
  Entity
  (save [this]
    (println "Saving product:" (:name this))
    this)
  (delete [this]
    (println "Deleting product:" (:id this))
    nil)
  (validate [this]
    (and (string? (:name this))
         (pos? (:price this)))))''',

    '''; Multimethods for polymorphism
(defmulti area :shape)

(defmethod area :circle [{:keys [radius]}]
  (* Math/PI radius radius))

(defmethod area :rectangle [{:keys [width height]}]
  (* width height))

(defmethod area :triangle [{:keys [base height]}]
  (/ (* base height) 2))

(defmethod area :default [shape]
  (throw (ex-info "Unknown shape" {:shape shape})))''',

    '''; Spec for validation
(require '[clojure.spec.alpha :as s])

(s/def ::name (s/and string? #(> (count %) 0)))
(s/def ::email (s/and string? #(re-matches #".+@.+\\..+" %)))
(s/def ::age (s/and int? #(> % 0) #(< % 150)))
(s/def ::user (s/keys :req-un [::name ::email] :opt-un [::age]))

(defn validate-user [user]
  (if (s/valid? ::user user)
    {:valid true :user user}
    {:valid false :errors (s/explain-data ::user user)}))

(defn generate-sample-users [n]
  (gen/sample (s/gen ::user) n))''',

    '''; Lazy sequences
(defn lazy-fibonacci []
  (letfn [(fib [a b]
            (lazy-seq (cons a (fib b (+ a b)))))]
    (fib 0 1)))

(defn sieve [s]
  (lazy-seq
    (cons (first s)
          (sieve (filter #(not (zero? (mod % (first s))))
                         (rest s))))))

(def primes (sieve (iterate inc 2)))

(defn take-while-sum [limit coll]
  (lazy-seq
    (when-let [s (seq coll)]
      (let [f (first s)]
        (when (<= f limit)
          (cons f (take-while-sum (- limit f) (rest s))))))))''',

    '''; Component system pattern
(defrecord Database [connection config]
  component/Lifecycle
  (start [this]
    (println "Starting database connection")
    (assoc this :connection (create-connection (:config this))))
  (stop [this]
    (println "Stopping database connection")
    (when-let [conn (:connection this)]
      (.close conn))
    (assoc this :connection nil)))

(defrecord WebServer [port database]
  component/Lifecycle
  (start [this]
    (println "Starting web server on port" (:port this))
    (assoc this :server (start-server (:port this) (:database this))))
  (stop [this]
    (println "Stopping web server")
    (when-let [server (:server this)]
      (.stop server))
    (assoc this :server nil)))

(defn create-system [config]
  (component/system-map
    :database (map->Database {:config (:db config)})
    :web-server (component/using
                  (map->WebServer {:port (:port config)})
                  [:database])))''',

    '''; Transducers
(def xf
  (comp
    (filter even?)
    (map #(* % 2))
    (take 10)))

(defn process-with-transducer [coll]
  (transduce xf + coll))

(defn async-pipeline [in-ch out-ch xf]
  (async/pipeline 4 out-ch xf in-ch))

(defn parallel-transform [coll xf]
  (into [] xf coll))''',
]

DART_SAMPLES = [
    '''// Classes and constructors
class User {
  final String id;
  final String name;
  final String email;
  final DateTime createdAt;

  User({
    required this.id,
    required this.name,
    required this.email,
    DateTime? createdAt,
  }) : createdAt = createdAt ?? DateTime.now();

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'] as String,
      name: json['name'] as String,
      email: json['email'] as String,
      createdAt: DateTime.parse(json['createdAt'] as String),
    );
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'email': email,
    'createdAt': createdAt.toIso8601String(),
  };
}''',

    '''// Mixins
mixin Loggable {
  void log(String message) {
    print('[${DateTime.now()}] $message');
  }
}

mixin Validatable {
  bool validate();
  
  void assertValid() {
    if (!validate()) {
      throw ValidationException('Validation failed');
    }
  }
}

class UserService with Loggable, Validatable {
  final String apiKey;
  
  UserService(this.apiKey);
  
  @override
  bool validate() => apiKey.isNotEmpty;
  
  Future<User> fetchUser(String id) async {
    log('Fetching user: $id');
    assertValid();
    return await _api.getUser(id);
  }
}''',

    '''// Async/await and Futures
Future<List<User>> fetchUsers() async {
  try {
    final response = await http.get(Uri.parse('$baseUrl/users'));
    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data.map((json) => User.fromJson(json)).toList();
    }
    throw HttpException('Failed to load users: ${response.statusCode}');
  } catch (e) {
    print('Error fetching users: $e');
    rethrow;
  }
}

Future<void> processUsersInParallel(List<String> userIds) async {
  final futures = userIds.map((id) => fetchUserDetails(id));
  final results = await Future.wait(futures);
  for (final user in results) {
    await processUser(user);
  }
}''',

    '''// Streams
Stream<int> countStream(int max) async* {
  for (int i = 1; i <= max; i++) {
    await Future.delayed(Duration(seconds: 1));
    yield i;
  }
}

class EventBus {
  final _controller = StreamController<Event>.broadcast();
  
  Stream<Event> get stream => _controller.stream;
  
  void emit(Event event) {
    _controller.add(event);
  }
  
  Stream<T> on<T extends Event>() {
    return stream.where((event) => event is T).cast<T>();
  }
  
  void dispose() {
    _controller.close();
  }
}

void listenToEvents() {
  final bus = EventBus();
  bus.on<UserCreatedEvent>().listen((event) {
    print('User created: ${event.userId}');
  });
}''',

    '''// Flutter StatefulWidget
class CounterWidget extends StatefulWidget {
  const CounterWidget({super.key});

  @override
  State<CounterWidget> createState() => _CounterWidgetState();
}

class _CounterWidgetState extends State<CounterWidget> {
  int _counter = 0;

  void _increment() {
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          'Count: $_counter',
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 16),
        ElevatedButton(
          onPressed: _increment,
          child: const Text('Increment'),
        ),
      ],
    );
  }
}''',

    '''// Flutter with Provider
class UserProvider extends ChangeNotifier {
  User? _user;
  bool _isLoading = false;
  String? _error;

  User? get user => _user;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> fetchUser(String id) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _user = await userRepository.getUser(id);
    } catch (e) {
      _error = e.toString();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void logout() {
    _user = null;
    notifyListeners();
  }
}''',

    '''// Flutter custom widget
class CustomCard extends StatelessWidget {
  final String title;
  final String? subtitle;
  final Widget? leading;
  final VoidCallback? onTap;

  const CustomCard({
    super.key,
    required this.title,
    this.subtitle,
    this.leading,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              if (leading != null) ...[
                leading!,
                const SizedBox(width: 16),
              ],
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: Theme.of(context).textTheme.titleMedium),
                    if (subtitle != null)
                      Text(subtitle!, style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}''',

    '''// Riverpod providers
final userRepositoryProvider = Provider<UserRepository>((ref) {
  return UserRepository(ref.watch(apiClientProvider));
});

final userProvider = FutureProvider.family<User, String>((ref, userId) async {
  final repository = ref.watch(userRepositoryProvider);
  return repository.getUser(userId);
});

final usersProvider = StreamProvider<List<User>>((ref) {
  final repository = ref.watch(userRepositoryProvider);
  return repository.watchUsers();
});

class UserNotifier extends StateNotifier<AsyncValue<User?>> {
  final UserRepository _repository;

  UserNotifier(this._repository) : super(const AsyncValue.loading());

  Future<void> login(String email, String password) async {
    state = const AsyncValue.loading();
    state = await AsyncValue.guard(() => _repository.login(email, password));
  }

  void logout() {
    state = const AsyncValue.data(null);
  }
}''',

    '''// Extension methods
extension StringExtension on String {
  String capitalize() {
    if (isEmpty) return this;
    return '${this[0].toUpperCase()}${substring(1)}';
  }
  
  bool get isValidEmail {
    return RegExp(r'^[\\w-\\.]+@([\\w-]+\\.)+[\\w-]{2,4}$').hasMatch(this);
  }
  
  String truncate(int maxLength, {String ellipsis = '...'}) {
    if (length <= maxLength) return this;
    return '${substring(0, maxLength - ellipsis.length)}$ellipsis';
  }
}

extension ListExtension<T> on List<T> {
  List<List<T>> chunk(int size) {
    return [
      for (var i = 0; i < length; i += size)
        sublist(i, (i + size > length) ? length : i + size)
    ];
  }
  
  T? firstWhereOrNull(bool Function(T) test) {
    for (final element in this) {
      if (test(element)) return element;
    }
    return null;
  }
}''',

    '''// Isolates for heavy computation
Future<int> computeExpensive(int value) async {
  return await Isolate.run(() {
    int result = 0;
    for (int i = 0; i < value; i++) {
      result += i * i;
    }
    return result;
  });
}

class IsolateWorker {
  late SendPort _sendPort;
  final _receivePort = ReceivePort();
  
  Future<void> start() async {
    await Isolate.spawn(_workerIsolate, _receivePort.sendPort);
    _sendPort = await _receivePort.first as SendPort;
  }
  
  static void _workerIsolate(SendPort sendPort) {
    final receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);
    
    receivePort.listen((message) {
      final result = _processMessage(message);
      sendPort.send(result);
    });
  }
  
  Future<dynamic> process(dynamic message) async {
    final responsePort = ReceivePort();
    _sendPort.send([message, responsePort.sendPort]);
    return await responsePort.first;
  }
}''',

    '''// Repository pattern
abstract class Repository<T, ID> {
  Future<T?> findById(ID id);
  Future<List<T>> findAll();
  Future<T> save(T entity);
  Future<void> delete(ID id);
}

class UserRepositoryImpl implements Repository<User, String> {
  final ApiClient _client;
  final LocalDatabase _db;
  
  UserRepositoryImpl(this._client, this._db);
  
  @override
  Future<User?> findById(String id) async {
    final cached = await _db.getUser(id);
    if (cached != null) return cached;
    
    final remote = await _client.getUser(id);
    if (remote != null) {
      await _db.saveUser(remote);
    }
    return remote;
  }
  
  @override
  Future<List<User>> findAll() async {
    try {
      final remote = await _client.getUsers();
      await _db.saveUsers(remote);
      return remote;
    } catch (e) {
      return await _db.getUsers();
    }
  }
  
  @override
  Future<User> save(User entity) async {
    final saved = await _client.saveUser(entity);
    await _db.saveUser(saved);
    return saved;
  }
  
  @override
  Future<void> delete(String id) async {
    await _client.deleteUser(id);
    await _db.deleteUser(id);
  }
}''',

    '''// Bloc pattern
abstract class UserEvent {}

class LoadUser extends UserEvent {
  final String userId;
  LoadUser(this.userId);
}

class UpdateUser extends UserEvent {
  final User user;
  UpdateUser(this.user);
}

abstract class UserState {}

class UserInitial extends UserState {}
class UserLoading extends UserState {}
class UserLoaded extends UserState {
  final User user;
  UserLoaded(this.user);
}
class UserError extends UserState {
  final String message;
  UserError(this.message);
}

class UserBloc extends Bloc<UserEvent, UserState> {
  final UserRepository _repository;
  
  UserBloc(this._repository) : super(UserInitial()) {
    on<LoadUser>(_onLoadUser);
    on<UpdateUser>(_onUpdateUser);
  }
  
  Future<void> _onLoadUser(LoadUser event, Emitter<UserState> emit) async {
    emit(UserLoading());
    try {
      final user = await _repository.getUser(event.userId);
      emit(UserLoaded(user));
    } catch (e) {
      emit(UserError(e.toString()));
    }
  }
  
  Future<void> _onUpdateUser(UpdateUser event, Emitter<UserState> emit) async {
    await _repository.updateUser(event.user);
    emit(UserLoaded(event.user));
  }
}''',
]

LUA_SAMPLES = [
    '''-- Tables as arrays and dictionaries
local function create_vector(x, y, z)
    return {x = x or 0, y = y or 0, z = z or 0}
end

local function vector_add(v1, v2)
    return {
        x = v1.x + v2.x,
        y = v1.y + v2.y,
        z = v1.z + v2.z
    }
end

local function vector_magnitude(v)
    return math.sqrt(v.x^2 + v.y^2 + v.z^2)
end

local function vector_normalize(v)
    local mag = vector_magnitude(v)
    if mag == 0 then return create_vector() end
    return {x = v.x / mag, y = v.y / mag, z = v.z / mag}
end''',

    '''-- Metatables for OOP
local Vector = {}
Vector.__index = Vector

function Vector.new(x, y, z)
    local self = setmetatable({}, Vector)
    self.x = x or 0
    self.y = y or 0
    self.z = z or 0
    return self
end

function Vector:__add(other)
    return Vector.new(self.x + other.x, self.y + other.y, self.z + other.z)
end

function Vector:__mul(scalar)
    return Vector.new(self.x * scalar, self.y * scalar, self.z * scalar)
end

function Vector:__tostring()
    return string.format("Vector(%g, %g, %g)", self.x, self.y, self.z)
end

function Vector:dot(other)
    return self.x * other.x + self.y * other.y + self.z * other.z
end

function Vector:cross(other)
    return Vector.new(
        self.y * other.z - self.z * other.y,
        self.z * other.x - self.x * other.z,
        self.x * other.y - self.y * other.x
    )
end''',

    '''-- Coroutines
local function producer()
    return coroutine.create(function()
        for i = 1, 10 do
            coroutine.yield(i * i)
        end
    end)
end

local function consumer(prod)
    while true do
        local status, value = coroutine.resume(prod)
        if not status or value == nil then break end
        print("Received:", value)
    end
end

local function async_task(name, steps)
    return coroutine.create(function()
        for i = 1, steps do
            print(string.format("%s: step %d/%d", name, i, steps))
            coroutine.yield()
        end
        return name .. " completed"
    end)
end

local function scheduler(tasks)
    while #tasks > 0 do
        for i = #tasks, 1, -1 do
            local status = coroutine.resume(tasks[i])
            if coroutine.status(tasks[i]) == "dead" then
                table.remove(tasks, i)
            end
        end
    end
end''',

    '''-- Module system
local M = {}

local private_data = {}

function M.create(name)
    local id = #private_data + 1
    private_data[id] = {name = name, created = os.time()}
    return id
end

function M.get_name(id)
    local data = private_data[id]
    return data and data.name or nil
end

function M.set_name(id, name)
    if private_data[id] then
        private_data[id].name = name
        return true
    end
    return false
end

function M.get_age(id)
    local data = private_data[id]
    if not data then return nil end
    return os.time() - data.created
end

return M''',

    '''-- Class system with inheritance
local Class = {}

function Class:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

local Entity = Class:new()

function Entity:init(x, y)
    self.x = x or 0
    self.y = y or 0
    self.active = true
end

function Entity:update(dt)
end

function Entity:draw()
end

local Player = Entity:new()

function Player:init(x, y)
    Entity.init(self, x, y)
    self.health = 100
    self.speed = 200
end

function Player:update(dt)
    if love.keyboard.isDown("left") then
        self.x = self.x - self.speed * dt
    end
    if love.keyboard.isDown("right") then
        self.x = self.x + self.speed * dt
    end
end

function Player:draw()
    love.graphics.rectangle("fill", self.x, self.y, 32, 32)
end''',

    '''-- Game loop pattern (Love2D style)
local Game = {}
Game.entities = {}
Game.dt = 0

function Game.load()
    Game.player = Player:new()
    Game.player:init(400, 300)
    table.insert(Game.entities, Game.player)
    
    for i = 1, 10 do
        local enemy = Enemy:new()
        enemy:init(math.random(800), math.random(600))
        table.insert(Game.entities, enemy)
    end
end

function Game.update(dt)
    Game.dt = dt
    for i = #Game.entities, 1, -1 do
        local entity = Game.entities[i]
        if entity.active then
            entity:update(dt)
        else
            table.remove(Game.entities, i)
        end
    end
    Game.check_collisions()
end

function Game.draw()
    for _, entity in ipairs(Game.entities) do
        if entity.active then
            entity:draw()
        end
    end
    love.graphics.print("Entities: " .. #Game.entities, 10, 10)
end

function Game.check_collisions()
    for i = 1, #Game.entities do
        for j = i + 1, #Game.entities do
            if Game.aabb(Game.entities[i], Game.entities[j]) then
                Game.entities[i]:on_collision(Game.entities[j])
                Game.entities[j]:on_collision(Game.entities[i])
            end
        end
    end
end''',

    '''-- Event system
local EventEmitter = {}
EventEmitter.__index = EventEmitter

function EventEmitter.new()
    return setmetatable({listeners = {}}, EventEmitter)
end

function EventEmitter:on(event, callback)
    if not self.listeners[event] then
        self.listeners[event] = {}
    end
    table.insert(self.listeners[event], callback)
    return function()
        self:off(event, callback)
    end
end

function EventEmitter:off(event, callback)
    local listeners = self.listeners[event]
    if not listeners then return end
    for i = #listeners, 1, -1 do
        if listeners[i] == callback then
            table.remove(listeners, i)
        end
    end
end

function EventEmitter:emit(event, ...)
    local listeners = self.listeners[event]
    if not listeners then return end
    for _, callback in ipairs(listeners) do
        callback(...)
    end
end

function EventEmitter:once(event, callback)
    local unsubscribe
    unsubscribe = self:on(event, function(...)
        unsubscribe()
        callback(...)
    end)
end''',

    '''-- State machine
local StateMachine = {}
StateMachine.__index = StateMachine

function StateMachine.new(states)
    local self = setmetatable({}, StateMachine)
    self.states = states
    self.current = nil
    return self
end

function StateMachine:change(state_name, ...)
    local new_state = self.states[state_name]
    if not new_state then
        error("State not found: " .. state_name)
    end
    
    if self.current and self.current.exit then
        self.current:exit()
    end
    
    self.current = new_state
    
    if self.current.enter then
        self.current:enter(...)
    end
end

function StateMachine:update(dt)
    if self.current and self.current.update then
        self.current:update(dt)
    end
end

function StateMachine:draw()
    if self.current and self.current.draw then
        self.current:draw()
    end
end

local game_states = {
    menu = {
        enter = function() print("Entering menu") end,
        update = function(dt) end,
        draw = function() love.graphics.print("MENU", 400, 300) end,
    },
    playing = {
        enter = function() Game.load() end,
        update = function(dt) Game.update(dt) end,
        draw = function() Game.draw() end,
    }
}''',

    '''-- Object pooling
local Pool = {}
Pool.__index = Pool

function Pool.new(factory, initial_size)
    local self = setmetatable({}, Pool)
    self.factory = factory
    self.available = {}
    self.active = {}
    
    for i = 1, initial_size do
        table.insert(self.available, factory())
    end
    
    return self
end

function Pool:acquire()
    local obj
    if #self.available > 0 then
        obj = table.remove(self.available)
    else
        obj = self.factory()
    end
    self.active[obj] = true
    return obj
end

function Pool:release(obj)
    if self.active[obj] then
        self.active[obj] = nil
        if obj.reset then
            obj:reset()
        end
        table.insert(self.available, obj)
    end
end

function Pool:update(dt)
    for obj in pairs(self.active) do
        if obj.update then
            obj:update(dt)
        end
        if obj.dead then
            self:release(obj)
        end
    end
end''',

    '''-- Component system (ECS-lite)
local Component = {}
local Entity = {}
local World = {}

function Component.new(name, data)
    return {name = name, data = data or {}}
end

function Entity.new()
    return {
        components = {},
        add = function(self, component)
            self.components[component.name] = component
            return self
        end,
        get = function(self, name)
            return self.components[name]
        end,
        has = function(self, name)
            return self.components[name] ~= nil
        end
    }
end

function World.new()
    local self = {entities = {}, systems = {}}
    
    function self:add_entity(entity)
        table.insert(self.entities, entity)
        return entity
    end
    
    function self:add_system(system)
        table.insert(self.systems, system)
        return system
    end
    
    function self:update(dt)
        for _, system in ipairs(self.systems) do
            system:update(self.entities, dt)
        end
    end
    
    function self:query(...)
        local required = {...}
        local result = {}
        for _, entity in ipairs(self.entities) do
            local match = true
            for _, comp_name in ipairs(required) do
                if not entity:has(comp_name) then
                    match = false
                    break
                end
            end
            if match then
                table.insert(result, entity)
            end
        end
        return result
    end
    
    return self
end''',

    '''-- Functional utilities
local F = {}

function F.map(t, fn)
    local result = {}
    for i, v in ipairs(t) do
        result[i] = fn(v, i)
    end
    return result
end

function F.filter(t, fn)
    local result = {}
    for i, v in ipairs(t) do
        if fn(v, i) then
            table.insert(result, v)
        end
    end
    return result
end

function F.reduce(t, fn, initial)
    local acc = initial
    for i, v in ipairs(t) do
        acc = fn(acc, v, i)
    end
    return acc
end

function F.compose(...)
    local fns = {...}
    return function(...)
        local result = {...}
        for i = #fns, 1, -1 do
            result = {fns[i](unpack(result))}
        end
        return unpack(result)
    end
end

function F.curry(fn, arity)
    arity = arity or debug.getinfo(fn, "u").nparams
    return function(...)
        local args = {...}
        if #args >= arity then
            return fn(unpack(args))
        else
            return F.curry(function(...)
                local more_args = {...}
                for _, v in ipairs(more_args) do
                    table.insert(args, v)
                end
                return fn(unpack(args))
            end, arity - #args)
        end
    end
end''',

    '''-- Timer system
local Timer = {}
Timer.__index = Timer

function Timer.new()
    return setmetatable({timers = {}}, Timer)
end

function Timer:after(delay, callback)
    local timer = {
        time = delay,
        callback = callback,
        type = "after"
    }
    table.insert(self.timers, timer)
    return timer
end

function Timer:every(interval, callback, limit)
    local timer = {
        time = interval,
        interval = interval,
        callback = callback,
        type = "every",
        count = 0,
        limit = limit
    }
    table.insert(self.timers, timer)
    return timer
end

function Timer:tween(duration, subject, target, easing, on_complete)
    local start_values = {}
    for k, v in pairs(target) do
        start_values[k] = subject[k]
    end
    
    local timer = {
        time = duration,
        duration = duration,
        subject = subject,
        target = target,
        start = start_values,
        easing = easing or function(t) return t end,
        on_complete = on_complete,
        type = "tween"
    }
    table.insert(self.timers, timer)
    return timer
end

function Timer:update(dt)
    for i = #self.timers, 1, -1 do
        local timer = self.timers[i]
        timer.time = timer.time - dt
        
        if timer.type == "tween" then
            local t = 1 - (timer.time / timer.duration)
            t = math.max(0, math.min(1, t))
            t = timer.easing(t)
            for k, v in pairs(timer.target) do
                timer.subject[k] = timer.start[k] + (v - timer.start[k]) * t
            end
        end
        
        if timer.time <= 0 then
            if timer.type == "after" then
                timer.callback()
                table.remove(self.timers, i)
            elseif timer.type == "every" then
                timer.callback()
                timer.count = timer.count + 1
                if timer.limit and timer.count >= timer.limit then
                    table.remove(self.timers, i)
                else
                    timer.time = timer.interval
                end
            elseif timer.type == "tween" then
                if timer.on_complete then
                    timer.on_complete()
                end
                table.remove(self.timers, i)
            end
        end
    end
end''',
]

R_SAMPLES = [
    '''# Data frames and tibbles
library(tidyverse)

create_sample_data <- function(n = 100) {
  tibble(
    id = 1:n,
    name = paste0("User_", 1:n),
    age = sample(18:65, n, replace = TRUE),
    income = rnorm(n, mean = 50000, sd = 15000),
    category = sample(c("A", "B", "C"), n, replace = TRUE),
    date = seq.Date(Sys.Date() - n + 1, Sys.Date(), by = "day")
  )
}

summarize_by_category <- function(df) {
  df %>%
    group_by(category) %>%
    summarise(
      count = n(),
      mean_age = mean(age),
      mean_income = mean(income),
      sd_income = sd(income),
      .groups = "drop"
    )
}''',

    '''# dplyr data manipulation
library(dplyr)

process_sales_data <- function(sales) {
  sales %>%
    filter(amount > 0) %>%
    mutate(
      month = lubridate::month(date, label = TRUE),
      year = lubridate::year(date),
      amount_log = log(amount),
      is_large = amount > median(amount)
    ) %>%
    arrange(desc(amount)) %>%
    select(id, customer, amount, month, year, is_large)
}

join_customer_orders <- function(customers, orders) {
  customers %>%
    left_join(orders, by = "customer_id") %>%
    group_by(customer_id, customer_name) %>%
    summarise(
      total_orders = n(),
      total_spent = sum(amount, na.rm = TRUE),
      avg_order = mean(amount, na.rm = TRUE),
      first_order = min(order_date),
      last_order = max(order_date),
      .groups = "drop"
    ) %>%
    mutate(
      customer_tier = case_when(
        total_spent > 10000 ~ "Gold",
        total_spent > 5000 ~ "Silver",
        TRUE ~ "Bronze"
      )
    )
}''',

    '''# ggplot2 visualizations
library(ggplot2)

plot_distribution <- function(data, column, title = NULL) {
  ggplot(data, aes(x = .data[[column]])) +
    geom_histogram(aes(y = after_stat(density)), 
                   bins = 30, fill = "steelblue", alpha = 0.7) +
    geom_density(color = "darkred", linewidth = 1) +
    labs(
      title = title %||% paste("Distribution of", column),
      x = column,
      y = "Density"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.text = element_text(size = 10)
    )
}

plot_time_series <- function(data, date_col, value_col, group_col = NULL) {
  p <- ggplot(data, aes(x = .data[[date_col]], y = .data[[value_col]]))
  
  if (!is.null(group_col)) {
    p <- p + 
      geom_line(aes(color = .data[[group_col]]), linewidth = 1) +
      scale_color_brewer(palette = "Set1")
  } else {
    p <- p + geom_line(color = "steelblue", linewidth = 1)
  }
  
  p +
    geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
    labs(title = paste(value_col, "over time")) +
    theme_minimal() +
    scale_x_date(date_labels = "%b %Y")
}''',

    '''# Statistical functions
run_linear_regression <- function(data, formula_str) {
  formula <- as.formula(formula_str)
  model <- lm(formula, data = data)
  
  summary_stats <- summary(model)
  
  list(
    model = model,
    coefficients = coef(model),
    r_squared = summary_stats$r.squared,
    adj_r_squared = summary_stats$adj.r.squared,
    p_values = summary_stats$coefficients[, "Pr(>|t|)"],
    residuals = residuals(model),
    fitted = fitted(model)
  )
}

perform_t_test <- function(data, group_col, value_col) {
  groups <- unique(data[[group_col]])
  if (length(groups) != 2) {
    stop("T-test requires exactly 2 groups")
  }
  
  group1 <- data[data[[group_col]] == groups[1], ][[value_col]]
  group2 <- data[data[[group_col]] == groups[2], ][[value_col]]
  
  t.test(group1, group2)
}

calculate_correlations <- function(data, method = "pearson") {
  numeric_cols <- data %>% select(where(is.numeric))
  cor_matrix <- cor(numeric_cols, use = "pairwise.complete.obs", method = method)
  
  cor_matrix
}''',

    '''# Data cleaning and transformation
clean_dataset <- function(df) {
  df %>%
    janitor::clean_names() %>%
    mutate(across(where(is.character), str_trim)) %>%
    mutate(across(where(is.character), 
                  ~ na_if(., ""))) %>%
    mutate(across(where(is.numeric), 
                  ~ ifelse(is.infinite(.), NA, .)))
}

handle_missing_values <- function(df, strategy = "median") {
  df %>%
    mutate(across(where(is.numeric), ~ {
      if (strategy == "median") {
        replace_na(., median(., na.rm = TRUE))
      } else if (strategy == "mean") {
        replace_na(., mean(., na.rm = TRUE))
      } else {
        .
      }
    }))
}

detect_outliers <- function(df, columns, method = "iqr", threshold = 1.5) {
  outlier_flags <- map_dfc(columns, function(col) {
    values <- df[[col]]
    if (method == "iqr") {
      q1 <- quantile(values, 0.25, na.rm = TRUE)
      q3 <- quantile(values, 0.75, na.rm = TRUE)
      iqr <- q3 - q1
      lower <- q1 - threshold * iqr
      upper <- q3 + threshold * iqr
      is_outlier <- values < lower | values > upper
    } else if (method == "zscore") {
      z_scores <- scale(values)
      is_outlier <- abs(z_scores) > threshold
    }
    tibble(!!paste0(col, "_outlier") := is_outlier)
  })
  
  bind_cols(df, outlier_flags)
}''',

    '''# Machine learning with tidymodels
library(tidymodels)

build_classification_model <- function(data, target, predictors) {
  split <- initial_split(data, prop = 0.8, strata = all_of(target))
  train_data <- training(split)
  test_data <- testing(split)
  
  recipe <- recipe(as.formula(paste(target, "~", paste(predictors, collapse = "+"))), 
                   data = train_data) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_dummy(all_nominal_predictors())
  
  model_spec <- logistic_reg() %>%
    set_engine("glm") %>%
    set_mode("classification")
  
  workflow <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(model_spec)
  
  fitted_model <- fit(workflow, data = train_data)
  
  predictions <- predict(fitted_model, test_data) %>%
    bind_cols(predict(fitted_model, test_data, type = "prob")) %>%
    bind_cols(test_data %>% select(all_of(target)))
  
  metrics <- predictions %>%
    metrics(truth = .data[[target]], estimate = .pred_class)
  
  list(
    model = fitted_model,
    predictions = predictions,
    metrics = metrics
  )
}''',

    '''# Functional programming with purrr
library(purrr)

process_multiple_files <- function(file_paths) {
  file_paths %>%
    map(~ read_csv(., show_col_types = FALSE)) %>%
    map(clean_dataset) %>%
    reduce(bind_rows)
}

apply_transformations <- function(data, transformations) {
  reduce(transformations, function(df, fn) fn(df), .init = data)
}

safe_divide <- function(x, y) {
  safely(function(a, b) a / b)(x, y)
}

batch_process <- function(data, batch_size, process_fn) {
  data %>%
    group_split(ceiling(row_number() / batch_size)) %>%
    map(process_fn) %>%
    bind_rows()
}

nested_analysis <- function(data, group_col) {
  data %>%
    group_by(.data[[group_col]]) %>%
    nest() %>%
    mutate(
      model = map(data, ~ lm(value ~ predictor, data = .)),
      summary = map(model, broom::tidy),
      r_squared = map_dbl(model, ~ summary(.)$r.squared)
    ) %>%
    unnest(summary)
}''',

    '''# Shiny application
library(shiny)

ui <- fluidPage(
  titlePanel("Data Explorer"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV"),
      selectInput("x_var", "X Variable", choices = NULL),
      selectInput("y_var", "Y Variable", choices = NULL),
      selectInput("color_var", "Color By", choices = c("None" = "")),
      actionButton("plot_btn", "Generate Plot")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Plot", plotOutput("main_plot")),
        tabPanel("Summary", verbatimTextOutput("summary")),
        tabPanel("Data", DT::dataTableOutput("data_table"))
      )
    )
  )
)

server <- function(input, output, session) {
  data <- reactive({
    req(input$file)
    read_csv(input$file$datapath)
  })
  
  observe({
    req(data())
    cols <- names(data())
    updateSelectInput(session, "x_var", choices = cols)
    updateSelectInput(session, "y_var", choices = cols)
    updateSelectInput(session, "color_var", choices = c("None" = "", cols))
  })
  
  output$main_plot <- renderPlot({
    req(input$x_var, input$y_var)
    
    p <- ggplot(data(), aes_string(x = input$x_var, y = input$y_var))
    
    if (input$color_var != "") {
      p <- p + geom_point(aes_string(color = input$color_var))
    } else {
      p <- p + geom_point()
    }
    
    p + theme_minimal()
  }) %>% bindEvent(input$plot_btn)
  
  output$summary <- renderPrint({
    req(data())
    summary(data())
  })
  
  output$data_table <- DT::renderDataTable({
    req(data())
    data()
  })
}''',

    '''# Time series analysis
library(forecast)
library(tseries)

analyze_time_series <- function(data, date_col, value_col, frequency = 12) {
  ts_data <- ts(data[[value_col]], 
                start = c(year(min(data[[date_col]])), 
                          month(min(data[[date_col]]))),
                frequency = frequency)
  
  decomposition <- decompose(ts_data)
  
  adf_test <- adf.test(ts_data)
  
  acf_result <- acf(ts_data, plot = FALSE)
  pacf_result <- pacf(ts_data, plot = FALSE)
  
  auto_model <- auto.arima(ts_data)
  
  forecast_result <- forecast(auto_model, h = 12)
  
  list(
    ts_data = ts_data,
    decomposition = decomposition,
    stationarity_test = adf_test,
    acf = acf_result,
    pacf = pacf_result,
    model = auto_model,
    forecast = forecast_result
  )
}

plot_forecast <- function(forecast_obj, title = "Time Series Forecast") {
  autoplot(forecast_obj) +
    labs(title = title, x = "Time", y = "Value") +
    theme_minimal()
}''',

    '''# Text analysis
library(tidytext)
library(wordcloud)

analyze_text <- function(text_df, text_col) {
  text_df %>%
    unnest_tokens(word, .data[[text_col]]) %>%
    anti_join(stop_words, by = "word") %>%
    count(word, sort = TRUE)
}

sentiment_analysis <- function(text_df, text_col) {
  text_df %>%
    unnest_tokens(word, .data[[text_col]]) %>%
    inner_join(get_sentiments("bing"), by = "word") %>%
    count(sentiment) %>%
    spread(sentiment, n, fill = 0) %>%
    mutate(sentiment_score = positive - negative)
}

create_wordcloud <- function(word_counts, max_words = 100) {
  wordcloud(
    words = word_counts$word,
    freq = word_counts$n,
    max.words = max_words,
    random.order = FALSE,
    colors = brewer.pal(8, "Dark2")
  )
}

tf_idf_analysis <- function(text_df, text_col, group_col) {
  text_df %>%
    unnest_tokens(word, .data[[text_col]]) %>%
    count(.data[[group_col]], word) %>%
    bind_tf_idf(word, .data[[group_col]], n) %>%
    arrange(desc(tf_idf))
}''',

    '''# Database operations
library(DBI)
library(dbplyr)

create_db_connection <- function(driver = "PostgreSQL", 
                                  dbname, host, port, user, password) {
  con <- dbConnect(
    RPostgres::Postgres(),
    dbname = dbname,
    host = host,
    port = port,
    user = user,
    password = password
  )
  
  con
}

query_database <- function(con, query) {
  result <- dbGetQuery(con, query)
  as_tibble(result)
}

lazy_query <- function(con, table_name) {
  tbl(con, table_name)
}

execute_with_transaction <- function(con, queries) {
  tryCatch({
    dbBegin(con)
    results <- map(queries, ~ dbExecute(con, .))
    dbCommit(con)
    results
  }, error = function(e) {
    dbRollback(con)
    stop(e)
  })
}

bulk_insert <- function(con, table_name, data, chunk_size = 1000) {
  data %>%
    group_split(ceiling(row_number() / chunk_size)) %>%
    walk(~ dbWriteTable(con, table_name, ., append = TRUE))
}''',

    '''# Reporting with R Markdown helpers
generate_summary_table <- function(data, group_var = NULL) {
  if (!is.null(group_var)) {
    data %>%
      group_by(.data[[group_var]]) %>%
      summarise(across(where(is.numeric), list(
        mean = ~ mean(., na.rm = TRUE),
        sd = ~ sd(., na.rm = TRUE),
        min = ~ min(., na.rm = TRUE),
        max = ~ max(., na.rm = TRUE)
      ))) %>%
      knitr::kable(digits = 2)
  } else {
    data %>%
      summarise(across(where(is.numeric), list(
        mean = ~ mean(., na.rm = TRUE),
        sd = ~ sd(., na.rm = TRUE),
        min = ~ min(., na.rm = TRUE),
        max = ~ max(., na.rm = TRUE)
      ))) %>%
      knitr::kable(digits = 2)
  }
}

create_report_plot <- function(data, plot_type, ...) {
  switch(plot_type,
    "histogram" = plot_distribution(data, ...),
    "scatter" = plot_scatter(data, ...),
    "time_series" = plot_time_series(data, ...),
    "bar" = plot_bar(data, ...),
    stop("Unknown plot type")
  )
}

save_results <- function(results, filename, format = "rds") {
  switch(format,
    "rds" = saveRDS(results, paste0(filename, ".rds")),
    "csv" = write_csv(results, paste0(filename, ".csv")),
    "xlsx" = writexl::write_xlsx(results, paste0(filename, ".xlsx")),
    stop("Unknown format")
  )
}''',
]

JULIA_SAMPLES = [
    '''# Multiple dispatch
abstract type Shape end

struct Circle <: Shape
    radius::Float64
end

struct Rectangle <: Shape
    width::Float64
    height::Float64
end

struct Triangle <: Shape
    base::Float64
    height::Float64
end

area(c::Circle) = pi * c.radius^2
area(r::Rectangle) = r.width * r.height
area(t::Triangle) = 0.5 * t.base * t.height

perimeter(c::Circle) = 2 * pi * c.radius
perimeter(r::Rectangle) = 2 * (r.width + r.height)

function total_area(shapes::Vector{<:Shape})
    sum(area, shapes)
end''',

    '''# Macros and metaprogramming
macro time_it(expr)
    quote
        local start = time_ns()
        local result = $(esc(expr))
        local elapsed = (time_ns() - start) / 1e9
        println("Elapsed time: ", elapsed, " seconds")
        result
    end
end

macro assert_type(var, expected_type)
    quote
        if !($(esc(var)) isa $(esc(expected_type)))
            error("Expected $($(string(var))) to be of type $($(string(expected_type)))")
        end
    end
end

macro define_getter(struct_type, field)
    fname = Symbol("get_", field)
    quote
        function $(esc(fname))(obj::$(esc(struct_type)))
            getfield(obj, $(QuoteNode(field)))
        end
    end
end''',

    '''# DataFrames operations
using DataFrames, CSV, Statistics

function load_and_process(filepath::String)
    df = CSV.read(filepath, DataFrame)
    
    transform!(df, :date => ByRow(x -> Date(x)) => :date)
    
    filter!(row -> !ismissing(row.value) && row.value > 0, df)
    
    df.log_value = log.(df.value)
    
    return df
end

function aggregate_by_group(df::DataFrame, group_col::Symbol)
    combine(groupby(df, group_col),
        :value => mean => :mean_value,
        :value => std => :std_value,
        :value => minimum => :min_value,
        :value => maximum => :max_value,
        nrow => :count
    )
end

function join_dataframes(df1::DataFrame, df2::DataFrame, on_col::Symbol)
    leftjoin(df1, df2, on = on_col)
end''',

    '''# Scientific computing
using LinearAlgebra

function solve_linear_system(A::Matrix{Float64}, b::Vector{Float64})
    if size(A, 1) != size(A, 2)
        throw(DimensionMismatch("Matrix must be square"))
    end
    
    lu_fact = lu(A)
    
    x = lu_fact \\ b
    
    residual = norm(A * x - b)
    
    return x, residual
end

function eigendecomposition(A::Matrix{Float64})
    vals, vecs = eigen(A)
    
    sorted_indices = sortperm(abs.(vals), rev=true)
    
    return vals[sorted_indices], vecs[:, sorted_indices]
end

function svd_compress(A::Matrix{Float64}, k::Int)
    U, S, V = svd(A)
    
    return U[:, 1:k] * Diagonal(S[1:k]) * V[:, 1:k]'
end''',

    '''# Parallel computing
using Distributed, SharedArrays

function parallel_map(f, data::Vector; nworkers::Int=4)
    if nprocs() < nworkers + 1
        addprocs(nworkers + 1 - nprocs())
    end
    
    results = pmap(f, data)
    
    return results
end

function threaded_computation(data::Vector{Float64})
    n = length(data)
    result = zeros(n)
    
    Threads.@threads for i in 1:n
        result[i] = expensive_computation(data[i])
    end
    
    return result
end

function shared_array_example(n::Int)
    S = SharedArray{Float64}(n)
    
    @sync @distributed for i in 1:n
        S[i] = sqrt(i)
    end
    
    return S
end''',

    '''# Differential equations
using DifferentialEquations

function lotka_volterra!(du, u, p, t)
    prey, predator = u
    alpha, beta, gamma, delta = p
    
    du[1] = alpha * prey - beta * prey * predator
    du[2] = delta * prey * predator - gamma * predator
end

function solve_ode()
    u0 = [10.0, 5.0]
    p = (1.5, 0.1, 0.8, 0.02)
    tspan = (0.0, 100.0)
    
    prob = ODEProblem(lotka_volterra!, u0, tspan, p)
    
    sol = solve(prob, Tsit5(), saveat=0.1)
    
    return sol
end

function heat_equation(N::Int, T::Float64, dt::Float64)
    dx = 1.0 / N
    alpha = 0.01
    
    u = zeros(N)
    u[div(N, 4):div(3N, 4)] .= 1.0
    
    t = 0.0
    while t < T
        u_new = copy(u)
        for i in 2:N-1
            u_new[i] = u[i] + alpha * dt / dx^2 * (u[i+1] - 2*u[i] + u[i-1])
        end
        u = u_new
        t += dt
    end
    
    return u
end''',

    '''# Optimization
using Optim, ForwardDiff

function optimize_function(f, x0::Vector{Float64}; method=BFGS())
    result = optimize(f, x0, method, autodiff=:forward)
    
    return Optim.minimizer(result), Optim.minimum(result)
end

function gradient_descent(f, x0::Vector{Float64}; 
                          learning_rate=0.01, 
                          max_iter=1000, 
                          tol=1e-6)
    x = copy(x0)
    
    for i in 1:max_iter
        grad = ForwardDiff.gradient(f, x)
        
        x_new = x - learning_rate * grad
        
        if norm(x_new - x) < tol
            return x_new, f(x_new), i
        end
        
        x = x_new
    end
    
    return x, f(x), max_iter
end

function constrained_optimization(f, g, x0)
    result = optimize(f, g, x0, IPNewton())
    return Optim.minimizer(result)
end''',

    '''# Neural network basics
using Flux

function create_neural_network(input_size::Int, hidden_sizes::Vector{Int}, output_size::Int)
    layers = []
    
    push!(layers, Dense(input_size, hidden_sizes[1], relu))
    
    for i in 1:length(hidden_sizes)-1
        push!(layers, Dense(hidden_sizes[i], hidden_sizes[i+1], relu))
    end
    
    push!(layers, Dense(hidden_sizes[end], output_size))
    
    return Chain(layers...)
end

function train_model!(model, X_train, y_train; epochs=100, lr=0.01)
    loss(x, y) = Flux.mse(model(x), y)
    
    opt = ADAM(lr)
    
    data = [(X_train, y_train)]
    
    losses = Float64[]
    
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), data, opt)
        push!(losses, loss(X_train, y_train))
        
        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $(losses[end])")
        end
    end
    
    return losses
end''',

    '''# Type system and parametric types
struct Stack{T}
    items::Vector{T}
    max_size::Int
    
    Stack{T}(max_size::Int=100) where T = new{T}(T[], max_size)
end

function Base.push!(s::Stack{T}, item::T) where T
    if length(s.items) >= s.max_size
        throw(OverflowError("Stack is full"))
    end
    push!(s.items, item)
    return s
end

function Base.pop!(s::Stack)
    if isempty(s.items)
        throw(ArgumentError("Stack is empty"))
    end
    return pop!(s.items)
end

Base.isempty(s::Stack) = isempty(s.items)
Base.length(s::Stack) = length(s.items)

struct Result{T, E}
    value::Union{T, Nothing}
    error::Union{E, Nothing}
    
    Result{T, E}(value::T) where {T, E} = new{T, E}(value, nothing)
    Result{T, E}(::Nothing, error::E) where {T, E} = new{T, E}(nothing, error)
end

isok(r::Result) = r.error === nothing
iserror(r::Result) = r.error !== nothing''',

    '''# Plotting with Plots.jl
using Plots

function plot_function(f, xrange::Tuple{Float64, Float64}; 
                       n_points::Int=100, 
                       title::String="Function Plot")
    x = range(xrange[1], xrange[2], length=n_points)
    y = f.(x)
    
    plot(x, y, 
         title=title, 
         xlabel="x", 
         ylabel="f(x)",
         linewidth=2,
         legend=false)
end

function plot_scatter_with_regression(x::Vector{Float64}, y::Vector{Float64})
    scatter(x, y, label="Data", alpha=0.6)
    
    A = hcat(ones(length(x)), x)
    coeffs = A \\ y
    
    x_line = range(minimum(x), maximum(x), length=100)
    y_line = coeffs[1] .+ coeffs[2] .* x_line
    
    plot!(x_line, y_line, label="Linear fit", linewidth=2, color=:red)
end

function create_subplot_grid(data_list::Vector, titles::Vector{String})
    n = length(data_list)
    cols = ceil(Int, sqrt(n))
    rows = ceil(Int, n / cols)
    
    plots = [plot(d, title=t) for (d, t) in zip(data_list, titles)]
    
    plot(plots..., layout=(rows, cols))
end''',

    '''# File I/O and serialization
using JSON, BSON, JLD2

function save_results(data, filename::String; format::Symbol=:jld2)
    if format == :json
        open(filename * ".json", "w") do f
            JSON.print(f, data, 2)
        end
    elseif format == :bson
        BSON.@save filename * ".bson" data
    elseif format == :jld2
        @save filename * ".jld2" data
    else
        error("Unknown format: $format")
    end
end

function load_results(filename::String; format::Symbol=:jld2)
    if format == :json
        return JSON.parsefile(filename * ".json")
    elseif format == :bson
        BSON.@load filename * ".bson" data
        return data
    elseif format == :jld2
        @load filename * ".jld2" data
        return data
    else
        error("Unknown format: $format")
    end
end

function process_large_file(filename::String, process_line::Function)
    results = []
    open(filename, "r") do f
        for line in eachline(f)
            result = process_line(line)
            if result !== nothing
                push!(results, result)
            end
        end
    end
    return results
end''',

    '''# Testing and benchmarking
using Test, BenchmarkTools

@testset "Math Functions" begin
    @test area(Circle(1.0)) ≈ pi
    @test area(Rectangle(2.0, 3.0)) ≈ 6.0
    @test area(Triangle(4.0, 3.0)) ≈ 6.0
    
    @test_throws DimensionMismatch solve_linear_system(rand(2, 3), rand(2))
    
    @testset "Edge Cases" begin
        @test area(Circle(0.0)) == 0.0
        @test area(Rectangle(0.0, 5.0)) == 0.0
    end
end

function benchmark_implementations()
    data = rand(1000)
    
    println("Implementation 1:")
    @btime sum($data)
    
    println("Implementation 2:")
    @btime reduce(+, $data)
    
    println("Implementation 3:")
    @btime foldl(+, $data)
end

function profile_function(f, args...)
    @time result = f(args...)
    @allocated f(args...)
    return result
end''',
]

PERL_SAMPLES = [
    '''# Regular expressions
use strict;
use warnings;

sub validate_email {
    my ($email) = @_;
    return $email =~ /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
}

sub extract_urls {
    my ($text) = @_;
    my @urls = $text =~ m{(https?://[^\s<>"{}|\\^`\[\]]+)}g;
    return @urls;
}

sub parse_log_line {
    my ($line) = @_;
    if ($line =~ /^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)$/) {
        return {
            date    => $1,
            time    => $2,
            level   => $3,
            message => $4
        };
    }
    return undef;
}

sub sanitize_html {
    my ($text) = @_;
    $text =~ s/<[^>]*>//g;
    $text =~ s/&nbsp;/ /g;
    $text =~ s/&amp;/&/g;
    $text =~ s/&lt;/</g;
    $text =~ s/&gt;/>/g;
    return $text;
}''',

    '''# Subroutines and references
use strict;
use warnings;

sub create_counter {
    my $count = 0;
    return sub {
        $count++;
        return $count;
    };
}

sub apply_to_all {
    my ($array_ref, $func) = @_;
    return [ map { $func->($_) } @$array_ref ];
}

sub memoize {
    my ($func) = @_;
    my %cache;
    return sub {
        my @args = @_;
        my $key = join(":", @args);
        unless (exists $cache{$key}) {
            $cache{$key} = $func->(@args);
        }
        return $cache{$key};
    };
}

sub compose {
    my @funcs = @_;
    return sub {
        my $result = shift;
        for my $func (reverse @funcs) {
            $result = $func->($result);
        }
        return $result;
    };
}''',

    '''# File handling
use strict;
use warnings;
use File::Find;
use File::Basename;

sub read_file {
    my ($filename) = @_;
    open(my $fh, '<:encoding(UTF-8)', $filename)
        or die "Cannot open file '$filename': $!";
    local $/;
    my $content = <$fh>;
    close($fh);
    return $content;
}

sub write_file {
    my ($filename, $content) = @_;
    open(my $fh, '>:encoding(UTF-8)', $filename)
        or die "Cannot write to file '$filename': $!";
    print $fh $content;
    close($fh);
}

sub process_csv {
    my ($filename, $callback) = @_;
    open(my $fh, '<:encoding(UTF-8)', $filename)
        or die "Cannot open file '$filename': $!";
    
    my $header = <$fh>;
    chomp $header;
    my @columns = split /,/, $header;
    
    while (my $line = <$fh>) {
        chomp $line;
        my @values = split /,/, $line;
        my %row;
        @row{@columns} = @values;
        $callback->(\%row);
    }
    close($fh);
}

sub find_files {
    my ($dir, $pattern) = @_;
    my @files;
    find(sub {
        push @files, $File::Find::name if -f && /$pattern/;
    }, $dir);
    return @files;
}''',

    '''# Hashes and arrays
use strict;
use warnings;

sub deep_copy {
    my ($ref) = @_;
    if (ref($ref) eq 'HASH') {
        return { map { $_ => deep_copy($ref->{$_}) } keys %$ref };
    } elsif (ref($ref) eq 'ARRAY') {
        return [ map { deep_copy($_) } @$ref ];
    } else {
        return $ref;
    }
}

sub merge_hashes {
    my ($hash1, $hash2) = @_;
    my %result = %$hash1;
    for my $key (keys %$hash2) {
        if (ref($result{$key}) eq 'HASH' && ref($hash2->{$key}) eq 'HASH') {
            $result{$key} = merge_hashes($result{$key}, $hash2->{$key});
        } else {
            $result{$key} = $hash2->{$key};
        }
    }
    return \%result;
}

sub group_by {
    my ($array_ref, $key_func) = @_;
    my %groups;
    for my $item (@$array_ref) {
        my $key = $key_func->($item);
        push @{$groups{$key}}, $item;
    }
    return \%groups;
}

sub flatten {
    my @result;
    for my $item (@_) {
        if (ref($item) eq 'ARRAY') {
            push @result, flatten(@$item);
        } else {
            push @result, $item;
        }
    }
    return @result;
}''',

    '''# Object-oriented Perl
package Person;
use strict;
use warnings;

sub new {
    my ($class, %args) = @_;
    my $self = {
        name  => $args{name}  // '',
        email => $args{email} // '',
        age   => $args{age}   // 0,
    };
    bless $self, $class;
    return $self;
}

sub name {
    my ($self, $value) = @_;
    if (defined $value) {
        $self->{name} = $value;
    }
    return $self->{name};
}

sub email {
    my ($self, $value) = @_;
    if (defined $value) {
        die "Invalid email" unless $value =~ /@/;
        $self->{email} = $value;
    }
    return $self->{email};
}

sub to_string {
    my ($self) = @_;
    return sprintf("Person: %s <%s>, age %d", 
                   $self->{name}, $self->{email}, $self->{age});
}

1;''',

    '''# Moose OOP
package User;
use Moose;
use namespace::autoclean;

has 'id' => (
    is       => 'ro',
    isa      => 'Int',
    required => 1,
);

has 'username' => (
    is       => 'rw',
    isa      => 'Str',
    required => 1,
);

has 'email' => (
    is       => 'rw',
    isa      => 'Str',
    required => 1,
    trigger  => sub {
        my ($self, $new_val) = @_;
        die "Invalid email format" unless $new_val =~ /@/;
    },
);

has 'created_at' => (
    is      => 'ro',
    isa     => 'DateTime',
    default => sub { DateTime->now },
);

has 'roles' => (
    is      => 'rw',
    isa     => 'ArrayRef[Str]',
    default => sub { [] },
    traits  => ['Array'],
    handles => {
        add_role    => 'push',
        has_role    => 'first',
        all_roles   => 'elements',
        role_count  => 'count',
    },
);

sub is_admin {
    my ($self) = @_;
    return $self->has_role(sub { $_ eq 'admin' });
}

__PACKAGE__->meta->make_immutable;
1;''',

    '''# Database operations with DBI
use strict;
use warnings;
use DBI;

sub get_db_connection {
    my ($dsn, $user, $password) = @_;
    my $dbh = DBI->connect($dsn, $user, $password, {
        RaiseError => 1,
        AutoCommit => 1,
        PrintError => 0,
    }) or die "Cannot connect to database: $DBI::errstr";
    return $dbh;
}

sub execute_query {
    my ($dbh, $sql, @params) = @_;
    my $sth = $dbh->prepare($sql);
    $sth->execute(@params);
    my @results;
    while (my $row = $sth->fetchrow_hashref) {
        push @results, $row;
    }
    return \@results;
}

sub insert_record {
    my ($dbh, $table, $data) = @_;
    my @columns = keys %$data;
    my @values = @{$data}{@columns};
    my $placeholders = join(', ', ('?') x @columns);
    my $columns_str = join(', ', @columns);
    
    my $sql = "INSERT INTO $table ($columns_str) VALUES ($placeholders)";
    my $sth = $dbh->prepare($sql);
    $sth->execute(@values);
    return $dbh->last_insert_id(undef, undef, $table, undef);
}

sub transaction {
    my ($dbh, $callback) = @_;
    eval {
        $dbh->begin_work;
        $callback->($dbh);
        $dbh->commit;
    };
    if ($@) {
        $dbh->rollback;
        die "Transaction failed: $@";
    }
}''',

    '''# Web scraping
use strict;
use warnings;
use LWP::UserAgent;
use HTML::TreeBuilder;
use JSON;

sub fetch_url {
    my ($url) = @_;
    my $ua = LWP::UserAgent->new(
        timeout => 30,
        agent   => 'Mozilla/5.0',
    );
    
    my $response = $ua->get($url);
    
    if ($response->is_success) {
        return $response->decoded_content;
    } else {
        die "Failed to fetch $url: " . $response->status_line;
    }
}

sub parse_html {
    my ($html) = @_;
    my $tree = HTML::TreeBuilder->new_from_content($html);
    return $tree;
}

sub extract_links {
    my ($tree, $base_url) = @_;
    my @links;
    for my $a ($tree->look_down(_tag => 'a')) {
        my $href = $a->attr('href');
        next unless $href;
        push @links, {
            url  => $href,
            text => $a->as_trimmed_text,
        };
    }
    return \@links;
}

sub fetch_json_api {
    my ($url) = @_;
    my $ua = LWP::UserAgent->new;
    my $response = $ua->get($url, 'Accept' => 'application/json');
    
    if ($response->is_success) {
        return decode_json($response->decoded_content);
    }
    die "API request failed: " . $response->status_line;
}''',

    '''# Async with AnyEvent
use strict;
use warnings;
use AnyEvent;
use AnyEvent::HTTP;

sub fetch_urls_async {
    my (@urls) = @_;
    my $cv = AnyEvent->condvar;
    my %results;
    
    $cv->begin(sub { $cv->send(\%results) });
    
    for my $url (@urls) {
        $cv->begin;
        http_get $url, sub {
            my ($body, $headers) = @_;
            $results{$url} = {
                status => $headers->{Status},
                body   => $body,
            };
            $cv->end;
        };
    }
    
    $cv->end;
    return $cv->recv;
}

sub create_timer {
    my ($interval, $callback) = @_;
    return AnyEvent->timer(
        after    => 0,
        interval => $interval,
        cb       => $callback,
    );
}

sub run_event_loop {
    my ($timeout) = @_;
    my $cv = AnyEvent->condvar;
    
    if ($timeout) {
        my $timer = AnyEvent->timer(
            after => $timeout,
            cb    => sub { $cv->send },
        );
    }
    
    $cv->recv;
}''',

    '''# Testing with Test::More
use strict;
use warnings;
use Test::More;
use Test::Exception;
use Test::Deep;

subtest 'validate_email tests' => sub {
    ok(validate_email('user@example.com'), 'Valid email accepted');
    ok(!validate_email('invalid-email'), 'Invalid email rejected');
    ok(!validate_email(''), 'Empty string rejected');
    ok(validate_email('user.name+tag@domain.co.uk'), 'Complex email accepted');
};

subtest 'data processing tests' => sub {
    my $data = [
        { name => 'Alice', age => 30 },
        { name => 'Bob', age => 25 },
    ];
    
    my $grouped = group_by($data, sub { $_[0]->{age} > 27 ? 'senior' : 'junior' });
    
    cmp_deeply($grouped, {
        senior => [ { name => 'Alice', age => 30 } ],
        junior => [ { name => 'Bob', age => 25 } ],
    }, 'Grouping works correctly');
};

subtest 'exception handling tests' => sub {
    throws_ok { die "Error!" } qr/Error/, 'Exception thrown as expected';
    lives_ok { my $x = 1 + 1 } 'No exception for valid code';
};

done_testing();''',

    '''# Command-line tool
use strict;
use warnings;
use Getopt::Long;
use Pod::Usage;

my %options = (
    verbose => 0,
    output  => '-',
    format  => 'text',
);

GetOptions(
    'verbose|v'  => \$options{verbose},
    'output|o=s' => \$options{output},
    'format|f=s' => \$options{format},
    'help|h'     => sub { pod2usage(1) },
    'man'        => sub { pod2usage(-verbose => 2) },
) or pod2usage(2);

my @files = @ARGV;
pod2usage("No input files specified") unless @files;

sub log_message {
    my ($msg) = @_;
    print STDERR "[INFO] $msg\n" if $options{verbose};
}

sub process_file {
    my ($filename) = @_;
    log_message("Processing $filename");
    
    open(my $fh, '<', $filename) or die "Cannot open $filename: $!";
    my @lines = <$fh>;
    close($fh);
    
    return \@lines;
}

sub format_output {
    my ($data, $format) = @_;
    if ($format eq 'json') {
        require JSON;
        return JSON::encode_json($data);
    } elsif ($format eq 'yaml') {
        require YAML;
        return YAML::Dump($data);
    }
    return join("\n", @$data);
}

my @results;
for my $file (@files) {
    push @results, @{process_file($file)};
}

my $output = format_output(\@results, $options{format});
if ($options{output} eq '-') {
    print $output;
} else {
    write_file($options{output}, $output);
}''',

    '''# Module with exports
package Utils::String;
use strict;
use warnings;
use Exporter 'import';

our @EXPORT_OK = qw(
    trim
    truncate
    slugify
    camelize
    underscore
);

our %EXPORT_TAGS = (
    all => \@EXPORT_OK,
);

sub trim {
    my ($str) = @_;
    $str =~ s/^\s+//;
    $str =~ s/\s+$//;
    return $str;
}

sub truncate {
    my ($str, $length, $suffix) = @_;
    $suffix //= '...';
    return $str if length($str) <= $length;
    return substr($str, 0, $length - length($suffix)) . $suffix;
}

sub slugify {
    my ($str) = @_;
    $str = lc($str);
    $str =~ s/[^a-z0-9]+/-/g;
    $str =~ s/^-|-$//g;
    return $str;
}

sub camelize {
    my ($str) = @_;
    $str =~ s/_([a-z])/uc($1)/ge;
    return $str;
}

sub underscore {
    my ($str) = @_;
    $str =~ s/([A-Z])/_\l$1/g;
    $str =~ s/^_//;
    return $str;
}

1;''',
]


def get_all_training_data(shuffle: bool = True, seed: int = 42) -> List[str]:
    """Get all training samples as a single shuffled list."""
    all_samples = []
    all_samples.extend(PYTHON_SAMPLES)
    all_samples.extend(JAVASCRIPT_SAMPLES)
    all_samples.extend(TYPESCRIPT_SAMPLES)
    all_samples.extend(REACT_SAMPLES)
    all_samples.extend(SQL_SAMPLES)
    all_samples.extend(TERRAFORM_SAMPLES)
    all_samples.extend(KUBERNETES_SAMPLES)
    all_samples.extend(DOCKER_SAMPLES)
    all_samples.extend(GO_SAMPLES)
    all_samples.extend(RUST_SAMPLES)
    all_samples.extend(JAVA_SAMPLES)
    all_samples.extend(CSHARP_SAMPLES)
    all_samples.extend(CPP_SAMPLES)
    all_samples.extend(C_SAMPLES)
    all_samples.extend(RUBY_SAMPLES)
    all_samples.extend(PHP_SAMPLES)
    all_samples.extend(SWIFT_SAMPLES)
    all_samples.extend(KOTLIN_SAMPLES)
    all_samples.extend(SCALA_SAMPLES)
    all_samples.extend(SHELL_SAMPLES)
    all_samples.extend(HASKELL_SAMPLES)
    all_samples.extend(ELIXIR_SAMPLES)
    all_samples.extend(CLOJURE_SAMPLES)
    all_samples.extend(DART_SAMPLES)
    all_samples.extend(LUA_SAMPLES)
    all_samples.extend(R_SAMPLES)
    all_samples.extend(JULIA_SAMPLES)
    all_samples.extend(PERL_SAMPLES)
    
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
        'go': GO_SAMPLES,
        'rust': RUST_SAMPLES,
        'java': JAVA_SAMPLES,
        'csharp': CSHARP_SAMPLES,
        'cpp': CPP_SAMPLES,
        'c': C_SAMPLES,
        'ruby': RUBY_SAMPLES,
        'php': PHP_SAMPLES,
        'swift': SWIFT_SAMPLES,
        'kotlin': KOTLIN_SAMPLES,
        'scala': SCALA_SAMPLES,
        'shell': SHELL_SAMPLES,
        'haskell': HASKELL_SAMPLES,
        'elixir': ELIXIR_SAMPLES,
        'clojure': CLOJURE_SAMPLES,
        'dart': DART_SAMPLES,
        'lua': LUA_SAMPLES,
        'r': R_SAMPLES,
        'julia': JULIA_SAMPLES,
        'perl': PERL_SAMPLES,
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


def load_training_data_from_source(
    source,
    languages: List[str] = None,
    min_length: int = 50,
    include_builtin: bool = True
) -> List[str]:
    """
    Load training data from external sources (directories, files, or zip archives).
    
    Args:
        source: Path to directory, file, or zip archive (str, Path, or bytes)
        languages: Optional list of languages to filter (e.g., ['python', 'javascript'])
        min_length: Minimum content length for samples
        include_builtin: Whether to include built-in training samples
        
    Returns:
        List of code samples for training
    """
    from .file_processor import UniversalFileProcessor, extract_code_blocks
    
    samples = []
    
    if include_builtin:
        samples.extend(get_all_training_data(shuffle=False))
    
    processor = UniversalFileProcessor(min_content_length=min_length)
    result = processor.process(source)
    
    for f in result.files:
        if languages:
            if not f.language or f.language == 'unknown' or f.language not in languages:
                continue
        
        blocks = extract_code_blocks(f.content, f.language)
        if blocks:
            samples.extend(blocks)
        elif len(f.content.strip()) >= min_length:
            samples.append(f.content.strip())
    
    return samples


def load_from_zip(
    zip_path,
    languages: List[str] = None,
    include_builtin: bool = True
) -> List[str]:
    """
    Load training data from a zip archive.
    
    Args:
        zip_path: Path to the zip file
        languages: Optional list of languages to include
        include_builtin: Whether to include built-in samples
        
    Returns:
        List of code samples for training
    """
    return load_training_data_from_source(
        zip_path,
        languages=languages,
        include_builtin=include_builtin
    )


def load_from_directory(
    directory,
    languages: List[str] = None,
    include_builtin: bool = True
) -> List[str]:
    """
    Load training data from a directory recursively.
    
    Args:
        directory: Path to the directory
        languages: Optional list of languages to include
        include_builtin: Whether to include built-in samples
        
    Returns:
        List of code samples for training
    """
    return load_training_data_from_source(
        directory,
        languages=languages,
        include_builtin=include_builtin
    )


def get_extended_training_stats(source=None) -> Dict:
    """
    Get comprehensive statistics including external sources.
    
    Args:
        source: Optional external source (directory, file, or zip)
        
    Returns:
        Dictionary with detailed statistics
    """
    stats = get_training_stats()
    
    if source:
        from .file_processor import UniversalFileProcessor
        
        processor = UniversalFileProcessor()
        result = processor.process(source)
        
        stats['external_source'] = {
            'total_files': result.total_files,
            'processed_files': result.processed_files,
            'skipped_files': result.skipped_files,
            'errors': len(result.errors),
            'languages': result.stats.get('languages', {}),
            'total_content': result.stats.get('total_content_length', 0),
        }
    
    return stats
