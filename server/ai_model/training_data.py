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
