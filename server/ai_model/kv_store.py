"""
Key-Value Store System for Platform Forge

This module provides a comprehensive key-value storage system matching
Replit's Key-Value Store functionality with persistence, caching, TTL support,
and atomic operations.

Key Components:
- KeyValueStore: Main store class with full CRUD operations
- StoreManager: Manages multiple isolated stores (namespaces)
- TTL Support: Automatic expiration of keys
- Atomic Operations: Thread-safe increment, append operations
- Batch Operations: Efficient bulk get/set/delete
- Pattern Matching: Prefix-based and wildcard key lookups

Storage Limits (Replit-compatible):
- 50 MiB per store
- 5000 keys maximum
- 1000 bytes per key
- 5 MiB per value

Usage:
    from server.ai_model.kv_store import (
        KeyValueStore,
        StoreManager,
        get, set, delete, keys, clear,
    )
    
    # Simple usage with helper functions
    set("user:123", {"name": "Alice", "score": 100})
    user = get("user:123")
    
    # Direct store usage
    store = KeyValueStore("my_namespace")
    store.set("key", "value", ttl=3600)  # Expires in 1 hour
    value = store.get("key")
    
    # Atomic operations
    store.increment("counter", 1)
    store.append("log", "new entry")
    
    # Batch operations
    store.set_many({"a": 1, "b": 2, "c": 3})
    values = store.get_many(["a", "b", "c"])
    
    # Pattern matching
    user_keys = store.keys("user:*")
    store.delete_prefix("temp:")
    
    # Multiple stores
    manager = StoreManager()
    cache_store = manager.get_store("cache")
    session_store = manager.get_store("sessions")
"""

import os
import re
import json
import time
import fnmatch
import hashlib
import threading
import builtins
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import copy


class StorageLimit(Enum):
    """Storage limits matching Replit's Key-Value Store."""
    MAX_STORE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MiB
    MAX_KEYS = 5000
    MAX_KEY_SIZE_BYTES = 1000
    MAX_VALUE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MiB


class StoreError(Exception):
    """Base exception for Key-Value Store errors."""
    pass


class KeyError(StoreError):
    """Error related to key operations."""
    pass


class KeyTooLargeError(StoreError):
    """Key exceeds maximum size limit."""
    def __init__(self, key: str, size: int, max_size: int = StorageLimit.MAX_KEY_SIZE_BYTES.value):
        self.key = key
        self.size = size
        self.max_size = max_size
        super().__init__(f"Key '{key[:50]}...' is {size} bytes, exceeds limit of {max_size} bytes")


class ValueTooLargeError(StoreError):
    """Value exceeds maximum size limit."""
    def __init__(self, key: str, size: int, max_size: int = StorageLimit.MAX_VALUE_SIZE_BYTES.value):
        self.key = key
        self.size = size
        self.max_size = max_size
        super().__init__(f"Value for key '{key}' is {size} bytes, exceeds limit of {max_size} bytes")


class StoreLimitExceededError(StoreError):
    """Store has exceeded its capacity limits."""
    def __init__(self, message: str, current: int, limit: int):
        self.current = current
        self.limit = limit
        super().__init__(f"{message}: {current}/{limit}")


class StoreNotFoundError(StoreError):
    """Store namespace not found."""
    def __init__(self, namespace: str):
        self.namespace = namespace
        super().__init__(f"Store '{namespace}' not found")


class SerializationError(StoreError):
    """Error serializing or deserializing data."""
    pass


class AtomicOperationError(StoreError):
    """Error during atomic operation."""
    pass


@dataclass
class StoredValue:
    """
    Wrapper for stored values with metadata.
    
    Attributes:
        value: The actual stored value
        created_at: Timestamp when the value was created
        updated_at: Timestamp of last update
        expires_at: Optional expiration timestamp
        access_count: Number of times the value was accessed
        last_accessed: Timestamp of last access
        size_bytes: Size of serialized value in bytes
        checksum: MD5 hash for integrity verification
    """
    value: Any
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: Optional[float] = None
    size_bytes: int = 0
    checksum: str = ""
    
    @property
    def is_expired(self) -> bool:
        """Check if the value has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    @property
    def ttl_remaining(self) -> Optional[float]:
        """Get remaining TTL in seconds, None if no expiration."""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0, remaining)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "value": self.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredValue':
        """Deserialize from dictionary."""
        return cls(
            value=data["value"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            expires_at=data.get("expires_at"),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
        )


@dataclass
class StoreStats:
    """Statistics about a Key-Value Store."""
    namespace: str
    key_count: int
    total_size_bytes: int
    size_limit_bytes: int
    key_limit: int
    oldest_key_age: Optional[float] = None
    newest_key_age: Optional[float] = None
    avg_value_size: float = 0.0
    expired_keys_purged: int = 0
    hit_rate: float = 0.0
    
    @property
    def size_utilization(self) -> float:
        """Calculate storage utilization percentage."""
        if self.size_limit_bytes == 0:
            return 0.0
        return (self.total_size_bytes / self.size_limit_bytes) * 100
    
    @property
    def key_utilization(self) -> float:
        """Calculate key count utilization percentage."""
        if self.key_limit == 0:
            return 0.0
        return (self.key_count / self.key_limit) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "key_count": self.key_count,
            "total_size_bytes": self.total_size_bytes,
            "size_limit_bytes": self.size_limit_bytes,
            "key_limit": self.key_limit,
            "oldest_key_age": self.oldest_key_age,
            "newest_key_age": self.newest_key_age,
            "avg_value_size": self.avg_value_size,
            "expired_keys_purged": self.expired_keys_purged,
            "hit_rate": self.hit_rate,
            "size_utilization_percent": self.size_utilization,
            "key_utilization_percent": self.key_utilization,
        }


@dataclass
class BatchResult:
    """Result of a batch operation."""
    successful: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    
    @property
    def all_successful(self) -> bool:
        return len(self.failed) == 0
    
    @property
    def success_count(self) -> int:
        return len(self.successful)
    
    @property
    def failure_count(self) -> int:
        return len(self.failed)


class Serializer(ABC):
    """Abstract base class for value serializers."""
    
    @abstractmethod
    def serialize(self, value: Any) -> str:
        """Serialize a value to string."""
        pass
    
    @abstractmethod
    def deserialize(self, data: str) -> Any:
        """Deserialize a string to value."""
        pass


class JSONSerializer(Serializer):
    """JSON-based serializer for values."""
    
    def serialize(self, value: Any) -> str:
        try:
            return json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise SerializationError(f"Failed to serialize value: {e}")
    
    def deserialize(self, data: str) -> Any:
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Failed to deserialize value: {e}")


class KeyValueStore:
    """
    Main Key-Value Store class with full CRUD operations.
    
    Features:
    - In-memory caching with disk persistence
    - TTL (time-to-live) support
    - Atomic operations (increment, append)
    - Batch operations
    - Prefix and pattern matching
    - Thread-safe operations
    - Replit-compatible storage limits
    
    Args:
        namespace: Unique identifier for this store
        storage_dir: Directory for persistent storage (default: ./kv_data)
        max_size: Maximum store size in bytes (default: 50 MiB)
        max_keys: Maximum number of keys (default: 5000)
        max_key_size: Maximum key size in bytes (default: 1000)
        max_value_size: Maximum value size in bytes (default: 5 MiB)
        auto_persist: Automatically persist changes to disk
        auto_expire: Automatically clean expired keys on access
    """
    
    def __init__(
        self,
        namespace: str = "default",
        storage_dir: Optional[str] = None,
        max_size: int = StorageLimit.MAX_STORE_SIZE_BYTES.value,
        max_keys: int = StorageLimit.MAX_KEYS.value,
        max_key_size: int = StorageLimit.MAX_KEY_SIZE_BYTES.value,
        max_value_size: int = StorageLimit.MAX_VALUE_SIZE_BYTES.value,
        auto_persist: bool = True,
        auto_expire: bool = True,
    ):
        self.namespace = namespace
        self.storage_dir = Path(storage_dir or "./kv_data")
        self.max_size = max_size
        self.max_keys = max_keys
        self.max_key_size = max_key_size
        self.max_value_size = max_value_size
        self.auto_persist = auto_persist
        self.auto_expire = auto_expire
        
        self._cache: Dict[str, StoredValue] = {}
        self._lock = threading.RLock()
        self._serializer = JSONSerializer()
        self._current_size = 0
        self._hits = 0
        self._misses = 0
        self._expired_purged = 0
        
        self._ensure_storage_dir()
        self._load_from_disk()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_storage_path(self) -> Path:
        """Get the file path for this store's data."""
        safe_namespace = re.sub(r'[^a-zA-Z0-9_-]', '_', self.namespace)
        return self.storage_dir / f"{safe_namespace}.json"
    
    def _calculate_checksum(self, value: Any) -> str:
        """Calculate MD5 checksum for value integrity."""
        serialized = self._serializer.serialize(value)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def _get_size(self, value: Any) -> int:
        """Get the size of a serialized value in bytes."""
        return len(self._serializer.serialize(value).encode('utf-8'))
    
    def _validate_key(self, key: str) -> None:
        """Validate key size and format."""
        if not isinstance(key, str):
            raise KeyError(f"Key must be a string, got {type(key).__name__}")
        
        key_bytes = len(key.encode('utf-8'))
        if key_bytes > self.max_key_size:
            raise KeyTooLargeError(key, key_bytes, self.max_key_size)
    
    def _validate_value(self, key: str, value: Any) -> int:
        """Validate value and return its size in bytes."""
        size = self._get_size(value)
        if size > self.max_value_size:
            raise ValueTooLargeError(key, size, self.max_value_size)
        return size
    
    def _check_limits(self, additional_size: int = 0, additional_keys: int = 0) -> None:
        """Check if operation would exceed store limits."""
        new_size = self._current_size + additional_size
        new_keys = len(self._cache) + additional_keys
        
        if new_size > self.max_size:
            raise StoreLimitExceededError(
                "Store size limit exceeded",
                new_size,
                self.max_size
            )
        
        if new_keys > self.max_keys:
            raise StoreLimitExceededError(
                "Key count limit exceeded",
                new_keys,
                self.max_keys
            )
    
    def _purge_expired(self) -> int:
        """Remove expired keys and return count of removed keys."""
        if not self.auto_expire:
            return 0
        
        expired_keys = [
            key for key, stored in self._cache.items()
            if stored.is_expired
        ]
        
        for key in expired_keys:
            self._current_size -= self._cache[key].size_bytes
            del self._cache[key]
        
        self._expired_purged += len(expired_keys)
        return len(expired_keys)
    
    def _load_from_disk(self) -> None:
        """Load store data from disk."""
        path = self._get_storage_path()
        if not path.exists():
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for key, value_data in data.items():
                stored = StoredValue.from_dict(value_data)
                if not stored.is_expired:
                    self._cache[key] = stored
                    self._current_size += stored.size_bytes
        except (json.JSONDecodeError, IOError) as e:
            pass
    
    def _persist_to_disk(self) -> None:
        """Persist store data to disk."""
        if not self.auto_persist:
            return
        
        path = self._get_storage_path()
        data = {
            key: stored.to_dict()
            for key, stored in self._cache.items()
            if not stored.is_expired
        }
        
        try:
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            temp_path.replace(path)
        except IOError:
            pass
    
    def get(
        self,
        key: str,
        default: Any = None,
        update_access: bool = True
    ) -> Any:
        """
        Get a value by key.
        
        Args:
            key: The key to retrieve
            default: Default value if key not found
            update_access: Whether to update access statistics
            
        Returns:
            The stored value or default if not found
        """
        self._validate_key(key)
        
        with self._lock:
            self._purge_expired()
            
            if key not in self._cache:
                self._misses += 1
                return default
            
            stored = self._cache[key]
            
            if stored.is_expired:
                self._current_size -= stored.size_bytes
                del self._cache[key]
                self._expired_purged += 1
                self._misses += 1
                return default
            
            self._hits += 1
            
            if update_access:
                stored.access_count += 1
                stored.last_accessed = time.time()
            
            return copy.deepcopy(stored.value)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        if_not_exists: bool = False,
        if_exists: bool = False
    ) -> bool:
        """
        Set a value by key.
        
        Args:
            key: The key to set
            value: The value to store
            ttl: Time-to-live in seconds (None for no expiration)
            if_not_exists: Only set if key doesn't exist (NX)
            if_exists: Only set if key exists (XX)
            
        Returns:
            True if value was set, False otherwise
        """
        self._validate_key(key)
        size = self._validate_value(key, value)
        
        with self._lock:
            self._purge_expired()
            
            exists = key in self._cache and not self._cache[key].is_expired
            
            if if_not_exists and exists:
                return False
            if if_exists and not exists:
                return False
            
            old_size = self._cache[key].size_bytes if key in self._cache else 0
            size_diff = size - old_size
            additional_keys = 0 if key in self._cache else 1
            
            self._check_limits(size_diff, additional_keys)
            
            expires_at = None
            if ttl is not None and ttl > 0:
                expires_at = time.time() + ttl
            
            stored = StoredValue(
                value=copy.deepcopy(value),
                expires_at=expires_at,
                size_bytes=size,
                checksum=self._calculate_checksum(value),
            )
            
            if key in self._cache:
                stored.created_at = self._cache[key].created_at
                stored.access_count = self._cache[key].access_count
            
            self._cache[key] = stored
            self._current_size += size_diff
            
            self._persist_to_disk()
            return True
    
    def delete(self, key: str) -> bool:
        """
        Delete a key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        self._validate_key(key)
        
        with self._lock:
            if key not in self._cache:
                return False
            
            self._current_size -= self._cache[key].size_bytes
            del self._cache[key]
            
            self._persist_to_disk()
            return True
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists and is not expired.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists and not expired
        """
        self._validate_key(key)
        
        with self._lock:
            if key not in self._cache:
                return False
            
            stored = self._cache[key]
            if stored.is_expired:
                self._current_size -= stored.size_bytes
                del self._cache[key]
                self._expired_purged += 1
                return False
            
            return True
    
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all keys matching an optional pattern.
        
        Args:
            pattern: Wildcard pattern (e.g., "user:*", "*:session")
            
        Returns:
            List of matching keys
        """
        with self._lock:
            self._purge_expired()
            
            all_keys = list(self._cache.keys())
            
            if pattern is None:
                return all_keys
            
            return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]
    
    def keys_prefix(self, prefix: str) -> List[str]:
        """
        Get all keys with a given prefix.
        
        Args:
            prefix: Key prefix to match
            
        Returns:
            List of keys starting with prefix
        """
        with self._lock:
            self._purge_expired()
            return [k for k in self._cache.keys() if k.startswith(prefix)]
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear all keys or keys matching a pattern.
        
        Args:
            pattern: Optional wildcard pattern
            
        Returns:
            Number of keys deleted
        """
        with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                self._current_size = 0
            else:
                matching = self.keys(pattern)
                count = len(matching)
                for key in matching:
                    self._current_size -= self._cache[key].size_bytes
                    del self._cache[key]
            
            self._persist_to_disk()
            return count
    
    def delete_prefix(self, prefix: str) -> int:
        """
        Delete all keys with a given prefix.
        
        Args:
            prefix: Key prefix to match
            
        Returns:
            Number of keys deleted
        """
        with self._lock:
            matching = self.keys_prefix(prefix)
            for key in matching:
                self._current_size -= self._cache[key].size_bytes
                del self._cache[key]
            
            self._persist_to_disk()
            return len(matching)
    
    def increment(
        self,
        key: str,
        delta: Union[int, float] = 1,
        default: Union[int, float] = 0
    ) -> Union[int, float]:
        """
        Atomically increment a numeric value.
        
        Args:
            key: The key to increment
            delta: Amount to add (can be negative)
            default: Default value if key doesn't exist
            
        Returns:
            New value after increment
        """
        self._validate_key(key)
        
        with self._lock:
            current = self.get(key, default=default, update_access=False)
            
            if not isinstance(current, (int, float)):
                raise AtomicOperationError(
                    f"Cannot increment non-numeric value of type {type(current).__name__}"
                )
            
            new_value = current + delta
            self.set(key, new_value)
            return new_value
    
    def decrement(
        self,
        key: str,
        delta: Union[int, float] = 1,
        default: Union[int, float] = 0
    ) -> Union[int, float]:
        """
        Atomically decrement a numeric value.
        
        Args:
            key: The key to decrement
            delta: Amount to subtract
            default: Default value if key doesn't exist
            
        Returns:
            New value after decrement
        """
        return self.increment(key, -delta, default)
    
    def append(self, key: str, value: Any, default: Optional[List] = None) -> List:
        """
        Atomically append to a list.
        
        Args:
            key: The key with list value
            value: Value to append
            default: Default list if key doesn't exist
            
        Returns:
            Updated list
        """
        self._validate_key(key)
        
        with self._lock:
            current = self.get(key, default=default or [], update_access=False)
            
            if not isinstance(current, list):
                raise AtomicOperationError(
                    f"Cannot append to non-list value of type {type(current).__name__}"
                )
            
            new_list = current + [value]
            self.set(key, new_list)
            return new_list
    
    def extend(self, key: str, values: List, default: Optional[List] = None) -> List:
        """
        Atomically extend a list with multiple values.
        
        Args:
            key: The key with list value
            values: Values to extend
            default: Default list if key doesn't exist
            
        Returns:
            Updated list
        """
        self._validate_key(key)
        
        with self._lock:
            current = self.get(key, default=default or [], update_access=False)
            
            if not isinstance(current, list):
                raise AtomicOperationError(
                    f"Cannot extend non-list value of type {type(current).__name__}"
                )
            
            new_list = current + list(values)
            self.set(key, new_list)
            return new_list
    
    def update_dict(self, key: str, updates: Dict, default: Optional[Dict] = None) -> Dict:
        """
        Atomically update a dictionary.
        
        Args:
            key: The key with dict value
            updates: Dictionary of updates to apply
            default: Default dict if key doesn't exist
            
        Returns:
            Updated dictionary
        """
        self._validate_key(key)
        
        with self._lock:
            current = self.get(key, default=default or {}, update_access=False)
            
            if not isinstance(current, dict):
                raise AtomicOperationError(
                    f"Cannot update non-dict value of type {type(current).__name__}"
                )
            
            new_dict = {**current, **updates}
            self.set(key, new_dict)
            return new_dict
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values at once.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            Dictionary of key-value pairs (missing keys are omitted)
        """
        result = {}
        with self._lock:
            for key in keys:
                value = self.get(key, default=None)
                if value is not None:
                    result[key] = value
        return result
    
    def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> BatchResult:
        """
        Set multiple values at once.
        
        Args:
            items: Dictionary of key-value pairs to set
            ttl: Optional TTL for all values
            
        Returns:
            BatchResult with success/failure details
        """
        result = BatchResult()
        
        with self._lock:
            for key, value in items.items():
                try:
                    self.set(key, value, ttl=ttl)
                    result.successful.append(key)
                except StoreError as e:
                    result.failed[key] = str(e)
        
        return result
    
    def delete_many(self, keys: List[str]) -> BatchResult:
        """
        Delete multiple keys at once.
        
        Args:
            keys: List of keys to delete
            
        Returns:
            BatchResult with success/failure details
        """
        result = BatchResult()
        
        with self._lock:
            for key in keys:
                try:
                    if self.delete(key):
                        result.successful.append(key)
                    else:
                        result.failed[key] = "Key not found"
                except StoreError as e:
                    result.failed[key] = str(e)
        
        return result
    
    def ttl(self, key: str) -> Optional[float]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: The key to check
            
        Returns:
            Remaining TTL in seconds, None if no expiration, -1 if not found
        """
        self._validate_key(key)
        
        with self._lock:
            if key not in self._cache:
                return -1
            
            stored = self._cache[key]
            if stored.is_expired:
                return -1
            
            return stored.ttl_remaining
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration on an existing key.
        
        Args:
            key: The key to set expiration on
            ttl: TTL in seconds
            
        Returns:
            True if expiration was set, False if key not found
        """
        self._validate_key(key)
        
        with self._lock:
            if key not in self._cache:
                return False
            
            stored = self._cache[key]
            if stored.is_expired:
                del self._cache[key]
                return False
            
            stored.expires_at = time.time() + ttl
            stored.updated_at = time.time()
            
            self._persist_to_disk()
            return True
    
    def persist(self, key: str) -> bool:
        """
        Remove expiration from a key.
        
        Args:
            key: The key to persist
            
        Returns:
            True if expiration was removed, False if key not found
        """
        self._validate_key(key)
        
        with self._lock:
            if key not in self._cache:
                return False
            
            stored = self._cache[key]
            if stored.is_expired:
                del self._cache[key]
                return False
            
            stored.expires_at = None
            stored.updated_at = time.time()
            
            self._persist_to_disk()
            return True
    
    def rename(self, old_key: str, new_key: str) -> bool:
        """
        Rename a key.
        
        Args:
            old_key: Current key name
            new_key: New key name
            
        Returns:
            True if renamed, False if old key not found
        """
        self._validate_key(old_key)
        self._validate_key(new_key)
        
        with self._lock:
            if old_key not in self._cache:
                return False
            
            stored = self._cache[old_key]
            if stored.is_expired:
                del self._cache[old_key]
                return False
            
            if new_key in self._cache:
                self._current_size -= self._cache[new_key].size_bytes
            
            self._cache[new_key] = stored
            del self._cache[old_key]
            
            self._persist_to_disk()
            return True
    
    def copy(self, source: str, dest: str) -> bool:
        """
        Copy a value to a new key.
        
        Args:
            source: Source key
            dest: Destination key
            
        Returns:
            True if copied, False if source not found
        """
        self._validate_key(source)
        self._validate_key(dest)
        
        with self._lock:
            value = self.get(source)
            if value is None:
                return False
            
            stored = self._cache[source]
            return self.set(
                dest,
                value,
                ttl=int(stored.ttl_remaining) if stored.ttl_remaining else None
            )
    
    def type(self, key: str) -> Optional[str]:
        """
        Get the type of a stored value.
        
        Args:
            key: The key to check
            
        Returns:
            Type name or None if key not found
        """
        value = self.get(key)
        if value is None:
            return None
        
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "list",
            dict: "object",
        }
        
        return type_map.get(type(value), type(value).__name__)
    
    def scan(
        self,
        cursor: int = 0,
        count: int = 10,
        pattern: Optional[str] = None
    ) -> Tuple[int, List[str]]:
        """
        Incrementally iterate over keys.
        
        Args:
            cursor: Cursor position (0 to start)
            count: Number of keys to return
            pattern: Optional pattern filter
            
        Returns:
            Tuple of (next_cursor, keys)
        """
        with self._lock:
            all_keys = self.keys(pattern)
            
            if cursor >= len(all_keys):
                return 0, []
            
            end = min(cursor + count, len(all_keys))
            next_cursor = end if end < len(all_keys) else 0
            
            return next_cursor, all_keys[cursor:end]
    
    def stats(self) -> StoreStats:
        """
        Get statistics about the store.
        
        Returns:
            StoreStats object with store metrics
        """
        with self._lock:
            self._purge_expired()
            
            total_access = self._hits + self._misses
            hit_rate = self._hits / total_access if total_access > 0 else 0.0
            
            ages = [
                time.time() - stored.created_at
                for stored in self._cache.values()
            ]
            
            return StoreStats(
                namespace=self.namespace,
                key_count=len(self._cache),
                total_size_bytes=self._current_size,
                size_limit_bytes=self.max_size,
                key_limit=self.max_keys,
                oldest_key_age=max(ages) if ages else None,
                newest_key_age=min(ages) if ages else None,
                avg_value_size=self._current_size / len(self._cache) if self._cache else 0.0,
                expired_keys_purged=self._expired_purged,
                hit_rate=hit_rate,
            )
    
    def flush(self) -> None:
        """Force persist all data to disk."""
        with self._lock:
            self._persist_to_disk()
    
    def __len__(self) -> int:
        """Get number of keys in store."""
        with self._lock:
            self._purge_expired()
            return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.exists(key)
    
    def __getitem__(self, key: str) -> Any:
        """Get value using bracket notation."""
        value = self.get(key)
        if value is None and not self.exists(key):
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set value using bracket notation."""
        self.set(key, value)
    
    def __delitem__(self, key: str) -> None:
        """Delete value using bracket notation."""
        if not self.delete(key):
            raise KeyError(key)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self.keys())
    
    def items(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over key-value pairs."""
        with self._lock:
            self._purge_expired()
            for key, stored in self._cache.items():
                yield key, copy.deepcopy(stored.value)
    
    def values(self) -> Iterator[Any]:
        """Iterate over values."""
        with self._lock:
            self._purge_expired()
            for stored in self._cache.values():
                yield copy.deepcopy(stored.value)


class StoreManager:
    """
    Manager for multiple isolated Key-Value stores (namespaces).
    
    Provides a centralized way to manage multiple stores with shared
    configuration and cleanup.
    
    Args:
        storage_dir: Base directory for all stores
        default_max_size: Default max size for new stores
        default_max_keys: Default max keys for new stores
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        default_max_size: int = StorageLimit.MAX_STORE_SIZE_BYTES.value,
        default_max_keys: int = StorageLimit.MAX_KEYS.value,
    ):
        self.storage_dir = Path(storage_dir or "./kv_data")
        self.default_max_size = default_max_size
        self.default_max_keys = default_max_keys
        self._stores: Dict[str, KeyValueStore] = {}
        self._lock = threading.Lock()
    
    def get_store(
        self,
        namespace: str = "default",
        create: bool = True,
        **kwargs
    ) -> KeyValueStore:
        """
        Get or create a store by namespace.
        
        Args:
            namespace: Store namespace/name
            create: Create store if it doesn't exist
            **kwargs: Additional arguments for KeyValueStore
            
        Returns:
            KeyValueStore instance
        """
        with self._lock:
            if namespace not in self._stores:
                if not create:
                    raise StoreNotFoundError(namespace)
                
                self._stores[namespace] = KeyValueStore(
                    namespace=namespace,
                    storage_dir=str(self.storage_dir),
                    max_size=kwargs.get('max_size', self.default_max_size),
                    max_keys=kwargs.get('max_keys', self.default_max_keys),
                    **{k: v for k, v in kwargs.items() 
                       if k not in ('max_size', 'max_keys')}
                )
            
            return self._stores[namespace]
    
    def list_stores(self) -> List[str]:
        """
        List all store namespaces.
        
        Returns:
            List of namespace names
        """
        with self._lock:
            on_disk: Set[str] = builtins.set()
            if self.storage_dir.exists():
                for f in self.storage_dir.glob("*.json"):
                    on_disk.add(f.stem)
            
            return list(builtins.set(self._stores.keys()) | on_disk)
    
    def delete_store(self, namespace: str) -> bool:
        """
        Delete a store and its data.
        
        Args:
            namespace: Store namespace to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if namespace in self._stores:
                store = self._stores[namespace]
                store.clear()
                del self._stores[namespace]
            
            safe_namespace = re.sub(r'[^a-zA-Z0-9_-]', '_', namespace)
            path = self.storage_dir / f"{safe_namespace}.json"
            
            if path.exists():
                path.unlink()
                return True
            
            return namespace in self._stores
    
    def store_exists(self, namespace: str) -> bool:
        """
        Check if a store exists.
        
        Args:
            namespace: Store namespace to check
            
        Returns:
            True if store exists
        """
        with self._lock:
            if namespace in self._stores:
                return True
            
            safe_namespace = re.sub(r'[^a-zA-Z0-9_-]', '_', namespace)
            path = self.storage_dir / f"{safe_namespace}.json"
            return path.exists()
    
    def get_all_stats(self) -> Dict[str, StoreStats]:
        """
        Get statistics for all stores.
        
        Returns:
            Dictionary of namespace to StoreStats
        """
        result = {}
        for namespace in self.list_stores():
            try:
                store = self.get_store(namespace, create=False)
                result[namespace] = store.stats()
            except StoreNotFoundError:
                pass
        return result
    
    def cleanup_expired(self) -> Dict[str, int]:
        """
        Cleanup expired keys in all stores.
        
        Returns:
            Dictionary of namespace to purged count
        """
        result = {}
        for namespace in self.list_stores():
            try:
                store = self.get_store(namespace, create=False)
                with store._lock:
                    count = store._purge_expired()
                    if count > 0:
                        store._persist_to_disk()
                    result[namespace] = count
            except StoreNotFoundError:
                pass
        return result
    
    def flush_all(self) -> None:
        """Flush all stores to disk."""
        with self._lock:
            for store in self._stores.values():
                store.flush()


_default_manager: Optional[StoreManager] = None
_default_store: Optional[KeyValueStore] = None


def get_default_manager() -> StoreManager:
    """Get the default StoreManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = StoreManager()
    return _default_manager


def get_default_store() -> KeyValueStore:
    """Get the default KeyValueStore instance."""
    global _default_store
    if _default_store is None:
        _default_store = get_default_manager().get_store("default")
    return _default_store


def set_default_store(store: KeyValueStore) -> None:
    """Set the default KeyValueStore instance."""
    global _default_store
    _default_store = store


def set_default_manager(manager: StoreManager) -> None:
    """Set the default StoreManager instance."""
    global _default_manager
    _default_manager = manager


def get(key: str, default: Any = None) -> Any:
    """
    Get a value from the default store.
    
    Args:
        key: The key to retrieve
        default: Default value if key not found
        
    Returns:
        The stored value or default
    """
    return get_default_store().get(key, default)


def set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """
    Set a value in the default store.
    
    Args:
        key: The key to set
        value: The value to store
        ttl: Optional TTL in seconds
        
    Returns:
        True if set successfully
    """
    return get_default_store().set(key, value, ttl=ttl)


def delete(key: str) -> bool:
    """
    Delete a key from the default store.
    
    Args:
        key: The key to delete
        
    Returns:
        True if deleted, False if not found
    """
    return get_default_store().delete(key)


def exists(key: str) -> bool:
    """
    Check if a key exists in the default store.
    
    Args:
        key: The key to check
        
    Returns:
        True if key exists
    """
    return get_default_store().exists(key)


def keys(pattern: Optional[str] = None) -> List[str]:
    """
    Get keys from the default store.
    
    Args:
        pattern: Optional wildcard pattern
        
    Returns:
        List of matching keys
    """
    return get_default_store().keys(pattern)


def clear(pattern: Optional[str] = None) -> int:
    """
    Clear keys from the default store.
    
    Args:
        pattern: Optional wildcard pattern
        
    Returns:
        Number of keys deleted
    """
    return get_default_store().clear(pattern)


def increment(key: str, delta: Union[int, float] = 1) -> Union[int, float]:
    """
    Atomically increment a value in the default store.
    
    Args:
        key: The key to increment
        delta: Amount to add
        
    Returns:
        New value after increment
    """
    return get_default_store().increment(key, delta)


def append(key: str, value: Any) -> List:
    """
    Atomically append to a list in the default store.
    
    Args:
        key: The key with list value
        value: Value to append
        
    Returns:
        Updated list
    """
    return get_default_store().append(key, value)


def get_many(keys_list: List[str]) -> Dict[str, Any]:
    """
    Get multiple values from the default store.
    
    Args:
        keys_list: List of keys to retrieve
        
    Returns:
        Dictionary of key-value pairs
    """
    return get_default_store().get_many(keys_list)


def set_many(items: Dict[str, Any], ttl: Optional[int] = None) -> BatchResult:
    """
    Set multiple values in the default store.
    
    Args:
        items: Dictionary of key-value pairs
        ttl: Optional TTL for all values
        
    Returns:
        BatchResult with success/failure details
    """
    return get_default_store().set_many(items, ttl=ttl)


def delete_many(keys_list: List[str]) -> BatchResult:
    """
    Delete multiple keys from the default store.
    
    Args:
        keys_list: List of keys to delete
        
    Returns:
        BatchResult with success/failure details
    """
    return get_default_store().delete_many(keys_list)


def stats() -> StoreStats:
    """
    Get statistics for the default store.
    
    Returns:
        StoreStats object
    """
    return get_default_store().stats()


def create_store(
    namespace: str,
    **kwargs
) -> KeyValueStore:
    """
    Create a new store with the given namespace.
    
    Args:
        namespace: Unique namespace for the store
        **kwargs: Additional arguments for KeyValueStore
        
    Returns:
        New KeyValueStore instance
    """
    return get_default_manager().get_store(namespace, **kwargs)


def list_stores() -> List[str]:
    """
    List all store namespaces.
    
    Returns:
        List of namespace names
    """
    return get_default_manager().list_stores()


def delete_store(namespace: str) -> bool:
    """
    Delete a store and its data.
    
    Args:
        namespace: Store namespace to delete
        
    Returns:
        True if deleted
    """
    return get_default_manager().delete_store(namespace)


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable size.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    for unit in ['B', 'KiB', 'MiB', 'GiB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TiB"
