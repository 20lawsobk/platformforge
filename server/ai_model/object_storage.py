"""
Object Storage System for Platform Forge

This module provides a comprehensive S3-compatible object storage system matching
Replit's App Storage functionality with buckets, file operations, metadata support,
and presigned URLs.

Key Components:
- StorageObject: Represents a stored object with full metadata
- Bucket: Container for organizing objects with quotas
- ObjectStorage: Main storage system managing multiple buckets
- StorageClient: High-level client interface matching Replit's SDK

Features:
- S3-compatible bucket system
- File upload from text, bytes, or filename
- File download as text, bytes, or to filename
- Object listing with prefix/delimiter support
- Content type detection
- Size limits and quotas
- Object metadata support
- Presigned URLs for temporary access
- Versioning support
- Multipart upload for large files

Storage Limits (Replit-compatible):
- 1 GiB per bucket (default)
- 100 MiB per object (default)
- 10,000 objects per bucket (default)
- 100 buckets per storage (default)

Usage:
    from server.ai_model.object_storage import (
        ObjectStorage,
        StorageClient,
        Bucket,
        StorageObject,
    )
    
    # Using the high-level client (recommended)
    client = StorageClient()
    
    # Upload operations
    client.upload_from_text("bucket/path/file.txt", "Hello, World!")
    client.upload_from_bytes("bucket/path/image.png", image_bytes)
    client.upload_from_filename("bucket/path/doc.pdf", "/local/path/doc.pdf")
    
    # Download operations
    text = client.download_as_text("bucket/path/file.txt")
    data = client.download_as_bytes("bucket/path/image.png")
    client.download_to_filename("bucket/path/doc.pdf", "/local/save/doc.pdf")
    
    # List and manage
    objects = client.list("bucket/prefix/")
    client.delete("bucket/path/file.txt")
    exists = client.exists("bucket/path/file.txt")
    
    # Generate presigned URL
    url = client.get_presigned_url("bucket/path/file.txt", expires_in=3600)
    
    # Low-level bucket operations
    storage = ObjectStorage()
    bucket = storage.create_bucket("my-bucket")
    bucket.put_object("key", b"data", content_type="application/octet-stream")
    obj = bucket.get_object("key")
"""

import os
import re
import json
import time
import mimetypes
import hashlib
import base64
import hmac
import secrets
import threading
import shutil
from typing import Dict, List, Optional, Any, Union, Iterator, Tuple, BinaryIO, TextIO
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import copy


class StorageLimit(Enum):
    """Storage limits matching Replit's Object Storage."""
    MAX_BUCKET_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GiB per bucket
    MAX_OBJECT_SIZE_BYTES = 100 * 1024 * 1024   # 100 MiB per object
    MAX_OBJECTS_PER_BUCKET = 10000
    MAX_BUCKETS = 100
    MAX_KEY_LENGTH = 1024
    MAX_METADATA_SIZE_BYTES = 2 * 1024  # 2 KiB per object metadata
    MULTIPART_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MiB chunks for multipart
    MIN_MULTIPART_SIZE = 5 * 1024 * 1024  # 5 MiB minimum for multipart


class StorageClass(Enum):
    """Storage classes for objects."""
    STANDARD = "STANDARD"
    INFREQUENT_ACCESS = "INFREQUENT_ACCESS"
    ARCHIVE = "ARCHIVE"
    DEEP_ARCHIVE = "DEEP_ARCHIVE"


class ObjectStorageError(Exception):
    """Base exception for Object Storage errors."""
    pass


class BucketNotFoundError(ObjectStorageError):
    """Bucket does not exist."""
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        super().__init__(f"Bucket '{bucket_name}' not found")


class BucketAlreadyExistsError(ObjectStorageError):
    """Bucket already exists."""
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        super().__init__(f"Bucket '{bucket_name}' already exists")


class BucketNotEmptyError(ObjectStorageError):
    """Bucket is not empty and cannot be deleted."""
    def __init__(self, bucket_name: str, object_count: int):
        self.bucket_name = bucket_name
        self.object_count = object_count
        super().__init__(f"Bucket '{bucket_name}' is not empty ({object_count} objects)")


class ObjectNotFoundError(ObjectStorageError):
    """Object does not exist."""
    def __init__(self, bucket_name: str, key: str):
        self.bucket_name = bucket_name
        self.key = key
        super().__init__(f"Object '{key}' not found in bucket '{bucket_name}'")


class ObjectTooLargeError(ObjectStorageError):
    """Object exceeds maximum size limit."""
    def __init__(self, key: str, size: int, max_size: int):
        self.key = key
        self.size = size
        self.max_size = max_size
        super().__init__(f"Object '{key}' is {size} bytes, exceeds limit of {max_size} bytes")


class KeyTooLongError(ObjectStorageError):
    """Object key exceeds maximum length."""
    def __init__(self, key: str, length: int, max_length: int):
        self.key = key
        self.length = length
        self.max_length = max_length
        super().__init__(f"Key length {length} exceeds maximum of {max_length}")


class QuotaExceededError(ObjectStorageError):
    """Storage quota exceeded."""
    def __init__(self, message: str, current: int, limit: int):
        self.current = current
        self.limit = limit
        super().__init__(f"{message}: {current}/{limit}")


class InvalidBucketNameError(ObjectStorageError):
    """Invalid bucket name."""
    def __init__(self, bucket_name: str, reason: str):
        self.bucket_name = bucket_name
        self.reason = reason
        super().__init__(f"Invalid bucket name '{bucket_name}': {reason}")


class PresignedUrlExpiredError(ObjectStorageError):
    """Presigned URL has expired."""
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Presigned URL for '{key}' has expired")


class InvalidPresignedUrlError(ObjectStorageError):
    """Invalid presigned URL signature."""
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Invalid signature for presigned URL accessing '{key}'")


class MultipartUploadError(ObjectStorageError):
    """Error during multipart upload."""
    pass


@dataclass
class ObjectMetadata:
    """
    Metadata for a stored object.
    
    Attributes:
        content_type: MIME type of the object
        content_length: Size of the object in bytes
        content_encoding: Content encoding (e.g., gzip)
        content_language: Content language
        content_disposition: Content disposition header
        cache_control: Cache control directives
        etag: Entity tag (MD5 hash of content)
        last_modified: Last modification timestamp
        version_id: Version identifier (if versioning enabled)
        storage_class: Storage class for the object
        user_metadata: Custom user-defined metadata
        checksum_sha256: SHA-256 checksum of content
    """
    content_type: str = "application/octet-stream"
    content_length: int = 0
    content_encoding: Optional[str] = None
    content_language: Optional[str] = None
    content_disposition: Optional[str] = None
    cache_control: Optional[str] = None
    etag: str = ""
    last_modified: float = field(default_factory=time.time)
    version_id: Optional[str] = None
    storage_class: StorageClass = StorageClass.STANDARD
    user_metadata: Dict[str, str] = field(default_factory=dict)
    checksum_sha256: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "content_type": self.content_type,
            "content_length": self.content_length,
            "content_encoding": self.content_encoding,
            "content_language": self.content_language,
            "content_disposition": self.content_disposition,
            "cache_control": self.cache_control,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "version_id": self.version_id,
            "storage_class": self.storage_class.value,
            "user_metadata": self.user_metadata,
            "checksum_sha256": self.checksum_sha256,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectMetadata':
        """Deserialize metadata from dictionary."""
        storage_class = data.get("storage_class", "STANDARD")
        if isinstance(storage_class, str):
            storage_class = StorageClass(storage_class)
        
        return cls(
            content_type=data.get("content_type", "application/octet-stream"),
            content_length=data.get("content_length", 0),
            content_encoding=data.get("content_encoding"),
            content_language=data.get("content_language"),
            content_disposition=data.get("content_disposition"),
            cache_control=data.get("cache_control"),
            etag=data.get("etag", ""),
            last_modified=data.get("last_modified", time.time()),
            version_id=data.get("version_id"),
            storage_class=storage_class,
            user_metadata=data.get("user_metadata", {}),
            checksum_sha256=data.get("checksum_sha256"),
        )


@dataclass
class StorageObject:
    """
    Represents a stored object with its data and metadata.
    
    Attributes:
        key: Object key (path within bucket)
        data: Object content as bytes
        metadata: Object metadata
        bucket_name: Name of the containing bucket
        created_at: Creation timestamp
        access_count: Number of times accessed
        last_accessed: Last access timestamp
    """
    key: str
    data: bytes
    metadata: ObjectMetadata
    bucket_name: str = ""
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: Optional[float] = None
    
    @property
    def size(self) -> int:
        """Get the size of the object in bytes."""
        return len(self.data)
    
    @property
    def content_type(self) -> str:
        """Get the content type of the object."""
        return self.metadata.content_type
    
    @property
    def etag(self) -> str:
        """Get the ETag of the object."""
        return self.metadata.etag
    
    def get_text(self, encoding: str = "utf-8") -> str:
        """Get object content as text."""
        return self.data.decode(encoding)
    
    def get_bytes(self) -> bytes:
        """Get object content as bytes."""
        return self.data
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize object to dictionary (excluding data)."""
        return {
            "key": self.key,
            "size": self.size,
            "metadata": self.metadata.to_dict(),
            "bucket_name": self.bucket_name,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }


@dataclass
class ListObjectsResult:
    """Result of listing objects in a bucket."""
    objects: List[StorageObject]
    prefixes: List[str] = field(default_factory=list)
    is_truncated: bool = False
    next_continuation_token: Optional[str] = None
    key_count: int = 0
    max_keys: int = 1000
    prefix: Optional[str] = None
    delimiter: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "prefixes": self.prefixes,
            "is_truncated": self.is_truncated,
            "next_continuation_token": self.next_continuation_token,
            "key_count": self.key_count,
            "max_keys": self.max_keys,
            "prefix": self.prefix,
            "delimiter": self.delimiter,
        }


@dataclass
class BucketInfo:
    """Information about a bucket."""
    name: str
    created_at: float
    object_count: int
    total_size_bytes: int
    versioning_enabled: bool = False
    lifecycle_rules: List[Dict[str, Any]] = field(default_factory=list)
    cors_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "object_count": self.object_count,
            "total_size_bytes": self.total_size_bytes,
            "versioning_enabled": self.versioning_enabled,
            "lifecycle_rules": self.lifecycle_rules,
            "cors_rules": self.cors_rules,
        }


@dataclass
class MultipartUpload:
    """Represents an in-progress multipart upload."""
    upload_id: str
    bucket_name: str
    key: str
    initiated_at: float
    parts: Dict[int, bytes] = field(default_factory=dict)
    part_etags: Dict[int, str] = field(default_factory=dict)
    metadata: Optional[ObjectMetadata] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "bucket_name": self.bucket_name,
            "key": self.key,
            "initiated_at": self.initiated_at,
            "part_count": len(self.parts),
            "total_size": sum(len(p) for p in self.parts.values()),
        }


@dataclass
class PresignedUrl:
    """Represents a presigned URL for object access."""
    url: str
    bucket_name: str
    key: str
    method: str
    expires_at: float
    signature: str
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "bucket_name": self.bucket_name,
            "key": self.key,
            "method": self.method,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired,
        }


class ContentTypeDetector:
    """Detects content types from filenames and data."""
    
    BINARY_SIGNATURES = {
        b'\x89PNG\r\n\x1a\n': 'image/png',
        b'\xff\xd8\xff': 'image/jpeg',
        b'GIF87a': 'image/gif',
        b'GIF89a': 'image/gif',
        b'%PDF': 'application/pdf',
        b'PK\x03\x04': 'application/zip',
        b'\x1f\x8b': 'application/gzip',
        b'RIFF': 'audio/wav',
        b'ID3': 'audio/mpeg',
        b'\xff\xfb': 'audio/mpeg',
        b'\x00\x00\x00\x1c\x66\x74\x79\x70': 'video/mp4',
        b'\x00\x00\x00\x20\x66\x74\x79\x70': 'video/mp4',
    }
    
    TEXT_EXTENSIONS = {
        '.txt': 'text/plain',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.csv': 'text/csv',
        '.md': 'text/markdown',
        '.yaml': 'application/x-yaml',
        '.yml': 'application/x-yaml',
        '.toml': 'application/toml',
        '.py': 'text/x-python',
        '.ts': 'application/typescript',
        '.tsx': 'application/typescript',
        '.jsx': 'text/jsx',
        '.rs': 'text/x-rust',
        '.go': 'text/x-go',
        '.java': 'text/x-java-source',
        '.c': 'text/x-c',
        '.cpp': 'text/x-c++src',
        '.h': 'text/x-c',
        '.hpp': 'text/x-c++hdr',
        '.sql': 'application/sql',
        '.sh': 'application/x-sh',
        '.bash': 'application/x-sh',
    }
    
    @classmethod
    def detect_from_filename(cls, filename: str) -> str:
        """Detect content type from filename extension."""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in cls.TEXT_EXTENSIONS:
            return cls.TEXT_EXTENSIONS[ext]
        
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"
    
    @classmethod
    def detect_from_data(cls, data: bytes, filename: Optional[str] = None) -> str:
        """Detect content type from data content and optional filename."""
        for signature, mime_type in cls.BINARY_SIGNATURES.items():
            if data.startswith(signature):
                return mime_type
        
        if filename:
            return cls.detect_from_filename(filename)
        
        try:
            data[:1024].decode('utf-8')
            return 'text/plain'
        except UnicodeDecodeError:
            pass
        
        return "application/octet-stream"


class Bucket:
    """
    Container for objects with quota management.
    
    A bucket provides:
    - Object storage with key-value access
    - Size and object count quotas
    - Optional versioning
    - Listing with prefix/delimiter support
    - Batch operations
    
    Args:
        name: Unique bucket name
        storage_dir: Directory for persistent storage
        max_size: Maximum bucket size in bytes
        max_objects: Maximum number of objects
        max_object_size: Maximum size per object
        versioning_enabled: Enable object versioning
        auto_persist: Automatically persist changes to disk
    """
    
    BUCKET_NAME_PATTERN = re.compile(r'^[a-z0-9][a-z0-9\-]{1,61}[a-z0-9]$')
    
    def __init__(
        self,
        name: str,
        storage_dir: Optional[str] = None,
        max_size: int = StorageLimit.MAX_BUCKET_SIZE_BYTES.value,
        max_objects: int = StorageLimit.MAX_OBJECTS_PER_BUCKET.value,
        max_object_size: int = StorageLimit.MAX_OBJECT_SIZE_BYTES.value,
        versioning_enabled: bool = False,
        auto_persist: bool = True,
    ):
        self._validate_bucket_name(name)
        
        self.name = name
        self.storage_dir = Path(storage_dir or "./object_storage")
        self.max_size = max_size
        self.max_objects = max_objects
        self.max_object_size = max_object_size
        self.versioning_enabled = versioning_enabled
        self.auto_persist = auto_persist
        
        self._objects: Dict[str, StorageObject] = {}
        self._versions: Dict[str, List[StorageObject]] = {}
        self._lock = threading.RLock()
        self._current_size = 0
        self._created_at = time.time()
        self._multipart_uploads: Dict[str, MultipartUpload] = {}
        
        self._ensure_storage_dir()
        self._load_from_disk()
    
    @classmethod
    def _validate_bucket_name(cls, name: str) -> None:
        """Validate bucket name follows S3 naming rules."""
        if len(name) < 3:
            raise InvalidBucketNameError(name, "Name must be at least 3 characters")
        if len(name) > 63:
            raise InvalidBucketNameError(name, "Name must be at most 63 characters")
        if not cls.BUCKET_NAME_PATTERN.match(name):
            raise InvalidBucketNameError(
                name, 
                "Name must contain only lowercase letters, numbers, and hyphens, "
                "and must start and end with a letter or number"
            )
        if '..' in name:
            raise InvalidBucketNameError(name, "Name cannot contain consecutive periods")
    
    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        bucket_dir = self.storage_dir / self.name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        (bucket_dir / "objects").mkdir(exist_ok=True)
        (bucket_dir / "metadata").mkdir(exist_ok=True)
    
    def _get_object_path(self, key: str) -> Path:
        """Get the file path for an object."""
        safe_key = base64.urlsafe_b64encode(key.encode()).decode()
        return self.storage_dir / self.name / "objects" / safe_key
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get the metadata file path for an object."""
        safe_key = base64.urlsafe_b64encode(key.encode()).decode()
        return self.storage_dir / self.name / "metadata" / f"{safe_key}.json"
    
    def _calculate_etag(self, data: bytes) -> str:
        """Calculate ETag (MD5 hash) for object data."""
        return hashlib.md5(data).hexdigest()
    
    def _calculate_sha256(self, data: bytes) -> str:
        """Calculate SHA-256 checksum for object data."""
        return hashlib.sha256(data).hexdigest()
    
    def _validate_key(self, key: str) -> None:
        """Validate object key."""
        if not key:
            raise KeyTooLongError(key, 0, StorageLimit.MAX_KEY_LENGTH.value)
        
        if len(key) > StorageLimit.MAX_KEY_LENGTH.value:
            raise KeyTooLongError(key, len(key), StorageLimit.MAX_KEY_LENGTH.value)
    
    def _check_quotas(self, additional_size: int = 0, additional_objects: int = 0) -> None:
        """Check if operation would exceed quotas."""
        new_size = self._current_size + additional_size
        new_count = len(self._objects) + additional_objects
        
        if new_size > self.max_size:
            raise QuotaExceededError(
                "Bucket size limit exceeded",
                new_size,
                self.max_size
            )
        
        if new_count > self.max_objects:
            raise QuotaExceededError(
                "Object count limit exceeded",
                new_count,
                self.max_objects
            )
    
    def _load_from_disk(self) -> None:
        """Load bucket data from disk."""
        metadata_dir = self.storage_dir / self.name / "metadata"
        if not metadata_dir.exists():
            return
        
        for meta_file in metadata_dir.glob("*.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                key = meta_data.get("key", "")
                obj_path = self._get_object_path(key)
                
                if obj_path.exists():
                    with open(obj_path, 'rb') as f:
                        data = f.read()
                    
                    metadata = ObjectMetadata.from_dict(meta_data.get("metadata", {}))
                    
                    obj = StorageObject(
                        key=key,
                        data=data,
                        metadata=metadata,
                        bucket_name=self.name,
                        created_at=meta_data.get("created_at", time.time()),
                        access_count=meta_data.get("access_count", 0),
                        last_accessed=meta_data.get("last_accessed"),
                    )
                    
                    self._objects[key] = obj
                    self._current_size += len(data)
                    
            except (json.JSONDecodeError, IOError):
                pass
    
    def _persist_object(self, obj: StorageObject) -> None:
        """Persist a single object to disk."""
        if not self.auto_persist:
            return
        
        obj_path = self._get_object_path(obj.key)
        meta_path = self._get_metadata_path(obj.key)
        
        try:
            with open(obj_path, 'wb') as f:
                f.write(obj.data)
            
            meta_data = {
                "key": obj.key,
                "metadata": obj.metadata.to_dict(),
                "created_at": obj.created_at,
                "access_count": obj.access_count,
                "last_accessed": obj.last_accessed,
            }
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
                
        except IOError:
            pass
    
    def _delete_object_files(self, key: str) -> None:
        """Delete object files from disk."""
        obj_path = self._get_object_path(key)
        meta_path = self._get_metadata_path(key)
        
        try:
            if obj_path.exists():
                obj_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
        except IOError:
            pass
    
    def put_object(
        self,
        key: str,
        data: Union[bytes, str],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        storage_class: StorageClass = StorageClass.STANDARD,
        cache_control: Optional[str] = None,
        content_encoding: Optional[str] = None,
        content_disposition: Optional[str] = None,
    ) -> StorageObject:
        """
        Store an object in the bucket.
        
        Args:
            key: Object key (path)
            data: Object content (bytes or string)
            content_type: MIME type (auto-detected if not provided)
            metadata: Custom user metadata
            storage_class: Storage class for the object
            cache_control: Cache control header
            content_encoding: Content encoding
            content_disposition: Content disposition
            
        Returns:
            The stored StorageObject
        """
        self._validate_key(key)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if len(data) > self.max_object_size:
            raise ObjectTooLargeError(key, len(data), self.max_object_size)
        
        if content_type is None:
            content_type = ContentTypeDetector.detect_from_data(data, key)
        
        with self._lock:
            old_size = 0
            if key in self._objects:
                old_size = len(self._objects[key].data)
                
                if self.versioning_enabled:
                    if key not in self._versions:
                        self._versions[key] = []
                    self._versions[key].append(self._objects[key])
            
            size_diff = len(data) - old_size
            additional_objects = 0 if key in self._objects else 1
            
            self._check_quotas(size_diff, additional_objects)
            
            etag = self._calculate_etag(data)
            sha256 = self._calculate_sha256(data)
            version_id = None
            if self.versioning_enabled:
                version_id = f"{int(time.time() * 1000)}-{secrets.token_hex(4)}"
            
            obj_metadata = ObjectMetadata(
                content_type=content_type,
                content_length=len(data),
                content_encoding=content_encoding,
                cache_control=cache_control,
                content_disposition=content_disposition,
                etag=etag,
                last_modified=time.time(),
                version_id=version_id,
                storage_class=storage_class,
                user_metadata=metadata or {},
                checksum_sha256=sha256,
            )
            
            obj = StorageObject(
                key=key,
                data=data,
                metadata=obj_metadata,
                bucket_name=self.name,
            )
            
            self._objects[key] = obj
            self._current_size += size_diff
            
            self._persist_object(obj)
            
            return obj
    
    def get_object(
        self,
        key: str,
        version_id: Optional[str] = None,
        update_access: bool = True,
    ) -> StorageObject:
        """
        Retrieve an object from the bucket.
        
        Args:
            key: Object key
            version_id: Specific version to retrieve
            update_access: Whether to update access statistics
            
        Returns:
            The StorageObject
            
        Raises:
            ObjectNotFoundError: If object doesn't exist
        """
        self._validate_key(key)
        
        with self._lock:
            if version_id and self.versioning_enabled:
                if key in self._versions:
                    for version in self._versions[key]:
                        if version.metadata.version_id == version_id:
                            return copy.deepcopy(version)
            
            if key not in self._objects:
                raise ObjectNotFoundError(self.name, key)
            
            obj = self._objects[key]
            
            if update_access:
                obj.access_count += 1
                obj.last_accessed = time.time()
            
            return copy.deepcopy(obj)
    
    def head_object(self, key: str) -> ObjectMetadata:
        """
        Get object metadata without retrieving content.
        
        Args:
            key: Object key
            
        Returns:
            Object metadata
        """
        self._validate_key(key)
        
        with self._lock:
            if key not in self._objects:
                raise ObjectNotFoundError(self.name, key)
            
            return copy.deepcopy(self._objects[key].metadata)
    
    def delete_object(self, key: str, version_id: Optional[str] = None) -> bool:
        """
        Delete an object from the bucket.
        
        Args:
            key: Object key
            version_id: Specific version to delete
            
        Returns:
            True if object was deleted
        """
        self._validate_key(key)
        
        with self._lock:
            if version_id and self.versioning_enabled:
                if key in self._versions:
                    self._versions[key] = [
                        v for v in self._versions[key]
                        if v.metadata.version_id != version_id
                    ]
                    return True
                return False
            
            if key not in self._objects:
                return False
            
            obj = self._objects[key]
            self._current_size -= len(obj.data)
            del self._objects[key]
            
            self._delete_object_files(key)
            
            return True
    
    def delete_objects(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple objects.
        
        Args:
            keys: List of object keys to delete
            
        Returns:
            Dictionary mapping keys to deletion success
        """
        results = {}
        with self._lock:
            for key in keys:
                try:
                    results[key] = self.delete_object(key)
                except Exception:
                    results[key] = False
        return results
    
    def object_exists(self, key: str) -> bool:
        """Check if an object exists."""
        self._validate_key(key)
        
        with self._lock:
            return key in self._objects
    
    def list_objects(
        self,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> ListObjectsResult:
        """
        List objects in the bucket.
        
        Args:
            prefix: Filter objects by prefix
            delimiter: Group objects by delimiter (e.g., "/" for folders)
            max_keys: Maximum number of keys to return
            continuation_token: Token for pagination
            
        Returns:
            ListObjectsResult with matching objects
        """
        with self._lock:
            all_keys = sorted(self._objects.keys())
            
            if prefix:
                all_keys = [k for k in all_keys if k.startswith(prefix)]
            
            start_idx = 0
            if continuation_token:
                try:
                    start_idx = all_keys.index(continuation_token) + 1
                except ValueError:
                    pass
            
            objects = []
            prefixes = set()
            
            for key in all_keys[start_idx:]:
                if len(objects) >= max_keys:
                    break
                
                if delimiter:
                    relative_key = key[len(prefix):] if prefix else key
                    if delimiter in relative_key:
                        prefix_end = relative_key.index(delimiter) + len(delimiter)
                        common_prefix = (prefix or "") + relative_key[:prefix_end]
                        prefixes.add(common_prefix)
                        continue
                
                objects.append(copy.deepcopy(self._objects[key]))
            
            is_truncated = len(all_keys) > start_idx + max_keys
            next_token = None
            if is_truncated and objects:
                next_token = objects[-1].key
            
            return ListObjectsResult(
                objects=objects,
                prefixes=sorted(prefixes),
                is_truncated=is_truncated,
                next_continuation_token=next_token,
                key_count=len(objects),
                max_keys=max_keys,
                prefix=prefix,
                delimiter=delimiter,
            )
    
    def list_object_versions(self, prefix: Optional[str] = None) -> List[StorageObject]:
        """List all versions of objects."""
        if not self.versioning_enabled:
            return []
        
        versions = []
        with self._lock:
            for key, version_list in self._versions.items():
                if prefix and not key.startswith(prefix):
                    continue
                versions.extend(version_list)
        
        return sorted(versions, key=lambda x: x.metadata.last_modified, reverse=True)
    
    def copy_object(
        self,
        source_key: str,
        dest_key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """
        Copy an object within the bucket.
        
        Args:
            source_key: Source object key
            dest_key: Destination object key
            metadata: New metadata (uses source if not provided)
            
        Returns:
            The copied StorageObject
        """
        source_obj = self.get_object(source_key, update_access=False)
        
        new_metadata = metadata if metadata is not None else source_obj.metadata.user_metadata
        
        return self.put_object(
            key=dest_key,
            data=source_obj.data,
            content_type=source_obj.metadata.content_type,
            metadata=new_metadata,
            storage_class=source_obj.metadata.storage_class,
            cache_control=source_obj.metadata.cache_control,
            content_encoding=source_obj.metadata.content_encoding,
            content_disposition=source_obj.metadata.content_disposition,
        )
    
    def initiate_multipart_upload(
        self,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Initiate a multipart upload.
        
        Args:
            key: Object key
            content_type: Content type
            metadata: User metadata
            
        Returns:
            Upload ID
        """
        self._validate_key(key)
        
        upload_id = secrets.token_hex(16)
        
        obj_metadata = None
        if content_type or metadata:
            obj_metadata = ObjectMetadata(
                content_type=content_type or "application/octet-stream",
                user_metadata=metadata or {},
            )
        
        upload = MultipartUpload(
            upload_id=upload_id,
            bucket_name=self.name,
            key=key,
            initiated_at=time.time(),
            metadata=obj_metadata,
        )
        
        with self._lock:
            self._multipart_uploads[upload_id] = upload
        
        return upload_id
    
    def upload_part(
        self,
        upload_id: str,
        part_number: int,
        data: bytes,
    ) -> str:
        """
        Upload a part for multipart upload.
        
        Args:
            upload_id: Upload ID from initiate_multipart_upload
            part_number: Part number (1-10000)
            data: Part data
            
        Returns:
            ETag for the part
        """
        if part_number < 1 or part_number > 10000:
            raise MultipartUploadError(f"Part number must be between 1 and 10000, got {part_number}")
        
        with self._lock:
            if upload_id not in self._multipart_uploads:
                raise MultipartUploadError(f"Upload ID '{upload_id}' not found")
            
            upload = self._multipart_uploads[upload_id]
            etag = self._calculate_etag(data)
            
            upload.parts[part_number] = data
            upload.part_etags[part_number] = etag
            
            return etag
    
    def complete_multipart_upload(
        self,
        upload_id: str,
        parts: Optional[List[Dict[str, Any]]] = None,
    ) -> StorageObject:
        """
        Complete a multipart upload.
        
        Args:
            upload_id: Upload ID
            parts: List of {"part_number": int, "etag": str} (optional verification)
            
        Returns:
            The completed StorageObject
        """
        with self._lock:
            if upload_id not in self._multipart_uploads:
                raise MultipartUploadError(f"Upload ID '{upload_id}' not found")
            
            upload = self._multipart_uploads[upload_id]
            
            if parts:
                for part in parts:
                    pn = part["part_number"]
                    etag = part["etag"]
                    if pn not in upload.part_etags or upload.part_etags[pn] != etag:
                        raise MultipartUploadError(f"ETag mismatch for part {pn}")
            
            sorted_parts = sorted(upload.parts.items())
            combined_data = b''.join(data for _, data in sorted_parts)
            
            content_type = None
            metadata = None
            if upload.metadata:
                content_type = upload.metadata.content_type
                metadata = upload.metadata.user_metadata
            
            obj = self.put_object(
                key=upload.key,
                data=combined_data,
                content_type=content_type,
                metadata=metadata,
            )
            
            del self._multipart_uploads[upload_id]
            
            return obj
    
    def abort_multipart_upload(self, upload_id: str) -> bool:
        """Abort a multipart upload."""
        with self._lock:
            if upload_id in self._multipart_uploads:
                del self._multipart_uploads[upload_id]
                return True
            return False
    
    def get_info(self) -> BucketInfo:
        """Get bucket information."""
        with self._lock:
            return BucketInfo(
                name=self.name,
                created_at=self._created_at,
                object_count=len(self._objects),
                total_size_bytes=self._current_size,
                versioning_enabled=self.versioning_enabled,
            )
    
    def clear(self) -> int:
        """Delete all objects in the bucket."""
        with self._lock:
            count = len(self._objects)
            
            for key in list(self._objects.keys()):
                self._delete_object_files(key)
            
            self._objects.clear()
            self._versions.clear()
            self._current_size = 0
            
            return count


class ObjectStorage:
    """
    Main storage system managing multiple buckets.
    
    This provides:
    - Bucket creation and management
    - Cross-bucket operations
    - Presigned URL generation
    - Global quotas and limits
    
    Args:
        storage_dir: Base directory for storage
        max_buckets: Maximum number of buckets
        secret_key: Secret key for presigned URL signing
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_buckets: int = StorageLimit.MAX_BUCKETS.value,
        secret_key: Optional[str] = None,
    ):
        self.storage_dir = Path(storage_dir or "./object_storage")
        self.max_buckets = max_buckets
        self._secret_key = secret_key or secrets.token_hex(32)
        
        self._buckets: Dict[str, Bucket] = {}
        self._lock = threading.RLock()
        
        self._ensure_storage_dir()
        self._load_buckets()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_buckets(self) -> None:
        """Load existing buckets from disk."""
        if not self.storage_dir.exists():
            return
        
        for bucket_dir in self.storage_dir.iterdir():
            if bucket_dir.is_dir():
                try:
                    bucket = Bucket(
                        name=bucket_dir.name,
                        storage_dir=str(self.storage_dir),
                    )
                    self._buckets[bucket_dir.name] = bucket
                except InvalidBucketNameError:
                    pass
    
    def create_bucket(
        self,
        name: str,
        max_size: int = StorageLimit.MAX_BUCKET_SIZE_BYTES.value,
        max_objects: int = StorageLimit.MAX_OBJECTS_PER_BUCKET.value,
        versioning_enabled: bool = False,
    ) -> Bucket:
        """
        Create a new bucket.
        
        Args:
            name: Bucket name
            max_size: Maximum bucket size
            max_objects: Maximum object count
            versioning_enabled: Enable versioning
            
        Returns:
            The created Bucket
        """
        with self._lock:
            if name in self._buckets:
                raise BucketAlreadyExistsError(name)
            
            if len(self._buckets) >= self.max_buckets:
                raise QuotaExceededError(
                    "Bucket limit exceeded",
                    len(self._buckets),
                    self.max_buckets
                )
            
            bucket = Bucket(
                name=name,
                storage_dir=str(self.storage_dir),
                max_size=max_size,
                max_objects=max_objects,
                versioning_enabled=versioning_enabled,
            )
            
            self._buckets[name] = bucket
            return bucket
    
    def get_bucket(self, name: str) -> Bucket:
        """Get a bucket by name."""
        with self._lock:
            if name not in self._buckets:
                raise BucketNotFoundError(name)
            return self._buckets[name]
    
    def delete_bucket(self, name: str, force: bool = False) -> bool:
        """
        Delete a bucket.
        
        Args:
            name: Bucket name
            force: Delete even if not empty
            
        Returns:
            True if bucket was deleted
        """
        with self._lock:
            if name not in self._buckets:
                raise BucketNotFoundError(name)
            
            bucket = self._buckets[name]
            info = bucket.get_info()
            
            if info.object_count > 0 and not force:
                raise BucketNotEmptyError(name, info.object_count)
            
            if force:
                bucket.clear()
            
            bucket_dir = self.storage_dir / name
            if bucket_dir.exists():
                shutil.rmtree(bucket_dir)
            
            del self._buckets[name]
            return True
    
    def bucket_exists(self, name: str) -> bool:
        """Check if a bucket exists."""
        with self._lock:
            return name in self._buckets
    
    def list_buckets(self) -> List[BucketInfo]:
        """List all buckets."""
        with self._lock:
            return [bucket.get_info() for bucket in self._buckets.values()]
    
    def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """
        Copy an object across buckets.
        
        Args:
            source_bucket: Source bucket name
            source_key: Source object key
            dest_bucket: Destination bucket name
            dest_key: Destination object key
            metadata: New metadata
            
        Returns:
            The copied StorageObject
        """
        src_bucket = self.get_bucket(source_bucket)
        dst_bucket = self.get_bucket(dest_bucket)
        
        source_obj = src_bucket.get_object(source_key, update_access=False)
        
        new_metadata = metadata if metadata is not None else source_obj.metadata.user_metadata
        
        return dst_bucket.put_object(
            key=dest_key,
            data=source_obj.data,
            content_type=source_obj.metadata.content_type,
            metadata=new_metadata,
            storage_class=source_obj.metadata.storage_class,
        )
    
    def generate_presigned_url(
        self,
        bucket_name: str,
        key: str,
        method: str = "GET",
        expires_in: int = 3600,
    ) -> PresignedUrl:
        """
        Generate a presigned URL for temporary access.
        
        Args:
            bucket_name: Bucket name
            key: Object key
            method: HTTP method (GET, PUT)
            expires_in: Expiration time in seconds
            
        Returns:
            PresignedUrl object
        """
        if bucket_name not in self._buckets:
            raise BucketNotFoundError(bucket_name)
        
        expires_at = time.time() + expires_in
        
        string_to_sign = f"{method}\n{bucket_name}\n{key}\n{expires_at}"
        signature = hmac.new(
            self._secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        url = f"/storage/{bucket_name}/{key}?signature={signature}&expires={int(expires_at)}&method={method}"
        
        return PresignedUrl(
            url=url,
            bucket_name=bucket_name,
            key=key,
            method=method,
            expires_at=expires_at,
            signature=signature,
        )
    
    def verify_presigned_url(
        self,
        bucket_name: str,
        key: str,
        method: str,
        signature: str,
        expires: float,
    ) -> bool:
        """Verify a presigned URL signature."""
        if time.time() > expires:
            raise PresignedUrlExpiredError(key)
        
        string_to_sign = f"{method}\n{bucket_name}\n{key}\n{expires}"
        expected_signature = hmac.new(
            self._secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise InvalidPresignedUrlError(key)
        
        return True
    
    def get_total_size(self) -> int:
        """Get total size across all buckets."""
        with self._lock:
            return sum(b.get_info().total_size_bytes for b in self._buckets.values())
    
    def get_total_objects(self) -> int:
        """Get total object count across all buckets."""
        with self._lock:
            return sum(b.get_info().object_count for b in self._buckets.values())


class StorageClient:
    """
    High-level client interface matching Replit's Object Storage SDK.
    
    Provides a simplified API for common storage operations using
    path-based access (bucket/key format).
    
    Args:
        storage_dir: Base directory for storage
        default_bucket: Default bucket name for operations
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        default_bucket: str = "default",
    ):
        self._storage = ObjectStorage(storage_dir=storage_dir)
        self._default_bucket = default_bucket
        
        if not self._storage.bucket_exists(default_bucket):
            self._storage.create_bucket(default_bucket)
    
    def _parse_path(self, path: str) -> Tuple[str, str]:
        """Parse a path into bucket and key."""
        parts = path.split("/", 1)
        if len(parts) == 1:
            return self._default_bucket, parts[0]
        return parts[0], parts[1]
    
    def _get_or_create_bucket(self, bucket_name: str) -> Bucket:
        """Get a bucket, creating it if necessary."""
        if not self._storage.bucket_exists(bucket_name):
            return self._storage.create_bucket(bucket_name)
        return self._storage.get_bucket(bucket_name)
    
    def upload_from_text(
        self,
        path: str,
        text: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        encoding: str = "utf-8",
    ) -> StorageObject:
        """
        Upload text content to storage.
        
        Args:
            path: Storage path (bucket/key or just key)
            text: Text content to upload
            content_type: MIME type (default: text/plain)
            metadata: Custom metadata
            encoding: Text encoding
            
        Returns:
            The stored StorageObject
        """
        bucket_name, key = self._parse_path(path)
        bucket = self._get_or_create_bucket(bucket_name)
        
        if content_type is None:
            content_type = ContentTypeDetector.detect_from_filename(key)
            if content_type == "application/octet-stream":
                content_type = "text/plain"
        
        return bucket.put_object(
            key=key,
            data=text.encode(encoding),
            content_type=content_type,
            metadata=metadata,
        )
    
    def upload_from_bytes(
        self,
        path: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """
        Upload binary content to storage.
        
        Args:
            path: Storage path (bucket/key or just key)
            data: Binary content to upload
            content_type: MIME type (auto-detected if not provided)
            metadata: Custom metadata
            
        Returns:
            The stored StorageObject
        """
        bucket_name, key = self._parse_path(path)
        bucket = self._get_or_create_bucket(bucket_name)
        
        if content_type is None:
            content_type = ContentTypeDetector.detect_from_data(data, key)
        
        return bucket.put_object(
            key=key,
            data=data,
            content_type=content_type,
            metadata=metadata,
        )
    
    def upload_from_filename(
        self,
        path: str,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """
        Upload a file from the local filesystem.
        
        Args:
            path: Storage path (bucket/key or just key)
            filename: Local file path to upload
            content_type: MIME type (auto-detected if not provided)
            metadata: Custom metadata
            
        Returns:
            The stored StorageObject
        """
        with open(filename, 'rb') as f:
            data = f.read()
        
        if content_type is None:
            content_type = ContentTypeDetector.detect_from_filename(filename)
        
        return self.upload_from_bytes(path, data, content_type, metadata)
    
    def download_as_text(
        self,
        path: str,
        encoding: str = "utf-8",
    ) -> str:
        """
        Download object content as text.
        
        Args:
            path: Storage path (bucket/key or just key)
            encoding: Text encoding
            
        Returns:
            Object content as string
        """
        bucket_name, key = self._parse_path(path)
        bucket = self._storage.get_bucket(bucket_name)
        obj = bucket.get_object(key)
        return obj.data.decode(encoding)
    
    def download_as_bytes(self, path: str) -> bytes:
        """
        Download object content as bytes.
        
        Args:
            path: Storage path (bucket/key or just key)
            
        Returns:
            Object content as bytes
        """
        bucket_name, key = self._parse_path(path)
        bucket = self._storage.get_bucket(bucket_name)
        obj = bucket.get_object(key)
        return obj.data
    
    def download_to_filename(self, path: str, filename: str) -> int:
        """
        Download object to a local file.
        
        Args:
            path: Storage path (bucket/key or just key)
            filename: Local file path to save to
            
        Returns:
            Number of bytes written
        """
        data = self.download_as_bytes(path)
        
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        
        with open(filename, 'wb') as f:
            return f.write(data)
    
    def list(
        self,
        path: str = "",
        recursive: bool = True,
        max_results: int = 1000,
    ) -> List[StorageObject]:
        """
        List objects in storage.
        
        Args:
            path: Path prefix to list (bucket/prefix or just prefix)
            recursive: If False, use "/" as delimiter
            max_results: Maximum number of results
            
        Returns:
            List of StorageObjects
        """
        if not path:
            results = []
            for bucket_info in self._storage.list_buckets():
                bucket = self._storage.get_bucket(bucket_info.name)
                list_result = bucket.list_objects(max_keys=max_results)
                results.extend(list_result.objects)
            return results[:max_results]
        
        bucket_name, prefix = self._parse_path(path)
        
        if not self._storage.bucket_exists(bucket_name):
            return []
        
        bucket = self._storage.get_bucket(bucket_name)
        delimiter = None if recursive else "/"
        
        result = bucket.list_objects(
            prefix=prefix if prefix else None,
            delimiter=delimiter,
            max_keys=max_results,
        )
        
        return result.objects
    
    def delete(self, path: str) -> bool:
        """
        Delete an object.
        
        Args:
            path: Storage path (bucket/key or just key)
            
        Returns:
            True if object was deleted
        """
        bucket_name, key = self._parse_path(path)
        
        if not self._storage.bucket_exists(bucket_name):
            return False
        
        bucket = self._storage.get_bucket(bucket_name)
        return bucket.delete_object(key)
    
    def exists(self, path: str) -> bool:
        """
        Check if an object exists.
        
        Args:
            path: Storage path (bucket/key or just key)
            
        Returns:
            True if object exists
        """
        bucket_name, key = self._parse_path(path)
        
        if not self._storage.bucket_exists(bucket_name):
            return False
        
        bucket = self._storage.get_bucket(bucket_name)
        return bucket.object_exists(key)
    
    def get_metadata(self, path: str) -> ObjectMetadata:
        """
        Get object metadata.
        
        Args:
            path: Storage path (bucket/key or just key)
            
        Returns:
            Object metadata
        """
        bucket_name, key = self._parse_path(path)
        bucket = self._storage.get_bucket(bucket_name)
        return bucket.head_object(key)
    
    def copy(
        self,
        source_path: str,
        dest_path: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """
        Copy an object.
        
        Args:
            source_path: Source path
            dest_path: Destination path
            metadata: New metadata (uses source if not provided)
            
        Returns:
            The copied StorageObject
        """
        src_bucket, src_key = self._parse_path(source_path)
        dst_bucket, dst_key = self._parse_path(dest_path)
        
        self._get_or_create_bucket(dst_bucket)
        
        return self._storage.copy_object(
            source_bucket=src_bucket,
            source_key=src_key,
            dest_bucket=dst_bucket,
            dest_key=dst_key,
            metadata=metadata,
        )
    
    def get_presigned_url(
        self,
        path: str,
        method: str = "GET",
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned URL for temporary access.
        
        Args:
            path: Storage path
            method: HTTP method (GET, PUT)
            expires_in: Expiration time in seconds
            
        Returns:
            Presigned URL string
        """
        bucket_name, key = self._parse_path(path)
        presigned = self._storage.generate_presigned_url(
            bucket_name=bucket_name,
            key=key,
            method=method,
            expires_in=expires_in,
        )
        return presigned.url
    
    def create_bucket(
        self,
        name: str,
        versioning_enabled: bool = False,
    ) -> Bucket:
        """
        Create a new bucket.
        
        Args:
            name: Bucket name
            versioning_enabled: Enable versioning
            
        Returns:
            The created Bucket
        """
        return self._storage.create_bucket(name, versioning_enabled=versioning_enabled)
    
    def delete_bucket(self, name: str, force: bool = False) -> bool:
        """
        Delete a bucket.
        
        Args:
            name: Bucket name
            force: Delete even if not empty
            
        Returns:
            True if bucket was deleted
        """
        return self._storage.delete_bucket(name, force=force)
    
    def list_buckets(self) -> List[BucketInfo]:
        """List all buckets."""
        return self._storage.list_buckets()
    
    @property
    def storage(self) -> ObjectStorage:
        """Access the underlying ObjectStorage instance."""
        return self._storage


_default_client: Optional[StorageClient] = None
_client_lock = threading.Lock()


def get_default_client() -> StorageClient:
    """Get or create the default storage client."""
    global _default_client
    with _client_lock:
        if _default_client is None:
            _default_client = StorageClient()
        return _default_client


def set_default_client(client: StorageClient) -> None:
    """Set the default storage client."""
    global _default_client
    with _client_lock:
        _default_client = client


def upload_from_text(
    path: str,
    text: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> StorageObject:
    """Upload text content to storage using the default client."""
    return get_default_client().upload_from_text(path, text, content_type, metadata)


def upload_from_bytes(
    path: str,
    data: bytes,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> StorageObject:
    """Upload binary content to storage using the default client."""
    return get_default_client().upload_from_bytes(path, data, content_type, metadata)


def upload_from_filename(
    path: str,
    filename: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> StorageObject:
    """Upload a file from the local filesystem using the default client."""
    return get_default_client().upload_from_filename(path, filename, content_type, metadata)


def download_as_text(path: str, encoding: str = "utf-8") -> str:
    """Download object content as text using the default client."""
    return get_default_client().download_as_text(path, encoding)


def download_as_bytes(path: str) -> bytes:
    """Download object content as bytes using the default client."""
    return get_default_client().download_as_bytes(path)


def download_to_filename(path: str, filename: str) -> int:
    """Download object to a local file using the default client."""
    return get_default_client().download_to_filename(path, filename)


def list_objects(
    path: str = "",
    recursive: bool = True,
    max_results: int = 1000,
) -> List[StorageObject]:
    """List objects in storage using the default client."""
    return get_default_client().list(path, recursive, max_results)


def delete_object(path: str) -> bool:
    """Delete an object using the default client."""
    return get_default_client().delete(path)


def object_exists(path: str) -> bool:
    """Check if an object exists using the default client."""
    return get_default_client().exists(path)


def get_presigned_url(path: str, method: str = "GET", expires_in: int = 3600) -> str:
    """Generate a presigned URL using the default client."""
    return get_default_client().get_presigned_url(path, method, expires_in)


def format_size(size_bytes: int) -> str:
    """Format a size in bytes as a human-readable string."""
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PiB"
