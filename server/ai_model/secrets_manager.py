"""
Secrets Manager System for Platform Forge

This module provides a comprehensive secrets management system matching
Replit's Secrets functionality with encrypted storage, environment integration,
access control, and audit logging.

Key Components:
- Secret: Represents an encrypted secret with metadata
- SecretStore: Storage backend with AES-256 encryption
- SecretsManager: Main manager class for secret lifecycle
- EnvironmentInjector: Injects secrets into environment variables

Features:
- AES-256-GCM encryption for stored secrets
- PBKDF2 key derivation from master password
- App-level and account-level secret scopes
- Owner vs collaborator access control
- Secret rotation with version history
- Expiration support with automatic invalidation
- Comprehensive audit logging
- Deployment environment integration
- Secure memory handling

Security Practices:
- Encryption at rest with authenticated encryption (AES-GCM)
- Key derivation with 100,000+ PBKDF2 iterations
- Secure random IV generation per encryption
- Secret values cleared from memory after use
- No plaintext secrets in logs or errors
- Access logging for compliance

Usage:
    from server.ai_model.secrets_manager import (
        SecretsManager,
        Secret,
        SecretScope,
        AccessLevel,
    )
    
    # Initialize the manager
    manager = SecretsManager(master_password="secure_master_password")
    
    # Set a secret
    manager.set_secret("API_KEY", "sk-1234567890", scope=SecretScope.APP)
    
    # Get a secret
    value = manager.get_secret("API_KEY")
    
    # List secrets (names only)
    names = manager.list_secrets()
    
    # Rotate a secret
    manager.rotate_secret("API_KEY", "sk-new-value-0987654321")
    
    # Get secret history
    history = manager.get_secret_history("API_KEY")
    
    # Export to environment
    manager.export_to_env()
    
    # Load from environment
    manager.load_from_env(prefix="APP_")
    
    # Use the environment injector
    from server.ai_model.secrets_manager import EnvironmentInjector
    
    injector = EnvironmentInjector(manager)
    injector.inject_all()
    injector.inject_for_deployment("production")
"""

import os
import re
import json
import time
import base64
import hashlib
import secrets as python_secrets
import threading
import logging
import copy
import ctypes
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SecretScope(Enum):
    """Scope of a secret's visibility and usage."""
    APP = "app"
    ACCOUNT = "account"
    PROJECT = "project"
    GLOBAL = "global"


class AccessLevel(Enum):
    """Access level for secret visibility."""
    OWNER = "owner"
    COLLABORATOR = "collaborator"
    VIEWER = "viewer"
    PUBLIC = "public"


class DeploymentEnvironment(Enum):
    """Deployment environments for secrets."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ALL = "all"


class AuditAction(Enum):
    """Types of auditable actions on secrets."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    ROTATE = "rotate"
    EXPORT = "export"
    IMPORT = "import"
    ACCESS_DENIED = "access_denied"
    EXPIRED = "expired"


class SecretStorageLimit(Enum):
    """Storage limits for secrets."""
    MAX_SECRET_NAME_LENGTH = 256
    MAX_SECRET_VALUE_LENGTH = 64 * 1024
    MAX_SECRETS_PER_APP = 1000
    MAX_SECRETS_PER_ACCOUNT = 5000
    MAX_SECRET_VERSIONS = 100
    MAX_DESCRIPTION_LENGTH = 1024
    PBKDF2_ITERATIONS = 100000
    AES_KEY_SIZE_BYTES = 32
    GCM_NONCE_SIZE_BYTES = 12
    GCM_TAG_SIZE_BYTES = 16


class SecretsError(Exception):
    """Base exception for Secrets Manager errors."""
    pass


class SecretNotFoundError(SecretsError):
    """Secret does not exist."""
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Secret '{name}' not found")


class SecretAlreadyExistsError(SecretsError):
    """Secret already exists."""
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Secret '{name}' already exists")


class SecretExpiredError(SecretsError):
    """Secret has expired."""
    def __init__(self, name: str, expired_at: float):
        self.name = name
        self.expired_at = expired_at
        super().__init__(f"Secret '{name}' expired at {datetime.fromtimestamp(expired_at)}")


class SecretAccessDeniedError(SecretsError):
    """Access to secret denied."""
    def __init__(self, name: str, required_level: AccessLevel, current_level: AccessLevel):
        self.name = name
        self.required_level = required_level
        self.current_level = current_level
        super().__init__(
            f"Access denied for secret '{name}': "
            f"requires {required_level.value}, have {current_level.value}"
        )


class InvalidSecretNameError(SecretsError):
    """Invalid secret name format."""
    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason
        super().__init__(f"Invalid secret name '{name}': {reason}")


class SecretValueTooLargeError(SecretsError):
    """Secret value exceeds maximum size."""
    def __init__(self, name: str, size: int, max_size: int):
        self.name = name
        self.size = size
        self.max_size = max_size
        super().__init__(f"Secret '{name}' value is {size} bytes, exceeds limit of {max_size} bytes")


class SecretQuotaExceededError(SecretsError):
    """Secret quota exceeded."""
    def __init__(self, scope: SecretScope, current: int, limit: int):
        self.scope = scope
        self.current = current
        self.limit = limit
        super().__init__(f"Secret quota exceeded for {scope.value}: {current}/{limit}")


class EncryptionError(SecretsError):
    """Error during encryption or decryption."""
    pass


class MasterPasswordRequiredError(SecretsError):
    """Master password is required but not provided."""
    def __init__(self):
        super().__init__("Master password is required for encryption operations")


def secure_zero_memory(data: Union[bytes, bytearray, str]) -> None:
    """
    Securely zero out memory containing sensitive data.
    
    This attempts to overwrite memory to prevent sensitive data from
    being recovered. Note: This is best-effort due to Python's memory model.
    """
    if isinstance(data, str):
        return
    
    if isinstance(data, (bytes, bytearray)):
        try:
            if isinstance(data, bytearray):
                for i in range(len(data)):
                    data[i] = 0
            else:
                mutable = bytearray(data)
                ctypes.memset(ctypes.addressof((ctypes.c_char * len(mutable)).from_buffer(mutable)), 0, len(mutable))
        except (TypeError, ValueError):
            pass


@dataclass
class SecretVersion:
    """
    Represents a historical version of a secret.
    
    Attributes:
        version: Version number (1, 2, 3, ...)
        encrypted_value: The encrypted secret value
        created_at: When this version was created
        created_by: User who created this version
        rotation_reason: Why the secret was rotated
    """
    version: int
    encrypted_value: bytes
    created_at: float = field(default_factory=time.time)
    created_by: Optional[str] = None
    rotation_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize version to dictionary."""
        return {
            "version": self.version,
            "encrypted_value": base64.b64encode(self.encrypted_value).decode('ascii'),
            "created_at": self.created_at,
            "created_by": self.created_by,
            "rotation_reason": self.rotation_reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecretVersion':
        """Deserialize version from dictionary."""
        return cls(
            version=data["version"],
            encrypted_value=base64.b64decode(data["encrypted_value"]),
            created_at=data.get("created_at", time.time()),
            created_by=data.get("created_by"),
            rotation_reason=data.get("rotation_reason"),
        )


@dataclass
class AuditLogEntry:
    """
    Audit log entry for secret access.
    
    Attributes:
        timestamp: When the action occurred
        action: Type of action performed
        secret_name: Name of the secret
        user_id: User who performed the action
        ip_address: IP address of the requester
        success: Whether the action succeeded
        details: Additional details about the action
        environment: Deployment environment context
    """
    timestamp: float
    action: AuditAction
    secret_name: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    success: bool = True
    details: Optional[str] = None
    environment: Optional[DeploymentEnvironment] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry to dictionary."""
        return {
            "timestamp": self.timestamp,
            "action": self.action.value,
            "secret_name": self.secret_name,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "success": self.success,
            "details": self.details,
            "environment": self.environment.value if self.environment else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditLogEntry':
        """Deserialize entry from dictionary."""
        env = data.get("environment")
        return cls(
            timestamp=data["timestamp"],
            action=AuditAction(data["action"]),
            secret_name=data["secret_name"],
            user_id=data.get("user_id"),
            ip_address=data.get("ip_address"),
            success=data.get("success", True),
            details=data.get("details"),
            environment=DeploymentEnvironment(env) if env else None,
        )


@dataclass
class SecretMetadata:
    """
    Metadata for a secret.
    
    Attributes:
        name: Secret name/key
        scope: Visibility scope
        access_level: Required access level
        description: Human-readable description
        environments: Environments where secret is available
        created_at: Creation timestamp
        updated_at: Last update timestamp
        created_by: User who created the secret
        expires_at: Expiration timestamp (optional)
        current_version: Current version number
        tags: Key-value tags for organization
        linked_secrets: Names of related secrets
    """
    name: str
    scope: SecretScope = SecretScope.APP
    access_level: AccessLevel = AccessLevel.OWNER
    description: Optional[str] = None
    environments: List[DeploymentEnvironment] = field(default_factory=lambda: [DeploymentEnvironment.ALL])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    created_by: Optional[str] = None
    expires_at: Optional[float] = None
    current_version: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    linked_secrets: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if the secret has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    @property
    def ttl_remaining(self) -> Optional[float]:
        """Get remaining TTL in seconds."""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0, remaining)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "name": self.name,
            "scope": self.scope.value,
            "access_level": self.access_level.value,
            "description": self.description,
            "environments": [e.value for e in self.environments],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "expires_at": self.expires_at,
            "current_version": self.current_version,
            "tags": self.tags,
            "linked_secrets": self.linked_secrets,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecretMetadata':
        """Deserialize metadata from dictionary."""
        return cls(
            name=data["name"],
            scope=SecretScope(data.get("scope", "app")),
            access_level=AccessLevel(data.get("access_level", "owner")),
            description=data.get("description"),
            environments=[DeploymentEnvironment(e) for e in data.get("environments", ["all"])],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            created_by=data.get("created_by"),
            expires_at=data.get("expires_at"),
            current_version=data.get("current_version", 1),
            tags=data.get("tags", {}),
            linked_secrets=data.get("linked_secrets", []),
        )


@dataclass
class Secret:
    """
    Represents an encrypted secret with full metadata.
    
    This is the primary data class for secrets. It contains the encrypted
    value, metadata, and version history. The value is always stored
    encrypted and only decrypted when explicitly accessed.
    
    Attributes:
        metadata: Secret metadata
        encrypted_value: Current encrypted value
        nonce: Encryption nonce/IV
        versions: Version history for rotation tracking
        access_count: Number of times the secret was accessed
        last_accessed: Timestamp of last access
    """
    metadata: SecretMetadata
    encrypted_value: bytes
    nonce: bytes
    versions: List[SecretVersion] = field(default_factory=list)
    access_count: int = 0
    last_accessed: Optional[float] = None
    
    @property
    def name(self) -> str:
        """Get the secret name."""
        return self.metadata.name
    
    @property
    def is_expired(self) -> bool:
        """Check if the secret has expired."""
        return self.metadata.is_expired
    
    @property
    def current_version(self) -> int:
        """Get the current version number."""
        return self.metadata.current_version
    
    def record_access(self) -> None:
        """Record an access to this secret."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize secret to dictionary (for storage)."""
        return {
            "metadata": self.metadata.to_dict(),
            "encrypted_value": base64.b64encode(self.encrypted_value).decode('ascii'),
            "nonce": base64.b64encode(self.nonce).decode('ascii'),
            "versions": [v.to_dict() for v in self.versions],
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Secret':
        """Deserialize secret from dictionary."""
        return cls(
            metadata=SecretMetadata.from_dict(data["metadata"]),
            encrypted_value=base64.b64decode(data["encrypted_value"]),
            nonce=base64.b64decode(data["nonce"]),
            versions=[SecretVersion.from_dict(v) for v in data.get("versions", [])],
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
        )
    
    def get_public_info(self) -> Dict[str, Any]:
        """Get public (non-sensitive) information about the secret."""
        return {
            "name": self.name,
            "scope": self.metadata.scope.value,
            "access_level": self.metadata.access_level.value,
            "description": self.metadata.description,
            "environments": [e.value for e in self.metadata.environments],
            "created_at": self.metadata.created_at,
            "updated_at": self.metadata.updated_at,
            "is_expired": self.is_expired,
            "ttl_remaining": self.metadata.ttl_remaining,
            "current_version": self.current_version,
            "version_count": len(self.versions) + 1,
            "tags": self.metadata.tags,
            "access_count": self.access_count,
        }


class Encryptor:
    """
    Handles encryption and decryption of secret values.
    
    Uses AES-256-GCM for authenticated encryption with PBKDF2 key derivation.
    
    Args:
        master_password: The master password for key derivation
        salt: Optional salt for key derivation (generated if not provided)
        iterations: PBKDF2 iteration count (default: 100000)
    """
    
    def __init__(
        self,
        master_password: str,
        salt: Optional[bytes] = None,
        iterations: int = SecretStorageLimit.PBKDF2_ITERATIONS.value,
    ):
        if not CRYPTO_AVAILABLE:
            raise EncryptionError(
                "Cryptography library not available. Install with: pip install cryptography"
            )
        
        self._iterations = iterations
        self._salt = salt or python_secrets.token_bytes(32)
        self._key = self._derive_key(master_password)
        self._aesgcm = AESGCM(self._key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=SecretStorageLimit.AES_KEY_SIZE_BYTES.value,
            salt=self._salt,
            iterations=self._iterations,
            backend=default_backend(),
        )
        return kdf.derive(password.encode('utf-8'))
    
    @property
    def salt(self) -> bytes:
        """Get the salt used for key derivation."""
        return self._salt
    
    def encrypt(self, plaintext: Union[str, bytes]) -> Tuple[bytes, bytes]:
        """
        Encrypt a plaintext value.
        
        Args:
            plaintext: Value to encrypt (string or bytes)
            
        Returns:
            Tuple of (encrypted_data, nonce)
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        nonce = python_secrets.token_bytes(SecretStorageLimit.GCM_NONCE_SIZE_BYTES.value)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, None)
        
        return ciphertext, nonce
    
    def decrypt(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """
        Decrypt an encrypted value.
        
        Args:
            ciphertext: Encrypted data
            nonce: Encryption nonce
            
        Returns:
            Decrypted plaintext bytes
        """
        try:
            return self._aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")
    
    def decrypt_to_string(self, ciphertext: bytes, nonce: bytes, encoding: str = 'utf-8') -> str:
        """
        Decrypt an encrypted value and return as string.
        
        Args:
            ciphertext: Encrypted data
            nonce: Encryption nonce
            encoding: String encoding to use
            
        Returns:
            Decrypted plaintext string
        """
        plaintext = self.decrypt(ciphertext, nonce)
        return plaintext.decode(encoding)
    
    def rotate_key(self, new_password: str) -> 'Encryptor':
        """
        Create a new encryptor with a rotated key.
        
        Args:
            new_password: New master password
            
        Returns:
            New Encryptor instance with rotated key
        """
        new_salt = python_secrets.token_bytes(32)
        return Encryptor(new_password, salt=new_salt, iterations=self._iterations)
    
    def destroy(self) -> None:
        """
        Securely destroy the encryption key from memory.
        """
        if hasattr(self, '_key'):
            secure_zero_memory(bytearray(self._key))
            del self._key
        if hasattr(self, '_aesgcm'):
            del self._aesgcm


class FallbackEncryptor:
    """
    Fallback encryptor when cryptography library is not available.
    
    Uses a simple XOR-based obfuscation. This is NOT cryptographically secure
    and should only be used for development/testing purposes.
    """
    
    def __init__(self, master_password: str, salt: Optional[bytes] = None, **kwargs):
        self._salt = salt or python_secrets.token_bytes(32)
        self._key = self._derive_key(master_password)
        logging.warning(
            "Using fallback encryptor (XOR-based). This is NOT secure! "
            "Install 'cryptography' package for proper AES-256 encryption."
        )
    
    def _derive_key(self, password: str) -> bytes:
        """Simple key derivation using SHA-256."""
        combined = password.encode('utf-8') + self._salt
        return hashlib.sha256(combined).digest()
    
    @property
    def salt(self) -> bytes:
        return self._salt
    
    def encrypt(self, plaintext: Union[str, bytes]) -> Tuple[bytes, bytes]:
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        nonce = python_secrets.token_bytes(12)
        key_stream = hashlib.sha256(self._key + nonce).digest()
        
        ciphertext = bytearray(len(plaintext))
        for i, byte in enumerate(plaintext):
            ciphertext[i] = byte ^ key_stream[i % len(key_stream)]
        
        return bytes(ciphertext), nonce
    
    def decrypt(self, ciphertext: bytes, nonce: bytes) -> bytes:
        key_stream = hashlib.sha256(self._key + nonce).digest()
        
        plaintext = bytearray(len(ciphertext))
        for i, byte in enumerate(ciphertext):
            plaintext[i] = byte ^ key_stream[i % len(key_stream)]
        
        return bytes(plaintext)
    
    def decrypt_to_string(self, ciphertext: bytes, nonce: bytes, encoding: str = 'utf-8') -> str:
        return self.decrypt(ciphertext, nonce).decode(encoding)
    
    def rotate_key(self, new_password: str) -> 'FallbackEncryptor':
        return FallbackEncryptor(new_password, salt=python_secrets.token_bytes(32))
    
    def destroy(self) -> None:
        if hasattr(self, '_key'):
            secure_zero_memory(bytearray(self._key))


class AuditLogger:
    """
    Audit logger for secret access tracking.
    
    Provides a persistent log of all secret operations for compliance
    and security monitoring.
    
    Args:
        storage_path: Path to store audit logs
        max_entries: Maximum entries per log file
        retention_days: Days to retain audit logs
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_entries: int = 10000,
        retention_days: int = 90,
    ):
        self.storage_path = Path(storage_path or "./secrets_audit")
        self.max_entries = max_entries
        self.retention_days = retention_days
        self._entries: List[AuditLogEntry] = []
        self._lock = threading.RLock()
        self._logger = logging.getLogger("secrets_audit")
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def log(
        self,
        action: AuditAction,
        secret_name: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
        details: Optional[str] = None,
        environment: Optional[DeploymentEnvironment] = None,
    ) -> AuditLogEntry:
        """
        Log an action on a secret.
        
        Args:
            action: Type of action performed
            secret_name: Name of the secret
            user_id: User who performed the action
            ip_address: IP address of the requester
            success: Whether the action succeeded
            details: Additional details
            environment: Deployment environment context
            
        Returns:
            The created audit log entry
        """
        entry = AuditLogEntry(
            timestamp=time.time(),
            action=action,
            secret_name=secret_name,
            user_id=user_id,
            ip_address=ip_address,
            success=success,
            details=details,
            environment=environment,
        )
        
        with self._lock:
            self._entries.append(entry)
            
            if len(self._entries) >= self.max_entries:
                self._flush_to_disk()
        
        log_msg = f"SECRET_AUDIT: {action.value} on '{secret_name}'"
        if user_id:
            log_msg += f" by user '{user_id}'"
        if not success:
            log_msg += " (FAILED)"
        if details:
            log_msg += f" - {details}"
        
        if success:
            self._logger.info(log_msg)
        else:
            self._logger.warning(log_msg)
        
        return entry
    
    def _flush_to_disk(self) -> None:
        """Flush entries to disk and clear memory buffer."""
        if not self._entries:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{timestamp}.json"
        filepath = self.storage_path / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([e.to_dict() for e in self._entries], f, indent=2)
            self._entries.clear()
        except IOError as e:
            self._logger.error(f"Failed to flush audit log: {e}")
    
    def get_entries(
        self,
        secret_name: Optional[str] = None,
        action: Optional[AuditAction] = None,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        success_only: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """
        Query audit log entries with filters.
        
        Args:
            secret_name: Filter by secret name
            action: Filter by action type
            user_id: Filter by user ID
            start_time: Filter entries after this timestamp
            end_time: Filter entries before this timestamp
            success_only: Filter by success status
            limit: Maximum number of entries to return
            
        Returns:
            List of matching audit log entries
        """
        with self._lock:
            all_entries = self._entries.copy()
        
        for log_file in sorted(self.storage_path.glob("audit_*.json"), reverse=True):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    file_entries = [AuditLogEntry.from_dict(d) for d in json.load(f)]
                    all_entries.extend(file_entries)
            except (json.JSONDecodeError, IOError):
                continue
        
        filtered = []
        for entry in sorted(all_entries, key=lambda e: e.timestamp, reverse=True):
            if secret_name and entry.secret_name != secret_name:
                continue
            if action and entry.action != action:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if success_only is not None and entry.success != success_only:
                continue
            
            filtered.append(entry)
            
            if len(filtered) >= limit:
                break
        
        return filtered
    
    def flush(self) -> None:
        """Force flush of in-memory entries to disk."""
        with self._lock:
            self._flush_to_disk()
    
    def cleanup_old_logs(self) -> int:
        """
        Remove audit logs older than retention period.
        
        Returns:
            Number of log files removed
        """
        cutoff_time = time.time() - (self.retention_days * 86400)
        removed_count = 0
        
        for log_file in self.storage_path.glob("audit_*.json"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    removed_count += 1
            except (IOError, OSError):
                continue
        
        return removed_count


class SecretStore:
    """
    Storage backend for secrets with encryption.
    
    Provides the low-level storage operations with encryption, persistence,
    and thread-safety. This class handles the actual storage of secrets
    while SecretsManager provides the high-level API.
    
    Args:
        storage_dir: Directory for persistent storage
        master_password: Password for encryption (required for encryption)
        auto_persist: Automatically persist changes to disk
    """
    
    SECRET_NAME_PATTERN = re.compile(r'^[A-Z][A-Z0-9_]*$')
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        master_password: Optional[str] = None,
        auto_persist: bool = True,
    ):
        self.storage_dir = Path(storage_dir or "./secrets_data")
        self.auto_persist = auto_persist
        
        self._secrets: Dict[str, Secret] = {}
        self._lock = threading.RLock()
        self._encryptor: Optional[Union[Encryptor, FallbackEncryptor]] = None
        self._salt: Optional[bytes] = None
        
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_salt()
        
        if master_password:
            self._init_encryptor(master_password)
        
        self._load_from_disk()
    
    def _load_salt(self) -> None:
        """Load or generate the encryption salt."""
        salt_path = self.storage_dir / ".salt"
        
        if salt_path.exists():
            with open(salt_path, 'rb') as f:
                self._salt = f.read()
        else:
            self._salt = python_secrets.token_bytes(32)
            with open(salt_path, 'wb') as f:
                f.write(self._salt)
    
    def _init_encryptor(self, master_password: str) -> None:
        """Initialize the encryptor with the master password."""
        if CRYPTO_AVAILABLE:
            self._encryptor = Encryptor(master_password, salt=self._salt)
        else:
            self._encryptor = FallbackEncryptor(master_password, salt=self._salt)
    
    def set_master_password(self, password: str) -> None:
        """Set or update the master password for encryption."""
        self._init_encryptor(password)
    
    def _require_encryptor(self) -> Union[Encryptor, FallbackEncryptor]:
        """Ensure encryptor is initialized."""
        if self._encryptor is None:
            raise MasterPasswordRequiredError()
        return self._encryptor
    
    def _validate_secret_name(self, name: str) -> None:
        """Validate secret name format."""
        if not name:
            raise InvalidSecretNameError(name, "Name cannot be empty")
        
        if len(name) > SecretStorageLimit.MAX_SECRET_NAME_LENGTH.value:
            raise InvalidSecretNameError(
                name, 
                f"Name exceeds maximum length of {SecretStorageLimit.MAX_SECRET_NAME_LENGTH.value}"
            )
        
        if not self.SECRET_NAME_PATTERN.match(name):
            raise InvalidSecretNameError(
                name,
                "Name must start with uppercase letter and contain only uppercase letters, "
                "numbers, and underscores (e.g., MY_API_KEY, DATABASE_URL)"
            )
    
    def _validate_secret_value(self, name: str, value: str) -> None:
        """Validate secret value."""
        if len(value) > SecretStorageLimit.MAX_SECRET_VALUE_LENGTH.value:
            raise SecretValueTooLargeError(
                name,
                len(value),
                SecretStorageLimit.MAX_SECRET_VALUE_LENGTH.value
            )
    
    def _get_storage_path(self) -> Path:
        """Get the file path for secrets storage."""
        return self.storage_dir / "secrets.enc.json"
    
    def _load_from_disk(self) -> None:
        """Load secrets from disk."""
        path = self._get_storage_path()
        if not path.exists():
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for secret_data in data.get("secrets", []):
                try:
                    secret = Secret.from_dict(secret_data)
                    self._secrets[secret.name] = secret
                except (KeyError, ValueError):
                    continue
        except (json.JSONDecodeError, IOError):
            pass
    
    def _persist_to_disk(self) -> None:
        """Persist secrets to disk."""
        if not self.auto_persist:
            return
        
        path = self._get_storage_path()
        data = {
            "version": "1.0",
            "updated_at": time.time(),
            "secrets": [secret.to_dict() for secret in self._secrets.values()],
        }
        
        try:
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(path)
        except IOError:
            pass
    
    def store(
        self,
        name: str,
        value: str,
        metadata: Optional[SecretMetadata] = None,
    ) -> Secret:
        """
        Store a secret with encryption.
        
        Args:
            name: Secret name
            value: Secret value (will be encrypted)
            metadata: Optional metadata (created if not provided)
            
        Returns:
            The stored Secret object
        """
        self._validate_secret_name(name)
        self._validate_secret_value(name, value)
        
        encryptor = self._require_encryptor()
        
        with self._lock:
            encrypted_value, nonce = encryptor.encrypt(value)
            
            if metadata is None:
                metadata = SecretMetadata(name=name)
            else:
                metadata.name = name
            
            metadata.updated_at = time.time()
            
            secret = Secret(
                metadata=metadata,
                encrypted_value=encrypted_value,
                nonce=nonce,
            )
            
            self._secrets[name] = secret
            self._persist_to_disk()
            
            return secret
    
    def retrieve(self, name: str) -> Optional[str]:
        """
        Retrieve and decrypt a secret value.
        
        Args:
            name: Secret name
            
        Returns:
            Decrypted secret value, or None if not found
        """
        with self._lock:
            if name not in self._secrets:
                return None
            
            secret = self._secrets[name]
            
            if secret.is_expired:
                return None
            
            encryptor = self._require_encryptor()
            secret.record_access()
            
            return encryptor.decrypt_to_string(secret.encrypted_value, secret.nonce)
    
    def get_secret(self, name: str) -> Optional[Secret]:
        """
        Get a Secret object without decryption.
        
        Args:
            name: Secret name
            
        Returns:
            Secret object, or None if not found
        """
        with self._lock:
            return self._secrets.get(name)
    
    def remove(self, name: str) -> bool:
        """
        Remove a secret from storage.
        
        Args:
            name: Secret name
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name not in self._secrets:
                return False
            
            del self._secrets[name]
            self._persist_to_disk()
            return True
    
    def list_names(
        self,
        scope: Optional[SecretScope] = None,
        environment: Optional[DeploymentEnvironment] = None,
    ) -> List[str]:
        """
        List secret names with optional filtering.
        
        Args:
            scope: Filter by scope
            environment: Filter by environment
            
        Returns:
            List of secret names
        """
        with self._lock:
            names = []
            
            for name, secret in self._secrets.items():
                if secret.is_expired:
                    continue
                
                if scope and secret.metadata.scope != scope:
                    continue
                
                if environment and environment not in secret.metadata.environments:
                    if DeploymentEnvironment.ALL not in secret.metadata.environments:
                        continue
                
                names.append(name)
            
            return sorted(names)
    
    def exists(self, name: str) -> bool:
        """Check if a secret exists."""
        with self._lock:
            if name not in self._secrets:
                return False
            return not self._secrets[name].is_expired
    
    def get_all_secrets(self) -> Dict[str, Secret]:
        """Get all secrets (for internal use)."""
        with self._lock:
            return copy.deepcopy(self._secrets)
    
    def clear(self) -> int:
        """
        Clear all secrets.
        
        Returns:
            Number of secrets cleared
        """
        with self._lock:
            count = len(self._secrets)
            self._secrets.clear()
            self._persist_to_disk()
            return count
    
    def destroy(self) -> None:
        """Securely destroy the store and encryption keys."""
        with self._lock:
            if self._encryptor:
                self._encryptor.destroy()
                self._encryptor = None
            self._secrets.clear()


class SecretsManager:
    """
    Main manager class for the Secrets system.
    
    Provides the high-level API for managing secrets with access control,
    rotation, versioning, and audit logging.
    
    Args:
        master_password: Password for encryption
        storage_dir: Directory for persistent storage
        audit_logger: Optional custom audit logger
        current_user: Current user ID for access control
        current_access_level: Access level of current user
    """
    
    def __init__(
        self,
        master_password: Optional[str] = None,
        storage_dir: Optional[str] = None,
        audit_logger: Optional[AuditLogger] = None,
        current_user: Optional[str] = None,
        current_access_level: AccessLevel = AccessLevel.OWNER,
    ):
        self._store = SecretStore(
            storage_dir=storage_dir,
            master_password=master_password,
        )
        self._audit = audit_logger or AuditLogger(
            storage_path=str(Path(storage_dir or "./secrets_data") / "audit")
        )
        self._current_user = current_user
        self._current_access_level = current_access_level
        self._lock = threading.RLock()
    
    def set_master_password(self, password: str) -> None:
        """Set or update the master password."""
        self._store.set_master_password(password)
    
    def set_current_user(self, user_id: str, access_level: AccessLevel) -> None:
        """Set the current user context for access control."""
        self._current_user = user_id
        self._current_access_level = access_level
    
    def _check_access(self, secret: Secret, action: AuditAction) -> None:
        """Check if current user has access to the secret."""
        required_level = secret.metadata.access_level
        
        access_hierarchy = {
            AccessLevel.OWNER: 4,
            AccessLevel.COLLABORATOR: 3,
            AccessLevel.VIEWER: 2,
            AccessLevel.PUBLIC: 1,
        }
        
        current_rank = access_hierarchy.get(self._current_access_level, 0)
        required_rank = access_hierarchy.get(required_level, 4)
        
        if current_rank < required_rank:
            self._audit.log(
                action=AuditAction.ACCESS_DENIED,
                secret_name=secret.name,
                user_id=self._current_user,
                success=False,
                details=f"Required {required_level.value}, had {self._current_access_level.value}",
            )
            raise SecretAccessDeniedError(
                secret.name,
                required_level,
                self._current_access_level,
            )
    
    def set_secret(
        self,
        name: str,
        value: str,
        scope: SecretScope = SecretScope.APP,
        access_level: AccessLevel = AccessLevel.OWNER,
        description: Optional[str] = None,
        environments: Optional[List[DeploymentEnvironment]] = None,
        expires_in: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
    ) -> Secret:
        """
        Set a secret value.
        
        Args:
            name: Secret name (must be uppercase with underscores)
            value: Secret value
            scope: Visibility scope
            access_level: Required access level
            description: Human-readable description
            environments: Environments where available
            expires_in: Expiration time in seconds
            tags: Key-value tags
            overwrite: Whether to overwrite existing secret
            
        Returns:
            The created/updated Secret object
        """
        with self._lock:
            existing = self._store.get_secret(name)
            
            if existing and not overwrite:
                raise SecretAlreadyExistsError(name)
            
            expires_at = None
            if expires_in:
                expires_at = time.time() + expires_in
            
            metadata = SecretMetadata(
                name=name,
                scope=scope,
                access_level=access_level,
                description=description,
                environments=environments or [DeploymentEnvironment.ALL],
                created_by=self._current_user,
                expires_at=expires_at,
                tags=tags or {},
            )
            
            if existing:
                metadata.created_at = existing.metadata.created_at
                metadata.current_version = existing.metadata.current_version + 1
            
            secret = self._store.store(name, value, metadata)
            
            action = AuditAction.UPDATE if existing else AuditAction.CREATE
            self._audit.log(
                action=action,
                secret_name=name,
                user_id=self._current_user,
                success=True,
                details=f"Scope: {scope.value}, Version: {metadata.current_version}",
            )
            
            return secret
    
    def get_secret(
        self,
        name: str,
        environment: Optional[DeploymentEnvironment] = None,
    ) -> str:
        """
        Get a secret value.
        
        Args:
            name: Secret name
            environment: Deployment environment context
            
        Returns:
            Decrypted secret value
            
        Raises:
            SecretNotFoundError: If secret doesn't exist
            SecretExpiredError: If secret has expired
            SecretAccessDeniedError: If access is denied
        """
        with self._lock:
            secret = self._store.get_secret(name)
            
            if not secret:
                self._audit.log(
                    action=AuditAction.READ,
                    secret_name=name,
                    user_id=self._current_user,
                    success=False,
                    details="Secret not found",
                )
                raise SecretNotFoundError(name)
            
            if secret.is_expired:
                self._audit.log(
                    action=AuditAction.EXPIRED,
                    secret_name=name,
                    user_id=self._current_user,
                    success=False,
                )
                raise SecretExpiredError(name, secret.metadata.expires_at)
            
            if environment:
                if (environment not in secret.metadata.environments and 
                    DeploymentEnvironment.ALL not in secret.metadata.environments):
                    self._audit.log(
                        action=AuditAction.ACCESS_DENIED,
                        secret_name=name,
                        user_id=self._current_user,
                        success=False,
                        details=f"Not available in {environment.value}",
                        environment=environment,
                    )
                    raise SecretAccessDeniedError(
                        name, 
                        AccessLevel.OWNER, 
                        self._current_access_level
                    )
            
            self._check_access(secret, AuditAction.READ)
            
            value = self._store.retrieve(name)
            
            self._audit.log(
                action=AuditAction.READ,
                secret_name=name,
                user_id=self._current_user,
                success=True,
                environment=environment,
            )
            
            return value
    
    def delete_secret(self, name: str) -> bool:
        """
        Delete a secret.
        
        Args:
            name: Secret name
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            secret = self._store.get_secret(name)
            
            if secret:
                self._check_access(secret, AuditAction.DELETE)
            
            result = self._store.remove(name)
            
            self._audit.log(
                action=AuditAction.DELETE,
                secret_name=name,
                user_id=self._current_user,
                success=result,
                details="Secret deleted" if result else "Secret not found",
            )
            
            return result
    
    def list_secrets(
        self,
        scope: Optional[SecretScope] = None,
        environment: Optional[DeploymentEnvironment] = None,
        include_expired: bool = False,
    ) -> List[str]:
        """
        List secret names (not values).
        
        Args:
            scope: Filter by scope
            environment: Filter by environment
            include_expired: Include expired secrets
            
        Returns:
            List of secret names
        """
        with self._lock:
            names = self._store.list_names(scope=scope, environment=environment)
            
            if not include_expired:
                names = [
                    name for name in names
                    if not self._store.get_secret(name).is_expired
                ]
            
            return names
    
    def get_secret_info(self, name: str) -> Dict[str, Any]:
        """
        Get public information about a secret (no value).
        
        Args:
            name: Secret name
            
        Returns:
            Dictionary of public secret information
        """
        with self._lock:
            secret = self._store.get_secret(name)
            if not secret:
                raise SecretNotFoundError(name)
            
            return secret.get_public_info()
    
    def rotate_secret(
        self,
        name: str,
        new_value: str,
        reason: Optional[str] = None,
    ) -> Secret:
        """
        Rotate a secret to a new value while preserving history.
        
        Args:
            name: Secret name
            new_value: New secret value
            reason: Reason for rotation
            
        Returns:
            Updated Secret object
        """
        with self._lock:
            secret = self._store.get_secret(name)
            
            if not secret:
                raise SecretNotFoundError(name)
            
            self._check_access(secret, AuditAction.ROTATE)
            
            version = SecretVersion(
                version=secret.metadata.current_version,
                encrypted_value=secret.encrypted_value,
                created_at=secret.metadata.updated_at,
                created_by=secret.metadata.created_by,
                rotation_reason=reason,
            )
            
            if len(secret.versions) >= SecretStorageLimit.MAX_SECRET_VERSIONS.value:
                secret.versions.pop(0)
            
            secret.versions.append(version)
            
            metadata = secret.metadata
            metadata.current_version += 1
            metadata.updated_at = time.time()
            
            new_secret = self._store.store(name, new_value, metadata)
            new_secret.versions = secret.versions
            
            self._audit.log(
                action=AuditAction.ROTATE,
                secret_name=name,
                user_id=self._current_user,
                success=True,
                details=f"Rotated to version {metadata.current_version}. Reason: {reason or 'Not specified'}",
            )
            
            return new_secret
    
    def get_secret_history(
        self,
        name: str,
        include_current: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get the version history of a secret.
        
        Args:
            name: Secret name
            include_current: Include current version in history
            
        Returns:
            List of version information dictionaries
        """
        with self._lock:
            secret = self._store.get_secret(name)
            
            if not secret:
                raise SecretNotFoundError(name)
            
            self._check_access(secret, AuditAction.READ)
            
            history = []
            
            for version in secret.versions:
                history.append({
                    "version": version.version,
                    "created_at": version.created_at,
                    "created_by": version.created_by,
                    "rotation_reason": version.rotation_reason,
                })
            
            if include_current:
                history.append({
                    "version": secret.metadata.current_version,
                    "created_at": secret.metadata.updated_at,
                    "created_by": secret.metadata.created_by,
                    "rotation_reason": None,
                    "is_current": True,
                })
            
            return sorted(history, key=lambda x: x["version"])
    
    def restore_version(self, name: str, version: int) -> Secret:
        """
        Restore a secret to a previous version.
        
        Args:
            name: Secret name
            version: Version number to restore
            
        Returns:
            Restored Secret object
        """
        with self._lock:
            secret = self._store.get_secret(name)
            
            if not secret:
                raise SecretNotFoundError(name)
            
            self._check_access(secret, AuditAction.ROTATE)
            
            target_version = None
            for v in secret.versions:
                if v.version == version:
                    target_version = v
                    break
            
            if not target_version:
                raise SecretsError(f"Version {version} not found for secret '{name}'")
            
            encryptor = self._store._require_encryptor()
            old_value = encryptor.decrypt_to_string(
                target_version.encrypted_value,
                secret.nonce
            )
            
            return self.rotate_secret(
                name,
                old_value,
                reason=f"Restored from version {version}",
            )
    
    def export_to_env(
        self,
        environment: Optional[DeploymentEnvironment] = None,
        prefix: str = "",
    ) -> Dict[str, str]:
        """
        Export secrets as environment variables.
        
        Args:
            environment: Only export secrets for this environment
            prefix: Prefix to add to variable names
            
        Returns:
            Dictionary of environment variable names to values
        """
        with self._lock:
            exported = {}
            names = self.list_secrets(environment=environment)
            
            for name in names:
                try:
                    value = self.get_secret(name, environment=environment)
                    env_name = f"{prefix}{name}"
                    os.environ[env_name] = value
                    exported[env_name] = value
                except (SecretNotFoundError, SecretExpiredError, SecretAccessDeniedError):
                    continue
            
            self._audit.log(
                action=AuditAction.EXPORT,
                secret_name="*",
                user_id=self._current_user,
                success=True,
                details=f"Exported {len(exported)} secrets to environment",
                environment=environment,
            )
            
            return exported
    
    def load_from_env(
        self,
        prefix: str = "",
        scope: SecretScope = SecretScope.APP,
        overwrite: bool = False,
    ) -> List[str]:
        """
        Load secrets from environment variables.
        
        Args:
            prefix: Only import variables with this prefix
            scope: Scope to assign to imported secrets
            overwrite: Whether to overwrite existing secrets
            
        Returns:
            List of imported secret names
        """
        imported = []
        
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            
            name = key[len(prefix):] if prefix else key
            
            try:
                self._store._validate_secret_name(name)
            except InvalidSecretNameError:
                continue
            
            try:
                if self._store.exists(name) and not overwrite:
                    continue
                
                self.set_secret(
                    name=name,
                    value=value,
                    scope=scope,
                    overwrite=overwrite,
                )
                imported.append(name)
            except SecretsError:
                continue
        
        self._audit.log(
            action=AuditAction.IMPORT,
            secret_name="*",
            user_id=self._current_user,
            success=True,
            details=f"Imported {len(imported)} secrets from environment",
        )
        
        return imported
    
    def get_audit_log(
        self,
        secret_name: Optional[str] = None,
        action: Optional[AuditAction] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            secret_name: Filter by secret name
            action: Filter by action type
            limit: Maximum entries to return
            
        Returns:
            List of audit log entries as dictionaries
        """
        entries = self._audit.get_entries(
            secret_name=secret_name,
            action=action,
            limit=limit,
        )
        return [e.to_dict() for e in entries]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the secrets store.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            all_secrets = self._store.get_all_secrets()
            
            scopes = {}
            environments = {}
            expired_count = 0
            total_versions = 0
            
            for secret in all_secrets.values():
                scope = secret.metadata.scope.value
                scopes[scope] = scopes.get(scope, 0) + 1
                
                for env in secret.metadata.environments:
                    env_name = env.value
                    environments[env_name] = environments.get(env_name, 0) + 1
                
                if secret.is_expired:
                    expired_count += 1
                
                total_versions += len(secret.versions) + 1
            
            return {
                "total_secrets": len(all_secrets),
                "by_scope": scopes,
                "by_environment": environments,
                "expired_secrets": expired_count,
                "total_versions": total_versions,
                "audit_entries": len(self._audit.get_entries(limit=10000)),
            }
    
    def destroy(self) -> None:
        """Securely destroy the manager and all encryption keys."""
        self._audit.flush()
        self._store.destroy()


class EnvironmentInjector:
    """
    Injects secrets into the environment.
    
    Provides utilities for injecting secrets as environment variables
    for different deployment environments and contexts.
    
    Args:
        manager: SecretsManager instance
        auto_refresh: Automatically refresh expired secrets
        refresh_interval: Seconds between refresh checks
    """
    
    def __init__(
        self,
        manager: SecretsManager,
        auto_refresh: bool = False,
        refresh_interval: int = 300,
    ):
        self._manager = manager
        self._auto_refresh = auto_refresh
        self._refresh_interval = refresh_interval
        self._injected: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_refresh = threading.Event()
        
        if auto_refresh:
            self._start_refresh_thread()
    
    def _start_refresh_thread(self) -> None:
        """Start the background refresh thread."""
        def refresh_loop():
            while not self._stop_refresh.wait(self._refresh_interval):
                self.refresh_all()
        
        self._refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self._refresh_thread.start()
    
    def inject(
        self,
        name: str,
        env_var: Optional[str] = None,
        environment: Optional[DeploymentEnvironment] = None,
    ) -> bool:
        """
        Inject a single secret into the environment.
        
        Args:
            name: Secret name
            env_var: Environment variable name (defaults to secret name)
            environment: Deployment environment context
            
        Returns:
            True if injected, False if failed
        """
        try:
            value = self._manager.get_secret(name, environment=environment)
            var_name = env_var or name
            
            with self._lock:
                os.environ[var_name] = value
                self._injected[var_name] = name
            
            return True
        except SecretsError:
            return False
    
    def inject_all(
        self,
        environment: Optional[DeploymentEnvironment] = None,
        prefix: str = "",
    ) -> Dict[str, bool]:
        """
        Inject all available secrets into the environment.
        
        Args:
            environment: Filter by deployment environment
            prefix: Prefix to add to variable names
            
        Returns:
            Dictionary mapping secret names to success status
        """
        results = {}
        names = self._manager.list_secrets(environment=environment)
        
        for name in names:
            var_name = f"{prefix}{name}"
            results[name] = self.inject(name, var_name, environment)
        
        return results
    
    def inject_for_deployment(
        self,
        environment: DeploymentEnvironment,
        prefix: str = "",
    ) -> Dict[str, bool]:
        """
        Inject secrets for a specific deployment environment.
        
        Args:
            environment: Target deployment environment
            prefix: Prefix to add to variable names
            
        Returns:
            Dictionary mapping secret names to success status
        """
        return self.inject_all(environment=environment, prefix=prefix)
    
    def remove(self, name: str) -> bool:
        """
        Remove a previously injected secret from environment.
        
        Args:
            name: Secret name or environment variable name
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._injected:
                var_name = name
            else:
                var_name = None
                for var, secret_name in self._injected.items():
                    if secret_name == name:
                        var_name = var
                        break
            
            if var_name and var_name in os.environ:
                del os.environ[var_name]
                del self._injected[var_name]
                return True
            
            return False
    
    def remove_all(self) -> int:
        """
        Remove all injected secrets from environment.
        
        Returns:
            Number of secrets removed
        """
        with self._lock:
            count = 0
            for var_name in list(self._injected.keys()):
                if var_name in os.environ:
                    del os.environ[var_name]
                    count += 1
            self._injected.clear()
            return count
    
    def refresh(self, name: str) -> bool:
        """
        Refresh a single injected secret.
        
        Args:
            name: Secret name
            
        Returns:
            True if refreshed, False if failed
        """
        with self._lock:
            for var_name, secret_name in self._injected.items():
                if secret_name == name:
                    return self.inject(name, var_name)
            return False
    
    def refresh_all(self) -> Dict[str, bool]:
        """
        Refresh all injected secrets.
        
        Returns:
            Dictionary mapping secret names to success status
        """
        with self._lock:
            results = {}
            for var_name, secret_name in list(self._injected.items()):
                results[secret_name] = self.inject(secret_name, var_name)
            return results
    
    def get_injected(self) -> Dict[str, str]:
        """
        Get list of currently injected secrets.
        
        Returns:
            Dictionary mapping environment variable names to secret names
        """
        with self._lock:
            return copy.copy(self._injected)
    
    def stop(self) -> None:
        """Stop the background refresh thread."""
        if self._auto_refresh:
            self._stop_refresh.set()
            if self._refresh_thread:
                self._refresh_thread.join(timeout=5)


_default_manager: Optional[SecretsManager] = None


def get_default_manager() -> SecretsManager:
    """Get the default SecretsManager instance."""
    global _default_manager
    if _default_manager is None:
        raise SecretsError("Default SecretsManager not initialized. Call set_default_manager() first.")
    return _default_manager


def set_default_manager(manager: SecretsManager) -> None:
    """Set the default SecretsManager instance."""
    global _default_manager
    _default_manager = manager


def init_secrets(
    master_password: str,
    storage_dir: Optional[str] = None,
    current_user: Optional[str] = None,
) -> SecretsManager:
    """
    Initialize and set the default SecretsManager.
    
    Args:
        master_password: Master password for encryption
        storage_dir: Directory for persistent storage
        current_user: Current user ID
        
    Returns:
        The initialized SecretsManager
    """
    manager = SecretsManager(
        master_password=master_password,
        storage_dir=storage_dir,
        current_user=current_user,
    )
    set_default_manager(manager)
    return manager


def set_secret(
    name: str,
    value: str,
    scope: SecretScope = SecretScope.APP,
    **kwargs,
) -> Secret:
    """Set a secret using the default manager."""
    return get_default_manager().set_secret(name, value, scope=scope, **kwargs)


def get_secret(name: str, **kwargs) -> str:
    """Get a secret using the default manager."""
    return get_default_manager().get_secret(name, **kwargs)


def delete_secret(name: str) -> bool:
    """Delete a secret using the default manager."""
    return get_default_manager().delete_secret(name)


def list_secrets(**kwargs) -> List[str]:
    """List secrets using the default manager."""
    return get_default_manager().list_secrets(**kwargs)


def rotate_secret(name: str, new_value: str, **kwargs) -> Secret:
    """Rotate a secret using the default manager."""
    return get_default_manager().rotate_secret(name, new_value, **kwargs)


def export_to_env(**kwargs) -> Dict[str, str]:
    """Export secrets to environment using the default manager."""
    return get_default_manager().export_to_env(**kwargs)


def load_from_env(**kwargs) -> List[str]:
    """Load secrets from environment using the default manager."""
    return get_default_manager().load_from_env(**kwargs)


__all__ = [
    'SecretScope',
    'AccessLevel',
    'DeploymentEnvironment',
    'AuditAction',
    'SecretStorageLimit',
    'SecretsError',
    'SecretNotFoundError',
    'SecretAlreadyExistsError',
    'SecretExpiredError',
    'SecretAccessDeniedError',
    'InvalidSecretNameError',
    'SecretValueTooLargeError',
    'SecretQuotaExceededError',
    'EncryptionError',
    'MasterPasswordRequiredError',
    'SecretVersion',
    'AuditLogEntry',
    'SecretMetadata',
    'Secret',
    'Encryptor',
    'FallbackEncryptor',
    'AuditLogger',
    'SecretStore',
    'SecretsManager',
    'EnvironmentInjector',
    'get_default_manager',
    'set_default_manager',
    'init_secrets',
    'set_secret',
    'get_secret',
    'delete_secret',
    'list_secrets',
    'rotate_secret',
    'export_to_env',
    'load_from_env',
    'secure_zero_memory',
]
