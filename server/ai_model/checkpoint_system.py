"""
Checkpoint and Rollback System for Platform Forge

This module provides a comprehensive checkpoint and rollback system enabling easy
recovery from mistakes. It includes efficient file snapshotting, database state
capture, automatic checkpoint creation, and visual diff capabilities.

Key Components:
- Checkpoint: Dataclass storing checkpoint state (files, database, metadata)
- CheckpointManager: Create, restore, list, and manage checkpoints
- FileSnapshot: Efficient file state capture with hash-based deduplication
- RestoreResult: Detailed results from restore operations
- AutoCheckpoint: Rules for automatic checkpoint creation
- DiffViewer: Visual differences between current state and checkpoints

Usage:
    from server.ai_model.checkpoint_system import (
        CheckpointManager,
        quick_checkpoint,
        rollback_to_last,
        get_recovery_options,
    )
    
    # Create a checkpoint manager
    manager = CheckpointManager(base_path="/project")
    
    # Quick checkpoint creation
    checkpoint = quick_checkpoint("Before major refactor")
    
    # Restore to last checkpoint
    result = rollback_to_last()
    
    # Get recovery suggestions for an error
    options = get_recovery_options(error)
"""

import os
import re
import json
import gzip
import shutil
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import difflib
import uuid
import time
import threading


class CheckpointType(Enum):
    """Types of checkpoints."""
    MANUAL = "manual"
    AUTO = "auto"
    SCHEDULED = "scheduled"
    PRE_DESTRUCTIVE = "pre_destructive"
    PRE_DEPLOYMENT = "pre_deployment"
    PRE_GIT = "pre_git"
    
    def __str__(self) -> str:
        return self.value


class TriggerType(Enum):
    """Types of triggers for auto-checkpoints."""
    FILE_DELETE = "file_delete"
    FILE_BULK_MODIFY = "file_bulk_modify"
    DATABASE_DROP = "database_drop"
    DATABASE_TRUNCATE = "database_truncate"
    DATABASE_ALTER = "database_alter"
    DATABASE_DELETE = "database_delete"
    GIT_FORCE_PUSH = "git_force_push"
    GIT_RESET = "git_reset"
    GIT_CLEAN = "git_clean"
    DEPLOYMENT = "deployment"
    CONFIG_CHANGE = "config_change"
    DEPENDENCY_UPDATE = "dependency_update"
    SCHEMA_MIGRATION = "schema_migration"
    
    def __str__(self) -> str:
        return self.value


class CompressionLevel(Enum):
    """Compression levels for snapshots."""
    NONE = 0
    FAST = 1
    BALANCED = 6
    MAXIMUM = 9


class DiffType(Enum):
    """Types of differences in files."""
    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    BINARY_CHANGED = "binary_changed"


@dataclass
class FileState:
    """Represents the state of a single file."""
    path: str
    content_hash: str
    size_bytes: int
    modified_time: float
    permissions: int
    is_binary: bool = False
    content: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "modified_time": self.modified_time,
            "permissions": self.permissions,
            "is_binary": self.is_binary,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileState":
        return cls(
            path=data["path"],
            content_hash=data["content_hash"],
            size_bytes=data["size_bytes"],
            modified_time=data["modified_time"],
            permissions=data["permissions"],
            is_binary=data.get("is_binary", False),
        )


@dataclass
class DatabaseState:
    """Represents the state of database tables."""
    tables: Dict[str, int]
    schema_hash: str
    row_counts: Dict[str, int]
    timestamp: float
    connection_info: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": self.tables,
            "schema_hash": self.schema_hash,
            "row_counts": self.row_counts,
            "timestamp": self.timestamp,
            "connection_info": self.connection_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseState":
        return cls(
            tables=data["tables"],
            schema_hash=data["schema_hash"],
            row_counts=data["row_counts"],
            timestamp=data["timestamp"],
            connection_info=data.get("connection_info", {}),
        )
    
    @classmethod
    def empty(cls) -> "DatabaseState":
        return cls(
            tables={},
            schema_hash="",
            row_counts={},
            timestamp=time.time(),
        )


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint."""
    action_that_triggered: str
    user_confirmed: bool = False
    trigger_type: Optional[TriggerType] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    created_by: str = "system"
    parent_checkpoint_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_that_triggered": self.action_that_triggered,
            "user_confirmed": self.user_confirmed,
            "trigger_type": self.trigger_type.value if self.trigger_type else None,
            "tags": self.tags,
            "notes": self.notes,
            "created_by": self.created_by,
            "parent_checkpoint_id": self.parent_checkpoint_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(
            action_that_triggered=data["action_that_triggered"],
            user_confirmed=data.get("user_confirmed", False),
            trigger_type=TriggerType(data["trigger_type"]) if data.get("trigger_type") else None,
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            created_by=data.get("created_by", "system"),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
        )


@dataclass
class Checkpoint:
    """
    Represents a complete checkpoint of system state.
    
    Attributes:
        id: Unique identifier for the checkpoint
        timestamp: When the checkpoint was created
        description: Human-readable description
        files_snapshot: Snapshot of file states
        database_state: State of database tables
        metadata: Additional metadata about the checkpoint
        size_bytes: Total size of the checkpoint data
        restore_time_estimate: Estimated seconds to restore
    """
    id: str
    timestamp: float
    description: str
    files_snapshot: Dict[str, FileState]
    database_state: DatabaseState
    metadata: CheckpointMetadata
    size_bytes: int = 0
    restore_time_estimate: float = 0.0
    checkpoint_type: CheckpointType = CheckpointType.MANUAL
    is_compressed: bool = False
    storage_path: Optional[str] = None
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self._calculate_size()
        if self.restore_time_estimate == 0.0:
            self._estimate_restore_time()
    
    def _calculate_size(self):
        """Calculate total size of checkpoint data."""
        total = 0
        for file_state in self.files_snapshot.values():
            total += file_state.size_bytes
        self.size_bytes = total
    
    def _estimate_restore_time(self):
        """Estimate time to restore this checkpoint."""
        file_count = len(self.files_snapshot)
        mb_size = self.size_bytes / (1024 * 1024)
        self.restore_time_estimate = (file_count * 0.01) + (mb_size * 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "description": self.description,
            "files_snapshot": {k: v.to_dict() for k, v in self.files_snapshot.items()},
            "database_state": self.database_state.to_dict(),
            "metadata": self.metadata.to_dict(),
            "size_bytes": self.size_bytes,
            "restore_time_estimate": self.restore_time_estimate,
            "checkpoint_type": self.checkpoint_type.value,
            "is_compressed": self.is_compressed,
            "storage_path": self.storage_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            description=data["description"],
            files_snapshot={k: FileState.from_dict(v) for k, v in data["files_snapshot"].items()},
            database_state=DatabaseState.from_dict(data["database_state"]),
            metadata=CheckpointMetadata.from_dict(data["metadata"]),
            size_bytes=data.get("size_bytes", 0),
            restore_time_estimate=data.get("restore_time_estimate", 0.0),
            checkpoint_type=CheckpointType(data.get("checkpoint_type", "manual")),
            is_compressed=data.get("is_compressed", False),
            storage_path=data.get("storage_path"),
        )
    
    def get_age(self) -> timedelta:
        """Get the age of this checkpoint."""
        return timedelta(seconds=time.time() - self.timestamp)
    
    def get_file_count(self) -> int:
        """Get number of files in snapshot."""
        return len(self.files_snapshot)
    
    def get_formatted_size(self) -> str:
        """Get human-readable size string."""
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        elif self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        elif self.size_bytes < 1024 * 1024 * 1024:
            return f"{self.size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{self.size_bytes / (1024 * 1024 * 1024):.2f} GB"


@dataclass
class RestoreResult:
    """
    Result of a checkpoint restore operation.
    
    Attributes:
        success: Whether the restore completed successfully
        restored_files: List of files that were restored
        errors: List of errors encountered during restore
        warnings: List of warnings (non-fatal issues)
        time_taken: Seconds taken to complete restore
        rollback_checkpoint: Checkpoint created before restore (for recovery)
    """
    success: bool
    restored_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    time_taken: float = 0.0
    rollback_checkpoint: Optional[str] = None
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    database_restored: bool = False
    partial_restore: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "restored_files": self.restored_files,
            "errors": self.errors,
            "warnings": self.warnings,
            "time_taken": self.time_taken,
            "rollback_checkpoint": self.rollback_checkpoint,
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "database_restored": self.database_restored,
            "partial_restore": self.partial_restore,
        }
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        if self.partial_restore:
            status = "PARTIAL"
        
        summary = [
            f"Restore Status: {status}",
            f"Time taken: {self.time_taken:.2f}s",
            f"Files restored: {len(self.restored_files)}",
            f"  - Added: {self.files_added}",
            f"  - Modified: {self.files_modified}",
            f"  - Deleted: {self.files_deleted}",
        ]
        
        if self.database_restored:
            summary.append("Database state: Restored")
        
        if self.errors:
            summary.append(f"Errors: {len(self.errors)}")
            for error in self.errors[:3]:
                summary.append(f"  - {error}")
        
        if self.warnings:
            summary.append(f"Warnings: {len(self.warnings)}")
        
        if self.rollback_checkpoint:
            summary.append(f"Rollback checkpoint: {self.rollback_checkpoint}")
        
        return "\n".join(summary)


@dataclass
class FileDiff:
    """Represents differences in a single file."""
    path: str
    diff_type: DiffType
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    old_size: int = 0
    new_size: int = 0
    diff_lines: List[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "diff_type": self.diff_type.value,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "old_size": self.old_size,
            "new_size": self.new_size,
            "additions": self.additions,
            "deletions": self.deletions,
        }


@dataclass
class DatabaseDiff:
    """Represents differences in database state."""
    tables_added: List[str] = field(default_factory=list)
    tables_removed: List[str] = field(default_factory=list)
    row_count_changes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    schema_changed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables_added": self.tables_added,
            "tables_removed": self.tables_removed,
            "row_count_changes": self.row_count_changes,
            "schema_changed": self.schema_changed,
        }


@dataclass
class DiffResult:
    """Complete diff between current state and checkpoint."""
    checkpoint_id: str
    checkpoint_timestamp: float
    file_diffs: List[FileDiff] = field(default_factory=list)
    database_diff: Optional[DatabaseDiff] = None
    total_additions: int = 0
    total_deletions: int = 0
    files_changed: int = 0
    files_added: int = 0
    files_deleted: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_timestamp": self.checkpoint_timestamp,
            "file_diffs": [d.to_dict() for d in self.file_diffs],
            "database_diff": self.database_diff.to_dict() if self.database_diff else None,
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "files_changed": self.files_changed,
            "files_added": self.files_added,
            "files_deleted": self.files_deleted,
        }
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Diff from checkpoint: {self.checkpoint_id}",
            f"Checkpoint time: {datetime.fromtimestamp(self.checkpoint_timestamp).isoformat()}",
            f"",
            f"Files changed: {self.files_changed}",
            f"Files added: {self.files_added}",
            f"Files deleted: {self.files_deleted}",
            f"",
            f"Total lines added: +{self.total_additions}",
            f"Total lines deleted: -{self.total_deletions}",
        ]
        
        if self.database_diff:
            lines.append("")
            lines.append("Database changes:")
            if self.database_diff.tables_added:
                lines.append(f"  Tables added: {', '.join(self.database_diff.tables_added)}")
            if self.database_diff.tables_removed:
                lines.append(f"  Tables removed: {', '.join(self.database_diff.tables_removed)}")
            if self.database_diff.schema_changed:
                lines.append("  Schema: Modified")
        
        return "\n".join(lines)


@dataclass
class RetentionPolicy:
    """Configuration for checkpoint retention."""
    keep_last_n: int = 10
    keep_for_days: int = 30
    keep_manual_forever: bool = True
    keep_pre_destructive: bool = True
    max_total_size_mb: int = 1000
    compress_after_days: int = 7
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "keep_last_n": self.keep_last_n,
            "keep_for_days": self.keep_for_days,
            "keep_manual_forever": self.keep_manual_forever,
            "keep_pre_destructive": self.keep_pre_destructive,
            "max_total_size_mb": self.max_total_size_mb,
            "compress_after_days": self.compress_after_days,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetentionPolicy":
        return cls(**data)


@dataclass
class AutoCheckpointRule:
    """Rule for automatic checkpoint creation."""
    trigger: TriggerType
    enabled: bool = True
    description: str = ""
    pattern: Optional[str] = None
    min_files_affected: int = 1
    cooldown_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger.value,
            "enabled": self.enabled,
            "description": self.description,
            "pattern": self.pattern,
            "min_files_affected": self.min_files_affected,
            "cooldown_seconds": self.cooldown_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoCheckpointRule":
        return cls(
            trigger=TriggerType(data["trigger"]),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            pattern=data.get("pattern"),
            min_files_affected=data.get("min_files_affected", 1),
            cooldown_seconds=data.get("cooldown_seconds", 60),
        )


class FileSnapshot:
    """
    Captures file states efficiently using hash-based deduplication.
    
    Features:
    - Content-addressable storage
    - Incremental snapshots (only changed files)
    - Compression for older snapshots
    """
    
    def __init__(
        self,
        base_path: str,
        storage_path: str,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
    ):
        self.base_path = Path(base_path)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.exclude_patterns = exclude_patterns or [
            r'\.git/',
            r'node_modules/',
            r'__pycache__/',
            r'\.pyc$',
            r'\.pyo$',
            r'\.egg-info/',
            r'dist/',
            r'build/',
            r'\.DS_Store',
            r'\.env$',
            r'\.venv/',
            r'venv/',
        ]
        self.include_patterns = include_patterns
        
        self._content_store = self.storage_path / "content"
        self._content_store.mkdir(exist_ok=True)
        
        self._hash_index: Dict[str, str] = {}
        self._load_hash_index()
    
    def _load_hash_index(self):
        """Load the hash index from disk."""
        index_file = self.storage_path / "hash_index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    self._hash_index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._hash_index = {}
    
    def _save_hash_index(self):
        """Save the hash index to disk."""
        index_file = self.storage_path / "hash_index.json"
        with open(index_file, 'w') as f:
            json.dump(self._hash_index, f)
    
    def _should_include(self, path: Path) -> bool:
        """Check if a path should be included in the snapshot."""
        rel_path = str(path.relative_to(self.base_path))
        
        for pattern in self.exclude_patterns:
            if re.search(pattern, rel_path):
                return False
        
        if self.include_patterns:
            for pattern in self.include_patterns:
                if re.search(pattern, rel_path):
                    return True
            return False
        
        return True
    
    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()
    
    def _is_binary(self, content: bytes) -> bool:
        """Check if content is binary."""
        if not content:
            return False
        try:
            content[:8192].decode('utf-8')
            return False
        except UnicodeDecodeError:
            return True
    
    def _store_content(self, content_hash: str, content: bytes) -> str:
        """Store content using content-addressable storage."""
        if content_hash in self._hash_index:
            return self._hash_index[content_hash]
        
        prefix = content_hash[:2]
        content_dir = self._content_store / prefix
        content_dir.mkdir(exist_ok=True)
        
        content_file = content_dir / content_hash
        with open(content_file, 'wb') as f:
            f.write(content)
        
        self._hash_index[content_hash] = str(content_file)
        return str(content_file)
    
    def _get_content(self, content_hash: str) -> Optional[bytes]:
        """Retrieve content by hash."""
        if content_hash not in self._hash_index:
            prefix = content_hash[:2]
            content_file = self._content_store / prefix / content_hash
            if content_file.exists():
                self._hash_index[content_hash] = str(content_file)
            else:
                return None
        
        content_path = Path(self._hash_index[content_hash])
        if content_path.exists():
            with open(content_path, 'rb') as f:
                return f.read()
        return None
    
    def capture(
        self,
        previous_snapshot: Optional[Dict[str, FileState]] = None,
        store_content: bool = True,
    ) -> Dict[str, FileState]:
        """
        Capture current file states.
        
        Args:
            previous_snapshot: Previous snapshot for incremental capture
            store_content: Whether to store file contents
            
        Returns:
            Dictionary mapping file paths to FileState objects
        """
        snapshot: Dict[str, FileState] = {}
        
        for path in self.base_path.rglob('*'):
            if not path.is_file():
                continue
            
            if not self._should_include(path):
                continue
            
            try:
                rel_path = str(path.relative_to(self.base_path))
                stat = path.stat()
                
                if previous_snapshot and rel_path in previous_snapshot:
                    prev_state = previous_snapshot[rel_path]
                    if prev_state.modified_time == stat.st_mtime:
                        snapshot[rel_path] = prev_state
                        continue
                
                with open(path, 'rb') as f:
                    content = f.read()
                
                content_hash = self._compute_hash(content)
                is_binary = self._is_binary(content)
                
                if store_content:
                    self._store_content(content_hash, content)
                
                snapshot[rel_path] = FileState(
                    path=rel_path,
                    content_hash=content_hash,
                    size_bytes=len(content),
                    modified_time=stat.st_mtime,
                    permissions=stat.st_mode,
                    is_binary=is_binary,
                )
                
            except (IOError, OSError) as e:
                continue
        
        self._save_hash_index()
        return snapshot
    
    def restore(
        self,
        snapshot: Dict[str, FileState],
        target_path: Optional[Path] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Restore files from a snapshot.
        
        Args:
            snapshot: Snapshot to restore from
            target_path: Target directory (defaults to base_path)
            
        Returns:
            Tuple of (restored_files, errors)
        """
        target = target_path or self.base_path
        restored: List[str] = []
        errors: List[str] = []
        
        for rel_path, file_state in snapshot.items():
            try:
                content = self._get_content(file_state.content_hash)
                if content is None:
                    errors.append(f"Content not found for {rel_path}")
                    continue
                
                full_path = target / rel_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(full_path, 'wb') as f:
                    f.write(content)
                
                os.chmod(full_path, file_state.permissions)
                
                restored.append(rel_path)
                
            except (IOError, OSError) as e:
                errors.append(f"Error restoring {rel_path}: {str(e)}")
        
        return restored, errors
    
    def compress_snapshot(
        self,
        snapshot: Dict[str, FileState],
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> int:
        """
        Compress a snapshot to a single file.
        
        Args:
            snapshot: Snapshot to compress
            output_path: Output file path
            level: Compression level
            
        Returns:
            Compressed size in bytes
        """
        data = {
            "snapshot": {k: v.to_dict() for k, v in snapshot.items()},
            "contents": {},
        }
        
        for rel_path, file_state in snapshot.items():
            content = self._get_content(file_state.content_hash)
            if content:
                if not file_state.is_binary:
                    try:
                        data["contents"][file_state.content_hash] = content.decode('utf-8')
                    except UnicodeDecodeError:
                        import base64
                        data["contents"][file_state.content_hash] = {
                            "binary": True,
                            "data": base64.b64encode(content).decode('ascii'),
                        }
                else:
                    import base64
                    data["contents"][file_state.content_hash] = {
                        "binary": True,
                        "data": base64.b64encode(content).decode('ascii'),
                    }
        
        json_data = json.dumps(data).encode('utf-8')
        
        with gzip.open(output_path, 'wb', compresslevel=level.value) as f:
            f.write(json_data)
        
        return output_path.stat().st_size
    
    def decompress_snapshot(self, input_path: Path) -> Dict[str, FileState]:
        """
        Decompress a snapshot from a compressed file.
        
        Args:
            input_path: Compressed file path
            
        Returns:
            Decompressed snapshot
        """
        with gzip.open(input_path, 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
        
        for content_hash, content_data in data.get("contents", {}).items():
            if isinstance(content_data, dict) and content_data.get("binary"):
                import base64
                content = base64.b64decode(content_data["data"])
            else:
                content = content_data.encode('utf-8')
            
            self._store_content(content_hash, content)
        
        snapshot = {k: FileState.from_dict(v) for k, v in data["snapshot"].items()}
        
        self._save_hash_index()
        return snapshot
    
    def get_diff(
        self,
        old_snapshot: Dict[str, FileState],
        new_snapshot: Dict[str, FileState],
        include_content: bool = False,
    ) -> List[FileDiff]:
        """
        Get differences between two snapshots.
        
        Args:
            old_snapshot: Previous snapshot
            new_snapshot: Current snapshot
            include_content: Whether to include diff content
            
        Returns:
            List of FileDiff objects
        """
        diffs: List[FileDiff] = []
        
        all_paths = set(old_snapshot.keys()) | set(new_snapshot.keys())
        
        for path in all_paths:
            old_state = old_snapshot.get(path)
            new_state = new_snapshot.get(path)
            
            if old_state is None:
                diffs.append(FileDiff(
                    path=path,
                    diff_type=DiffType.ADDED,
                    new_hash=new_state.content_hash if new_state else None,
                    new_size=new_state.size_bytes if new_state else 0,
                ))
            elif new_state is None:
                diffs.append(FileDiff(
                    path=path,
                    diff_type=DiffType.DELETED,
                    old_hash=old_state.content_hash,
                    old_size=old_state.size_bytes,
                ))
            elif old_state.content_hash != new_state.content_hash:
                diff = FileDiff(
                    path=path,
                    diff_type=DiffType.BINARY_CHANGED if old_state.is_binary or new_state.is_binary else DiffType.MODIFIED,
                    old_hash=old_state.content_hash,
                    new_hash=new_state.content_hash,
                    old_size=old_state.size_bytes,
                    new_size=new_state.size_bytes,
                )
                
                if include_content and not (old_state.is_binary or new_state.is_binary):
                    old_content = self._get_content(old_state.content_hash)
                    new_content = self._get_content(new_state.content_hash)
                    
                    if old_content and new_content:
                        try:
                            old_lines = old_content.decode('utf-8').splitlines(keepends=True)
                            new_lines = new_content.decode('utf-8').splitlines(keepends=True)
                            
                            differ = difflib.unified_diff(old_lines, new_lines, lineterm='')
                            diff.diff_lines = list(differ)
                            diff.additions = sum(1 for line in diff.diff_lines if line.startswith('+') and not line.startswith('+++'))
                            diff.deletions = sum(1 for line in diff.diff_lines if line.startswith('-') and not line.startswith('---'))
                        except UnicodeDecodeError:
                            pass
                
                diffs.append(diff)
        
        return diffs


class AutoCheckpoint:
    """
    Manages automatic checkpoint creation based on configurable rules.
    
    Features:
    - Trigger-based checkpoint creation
    - Configurable rules for different actions
    - Cooldown to prevent checkpoint spam
    """
    
    DEFAULT_RULES: List[AutoCheckpointRule] = [
        AutoCheckpointRule(
            trigger=TriggerType.FILE_DELETE,
            description="Before file deletions",
            min_files_affected=1,
            cooldown_seconds=30,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.FILE_BULK_MODIFY,
            description="Before bulk file modifications",
            min_files_affected=5,
            cooldown_seconds=60,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.DATABASE_DROP,
            description="Before dropping database objects",
            cooldown_seconds=10,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.DATABASE_TRUNCATE,
            description="Before truncating tables",
            cooldown_seconds=10,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.DATABASE_ALTER,
            description="Before altering database schema",
            cooldown_seconds=30,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.DATABASE_DELETE,
            description="Before bulk data deletions",
            cooldown_seconds=30,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.GIT_FORCE_PUSH,
            description="Before git force push",
            cooldown_seconds=10,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.GIT_RESET,
            description="Before git reset --hard",
            cooldown_seconds=10,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.GIT_CLEAN,
            description="Before git clean",
            cooldown_seconds=10,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.DEPLOYMENT,
            description="Before deployment",
            cooldown_seconds=300,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.CONFIG_CHANGE,
            description="Before config changes",
            pattern=r'\.(env|config|yml|yaml|json|toml)$',
            cooldown_seconds=60,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.DEPENDENCY_UPDATE,
            description="Before dependency updates",
            pattern=r'(package\.json|requirements\.txt|Cargo\.toml|go\.mod)$',
            cooldown_seconds=120,
        ),
        AutoCheckpointRule(
            trigger=TriggerType.SCHEMA_MIGRATION,
            description="Before database migrations",
            cooldown_seconds=60,
        ),
    ]
    
    def __init__(
        self,
        rules: Optional[List[AutoCheckpointRule]] = None,
        enabled: bool = True,
    ):
        self.rules = rules if rules is not None else self.DEFAULT_RULES.copy()
        self.enabled = enabled
        self._last_trigger_times: Dict[TriggerType, float] = {}
        self._lock = threading.Lock()
    
    def should_create_checkpoint(
        self,
        trigger: TriggerType,
        affected_files: Optional[List[str]] = None,
        action_details: Optional[str] = None,
    ) -> Tuple[bool, Optional[AutoCheckpointRule]]:
        """
        Check if a checkpoint should be created for the given trigger.
        
        Args:
            trigger: The type of trigger
            affected_files: List of affected file paths
            action_details: Additional details about the action
            
        Returns:
            Tuple of (should_create, matched_rule)
        """
        if not self.enabled:
            return False, None
        
        affected_files = affected_files or []
        
        for rule in self.rules:
            if rule.trigger != trigger or not rule.enabled:
                continue
            
            if len(affected_files) < rule.min_files_affected:
                continue
            
            if rule.pattern:
                if not any(re.search(rule.pattern, f) for f in affected_files):
                    continue
            
            with self._lock:
                last_time = self._last_trigger_times.get(trigger, 0)
                current_time = time.time()
                
                if current_time - last_time < rule.cooldown_seconds:
                    continue
                
                self._last_trigger_times[trigger] = current_time
            
            return True, rule
        
        return False, None
    
    def get_rule(self, trigger: TriggerType) -> Optional[AutoCheckpointRule]:
        """Get the rule for a specific trigger type."""
        for rule in self.rules:
            if rule.trigger == trigger:
                return rule
        return None
    
    def enable_rule(self, trigger: TriggerType):
        """Enable a specific rule."""
        rule = self.get_rule(trigger)
        if rule:
            rule.enabled = True
    
    def disable_rule(self, trigger: TriggerType):
        """Disable a specific rule."""
        rule = self.get_rule(trigger)
        if rule:
            rule.enabled = False
    
    def add_rule(self, rule: AutoCheckpointRule):
        """Add a new rule."""
        for existing in self.rules:
            if existing.trigger == rule.trigger:
                self.rules.remove(existing)
                break
        self.rules.append(rule)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rules": [r.to_dict() for r in self.rules],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoCheckpoint":
        return cls(
            rules=[AutoCheckpointRule.from_dict(r) for r in data.get("rules", [])],
            enabled=data.get("enabled", True),
        )


class DiffViewer:
    """
    Shows differences between current state and checkpoints.
    
    Features:
    - File-by-file comparison
    - Database changes summary
    - Visual diff formatting
    """
    
    def __init__(self, file_snapshot: FileSnapshot):
        self.file_snapshot = file_snapshot
    
    def get_checkpoint_diff(
        self,
        checkpoint: Checkpoint,
        current_snapshot: Optional[Dict[str, FileState]] = None,
        include_content: bool = True,
    ) -> DiffResult:
        """
        Get differences between current state and a checkpoint.
        
        Args:
            checkpoint: The checkpoint to compare against
            current_snapshot: Current file snapshot (captures if not provided)
            include_content: Whether to include diff content
            
        Returns:
            DiffResult with all differences
        """
        if current_snapshot is None:
            current_snapshot = self.file_snapshot.capture(store_content=False)
        
        file_diffs = self.file_snapshot.get_diff(
            checkpoint.files_snapshot,
            current_snapshot,
            include_content=include_content,
        )
        
        result = DiffResult(
            checkpoint_id=checkpoint.id,
            checkpoint_timestamp=checkpoint.timestamp,
            file_diffs=file_diffs,
        )
        
        for diff in file_diffs:
            if diff.diff_type == DiffType.ADDED:
                result.files_added += 1
            elif diff.diff_type == DiffType.DELETED:
                result.files_deleted += 1
            elif diff.diff_type in (DiffType.MODIFIED, DiffType.BINARY_CHANGED):
                result.files_changed += 1
            
            result.total_additions += diff.additions
            result.total_deletions += diff.deletions
        
        return result
    
    def format_diff(
        self,
        diff_result: DiffResult,
        color: bool = True,
        context_lines: int = 3,
        max_files: int = 50,
    ) -> str:
        """
        Format diff result for display.
        
        Args:
            diff_result: The diff result to format
            color: Whether to include ANSI color codes
            context_lines: Number of context lines around changes
            max_files: Maximum number of files to show
            
        Returns:
            Formatted diff string
        """
        lines = []
        
        if color:
            RED = "\033[91m"
            GREEN = "\033[92m"
            YELLOW = "\033[93m"
            BLUE = "\033[94m"
            RESET = "\033[0m"
        else:
            RED = GREEN = YELLOW = BLUE = RESET = ""
        
        lines.append(f"{BLUE}Checkpoint: {diff_result.checkpoint_id}{RESET}")
        lines.append(f"Time: {datetime.fromtimestamp(diff_result.checkpoint_timestamp).isoformat()}")
        lines.append("")
        lines.append(f"{YELLOW}Summary:{RESET}")
        lines.append(f"  {GREEN}+{diff_result.files_added} files added{RESET}")
        lines.append(f"  {RED}-{diff_result.files_deleted} files deleted{RESET}")
        lines.append(f"  {YELLOW}~{diff_result.files_changed} files modified{RESET}")
        lines.append(f"  {GREEN}+{diff_result.total_additions} lines{RESET} / {RED}-{diff_result.total_deletions} lines{RESET}")
        lines.append("")
        
        shown_files = 0
        for diff in diff_result.file_diffs:
            if shown_files >= max_files:
                remaining = len(diff_result.file_diffs) - max_files
                lines.append(f"\n{YELLOW}... and {remaining} more files{RESET}")
                break
            
            shown_files += 1
            
            if diff.diff_type == DiffType.ADDED:
                lines.append(f"{GREEN}+ {diff.path}{RESET}")
            elif diff.diff_type == DiffType.DELETED:
                lines.append(f"{RED}- {diff.path}{RESET}")
            elif diff.diff_type == DiffType.MODIFIED:
                lines.append(f"{YELLOW}M {diff.path}{RESET} ({GREEN}+{diff.additions}{RESET}/{RED}-{diff.deletions}{RESET})")
                
                if diff.diff_lines:
                    lines.append("  " + "-" * 40)
                    for line in diff.diff_lines[:20]:
                        if line.startswith('+') and not line.startswith('+++'):
                            lines.append(f"  {GREEN}{line.rstrip()}{RESET}")
                        elif line.startswith('-') and not line.startswith('---'):
                            lines.append(f"  {RED}{line.rstrip()}{RESET}")
                        elif line.startswith('@@'):
                            lines.append(f"  {BLUE}{line.rstrip()}{RESET}")
                    
                    if len(diff.diff_lines) > 20:
                        lines.append(f"  {YELLOW}... ({len(diff.diff_lines) - 20} more lines){RESET}")
                    
                    lines.append("  " + "-" * 40)
            elif diff.diff_type == DiffType.BINARY_CHANGED:
                lines.append(f"{YELLOW}B {diff.path}{RESET} (binary file changed)")
        
        if diff_result.database_diff:
            lines.append("")
            lines.append(f"{BLUE}Database Changes:{RESET}")
            
            for table in diff_result.database_diff.tables_added:
                lines.append(f"  {GREEN}+ Table: {table}{RESET}")
            
            for table in diff_result.database_diff.tables_removed:
                lines.append(f"  {RED}- Table: {table}{RESET}")
            
            for table, (old, new) in diff_result.database_diff.row_count_changes.items():
                diff_count = new - old
                if diff_count > 0:
                    lines.append(f"  {YELLOW}{table}: +{diff_count} rows{RESET}")
                else:
                    lines.append(f"  {YELLOW}{table}: {diff_count} rows{RESET}")
            
            if diff_result.database_diff.schema_changed:
                lines.append(f"  {YELLOW}Schema has been modified{RESET}")
        
        return "\n".join(lines)
    
    def get_file_diff_content(
        self,
        checkpoint: Checkpoint,
        file_path: str,
    ) -> Optional[str]:
        """
        Get detailed diff content for a specific file.
        
        Args:
            checkpoint: The checkpoint to compare against
            file_path: Path of the file to diff
            
        Returns:
            Unified diff string or None
        """
        if file_path not in checkpoint.files_snapshot:
            return None
        
        old_state = checkpoint.files_snapshot[file_path]
        old_content = self.file_snapshot._get_content(old_state.content_hash)
        
        current_path = self.file_snapshot.base_path / file_path
        if not current_path.exists():
            if old_content:
                lines = old_content.decode('utf-8', errors='replace').splitlines(keepends=True)
                return "".join(f"-{line}" for line in lines)
            return None
        
        try:
            with open(current_path, 'rb') as f:
                new_content = f.read()
        except IOError:
            return None
        
        if not old_content:
            return None
        
        try:
            old_lines = old_content.decode('utf-8').splitlines(keepends=True)
            new_lines = new_content.decode('utf-8').splitlines(keepends=True)
            
            diff = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
            return "".join(diff)
        except UnicodeDecodeError:
            return "[Binary file differs]"


class CheckpointManager:
    """
    Main class for creating, restoring, and managing checkpoints.
    
    Features:
    - Create checkpoints (manual and automatic)
    - Restore from checkpoints
    - List and delete checkpoints
    - Get diffs between current state and checkpoints
    - Configurable retention policy
    
    Usage:
        manager = CheckpointManager(base_path="/project")
        
        # Create a checkpoint
        checkpoint = manager.create_checkpoint("Before refactoring")
        
        # List checkpoints
        for cp in manager.list_checkpoints():
            print(f"{cp.id}: {cp.description}")
        
        # Restore
        result = manager.restore_checkpoint(checkpoint.id)
        print(result.get_summary())
    """
    
    def __init__(
        self,
        base_path: str,
        storage_path: Optional[str] = None,
        retention_policy: Optional[RetentionPolicy] = None,
        auto_checkpoint: Optional[AutoCheckpoint] = None,
    ):
        self.base_path = Path(base_path)
        self.storage_path = Path(storage_path) if storage_path else self.base_path / ".checkpoints"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.retention_policy = retention_policy or RetentionPolicy()
        self.auto_checkpoint = auto_checkpoint or AutoCheckpoint()
        
        self.file_snapshot = FileSnapshot(
            base_path=str(self.base_path),
            storage_path=str(self.storage_path / "files"),
        )
        
        self.diff_viewer = DiffViewer(self.file_snapshot)
        
        self._checkpoints_file = self.storage_path / "checkpoints.json"
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._load_checkpoints()
        
        self._lock = threading.Lock()
    
    def _load_checkpoints(self):
        """Load checkpoints from disk."""
        if self._checkpoints_file.exists():
            try:
                with open(self._checkpoints_file) as f:
                    data = json.load(f)
                self._checkpoints = {
                    k: Checkpoint.from_dict(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, IOError, KeyError):
                self._checkpoints = {}
    
    def _save_checkpoints(self):
        """Save checkpoints to disk."""
        with open(self._checkpoints_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self._checkpoints.items()}, f, indent=2)
    
    def _generate_id(self) -> str:
        """Generate a unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"cp_{timestamp}_{short_uuid}"
    
    def _capture_database_state(self) -> DatabaseState:
        """Capture current database state."""
        return DatabaseState.empty()
    
    def create_checkpoint(
        self,
        description: str,
        auto: bool = False,
        trigger_type: Optional[TriggerType] = None,
        tags: Optional[List[str]] = None,
        user_confirmed: bool = False,
    ) -> Checkpoint:
        """
        Create a new checkpoint.
        
        Args:
            description: Human-readable description
            auto: Whether this is an automatic checkpoint
            trigger_type: Type of trigger that caused the checkpoint
            tags: Optional tags for categorization
            user_confirmed: Whether user confirmed this action
            
        Returns:
            The created Checkpoint
        """
        with self._lock:
            checkpoint_id = self._generate_id()
            timestamp = time.time()
            
            previous_snapshot = None
            if self._checkpoints:
                most_recent = max(self._checkpoints.values(), key=lambda c: c.timestamp)
                previous_snapshot = most_recent.files_snapshot
            
            files_snapshot = self.file_snapshot.capture(previous_snapshot=previous_snapshot)
            database_state = self._capture_database_state()
            
            checkpoint_type = CheckpointType.AUTO if auto else CheckpointType.MANUAL
            if trigger_type:
                if trigger_type in (TriggerType.DATABASE_DROP, TriggerType.DATABASE_TRUNCATE,
                                   TriggerType.FILE_DELETE, TriggerType.GIT_FORCE_PUSH,
                                   TriggerType.GIT_RESET):
                    checkpoint_type = CheckpointType.PRE_DESTRUCTIVE
                elif trigger_type == TriggerType.DEPLOYMENT:
                    checkpoint_type = CheckpointType.PRE_DEPLOYMENT
                elif trigger_type in (TriggerType.GIT_CLEAN,):
                    checkpoint_type = CheckpointType.PRE_GIT
            
            parent_id = None
            if self._checkpoints:
                most_recent = max(self._checkpoints.values(), key=lambda c: c.timestamp)
                parent_id = most_recent.id
            
            metadata = CheckpointMetadata(
                action_that_triggered=description,
                user_confirmed=user_confirmed,
                trigger_type=trigger_type,
                tags=tags or [],
                parent_checkpoint_id=parent_id,
            )
            
            checkpoint = Checkpoint(
                id=checkpoint_id,
                timestamp=timestamp,
                description=description,
                files_snapshot=files_snapshot,
                database_state=database_state,
                metadata=metadata,
                checkpoint_type=checkpoint_type,
                storage_path=str(self.storage_path),
            )
            
            self._checkpoints[checkpoint_id] = checkpoint
            self._save_checkpoints()
            
            self._apply_retention_policy()
            
            return checkpoint
    
    def restore_checkpoint(
        self,
        checkpoint_id: str,
        create_rollback: bool = True,
        restore_database: bool = True,
        file_filter: Optional[Callable[[str], bool]] = None,
    ) -> RestoreResult:
        """
        Restore from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            create_rollback: Create a checkpoint before restoring
            restore_database: Whether to restore database state
            file_filter: Optional filter function for files to restore
            
        Returns:
            RestoreResult with details of the operation
        """
        start_time = time.time()
        
        if checkpoint_id not in self._checkpoints:
            return RestoreResult(
                success=False,
                errors=[f"Checkpoint not found: {checkpoint_id}"],
                time_taken=time.time() - start_time,
            )
        
        checkpoint = self._checkpoints[checkpoint_id]
        
        rollback_id = None
        if create_rollback:
            rollback_checkpoint = self.create_checkpoint(
                description=f"Pre-restore backup (restoring to {checkpoint_id})",
                auto=True,
                trigger_type=TriggerType.FILE_BULK_MODIFY,
            )
            rollback_id = rollback_checkpoint.id
        
        result = RestoreResult(
            success=True,
            rollback_checkpoint=rollback_id,
        )
        
        try:
            snapshot_to_restore = checkpoint.files_snapshot
            if file_filter:
                snapshot_to_restore = {
                    k: v for k, v in snapshot_to_restore.items()
                    if file_filter(k)
                }
            
            current_snapshot = self.file_snapshot.capture(store_content=False)
            
            for path in current_snapshot:
                if path not in snapshot_to_restore:
                    full_path = self.base_path / path
                    try:
                        if full_path.exists():
                            full_path.unlink()
                            result.files_deleted += 1
                    except (IOError, OSError) as e:
                        result.warnings.append(f"Could not delete {path}: {str(e)}")
            
            restored, errors = self.file_snapshot.restore(snapshot_to_restore)
            
            for path in restored:
                if path in current_snapshot:
                    result.files_modified += 1
                else:
                    result.files_added += 1
            
            result.restored_files = restored
            result.errors.extend(errors)
            
            if errors:
                result.partial_restore = True
                if len(errors) > len(restored):
                    result.success = False
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Restore failed: {str(e)}")
        
        result.time_taken = time.time() - start_time
        return result
    
    def list_checkpoints(
        self,
        limit: int = 10,
        checkpoint_type: Optional[CheckpointType] = None,
        tags: Optional[List[str]] = None,
        since: Optional[float] = None,
    ) -> List[Checkpoint]:
        """
        List available checkpoints.
        
        Args:
            limit: Maximum number to return
            checkpoint_type: Filter by type
            tags: Filter by tags (any match)
            since: Only checkpoints after this timestamp
            
        Returns:
            List of Checkpoint objects, newest first
        """
        checkpoints = list(self._checkpoints.values())
        
        if checkpoint_type:
            checkpoints = [c for c in checkpoints if c.checkpoint_type == checkpoint_type]
        
        if tags:
            checkpoints = [c for c in checkpoints if any(t in c.metadata.tags for t in tags)]
        
        if since:
            checkpoints = [c for c in checkpoints if c.timestamp >= since]
        
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        
        return checkpoints[:limit]
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a specific checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if checkpoint_id not in self._checkpoints:
                return False
            
            del self._checkpoints[checkpoint_id]
            self._save_checkpoints()
            return True
    
    def get_checkpoint_diff(
        self,
        checkpoint_id: str,
        include_content: bool = True,
    ) -> Optional[DiffResult]:
        """
        Get differences between current state and a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to compare
            include_content: Whether to include diff content
            
        Returns:
            DiffResult or None if checkpoint not found
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return None
        
        return self.diff_viewer.get_checkpoint_diff(
            checkpoint,
            include_content=include_content,
        )
    
    def maybe_create_auto_checkpoint(
        self,
        trigger: TriggerType,
        affected_files: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> Optional[Checkpoint]:
        """
        Create an automatic checkpoint if rules match.
        
        Args:
            trigger: The trigger type
            affected_files: List of affected files
            description: Optional description
            
        Returns:
            Created Checkpoint or None
        """
        should_create, rule = self.auto_checkpoint.should_create_checkpoint(
            trigger,
            affected_files=affected_files,
        )
        
        if not should_create or not rule:
            return None
        
        desc = description or rule.description
        return self.create_checkpoint(
            description=desc,
            auto=True,
            trigger_type=trigger,
        )
    
    def _apply_retention_policy(self):
        """Apply retention policy to clean up old checkpoints."""
        policy = self.retention_policy
        
        checkpoints = list(self._checkpoints.values())
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        
        to_delete: Set[str] = set()
        now = time.time()
        
        for i, cp in enumerate(checkpoints):
            if policy.keep_manual_forever and cp.checkpoint_type == CheckpointType.MANUAL:
                continue
            
            if policy.keep_pre_destructive and cp.checkpoint_type == CheckpointType.PRE_DESTRUCTIVE:
                continue
            
            age_days = (now - cp.timestamp) / 86400
            if age_days > policy.keep_for_days:
                to_delete.add(cp.id)
                continue
            
            auto_checkpoints = [c for c in checkpoints 
                               if c.checkpoint_type in (CheckpointType.AUTO, CheckpointType.SCHEDULED)]
            auto_checkpoints = [c for c in auto_checkpoints if c.id not in to_delete]
            
            if len(auto_checkpoints) > policy.keep_last_n:
                excess = auto_checkpoints[policy.keep_last_n:]
                for c in excess:
                    to_delete.add(c.id)
        
        with self._lock:
            for cp_id in to_delete:
                if cp_id in self._checkpoints:
                    del self._checkpoints[cp_id]
            
            if to_delete:
                self._save_checkpoints()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(cp.size_bytes for cp in self._checkpoints.values())
        
        return {
            "checkpoint_count": len(self._checkpoints),
            "total_size_bytes": total_size,
            "total_size_formatted": self._format_size(total_size),
            "by_type": {
                t.value: len([c for c in self._checkpoints.values() if c.checkpoint_type == t])
                for t in CheckpointType
            },
            "oldest": min((c.timestamp for c in self._checkpoints.values()), default=None),
            "newest": max((c.timestamp for c in self._checkpoints.values()), default=None),
        }
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format size in human-readable form."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


_default_manager: Optional[CheckpointManager] = None


def _get_default_manager() -> CheckpointManager:
    """Get or create the default checkpoint manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CheckpointManager(base_path=os.getcwd())
    return _default_manager


def set_default_manager(manager: CheckpointManager):
    """Set the default checkpoint manager."""
    global _default_manager
    _default_manager = manager


def quick_checkpoint(description: str = "Quick checkpoint") -> Checkpoint:
    """
    Create a checkpoint with minimal configuration.
    
    Args:
        description: Description for the checkpoint
        
    Returns:
        The created Checkpoint
        
    Usage:
        cp = quick_checkpoint("Before making changes")
    """
    manager = _get_default_manager()
    return manager.create_checkpoint(description=description)


def rollback_to_last() -> RestoreResult:
    """
    Restore to the most recent checkpoint.
    
    Returns:
        RestoreResult with operation details
        
    Usage:
        result = rollback_to_last()
        if result.success:
            print("Restored successfully!")
    """
    manager = _get_default_manager()
    checkpoints = manager.list_checkpoints(limit=1)
    
    if not checkpoints:
        return RestoreResult(
            success=False,
            errors=["No checkpoints available"],
        )
    
    return manager.restore_checkpoint(checkpoints[0].id)


@dataclass
class RecoveryOption:
    """A suggested recovery option."""
    checkpoint_id: str
    checkpoint_description: str
    checkpoint_age: timedelta
    relevance_score: float
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_description": self.checkpoint_description,
            "checkpoint_age_seconds": self.checkpoint_age.total_seconds(),
            "relevance_score": self.relevance_score,
            "reason": self.reason,
        }


def get_recovery_options(
    error: Optional[Exception] = None,
    affected_files: Optional[List[str]] = None,
    error_message: Optional[str] = None,
    limit: int = 5,
) -> List[RecoveryOption]:
    """
    Suggest checkpoints that might help recover from an error.
    
    Args:
        error: The exception that occurred
        affected_files: Files that might be related to the error
        error_message: Error message string
        limit: Maximum number of suggestions
        
    Returns:
        List of RecoveryOption suggestions
        
    Usage:
        try:
            run_risky_operation()
        except Exception as e:
            options = get_recovery_options(error=e)
            for opt in options:
                print(f"{opt.checkpoint_id}: {opt.reason}")
    """
    manager = _get_default_manager()
    checkpoints = manager.list_checkpoints(limit=20)
    
    if not checkpoints:
        return []
    
    error_str = ""
    if error:
        error_str = str(error).lower()
    if error_message:
        error_str += " " + error_message.lower()
    
    options: List[RecoveryOption] = []
    
    for cp in checkpoints:
        score = 0.0
        reasons = []
        
        age = cp.get_age()
        age_hours = age.total_seconds() / 3600
        
        if age_hours < 1:
            score += 0.4
            reasons.append("Very recent")
        elif age_hours < 24:
            score += 0.2
            reasons.append("Recent (within 24h)")
        
        if cp.checkpoint_type == CheckpointType.MANUAL:
            score += 0.2
            reasons.append("Manually created")
        elif cp.checkpoint_type == CheckpointType.PRE_DESTRUCTIVE:
            score += 0.3
            reasons.append("Created before destructive action")
        
        if affected_files:
            matching_files = sum(1 for f in affected_files if f in cp.files_snapshot)
            if matching_files > 0:
                score += 0.3 * (matching_files / len(affected_files))
                reasons.append(f"Contains {matching_files} affected file(s)")
        
        if error_str:
            desc_lower = cp.description.lower()
            if any(word in desc_lower for word in error_str.split()):
                score += 0.1
                reasons.append("Description may be relevant")
        
        if score > 0:
            options.append(RecoveryOption(
                checkpoint_id=cp.id,
                checkpoint_description=cp.description,
                checkpoint_age=age,
                relevance_score=min(score, 1.0),
                reason="; ".join(reasons),
            ))
    
    options.sort(key=lambda o: o.relevance_score, reverse=True)
    
    return options[:limit]


def create_checkpoint_before_action(
    trigger: TriggerType,
    affected_files: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Optional[Checkpoint]:
    """
    Create an automatic checkpoint before a potentially destructive action.
    
    Args:
        trigger: The type of action about to be performed
        affected_files: Files that will be affected
        description: Optional description
        
    Returns:
        Created Checkpoint or None if rules don't require one
        
    Usage:
        cp = create_checkpoint_before_action(
            TriggerType.FILE_DELETE,
            affected_files=["important.py"]
        )
        if cp:
            print(f"Created backup checkpoint: {cp.id}")
    """
    manager = _get_default_manager()
    return manager.maybe_create_auto_checkpoint(
        trigger=trigger,
        affected_files=affected_files,
        description=description,
    )


__all__ = [
    'CheckpointType',
    'TriggerType',
    'CompressionLevel',
    'DiffType',
    'FileState',
    'DatabaseState',
    'CheckpointMetadata',
    'Checkpoint',
    'RestoreResult',
    'FileDiff',
    'DatabaseDiff',
    'DiffResult',
    'RetentionPolicy',
    'AutoCheckpointRule',
    'FileSnapshot',
    'AutoCheckpoint',
    'DiffViewer',
    'CheckpointManager',
    'RecoveryOption',
    'quick_checkpoint',
    'rollback_to_last',
    'get_recovery_options',
    'create_checkpoint_before_action',
    'set_default_manager',
]
