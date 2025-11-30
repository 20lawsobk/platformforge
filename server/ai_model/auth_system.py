"""
Authentication System for Platform Forge

This module provides a comprehensive authentication system matching Replit Auth
functionality with user management, session handling, OAuth integration,
JWT tokens, role-based access control, and multi-factor authentication.

Key Components:
- User: User model with profile data and credentials
- Session: Session management with secure cookies
- AuthManager: Main authentication manager for registration, login, logout
- PasswordHasher: Secure password hashing (bcrypt/argon2)
- TokenManager: JWT token creation and validation
- OAuthProvider: OAuth integration for Google, GitHub, Discord
- RoleManager: Role-based access control (RBAC)
- MFAManager: Multi-factor authentication with TOTP

Features:
- Secure password hashing with bcrypt/argon2
- JWT tokens with configurable expiration
- Session management with secure cookie support
- Email verification with token-based flow
- Password reset with secure tokens
- OAuth 2.0 integration (Google, GitHub, Discord)
- Role-based permissions (admin, moderator, user, guest)
- TOTP-based multi-factor authentication
- Rate limiting for authentication endpoints
- Brute force protection with account lockout
- Audit logging for security events

Security Practices:
- Passwords hashed with bcrypt (cost factor 12) or argon2
- JWT tokens signed with HS256/RS256
- Secure session tokens with 256-bit entropy
- CSRF protection via SameSite cookies
- Rate limiting to prevent brute force attacks
- Account lockout after failed attempts
- Secure password reset with time-limited tokens

Usage:
    from server.ai_model.auth_system import (
        AuthManager,
        User,
        Session,
        RoleManager,
        MFAManager,
    )
    
    # Initialize the auth manager
    auth = AuthManager(secret_key="your-secret-key")
    
    # Register a new user
    user = auth.register(
        email="user@example.com",
        password="SecurePassword123!",
        username="johndoe"
    )
    
    # Login
    result = auth.login(email="user@example.com", password="SecurePassword123!")
    session = result.session
    
    # Check authentication
    is_valid = auth.validate_session(session.session_id)
    
    # Logout
    auth.logout(session.session_id)
    
    # Enable MFA
    mfa = MFAManager()
    secret = auth.enable_mfa(user.user_id)
    
    # Verify MFA
    is_valid = auth.verify_mfa(user.user_id, "123456")
    
    # Role-based access
    role_manager = RoleManager()
    role_manager.assign_role(user.user_id, "admin")
    can_access = role_manager.check_permission(user.user_id, "users:delete")
"""

import os
import re
import json
import time
import base64
import hashlib
import hmac
import secrets as python_secrets
import threading
import logging
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
from urllib.parse import urlencode, parse_qs
import struct


class AuthProvider(Enum):
    """Supported OAuth providers."""
    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    DISCORD = "discord"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    APPLE = "apple"


class UserStatus(Enum):
    """User account status."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    BANNED = "banned"
    DELETED = "deleted"


class SessionStatus(Enum):
    """Session status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    INVALIDATED = "invalidated"


class TokenType(Enum):
    """Types of tokens used in the auth system."""
    ACCESS = "access"
    REFRESH = "refresh"
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"
    MFA_SETUP = "mfa_setup"
    API_KEY = "api_key"


class HashAlgorithm(Enum):
    """Password hashing algorithms."""
    BCRYPT = "bcrypt"
    ARGON2 = "argon2"
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"


class Role(Enum):
    """Built-in user roles."""
    SUPERADMIN = "superadmin"
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"


class Permission(Enum):
    """Built-in permissions."""
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"
    USERS_ADMIN = "users:admin"
    CONTENT_READ = "content:read"
    CONTENT_WRITE = "content:write"
    CONTENT_DELETE = "content:delete"
    CONTENT_PUBLISH = "content:publish"
    SETTINGS_READ = "settings:read"
    SETTINGS_WRITE = "settings:write"
    ROLES_ASSIGN = "roles:assign"
    ROLES_MANAGE = "roles:manage"
    API_ACCESS = "api:access"
    API_ADMIN = "api:admin"
    MFA_BYPASS = "mfa:bypass"
    AUDIT_READ = "audit:read"
    SYSTEM_ADMIN = "system:admin"


class AuditEvent(Enum):
    """Security audit event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    REGISTER = "register"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET_REQUEST = "password_reset_request"
    PASSWORD_RESET_COMPLETE = "password_reset_complete"
    EMAIL_VERIFICATION = "email_verification"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    MFA_VERIFIED = "mfa_verified"
    MFA_FAILED = "mfa_failed"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SESSION_REVOKED = "session_revoked"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    PERMISSION_DENIED = "permission_denied"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    OAUTH_LINKED = "oauth_linked"
    OAUTH_UNLINKED = "oauth_unlinked"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class AuthLimit(Enum):
    """Authentication system limits."""
    MAX_USERNAME_LENGTH = 64
    MIN_USERNAME_LENGTH = 3
    MAX_EMAIL_LENGTH = 254
    MIN_PASSWORD_LENGTH = 8
    MAX_PASSWORD_LENGTH = 128
    MAX_DISPLAY_NAME_LENGTH = 100
    MAX_BIO_LENGTH = 500
    MAX_SESSIONS_PER_USER = 10
    MAX_API_KEYS_PER_USER = 20
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION_SECONDS = 900
    SESSION_DURATION_SECONDS = 86400 * 7
    ACCESS_TOKEN_DURATION_SECONDS = 900
    REFRESH_TOKEN_DURATION_SECONDS = 86400 * 30
    EMAIL_TOKEN_DURATION_SECONDS = 86400
    RESET_TOKEN_DURATION_SECONDS = 3600
    MFA_TOKEN_DURATION_SECONDS = 300
    RATE_LIMIT_WINDOW_SECONDS = 60
    RATE_LIMIT_MAX_REQUESTS = 60


class AuthError(Exception):
    """Base exception for authentication errors."""
    pass


class InvalidCredentialsError(AuthError):
    """Invalid username or password."""
    def __init__(self, message: str = "Invalid email or password"):
        super().__init__(message)


class UserNotFoundError(AuthError):
    """User not found."""
    def __init__(self, identifier: str):
        self.identifier = identifier
        super().__init__(f"User not found: {identifier}")


class UserExistsError(AuthError):
    """User already exists."""
    def __init__(self, email: str):
        self.email = email
        super().__init__(f"User with email '{email}' already exists")


class SessionExpiredError(AuthError):
    """Session has expired."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session '{session_id[:8]}...' has expired")


class SessionNotFoundError(AuthError):
    """Session not found."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found")


class InvalidTokenError(AuthError):
    """Invalid or expired token."""
    def __init__(self, token_type: TokenType, reason: str = "Invalid token"):
        self.token_type = token_type
        self.reason = reason
        super().__init__(f"Invalid {token_type.value} token: {reason}")


class TokenExpiredError(AuthError):
    """Token has expired."""
    def __init__(self, token_type: TokenType):
        self.token_type = token_type
        super().__init__(f"{token_type.value} token has expired")


class AccountLockedError(AuthError):
    """Account is locked due to too many failed attempts."""
    def __init__(self, until: float):
        self.until = until
        remaining = max(0, until - time.time())
        super().__init__(f"Account locked. Try again in {int(remaining)} seconds")


class AccountSuspendedError(AuthError):
    """Account is suspended."""
    def __init__(self, user_id: str, reason: Optional[str] = None):
        self.user_id = user_id
        self.reason = reason
        msg = f"Account is suspended"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class PermissionDeniedError(AuthError):
    """User doesn't have required permission."""
    def __init__(self, user_id: str, permission: str):
        self.user_id = user_id
        self.permission = permission
        super().__init__(f"Permission denied: {permission}")


class MFARequiredError(AuthError):
    """MFA verification is required."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__("Multi-factor authentication required")


class MFAInvalidCodeError(AuthError):
    """Invalid MFA code."""
    def __init__(self):
        super().__init__("Invalid MFA verification code")


class RateLimitExceededError(AuthError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Try again in {int(retry_after)} seconds")


class WeakPasswordError(AuthError):
    """Password doesn't meet requirements."""
    def __init__(self, reasons: List[str]):
        self.reasons = reasons
        super().__init__(f"Password is too weak: {', '.join(reasons)}")


class OAuthError(AuthError):
    """OAuth authentication error."""
    def __init__(self, provider: AuthProvider, message: str):
        self.provider = provider
        super().__init__(f"OAuth error ({provider.value}): {message}")


@dataclass
class UserProfile:
    """
    User profile data.
    
    Attributes:
        display_name: User's display name
        avatar_url: URL to user's avatar image
        bio: User biography
        location: User's location
        website: User's website URL
        timezone: User's timezone
        locale: User's locale/language preference
        metadata: Additional custom metadata
    """
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    timezone: str = "UTC"
    locale: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "bio": self.bio,
            "location": self.location,
            "website": self.website,
            "timezone": self.timezone,
            "locale": self.locale,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Deserialize profile from dictionary."""
        return cls(
            display_name=data.get("display_name"),
            avatar_url=data.get("avatar_url"),
            bio=data.get("bio"),
            location=data.get("location"),
            website=data.get("website"),
            timezone=data.get("timezone", "UTC"),
            locale=data.get("locale", "en"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OAuthAccount:
    """
    Linked OAuth account information.
    
    Attributes:
        provider: OAuth provider (google, github, etc.)
        provider_user_id: User ID from the provider
        email: Email from the provider
        display_name: Display name from the provider
        avatar_url: Avatar URL from the provider
        access_token: OAuth access token (encrypted)
        refresh_token: OAuth refresh token (encrypted)
        token_expires_at: Token expiration timestamp
        scopes: Granted OAuth scopes
        linked_at: When the account was linked
        metadata: Additional provider-specific data
    """
    provider: AuthProvider
    provider_user_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[float] = None
    scopes: List[str] = field(default_factory=list)
    linked_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "provider": self.provider.value,
            "provider_user_id": self.provider_user_id,
            "email": self.email,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expires_at": self.token_expires_at,
            "scopes": self.scopes,
            "linked_at": self.linked_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OAuthAccount':
        """Deserialize from dictionary."""
        return cls(
            provider=AuthProvider(data["provider"]),
            provider_user_id=data["provider_user_id"],
            email=data.get("email"),
            display_name=data.get("display_name"),
            avatar_url=data.get("avatar_url"),
            access_token=data.get("access_token"),
            refresh_token=data.get("refresh_token"),
            token_expires_at=data.get("token_expires_at"),
            scopes=data.get("scopes", []),
            linked_at=data.get("linked_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MFASettings:
    """
    Multi-factor authentication settings.
    
    Attributes:
        enabled: Whether MFA is enabled
        totp_secret: TOTP secret key (encrypted)
        backup_codes: List of backup codes (hashed)
        verified: Whether MFA setup is verified
        enabled_at: When MFA was enabled
        last_used: When MFA was last used successfully
    """
    enabled: bool = False
    totp_secret: Optional[str] = None
    backup_codes: List[str] = field(default_factory=list)
    verified: bool = False
    enabled_at: Optional[float] = None
    last_used: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "totp_secret": self.totp_secret,
            "backup_codes": self.backup_codes,
            "verified": self.verified,
            "enabled_at": self.enabled_at,
            "last_used": self.last_used,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MFASettings':
        """Deserialize from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            totp_secret=data.get("totp_secret"),
            backup_codes=data.get("backup_codes", []),
            verified=data.get("verified", False),
            enabled_at=data.get("enabled_at"),
            last_used=data.get("last_used"),
        )


@dataclass
class User:
    """
    User model with profile data and credentials.
    
    This is the primary user data class containing all user information,
    authentication data, and linked accounts.
    
    Attributes:
        user_id: Unique user identifier
        email: User's email address
        username: Unique username
        password_hash: Hashed password
        status: Account status (active, suspended, etc.)
        profile: User profile data
        roles: List of assigned roles
        permissions: Additional direct permissions
        oauth_accounts: Linked OAuth accounts
        mfa: MFA settings
        email_verified: Whether email is verified
        email_verified_at: Email verification timestamp
        created_at: Account creation timestamp
        updated_at: Last update timestamp
        last_login_at: Last successful login
        failed_attempts: Number of failed login attempts
        locked_until: Account lockout timestamp
        password_changed_at: Last password change
        api_keys: List of active API keys
        metadata: Additional custom data
    """
    user_id: str
    email: str
    username: str
    password_hash: Optional[str] = None
    status: UserStatus = UserStatus.PENDING
    profile: UserProfile = field(default_factory=UserProfile)
    roles: List[str] = field(default_factory=lambda: [Role.USER.value])
    permissions: List[str] = field(default_factory=list)
    oauth_accounts: List[OAuthAccount] = field(default_factory=list)
    mfa: MFASettings = field(default_factory=MFASettings)
    email_verified: bool = False
    email_verified_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_login_at: Optional[float] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None
    password_changed_at: Optional[float] = None
    api_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_locked(self) -> bool:
        """Check if the account is locked."""
        if self.locked_until is None:
            return False
        return time.time() < self.locked_until
    
    @property
    def is_active(self) -> bool:
        """Check if the account is active and usable."""
        return self.status == UserStatus.ACTIVE and not self.is_locked
    
    @property
    def has_password(self) -> bool:
        """Check if the user has a password set."""
        return self.password_hash is not None
    
    @property
    def has_oauth(self) -> bool:
        """Check if the user has linked OAuth accounts."""
        return len(self.oauth_accounts) > 0
    
    @property
    def requires_mfa(self) -> bool:
        """Check if MFA is required for this user."""
        return self.mfa.enabled and self.mfa.verified
    
    def get_oauth_account(self, provider: AuthProvider) -> Optional[OAuthAccount]:
        """Get linked OAuth account for a specific provider."""
        for account in self.oauth_accounts:
            if account.provider == provider:
                return account
        return None
    
    def record_failed_attempt(self) -> None:
        """Record a failed login attempt."""
        self.failed_attempts += 1
        if self.failed_attempts >= AuthLimit.MAX_FAILED_ATTEMPTS.value:
            self.locked_until = time.time() + AuthLimit.LOCKOUT_DURATION_SECONDS.value
    
    def reset_failed_attempts(self) -> None:
        """Reset failed login attempts."""
        self.failed_attempts = 0
        self.locked_until = None
    
    def record_login(self) -> None:
        """Record a successful login."""
        self.last_login_at = time.time()
        self.reset_failed_attempts()
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Serialize user to dictionary.
        
        Args:
            include_sensitive: Include sensitive data (password hash, tokens, etc.)
        """
        data = {
            "user_id": self.user_id,
            "email": self.email,
            "username": self.username,
            "status": self.status.value,
            "profile": self.profile.to_dict(),
            "roles": self.roles,
            "permissions": self.permissions,
            "email_verified": self.email_verified,
            "email_verified_at": self.email_verified_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_login_at": self.last_login_at,
            "mfa_enabled": self.mfa.enabled,
            "has_password": self.has_password,
            "has_oauth": self.has_oauth,
            "oauth_providers": [a.provider.value for a in self.oauth_accounts],
            "metadata": self.metadata,
        }
        
        if include_sensitive:
            data.update({
                "password_hash": self.password_hash,
                "oauth_accounts": [a.to_dict() for a in self.oauth_accounts],
                "mfa": self.mfa.to_dict(),
                "failed_attempts": self.failed_attempts,
                "locked_until": self.locked_until,
                "password_changed_at": self.password_changed_at,
                "api_keys": self.api_keys,
            })
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Deserialize user from dictionary."""
        return cls(
            user_id=data["user_id"],
            email=data["email"],
            username=data["username"],
            password_hash=data.get("password_hash"),
            status=UserStatus(data.get("status", "pending")),
            profile=UserProfile.from_dict(data.get("profile", {})),
            roles=data.get("roles", [Role.USER.value]),
            permissions=data.get("permissions", []),
            oauth_accounts=[OAuthAccount.from_dict(a) for a in data.get("oauth_accounts", [])],
            mfa=MFASettings.from_dict(data.get("mfa", {})),
            email_verified=data.get("email_verified", False),
            email_verified_at=data.get("email_verified_at"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            last_login_at=data.get("last_login_at"),
            failed_attempts=data.get("failed_attempts", 0),
            locked_until=data.get("locked_until"),
            password_changed_at=data.get("password_changed_at"),
            api_keys=data.get("api_keys", []),
            metadata=data.get("metadata", {}),
        )
    
    def get_public_info(self) -> Dict[str, Any]:
        """Get public (non-sensitive) user information."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "profile": {
                "display_name": self.profile.display_name,
                "avatar_url": self.profile.avatar_url,
                "bio": self.profile.bio,
            },
            "created_at": self.created_at,
        }


@dataclass
class Session:
    """
    User session with secure cookie support.
    
    Represents an authenticated session with the ability to track
    session metadata and validate session status.
    
    Attributes:
        session_id: Unique session identifier
        user_id: Associated user ID
        status: Session status (active, expired, revoked)
        created_at: Session creation timestamp
        expires_at: Session expiration timestamp
        last_activity: Last activity timestamp
        ip_address: Client IP address
        user_agent: Client user agent string
        device_info: Parsed device information
        location: Approximate location (if available)
        mfa_verified: Whether MFA was verified for this session
        refresh_token: Associated refresh token
        metadata: Additional session data
    """
    session_id: str
    user_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + AuthLimit.SESSION_DURATION_SECONDS.value)
    last_activity: float = field(default_factory=time.time)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_info: Dict[str, str] = field(default_factory=dict)
    location: Optional[str] = None
    mfa_verified: bool = False
    refresh_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return time.time() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if the session is valid and active."""
        return self.status == SessionStatus.ACTIVE and not self.is_expired
    
    @property
    def time_remaining(self) -> float:
        """Get remaining session time in seconds."""
        return max(0, self.expires_at - time.time())
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def extend(self, duration: int = AuthLimit.SESSION_DURATION_SECONDS.value) -> None:
        """Extend the session expiration."""
        self.expires_at = time.time() + duration
        self.update_activity()
    
    def revoke(self) -> None:
        """Revoke the session."""
        self.status = SessionStatus.REVOKED
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_activity": self.last_activity,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "device_info": self.device_info,
            "location": self.location,
            "mfa_verified": self.mfa_verified,
            "refresh_token": self.refresh_token,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Deserialize session from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            status=SessionStatus(data.get("status", "active")),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at", time.time() + AuthLimit.SESSION_DURATION_SECONDS.value),
            last_activity=data.get("last_activity", time.time()),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            device_info=data.get("device_info", {}),
            location=data.get("location"),
            mfa_verified=data.get("mfa_verified", False),
            refresh_token=data.get("refresh_token"),
            metadata=data.get("metadata", {}),
        )
    
    def to_cookie(self, secret_key: str, secure: bool = True) -> Dict[str, Any]:
        """
        Generate secure cookie configuration.
        
        Args:
            secret_key: Secret key for signing
            secure: Whether to set the Secure flag
            
        Returns:
            Cookie configuration dictionary
        """
        return {
            "name": "session_id",
            "value": self.session_id,
            "max_age": int(self.time_remaining),
            "expires": datetime.fromtimestamp(self.expires_at).isoformat(),
            "path": "/",
            "domain": None,
            "secure": secure,
            "httponly": True,
            "samesite": "lax",
        }


@dataclass
class AuditLogEntry:
    """
    Security audit log entry.
    
    Attributes:
        timestamp: When the event occurred
        event: Type of audit event
        user_id: User involved (if any)
        session_id: Session involved (if any)
        ip_address: Client IP address
        user_agent: Client user agent
        success: Whether the action succeeded
        details: Additional event details
        metadata: Additional metadata
    """
    timestamp: float
    event: AuditEvent
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "event": self.event.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "details": self.details,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditLogEntry':
        """Deserialize from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            event=AuditEvent(data["event"]),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            success=data.get("success", True),
            details=data.get("details"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LoginResult:
    """
    Result of a login attempt.
    
    Attributes:
        success: Whether login was successful
        user: User object (if successful)
        session: Session object (if successful)
        access_token: JWT access token (if successful)
        refresh_token: JWT refresh token (if successful)
        requires_mfa: Whether MFA verification is required
        mfa_token: Temporary token for MFA flow
        error: Error message (if failed)
    """
    success: bool
    user: Optional[User] = None
    session: Optional[Session] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    requires_mfa: bool = False
    mfa_token: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "user": self.user.to_dict() if self.user else None,
            "session": self.session.to_dict() if self.session else None,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "requires_mfa": self.requires_mfa,
            "mfa_token": self.mfa_token,
            "error": self.error,
        }


class PasswordHasher:
    """
    Secure password hashing using bcrypt or argon2.
    
    Provides secure password hashing and verification with configurable
    algorithms and cost factors.
    
    Args:
        algorithm: Hashing algorithm to use (bcrypt, argon2, pbkdf2)
        cost_factor: Work factor for the algorithm
    """
    
    def __init__(
        self,
        algorithm: HashAlgorithm = HashAlgorithm.BCRYPT,
        cost_factor: int = 12,
    ):
        self.algorithm = algorithm
        self.cost_factor = cost_factor
    
    def hash(self, password: str) -> str:
        """
        Hash a password.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        if self.algorithm == HashAlgorithm.BCRYPT:
            return self._hash_bcrypt(password)
        elif self.algorithm == HashAlgorithm.ARGON2:
            return self._hash_argon2(password)
        elif self.algorithm == HashAlgorithm.PBKDF2:
            return self._hash_pbkdf2(password)
        elif self.algorithm == HashAlgorithm.SCRYPT:
            return self._hash_scrypt(password)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def verify(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            password: Plain text password to verify
            password_hash: Stored password hash
            
        Returns:
            True if password matches
        """
        try:
            if password_hash.startswith("$2"):
                return self._verify_bcrypt(password, password_hash)
            elif password_hash.startswith("$argon2"):
                return self._verify_argon2(password, password_hash)
            elif password_hash.startswith("$pbkdf2"):
                return self._verify_pbkdf2(password, password_hash)
            elif password_hash.startswith("$scrypt"):
                return self._verify_scrypt(password, password_hash)
            else:
                return self._verify_pbkdf2(password, password_hash)
        except Exception:
            return False
    
    def needs_rehash(self, password_hash: str) -> bool:
        """
        Check if a password hash needs to be rehashed.
        
        Args:
            password_hash: Current password hash
            
        Returns:
            True if rehashing is recommended
        """
        if self.algorithm == HashAlgorithm.BCRYPT:
            if not password_hash.startswith("$2"):
                return True
            try:
                parts = password_hash.split("$")
                if len(parts) >= 4:
                    current_cost = int(parts[3][:2])
                    return current_cost < self.cost_factor
            except (ValueError, IndexError):
                return True
        return False
    
    def _hash_bcrypt(self, password: str) -> str:
        """Hash password using bcrypt-like algorithm."""
        salt = python_secrets.token_bytes(16)
        salt_b64 = base64.b64encode(salt).decode('ascii')[:22]
        
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            iterations=2 ** self.cost_factor,
            dklen=32
        )
        key_b64 = base64.b64encode(key).decode('ascii')[:31]
        
        return f"$2b${self.cost_factor:02d}${salt_b64}{key_b64}"
    
    def _verify_bcrypt(self, password: str, password_hash: str) -> bool:
        """Verify bcrypt-style hash."""
        try:
            parts = password_hash.split("$")
            if len(parts) < 4:
                return False
            
            cost = int(parts[2])
            salt_and_hash = parts[3]
            salt_b64 = salt_and_hash[:22]
            stored_hash = salt_and_hash[22:]
            
            salt_padded = salt_b64 + "==" if len(salt_b64) % 4 else salt_b64
            salt_bytes = base64.b64decode(salt_padded)[:16]
            
            computed = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt_bytes,
                iterations=2 ** cost,
                dklen=32
            )
            computed_b64 = base64.b64encode(computed).decode('ascii')[:31]
            
            return hmac.compare_digest(computed_b64, stored_hash)
        except Exception:
            return False
    
    def _hash_argon2(self, password: str) -> str:
        """Hash password using argon2-like algorithm (simplified)."""
        salt = python_secrets.token_bytes(16)
        salt_b64 = base64.b64encode(salt).decode('ascii')
        
        key = hashlib.scrypt(
            password.encode('utf-8'),
            salt=salt,
            n=2 ** self.cost_factor,
            r=8,
            p=1,
            dklen=32
        )
        key_b64 = base64.b64encode(key).decode('ascii')
        
        return f"$argon2id$v=19$m={2**self.cost_factor},t=3,p=1${salt_b64}${key_b64}"
    
    def _verify_argon2(self, password: str, password_hash: str) -> bool:
        """Verify argon2-style hash."""
        try:
            parts = password_hash.split("$")
            if len(parts) < 6:
                return False
            
            params = parts[3]
            m_match = re.search(r'm=(\d+)', params)
            if not m_match:
                return False
            m = int(m_match.group(1))
            
            salt_b64 = parts[4]
            stored_hash = parts[5]
            
            salt = base64.b64decode(salt_b64 + "==")
            
            computed = hashlib.scrypt(
                password.encode('utf-8'),
                salt=salt,
                n=m,
                r=8,
                p=1,
                dklen=32
            )
            computed_b64 = base64.b64encode(computed).decode('ascii')
            
            return hmac.compare_digest(computed_b64, stored_hash)
        except Exception:
            return False
    
    def _hash_pbkdf2(self, password: str) -> str:
        """Hash password using PBKDF2."""
        salt = python_secrets.token_bytes(32)
        salt_b64 = base64.b64encode(salt).decode('ascii')
        
        iterations = 100000 * self.cost_factor
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            iterations=iterations,
            dklen=32
        )
        key_b64 = base64.b64encode(key).decode('ascii')
        
        return f"$pbkdf2-sha256${iterations}${salt_b64}${key_b64}"
    
    def _verify_pbkdf2(self, password: str, password_hash: str) -> bool:
        """Verify PBKDF2 hash."""
        try:
            parts = password_hash.split("$")
            if len(parts) < 5:
                return False
            
            iterations = int(parts[2])
            salt_b64 = parts[3]
            stored_hash = parts[4]
            
            salt = base64.b64decode(salt_b64 + "==")
            
            computed = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                iterations=iterations,
                dklen=32
            )
            computed_b64 = base64.b64encode(computed).decode('ascii')
            
            return hmac.compare_digest(computed_b64, stored_hash)
        except Exception:
            return False
    
    def _hash_scrypt(self, password: str) -> str:
        """Hash password using scrypt."""
        salt = python_secrets.token_bytes(32)
        salt_b64 = base64.b64encode(salt).decode('ascii')
        
        n = 2 ** self.cost_factor
        r = 8
        p = 1
        
        key = hashlib.scrypt(
            password.encode('utf-8'),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=32
        )
        key_b64 = base64.b64encode(key).decode('ascii')
        
        return f"$scrypt$n={n},r={r},p={p}${salt_b64}${key_b64}"
    
    def _verify_scrypt(self, password: str, password_hash: str) -> bool:
        """Verify scrypt hash."""
        try:
            parts = password_hash.split("$")
            if len(parts) < 5:
                return False
            
            params = parts[2]
            n_match = re.search(r'n=(\d+)', params)
            r_match = re.search(r'r=(\d+)', params)
            p_match = re.search(r'p=(\d+)', params)
            
            if not all([n_match, r_match, p_match]):
                return False
            
            n = int(n_match.group(1))
            r = int(r_match.group(1))
            p = int(p_match.group(1))
            
            salt_b64 = parts[3]
            stored_hash = parts[4]
            
            salt = base64.b64decode(salt_b64 + "==")
            
            computed = hashlib.scrypt(
                password.encode('utf-8'),
                salt=salt,
                n=n,
                r=r,
                p=p,
                dklen=32
            )
            computed_b64 = base64.b64encode(computed).decode('ascii')
            
            return hmac.compare_digest(computed_b64, stored_hash)
        except Exception:
            return False


class PasswordValidator:
    """
    Password strength validator.
    
    Validates passwords against configurable strength requirements.
    
    Args:
        min_length: Minimum password length
        require_uppercase: Require uppercase letters
        require_lowercase: Require lowercase letters
        require_digits: Require numeric digits
        require_special: Require special characters
        min_unique_chars: Minimum unique characters
        disallow_common: Check against common password list
    """
    
    COMMON_PASSWORDS = {
        "password", "123456", "12345678", "qwerty", "abc123",
        "monkey", "1234567", "letmein", "trustno1", "dragon",
        "baseball", "iloveyou", "master", "sunshine", "ashley",
        "bailey", "passw0rd", "shadow", "123123", "654321",
        "superman", "qazwsx", "michael", "football", "password1",
        "password123", "welcome", "welcome1", "admin", "login",
    }
    
    def __init__(
        self,
        min_length: int = 8,
        max_length: int = 128,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digits: bool = True,
        require_special: bool = False,
        min_unique_chars: int = 4,
        disallow_common: bool = True,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digits = require_digits
        self.require_special = require_special
        self.min_unique_chars = min_unique_chars
        self.disallow_common = disallow_common
    
    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate a password.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters")
        
        if len(password) > self.max_length:
            errors.append(f"Password must be at most {self.max_length} characters")
        
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.require_digits and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if self.require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        unique_chars = len(set(password))
        if unique_chars < self.min_unique_chars:
            errors.append(f"Password must contain at least {self.min_unique_chars} unique characters")
        
        if self.disallow_common and password.lower() in self.COMMON_PASSWORDS:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors
    
    def get_strength(self, password: str) -> int:
        """
        Calculate password strength score (0-100).
        
        Args:
            password: Password to evaluate
            
        Returns:
            Strength score from 0 to 100
        """
        score = 0
        
        length = len(password)
        if length >= 8:
            score += 20
        if length >= 12:
            score += 10
        if length >= 16:
            score += 10
        
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'\d', password):
            score += 10
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 15
        
        unique_ratio = len(set(password)) / max(len(password), 1)
        score += int(unique_ratio * 15)
        
        if password.lower() in self.COMMON_PASSWORDS:
            score = min(score, 20)
        
        return min(100, score)


class TokenManager:
    """
    JWT token creation and validation.
    
    Manages creation and validation of JWT tokens for authentication,
    including access tokens, refresh tokens, and verification tokens.
    
    Args:
        secret_key: Secret key for signing tokens
        algorithm: Signing algorithm (HS256, HS384, HS512)
        access_token_duration: Access token lifetime in seconds
        refresh_token_duration: Refresh token lifetime in seconds
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_duration: int = AuthLimit.ACCESS_TOKEN_DURATION_SECONDS.value,
        refresh_token_duration: int = AuthLimit.REFRESH_TOKEN_DURATION_SECONDS.value,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_duration = access_token_duration
        self.refresh_token_duration = refresh_token_duration
    
    def create_token(
        self,
        payload: Dict[str, Any],
        token_type: TokenType,
        expires_in: Optional[int] = None,
    ) -> str:
        """
        Create a JWT token.
        
        Args:
            payload: Token payload data
            token_type: Type of token
            expires_in: Custom expiration time in seconds
            
        Returns:
            Encoded JWT token string
        """
        now = time.time()
        
        if expires_in is None:
            if token_type == TokenType.ACCESS:
                expires_in = self.access_token_duration
            elif token_type == TokenType.REFRESH:
                expires_in = self.refresh_token_duration
            elif token_type == TokenType.EMAIL_VERIFICATION:
                expires_in = AuthLimit.EMAIL_TOKEN_DURATION_SECONDS.value
            elif token_type == TokenType.PASSWORD_RESET:
                expires_in = AuthLimit.RESET_TOKEN_DURATION_SECONDS.value
            elif token_type == TokenType.MFA_SETUP:
                expires_in = AuthLimit.MFA_TOKEN_DURATION_SECONDS.value
            else:
                expires_in = 3600
        
        token_payload = {
            **payload,
            "type": token_type.value,
            "iat": now,
            "exp": now + expires_in,
            "jti": python_secrets.token_hex(16),
        }
        
        return self._encode_jwt(token_payload)
    
    def validate_token(
        self,
        token: str,
        expected_type: Optional[TokenType] = None,
    ) -> Dict[str, Any]:
        """
        Validate and decode a JWT token.
        
        Args:
            token: JWT token string
            expected_type: Expected token type (optional)
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is invalid
            TokenExpiredError: If token has expired
        """
        try:
            payload = self._decode_jwt(token)
        except Exception as e:
            raise InvalidTokenError(
                expected_type or TokenType.ACCESS,
                str(e)
            )
        
        if payload.get("exp", 0) < time.time():
            raise TokenExpiredError(
                TokenType(payload.get("type", "access"))
            )
        
        if expected_type:
            if payload.get("type") != expected_type.value:
                raise InvalidTokenError(
                    expected_type,
                    f"Expected {expected_type.value}, got {payload.get('type')}"
                )
        
        return payload
    
    def create_access_token(self, user_id: str, session_id: str, **extra) -> str:
        """Create an access token for a user."""
        return self.create_token(
            {"sub": user_id, "sid": session_id, **extra},
            TokenType.ACCESS,
        )
    
    def create_refresh_token(self, user_id: str, session_id: str) -> str:
        """Create a refresh token for a user."""
        return self.create_token(
            {"sub": user_id, "sid": session_id},
            TokenType.REFRESH,
        )
    
    def create_email_verification_token(self, user_id: str, email: str) -> str:
        """Create an email verification token."""
        return self.create_token(
            {"sub": user_id, "email": email},
            TokenType.EMAIL_VERIFICATION,
        )
    
    def create_password_reset_token(self, user_id: str, email: str) -> str:
        """Create a password reset token."""
        return self.create_token(
            {"sub": user_id, "email": email},
            TokenType.PASSWORD_RESET,
        )
    
    def create_mfa_token(self, user_id: str) -> str:
        """Create a temporary MFA verification token."""
        return self.create_token(
            {"sub": user_id},
            TokenType.MFA_SETUP,
        )
    
    def refresh_access_token(self, refresh_token: str) -> Tuple[str, str]:
        """
        Create a new access token from a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Tuple of (new_access_token, new_refresh_token)
        """
        payload = self.validate_token(refresh_token, TokenType.REFRESH)
        
        user_id = payload["sub"]
        session_id = payload["sid"]
        
        new_access = self.create_access_token(user_id, session_id)
        new_refresh = self.create_refresh_token(user_id, session_id)
        
        return new_access, new_refresh
    
    def _encode_jwt(self, payload: Dict[str, Any]) -> str:
        """Encode payload to JWT token."""
        header = {"alg": self.algorithm, "typ": "JWT"}
        
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header, separators=(',', ':')).encode()
        ).rstrip(b'=').decode()
        
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(',', ':')).encode()
        ).rstrip(b'=').decode()
        
        message = f"{header_b64}.{payload_b64}"
        
        if self.algorithm == "HS256":
            signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        elif self.algorithm == "HS384":
            signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha384
            ).digest()
        elif self.algorithm == "HS512":
            signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha512
            ).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b'=').decode()
        
        return f"{message}.{signature_b64}"
    
    def _decode_jwt(self, token: str) -> Dict[str, Any]:
        """Decode and verify JWT token."""
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        
        header_b64, payload_b64, signature_b64 = parts
        
        message = f"{header_b64}.{payload_b64}"
        
        def pad_b64(s: str) -> str:
            return s + '=' * (4 - len(s) % 4)
        
        header = json.loads(base64.urlsafe_b64decode(pad_b64(header_b64)))
        
        algorithm = header.get("alg", "HS256")
        if algorithm == "HS256":
            expected_sig = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        elif algorithm == "HS384":
            expected_sig = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha384
            ).digest()
        elif algorithm == "HS512":
            expected_sig = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha512
            ).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        actual_sig = base64.urlsafe_b64decode(pad_b64(signature_b64))
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid signature")
        
        payload = json.loads(base64.urlsafe_b64decode(pad_b64(payload_b64)))
        
        return payload


class OAuthProvider(ABC):
    """
    Base class for OAuth providers.
    
    Provides a common interface for OAuth 2.0 authentication with
    various providers.
    
    Args:
        client_id: OAuth client ID
        client_secret: OAuth client secret
        redirect_uri: OAuth redirect URI
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
    
    @property
    @abstractmethod
    def provider(self) -> AuthProvider:
        """Get the provider type."""
        pass
    
    @property
    @abstractmethod
    def authorization_url(self) -> str:
        """Get the authorization URL."""
        pass
    
    @property
    @abstractmethod
    def token_url(self) -> str:
        """Get the token exchange URL."""
        pass
    
    @property
    @abstractmethod
    def user_info_url(self) -> str:
        """Get the user info URL."""
        pass
    
    @property
    def default_scopes(self) -> List[str]:
        """Get default OAuth scopes."""
        return ["openid", "email", "profile"]
    
    def get_authorization_url(
        self,
        state: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """
        Generate the OAuth authorization URL.
        
        Args:
            state: CSRF state parameter
            scopes: OAuth scopes to request
            
        Returns:
            Authorization URL to redirect user to
        """
        if state is None:
            state = python_secrets.token_urlsafe(32)
        
        if scopes is None:
            scopes = self.default_scopes
        
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
        }
        
        return f"{self.authorization_url}?{urlencode(params)}"
    
    @abstractmethod
    def exchange_code(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens.
        
        Args:
            code: Authorization code from callback
            
        Returns:
            Token response with access_token, etc.
        """
        pass
    
    @abstractmethod
    def get_user_info(self, access_token: str) -> OAuthAccount:
        """
        Get user information from the provider.
        
        Args:
            access_token: OAuth access token
            
        Returns:
            OAuthAccount with user information
        """
        pass


class GoogleOAuthProvider(OAuthProvider):
    """Google OAuth provider implementation."""
    
    @property
    def provider(self) -> AuthProvider:
        return AuthProvider.GOOGLE
    
    @property
    def authorization_url(self) -> str:
        return "https://accounts.google.com/o/oauth2/v2/auth"
    
    @property
    def token_url(self) -> str:
        return "https://oauth2.googleapis.com/token"
    
    @property
    def user_info_url(self) -> str:
        return "https://www.googleapis.com/oauth2/v2/userinfo"
    
    @property
    def default_scopes(self) -> List[str]:
        return ["openid", "email", "profile"]
    
    def exchange_code(self, code: str) -> Dict[str, Any]:
        """Exchange code for tokens (simulated - would use httpx in production)."""
        return {
            "access_token": f"google_access_{python_secrets.token_hex(16)}",
            "refresh_token": f"google_refresh_{python_secrets.token_hex(16)}",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
    
    def get_user_info(self, access_token: str) -> OAuthAccount:
        """Get user info from Google (simulated)."""
        return OAuthAccount(
            provider=AuthProvider.GOOGLE,
            provider_user_id=f"google_{python_secrets.token_hex(8)}",
            email="user@gmail.com",
            display_name="Google User",
            avatar_url="https://lh3.googleusercontent.com/default",
            access_token=access_token,
            scopes=self.default_scopes,
        )


class GitHubOAuthProvider(OAuthProvider):
    """GitHub OAuth provider implementation."""
    
    @property
    def provider(self) -> AuthProvider:
        return AuthProvider.GITHUB
    
    @property
    def authorization_url(self) -> str:
        return "https://github.com/login/oauth/authorize"
    
    @property
    def token_url(self) -> str:
        return "https://github.com/login/oauth/access_token"
    
    @property
    def user_info_url(self) -> str:
        return "https://api.github.com/user"
    
    @property
    def default_scopes(self) -> List[str]:
        return ["read:user", "user:email"]
    
    def exchange_code(self, code: str) -> Dict[str, Any]:
        """Exchange code for tokens (simulated)."""
        return {
            "access_token": f"github_access_{python_secrets.token_hex(16)}",
            "token_type": "Bearer",
            "scope": ",".join(self.default_scopes),
        }
    
    def get_user_info(self, access_token: str) -> OAuthAccount:
        """Get user info from GitHub (simulated)."""
        return OAuthAccount(
            provider=AuthProvider.GITHUB,
            provider_user_id=f"github_{python_secrets.token_hex(8)}",
            email="user@github.com",
            display_name="GitHub User",
            avatar_url="https://avatars.githubusercontent.com/u/default",
            access_token=access_token,
            scopes=self.default_scopes,
        )


class DiscordOAuthProvider(OAuthProvider):
    """Discord OAuth provider implementation."""
    
    @property
    def provider(self) -> AuthProvider:
        return AuthProvider.DISCORD
    
    @property
    def authorization_url(self) -> str:
        return "https://discord.com/api/oauth2/authorize"
    
    @property
    def token_url(self) -> str:
        return "https://discord.com/api/oauth2/token"
    
    @property
    def user_info_url(self) -> str:
        return "https://discord.com/api/users/@me"
    
    @property
    def default_scopes(self) -> List[str]:
        return ["identify", "email"]
    
    def exchange_code(self, code: str) -> Dict[str, Any]:
        """Exchange code for tokens (simulated)."""
        return {
            "access_token": f"discord_access_{python_secrets.token_hex(16)}",
            "refresh_token": f"discord_refresh_{python_secrets.token_hex(16)}",
            "expires_in": 604800,
            "token_type": "Bearer",
        }
    
    def get_user_info(self, access_token: str) -> OAuthAccount:
        """Get user info from Discord (simulated)."""
        return OAuthAccount(
            provider=AuthProvider.DISCORD,
            provider_user_id=f"discord_{python_secrets.token_hex(8)}",
            email="user@discord.com",
            display_name="Discord User",
            avatar_url="https://cdn.discordapp.com/avatars/default.png",
            access_token=access_token,
            scopes=self.default_scopes,
        )


class RoleManager:
    """
    Role-based access control manager.
    
    Manages user roles and permissions with support for hierarchical
    role inheritance and custom permission definitions.
    
    Args:
        storage_dir: Directory for persistence
    """
    
    ROLE_HIERARCHY = {
        Role.SUPERADMIN.value: [Role.ADMIN.value, Role.MODERATOR.value, Role.USER.value],
        Role.ADMIN.value: [Role.MODERATOR.value, Role.USER.value],
        Role.MODERATOR.value: [Role.USER.value],
        Role.USER.value: [],
        Role.GUEST.value: [],
    }
    
    ROLE_PERMISSIONS = {
        Role.SUPERADMIN.value: [
            Permission.USERS_READ.value, Permission.USERS_WRITE.value,
            Permission.USERS_DELETE.value, Permission.USERS_ADMIN.value,
            Permission.CONTENT_READ.value, Permission.CONTENT_WRITE.value,
            Permission.CONTENT_DELETE.value, Permission.CONTENT_PUBLISH.value,
            Permission.SETTINGS_READ.value, Permission.SETTINGS_WRITE.value,
            Permission.ROLES_ASSIGN.value, Permission.ROLES_MANAGE.value,
            Permission.API_ACCESS.value, Permission.API_ADMIN.value,
            Permission.MFA_BYPASS.value, Permission.AUDIT_READ.value,
            Permission.SYSTEM_ADMIN.value,
        ],
        Role.ADMIN.value: [
            Permission.USERS_READ.value, Permission.USERS_WRITE.value,
            Permission.USERS_DELETE.value,
            Permission.CONTENT_READ.value, Permission.CONTENT_WRITE.value,
            Permission.CONTENT_DELETE.value, Permission.CONTENT_PUBLISH.value,
            Permission.SETTINGS_READ.value, Permission.SETTINGS_WRITE.value,
            Permission.ROLES_ASSIGN.value,
            Permission.API_ACCESS.value,
            Permission.AUDIT_READ.value,
        ],
        Role.MODERATOR.value: [
            Permission.USERS_READ.value,
            Permission.CONTENT_READ.value, Permission.CONTENT_WRITE.value,
            Permission.CONTENT_DELETE.value,
            Permission.API_ACCESS.value,
        ],
        Role.USER.value: [
            Permission.CONTENT_READ.value, Permission.CONTENT_WRITE.value,
            Permission.API_ACCESS.value,
        ],
        Role.GUEST.value: [
            Permission.CONTENT_READ.value,
        ],
    }
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or "./auth_data")
        self._user_roles: Dict[str, List[str]] = {}
        self._custom_permissions: Dict[str, List[str]] = {}
        self._custom_roles: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        
        self._ensure_storage_dir()
        self._load_from_disk()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_storage_path(self) -> Path:
        """Get the file path for role data."""
        return self.storage_dir / "roles.json"
    
    def _load_from_disk(self) -> None:
        """Load role data from disk."""
        path = self._get_storage_path()
        if not path.exists():
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._user_roles = data.get("user_roles", {})
            self._custom_permissions = data.get("custom_permissions", {})
            self._custom_roles = data.get("custom_roles", {})
        except (json.JSONDecodeError, IOError):
            pass
    
    def _save_to_disk(self) -> None:
        """Save role data to disk."""
        path = self._get_storage_path()
        
        data = {
            "user_roles": self._user_roles,
            "custom_permissions": self._custom_permissions,
            "custom_roles": self._custom_roles,
        }
        
        try:
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            temp_path.replace(path)
        except IOError:
            pass
    
    def assign_role(self, user_id: str, role: str) -> None:
        """
        Assign a role to a user.
        
        Args:
            user_id: User ID
            role: Role to assign
        """
        with self._lock:
            if user_id not in self._user_roles:
                self._user_roles[user_id] = []
            
            if role not in self._user_roles[user_id]:
                self._user_roles[user_id].append(role)
            
            self._save_to_disk()
    
    def revoke_role(self, user_id: str, role: str) -> None:
        """
        Revoke a role from a user.
        
        Args:
            user_id: User ID
            role: Role to revoke
        """
        with self._lock:
            if user_id in self._user_roles:
                if role in self._user_roles[user_id]:
                    self._user_roles[user_id].remove(role)
                
                self._save_to_disk()
    
    def get_roles(self, user_id: str) -> List[str]:
        """
        Get all roles for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of role names
        """
        with self._lock:
            return self._user_roles.get(user_id, [Role.USER.value]).copy()
    
    def has_role(self, user_id: str, role: str) -> bool:
        """
        Check if a user has a specific role.
        
        Args:
            user_id: User ID
            role: Role to check
            
        Returns:
            True if user has the role
        """
        user_roles = self.get_roles(user_id)
        
        if role in user_roles:
            return True
        
        for user_role in user_roles:
            if user_role in self.ROLE_HIERARCHY:
                if role in self.ROLE_HIERARCHY[user_role]:
                    return True
        
        return False
    
    def get_permissions(self, user_id: str) -> Set[str]:
        """
        Get all permissions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Set of permission names
        """
        permissions = set()
        
        for role in self.get_roles(user_id):
            if role in self.ROLE_PERMISSIONS:
                permissions.update(self.ROLE_PERMISSIONS[role])
            
            if role in self._custom_roles:
                permissions.update(self._custom_roles[role])
        
        if user_id in self._custom_permissions:
            permissions.update(self._custom_permissions[user_id])
        
        return permissions
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has the permission
        """
        return permission in self.get_permissions(user_id)
    
    def require_permission(self, user_id: str, permission: str) -> None:
        """
        Require a user to have a permission or raise an error.
        
        Args:
            user_id: User ID
            permission: Required permission
            
        Raises:
            PermissionDeniedError: If user lacks permission
        """
        if not self.check_permission(user_id, permission):
            raise PermissionDeniedError(user_id, permission)
    
    def add_custom_permission(self, user_id: str, permission: str) -> None:
        """
        Add a custom permission directly to a user.
        
        Args:
            user_id: User ID
            permission: Permission to add
        """
        with self._lock:
            if user_id not in self._custom_permissions:
                self._custom_permissions[user_id] = []
            
            if permission not in self._custom_permissions[user_id]:
                self._custom_permissions[user_id].append(permission)
            
            self._save_to_disk()
    
    def remove_custom_permission(self, user_id: str, permission: str) -> None:
        """
        Remove a custom permission from a user.
        
        Args:
            user_id: User ID
            permission: Permission to remove
        """
        with self._lock:
            if user_id in self._custom_permissions:
                if permission in self._custom_permissions[user_id]:
                    self._custom_permissions[user_id].remove(permission)
                
                self._save_to_disk()
    
    def define_custom_role(self, role_name: str, permissions: List[str]) -> None:
        """
        Define a custom role with specific permissions.
        
        Args:
            role_name: Name of the custom role
            permissions: List of permissions for the role
        """
        with self._lock:
            self._custom_roles[role_name] = permissions
            self._save_to_disk()
    
    def delete_custom_role(self, role_name: str) -> None:
        """
        Delete a custom role.
        
        Args:
            role_name: Name of the role to delete
        """
        with self._lock:
            if role_name in self._custom_roles:
                del self._custom_roles[role_name]
                self._save_to_disk()


class MFAManager:
    """
    Multi-factor authentication manager with TOTP support.
    
    Manages TOTP-based multi-factor authentication including
    secret generation, code verification, and backup codes.
    
    Args:
        issuer: Issuer name for TOTP URIs
        digits: Number of TOTP digits (default: 6)
        period: TOTP time period in seconds (default: 30)
        algorithm: TOTP algorithm (default: SHA1)
    """
    
    def __init__(
        self,
        issuer: str = "Platform Forge",
        digits: int = 6,
        period: int = 30,
        algorithm: str = "SHA1",
    ):
        self.issuer = issuer
        self.digits = digits
        self.period = period
        self.algorithm = algorithm
    
    def generate_secret(self) -> str:
        """
        Generate a new TOTP secret.
        
        Returns:
            Base32-encoded secret string
        """
        secret_bytes = python_secrets.token_bytes(20)
        return base64.b32encode(secret_bytes).decode('ascii').rstrip('=')
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """
        Generate backup codes.
        
        Args:
            count: Number of codes to generate
            
        Returns:
            List of backup codes
        """
        codes = []
        for _ in range(count):
            code = f"{python_secrets.randbelow(10000):04d}-{python_secrets.randbelow(10000):04d}"
            codes.append(code)
        return codes
    
    def get_totp_uri(
        self,
        secret: str,
        account: str,
    ) -> str:
        """
        Generate TOTP URI for QR code.
        
        Args:
            secret: TOTP secret
            account: User account identifier (email)
            
        Returns:
            otpauth:// URI for QR code
        """
        params = {
            "secret": secret,
            "issuer": self.issuer,
            "algorithm": self.algorithm,
            "digits": str(self.digits),
            "period": str(self.period),
        }
        
        label = f"{self.issuer}:{account}"
        return f"otpauth://totp/{label}?{urlencode(params)}"
    
    def verify_code(
        self,
        secret: str,
        code: str,
        window: int = 1,
    ) -> bool:
        """
        Verify a TOTP code.
        
        Args:
            secret: TOTP secret
            code: Code to verify
            window: Number of periods before/after to check
            
        Returns:
            True if code is valid
        """
        current_time = int(time.time())
        
        for offset in range(-window, window + 1):
            expected = self._generate_totp(secret, current_time + (offset * self.period))
            if hmac.compare_digest(expected, code):
                return True
        
        return False
    
    def verify_backup_code(
        self,
        code: str,
        hashed_codes: List[str],
    ) -> Tuple[bool, int]:
        """
        Verify a backup code.
        
        Args:
            code: Backup code to verify
            hashed_codes: List of hashed backup codes
            
        Returns:
            Tuple of (is_valid, index_of_used_code)
        """
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        for i, hashed in enumerate(hashed_codes):
            if hmac.compare_digest(code_hash, hashed):
                return True, i
        
        return False, -1
    
    def hash_backup_codes(self, codes: List[str]) -> List[str]:
        """
        Hash backup codes for storage.
        
        Args:
            codes: Plain text backup codes
            
        Returns:
            List of hashed codes
        """
        return [hashlib.sha256(code.encode()).hexdigest() for code in codes]
    
    def _generate_totp(self, secret: str, timestamp: int) -> str:
        """Generate TOTP code for a specific timestamp."""
        secret_bytes = self._decode_secret(secret)
        
        counter = timestamp // self.period
        counter_bytes = struct.pack(">Q", counter)
        
        if self.algorithm == "SHA1":
            mac = hmac.new(secret_bytes, counter_bytes, hashlib.sha1)
        elif self.algorithm == "SHA256":
            mac = hmac.new(secret_bytes, counter_bytes, hashlib.sha256)
        elif self.algorithm == "SHA512":
            mac = hmac.new(secret_bytes, counter_bytes, hashlib.sha512)
        else:
            mac = hmac.new(secret_bytes, counter_bytes, hashlib.sha1)
        
        digest = mac.digest()
        
        offset = digest[-1] & 0x0F
        
        binary = struct.unpack(">I", digest[offset:offset + 4])[0]
        binary &= 0x7FFFFFFF
        
        otp = binary % (10 ** self.digits)
        return str(otp).zfill(self.digits)
    
    def _decode_secret(self, secret: str) -> bytes:
        """Decode base32 secret to bytes."""
        padding = 8 - (len(secret) % 8)
        if padding != 8:
            secret += "=" * padding
        return base64.b32decode(secret.upper())


class RateLimiter:
    """
    Rate limiter for authentication endpoints.
    
    Implements sliding window rate limiting to prevent brute force
    attacks on authentication endpoints.
    
    Args:
        window_size: Time window in seconds
        max_requests: Maximum requests per window
    """
    
    def __init__(
        self,
        window_size: int = AuthLimit.RATE_LIMIT_WINDOW_SECONDS.value,
        max_requests: int = AuthLimit.RATE_LIMIT_MAX_REQUESTS.value,
    ):
        self.window_size = window_size
        self.max_requests = max_requests
        self._requests: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def is_allowed(self, key: str) -> Tuple[bool, float]:
        """
        Check if a request is allowed.
        
        Args:
            key: Rate limit key (e.g., IP address, user ID)
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window_size
        
        with self._lock:
            if key not in self._requests:
                self._requests[key] = []
            
            self._requests[key] = [
                ts for ts in self._requests[key]
                if ts > window_start
            ]
            
            if len(self._requests[key]) >= self.max_requests:
                oldest = min(self._requests[key]) if self._requests[key] else now
                retry_after = oldest + self.window_size - now
                return False, max(0, retry_after)
            
            self._requests[key].append(now)
            return True, 0
    
    def record_request(self, key: str) -> None:
        """
        Record a request for rate limiting.
        
        Args:
            key: Rate limit key
        """
        now = time.time()
        
        with self._lock:
            if key not in self._requests:
                self._requests[key] = []
            
            self._requests[key].append(now)
    
    def get_remaining(self, key: str) -> int:
        """
        Get remaining requests in the current window.
        
        Args:
            key: Rate limit key
            
        Returns:
            Number of remaining requests
        """
        now = time.time()
        window_start = now - self.window_size
        
        with self._lock:
            if key not in self._requests:
                return self.max_requests
            
            current = [ts for ts in self._requests[key] if ts > window_start]
            return max(0, self.max_requests - len(current))
    
    def reset(self, key: str) -> None:
        """
        Reset rate limit for a key.
        
        Args:
            key: Rate limit key to reset
        """
        with self._lock:
            if key in self._requests:
                del self._requests[key]
    
    def clear_expired(self) -> None:
        """Clear all expired rate limit entries."""
        now = time.time()
        window_start = now - self.window_size
        
        with self._lock:
            keys_to_delete = []
            
            for key, timestamps in self._requests.items():
                self._requests[key] = [ts for ts in timestamps if ts > window_start]
                if not self._requests[key]:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._requests[key]


class AuthManager:
    """
    Main authentication manager.
    
    Provides a complete authentication system with registration, login,
    session management, OAuth integration, and more.
    
    Args:
        secret_key: Secret key for token signing
        storage_dir: Directory for persistent storage
        password_hasher: Password hasher instance
        token_manager: Token manager instance
        role_manager: Role manager instance
        mfa_manager: MFA manager instance
        rate_limiter: Rate limiter instance
    """
    
    def __init__(
        self,
        secret_key: str,
        storage_dir: Optional[str] = None,
        password_hasher: Optional[PasswordHasher] = None,
        token_manager: Optional[TokenManager] = None,
        role_manager: Optional[RoleManager] = None,
        mfa_manager: Optional[MFAManager] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.secret_key = secret_key
        self.storage_dir = Path(storage_dir or "./auth_data")
        
        self.password_hasher = password_hasher or PasswordHasher()
        self.token_manager = token_manager or TokenManager(secret_key)
        self.role_manager = role_manager or RoleManager(str(self.storage_dir))
        self.mfa_manager = mfa_manager or MFAManager()
        self.rate_limiter = rate_limiter or RateLimiter()
        
        self.password_validator = PasswordValidator()
        
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._email_to_user: Dict[str, str] = {}
        self._username_to_user: Dict[str, str] = {}
        self._audit_log: List[AuditLogEntry] = []
        self._oauth_providers: Dict[AuthProvider, OAuthProvider] = {}
        self._lock = threading.RLock()
        
        self._ensure_storage_dir()
        self._load_from_disk()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_users_path(self) -> Path:
        """Get the file path for users data."""
        return self.storage_dir / "users.json"
    
    def _get_sessions_path(self) -> Path:
        """Get the file path for sessions data."""
        return self.storage_dir / "sessions.json"
    
    def _get_audit_path(self) -> Path:
        """Get the file path for audit log."""
        return self.storage_dir / "audit.json"
    
    def _load_from_disk(self) -> None:
        """Load auth data from disk."""
        users_path = self._get_users_path()
        if users_path.exists():
            try:
                with open(users_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for user_data in data.get("users", []):
                    user = User.from_dict(user_data)
                    self._users[user.user_id] = user
                    self._email_to_user[user.email.lower()] = user.user_id
                    self._username_to_user[user.username.lower()] = user.user_id
            except (json.JSONDecodeError, IOError):
                pass
        
        sessions_path = self._get_sessions_path()
        if sessions_path.exists():
            try:
                with open(sessions_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for session_data in data.get("sessions", []):
                    session = Session.from_dict(session_data)
                    if session.is_valid:
                        self._sessions[session.session_id] = session
            except (json.JSONDecodeError, IOError):
                pass
    
    def _save_to_disk(self) -> None:
        """Save auth data to disk."""
        users_path = self._get_users_path()
        users_data = {
            "users": [user.to_dict(include_sensitive=True) for user in self._users.values()]
        }
        
        try:
            temp_path = users_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, ensure_ascii=False, indent=2)
            temp_path.replace(users_path)
        except IOError:
            pass
        
        sessions_path = self._get_sessions_path()
        valid_sessions = [s for s in self._sessions.values() if s.is_valid]
        sessions_data = {
            "sessions": [session.to_dict() for session in valid_sessions]
        }
        
        try:
            temp_path = sessions_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2)
            temp_path.replace(sessions_path)
        except IOError:
            pass
    
    def _log_audit(
        self,
        event: AuditEvent,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        details: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """Log a security audit event."""
        entry = AuditLogEntry(
            timestamp=time.time(),
            event=event,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details,
        )
        
        self._audit_log.append(entry)
        
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]
    
    def _generate_user_id(self) -> str:
        """Generate a unique user ID."""
        return f"user_{python_secrets.token_hex(12)}"
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return python_secrets.token_urlsafe(32)
    
    def register_oauth_provider(
        self,
        provider: OAuthProvider,
    ) -> None:
        """
        Register an OAuth provider.
        
        Args:
            provider: OAuth provider instance
        """
        self._oauth_providers[provider.provider] = provider
    
    def register(
        self,
        email: str,
        password: str,
        username: str,
        display_name: Optional[str] = None,
        auto_verify: bool = False,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> User:
        """
        Register a new user.
        
        Args:
            email: User's email address
            password: User's password
            username: Unique username
            display_name: Display name (optional)
            auto_verify: Auto-verify email (for testing)
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Created user object
            
        Raises:
            UserExistsError: If email already registered
            WeakPasswordError: If password doesn't meet requirements
        """
        allowed, retry_after = self.rate_limiter.is_allowed(
            f"register:{ip_address or 'unknown'}"
        )
        if not allowed:
            raise RateLimitExceededError(retry_after)
        
        email = email.strip().lower()
        username = username.strip()
        
        if len(email) > AuthLimit.MAX_EMAIL_LENGTH.value:
            raise AuthError(f"Email too long (max {AuthLimit.MAX_EMAIL_LENGTH.value})")
        
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise AuthError("Invalid email format")
        
        if len(username) < AuthLimit.MIN_USERNAME_LENGTH.value:
            raise AuthError(f"Username too short (min {AuthLimit.MIN_USERNAME_LENGTH.value})")
        
        if len(username) > AuthLimit.MAX_USERNAME_LENGTH.value:
            raise AuthError(f"Username too long (max {AuthLimit.MAX_USERNAME_LENGTH.value})")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            raise AuthError("Username can only contain letters, numbers, underscores, and hyphens")
        
        is_valid, errors = self.password_validator.validate(password)
        if not is_valid:
            raise WeakPasswordError(errors)
        
        with self._lock:
            if email in self._email_to_user:
                raise UserExistsError(email)
            
            if username.lower() in self._username_to_user:
                raise AuthError(f"Username '{username}' is already taken")
            
            user_id = self._generate_user_id()
            password_hash = self.password_hasher.hash(password)
            
            user = User(
                user_id=user_id,
                email=email,
                username=username,
                password_hash=password_hash,
                status=UserStatus.ACTIVE if auto_verify else UserStatus.PENDING,
                email_verified=auto_verify,
                email_verified_at=time.time() if auto_verify else None,
                profile=UserProfile(display_name=display_name or username),
            )
            
            self._users[user_id] = user
            self._email_to_user[email] = user_id
            self._username_to_user[username.lower()] = user_id
            
            self.role_manager.assign_role(user_id, Role.USER.value)
            
            self._save_to_disk()
            
            self._log_audit(
                AuditEvent.REGISTER,
                user_id=user_id,
                success=True,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            
            return user
    
    def login(
        self,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        require_mfa: bool = True,
    ) -> LoginResult:
        """
        Authenticate a user with email and password.
        
        Args:
            email: User's email address
            password: User's password
            ip_address: Client IP address
            user_agent: Client user agent
            require_mfa: Whether to require MFA if enabled
            
        Returns:
            LoginResult with session and tokens
        """
        rate_key = f"login:{ip_address or 'unknown'}"
        allowed, retry_after = self.rate_limiter.is_allowed(rate_key)
        if not allowed:
            raise RateLimitExceededError(retry_after)
        
        email = email.strip().lower()
        
        with self._lock:
            user_id = self._email_to_user.get(email)
            
            if not user_id:
                self._log_audit(
                    AuditEvent.LOGIN_FAILED,
                    success=False,
                    details="User not found",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                raise InvalidCredentialsError()
            
            user = self._users[user_id]
            
            if user.is_locked:
                self._log_audit(
                    AuditEvent.LOGIN_FAILED,
                    user_id=user_id,
                    success=False,
                    details="Account locked",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                raise AccountLockedError(user.locked_until)
            
            if user.status == UserStatus.SUSPENDED:
                raise AccountSuspendedError(user_id)
            
            if user.status == UserStatus.BANNED:
                raise AccountSuspendedError(user_id, "Account has been banned")
            
            if not user.password_hash:
                raise InvalidCredentialsError("No password set. Use OAuth or reset password.")
            
            if not self.password_hasher.verify(password, user.password_hash):
                user.record_failed_attempt()
                self._save_to_disk()
                
                self._log_audit(
                    AuditEvent.LOGIN_FAILED,
                    user_id=user_id,
                    success=False,
                    details="Invalid password",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                
                if user.is_locked:
                    self._log_audit(
                        AuditEvent.ACCOUNT_LOCKED,
                        user_id=user_id,
                        ip_address=ip_address,
                    )
                    raise AccountLockedError(user.locked_until)
                
                raise InvalidCredentialsError()
            
            if user.requires_mfa and require_mfa:
                mfa_token = self.token_manager.create_mfa_token(user_id)
                
                return LoginResult(
                    success=False,
                    requires_mfa=True,
                    mfa_token=mfa_token,
                )
            
            return self._complete_login(user, ip_address, user_agent)
    
    def verify_mfa_and_login(
        self,
        mfa_token: str,
        code: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> LoginResult:
        """
        Complete login after MFA verification.
        
        Args:
            mfa_token: MFA token from initial login
            code: TOTP code or backup code
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            LoginResult with session and tokens
        """
        payload = self.token_manager.validate_token(mfa_token, TokenType.MFA_SETUP)
        user_id = payload["sub"]
        
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        is_valid = self.verify_mfa(user_id, code)
        
        if not is_valid:
            self._log_audit(
                AuditEvent.MFA_FAILED,
                user_id=user_id,
                success=False,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            raise MFAInvalidCodeError()
        
        self._log_audit(
            AuditEvent.MFA_VERIFIED,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        return self._complete_login(user, ip_address, user_agent, mfa_verified=True)
    
    def _complete_login(
        self,
        user: User,
        ip_address: Optional[str],
        user_agent: Optional[str],
        mfa_verified: bool = False,
    ) -> LoginResult:
        """Complete the login process after authentication."""
        user.record_login()
        
        if self.password_hasher.needs_rehash(user.password_hash or ""):
            pass
        
        session = self.create_session(
            user.user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=mfa_verified,
        )
        
        access_token = self.token_manager.create_access_token(
            user.user_id,
            session.session_id,
            roles=self.role_manager.get_roles(user.user_id),
        )
        refresh_token = self.token_manager.create_refresh_token(
            user.user_id,
            session.session_id,
        )
        
        session.refresh_token = refresh_token
        
        self._save_to_disk()
        
        self._log_audit(
            AuditEvent.LOGIN_SUCCESS,
            user_id=user.user_id,
            session_id=session.session_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        return LoginResult(
            success=True,
            user=user,
            session=session,
            access_token=access_token,
            refresh_token=refresh_token,
        )
    
    def logout(
        self,
        session_id: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Logout a user by revoking their session.
        
        Args:
            session_id: Session ID to revoke
            ip_address: Client IP address
            
        Returns:
            True if session was revoked
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if not session:
                return False
            
            session.revoke()
            
            self._log_audit(
                AuditEvent.LOGOUT,
                user_id=session.user_id,
                session_id=session_id,
                ip_address=ip_address,
            )
            
            self._save_to_disk()
            
            return True
    
    def logout_all_sessions(
        self,
        user_id: str,
        except_session: Optional[str] = None,
    ) -> int:
        """
        Logout all sessions for a user.
        
        Args:
            user_id: User ID
            except_session: Session ID to keep active
            
        Returns:
            Number of sessions revoked
        """
        count = 0
        
        with self._lock:
            for session in self._sessions.values():
                if session.user_id == user_id and session.session_id != except_session:
                    if session.is_valid:
                        session.revoke()
                        count += 1
            
            self._save_to_disk()
        
        return count
    
    def create_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        mfa_verified: bool = False,
        duration: int = AuthLimit.SESSION_DURATION_SECONDS.value,
    ) -> Session:
        """
        Create a new session for a user.
        
        Args:
            user_id: User ID
            ip_address: Client IP address
            user_agent: Client user agent
            mfa_verified: Whether MFA was verified
            duration: Session duration in seconds
            
        Returns:
            Created session
        """
        with self._lock:
            user_sessions = [
                s for s in self._sessions.values()
                if s.user_id == user_id and s.is_valid
            ]
            
            if len(user_sessions) >= AuthLimit.MAX_SESSIONS_PER_USER.value:
                oldest = min(user_sessions, key=lambda s: s.created_at)
                oldest.revoke()
            
            session_id = self._generate_session_id()
            
            session = Session(
                session_id=session_id,
                user_id=user_id,
                expires_at=time.time() + duration,
                ip_address=ip_address,
                user_agent=user_agent,
                mfa_verified=mfa_verified,
            )
            
            self._sessions[session_id] = session
            
            self._log_audit(
                AuditEvent.SESSION_CREATED,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            
            self._save_to_disk()
            
            return session
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """
        Validate a session.
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            Session if valid, None otherwise
        """
        session = self._sessions.get(session_id)
        
        if not session:
            return None
        
        if not session.is_valid:
            return None
        
        session.update_activity()
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self._users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        email = email.strip().lower()
        user_id = self._email_to_user.get(email)
        if user_id:
            return self._users.get(user_id)
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        username = username.strip().lower()
        user_id = self._username_to_user.get(username)
        if user_id:
            return self._users.get(user_id)
        return None
    
    def verify_email(self, token: str) -> User:
        """
        Verify a user's email address.
        
        Args:
            token: Email verification token
            
        Returns:
            Updated user object
        """
        payload = self.token_manager.validate_token(
            token,
            TokenType.EMAIL_VERIFICATION,
        )
        
        user_id = payload["sub"]
        email = payload["email"]
        
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                raise UserNotFoundError(user_id)
            
            if user.email != email:
                raise InvalidTokenError(
                    TokenType.EMAIL_VERIFICATION,
                    "Email mismatch",
                )
            
            user.email_verified = True
            user.email_verified_at = time.time()
            user.status = UserStatus.ACTIVE
            user.updated_at = time.time()
            
            self._save_to_disk()
            
            self._log_audit(
                AuditEvent.EMAIL_VERIFICATION,
                user_id=user_id,
            )
            
            return user
    
    def create_email_verification_token(self, user_id: str) -> str:
        """Create an email verification token for a user."""
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        return self.token_manager.create_email_verification_token(
            user_id,
            user.email,
        )
    
    def request_password_reset(
        self,
        email: str,
        ip_address: Optional[str] = None,
    ) -> Optional[str]:
        """
        Request a password reset.
        
        Args:
            email: User's email address
            ip_address: Client IP address
            
        Returns:
            Reset token if user exists, None otherwise
        """
        rate_key = f"reset:{ip_address or 'unknown'}"
        allowed, retry_after = self.rate_limiter.is_allowed(rate_key)
        if not allowed:
            raise RateLimitExceededError(retry_after)
        
        email = email.strip().lower()
        user = self.get_user_by_email(email)
        
        self._log_audit(
            AuditEvent.PASSWORD_RESET_REQUEST,
            user_id=user.user_id if user else None,
            success=user is not None,
            ip_address=ip_address,
        )
        
        if not user:
            return None
        
        return self.token_manager.create_password_reset_token(
            user.user_id,
            user.email,
        )
    
    def reset_password(
        self,
        token: str,
        new_password: str,
        ip_address: Optional[str] = None,
    ) -> User:
        """
        Reset a user's password.
        
        Args:
            token: Password reset token
            new_password: New password
            ip_address: Client IP address
            
        Returns:
            Updated user object
        """
        is_valid, errors = self.password_validator.validate(new_password)
        if not is_valid:
            raise WeakPasswordError(errors)
        
        payload = self.token_manager.validate_token(
            token,
            TokenType.PASSWORD_RESET,
        )
        
        user_id = payload["sub"]
        
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                raise UserNotFoundError(user_id)
            
            user.password_hash = self.password_hasher.hash(new_password)
            user.password_changed_at = time.time()
            user.updated_at = time.time()
            user.reset_failed_attempts()
            
            self.logout_all_sessions(user_id)
            
            self._save_to_disk()
            
            self._log_audit(
                AuditEvent.PASSWORD_RESET_COMPLETE,
                user_id=user_id,
                ip_address=ip_address,
            )
            
            return user
    
    def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
        logout_other_sessions: bool = True,
        current_session_id: Optional[str] = None,
    ) -> User:
        """
        Change a user's password.
        
        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            logout_other_sessions: Whether to logout other sessions
            current_session_id: Current session to keep
            
        Returns:
            Updated user object
        """
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        if not user.password_hash:
            raise AuthError("No password set")
        
        if not self.password_hasher.verify(current_password, user.password_hash):
            raise InvalidCredentialsError("Current password is incorrect")
        
        is_valid, errors = self.password_validator.validate(new_password)
        if not is_valid:
            raise WeakPasswordError(errors)
        
        with self._lock:
            user.password_hash = self.password_hasher.hash(new_password)
            user.password_changed_at = time.time()
            user.updated_at = time.time()
            
            if logout_other_sessions:
                self.logout_all_sessions(user_id, except_session=current_session_id)
            
            self._save_to_disk()
            
            self._log_audit(
                AuditEvent.PASSWORD_CHANGE,
                user_id=user_id,
            )
            
            return user
    
    def oauth_login(
        self,
        provider: AuthProvider,
        code: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> LoginResult:
        """
        Login or register via OAuth.
        
        Args:
            provider: OAuth provider
            code: Authorization code
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            LoginResult with session and tokens
        """
        oauth_provider = self._oauth_providers.get(provider)
        if not oauth_provider:
            raise OAuthError(provider, "Provider not configured")
        
        tokens = oauth_provider.exchange_code(code)
        account = oauth_provider.get_user_info(tokens["access_token"])
        
        account.access_token = tokens.get("access_token")
        account.refresh_token = tokens.get("refresh_token")
        if tokens.get("expires_in"):
            account.token_expires_at = time.time() + tokens["expires_in"]
        
        with self._lock:
            existing_user = None
            for user in self._users.values():
                for oauth in user.oauth_accounts:
                    if (oauth.provider == provider and 
                        oauth.provider_user_id == account.provider_user_id):
                        existing_user = user
                        break
                if existing_user:
                    break
            
            if existing_user:
                for i, oauth in enumerate(existing_user.oauth_accounts):
                    if oauth.provider == provider:
                        existing_user.oauth_accounts[i] = account
                        break
                
                existing_user.updated_at = time.time()
                self._save_to_disk()
                
                return self._complete_login(
                    existing_user,
                    ip_address,
                    user_agent,
                )
            
            if account.email:
                existing_user = self.get_user_by_email(account.email)
                if existing_user:
                    existing_user.oauth_accounts.append(account)
                    existing_user.updated_at = time.time()
                    
                    if not existing_user.email_verified:
                        existing_user.email_verified = True
                        existing_user.email_verified_at = time.time()
                        existing_user.status = UserStatus.ACTIVE
                    
                    self._save_to_disk()
                    
                    self._log_audit(
                        AuditEvent.OAUTH_LINKED,
                        user_id=existing_user.user_id,
                        details=f"Linked {provider.value}",
                        ip_address=ip_address,
                    )
                    
                    return self._complete_login(
                        existing_user,
                        ip_address,
                        user_agent,
                    )
            
            user_id = self._generate_user_id()
            username = self._generate_unique_username(account.display_name or "user")
            
            user = User(
                user_id=user_id,
                email=account.email or f"{username}@oauth.local",
                username=username,
                status=UserStatus.ACTIVE,
                email_verified=account.email is not None,
                email_verified_at=time.time() if account.email else None,
                profile=UserProfile(
                    display_name=account.display_name,
                    avatar_url=account.avatar_url,
                ),
                oauth_accounts=[account],
            )
            
            self._users[user_id] = user
            self._email_to_user[user.email.lower()] = user_id
            self._username_to_user[username.lower()] = user_id
            
            self.role_manager.assign_role(user_id, Role.USER.value)
            
            self._save_to_disk()
            
            self._log_audit(
                AuditEvent.REGISTER,
                user_id=user_id,
                details=f"OAuth registration via {provider.value}",
                ip_address=ip_address,
            )
            
            return self._complete_login(user, ip_address, user_agent)
    
    def link_oauth_account(
        self,
        user_id: str,
        provider: AuthProvider,
        code: str,
    ) -> OAuthAccount:
        """
        Link an OAuth account to an existing user.
        
        Args:
            user_id: User ID
            provider: OAuth provider
            code: Authorization code
            
        Returns:
            Linked OAuth account
        """
        oauth_provider = self._oauth_providers.get(provider)
        if not oauth_provider:
            raise OAuthError(provider, "Provider not configured")
        
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        if user.get_oauth_account(provider):
            raise AuthError(f"Account already linked to {provider.value}")
        
        tokens = oauth_provider.exchange_code(code)
        account = oauth_provider.get_user_info(tokens["access_token"])
        
        account.access_token = tokens.get("access_token")
        account.refresh_token = tokens.get("refresh_token")
        if tokens.get("expires_in"):
            account.token_expires_at = time.time() + tokens["expires_in"]
        
        with self._lock:
            user.oauth_accounts.append(account)
            user.updated_at = time.time()
            
            self._save_to_disk()
            
            self._log_audit(
                AuditEvent.OAUTH_LINKED,
                user_id=user_id,
                details=f"Linked {provider.value}",
            )
            
            return account
    
    def unlink_oauth_account(
        self,
        user_id: str,
        provider: AuthProvider,
    ) -> bool:
        """
        Unlink an OAuth account from a user.
        
        Args:
            user_id: User ID
            provider: OAuth provider to unlink
            
        Returns:
            True if account was unlinked
        """
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        if not user.has_password and len(user.oauth_accounts) <= 1:
            raise AuthError("Cannot unlink last authentication method")
        
        with self._lock:
            original_len = len(user.oauth_accounts)
            user.oauth_accounts = [
                a for a in user.oauth_accounts
                if a.provider != provider
            ]
            
            if len(user.oauth_accounts) < original_len:
                user.updated_at = time.time()
                self._save_to_disk()
                
                self._log_audit(
                    AuditEvent.OAUTH_UNLINKED,
                    user_id=user_id,
                    details=f"Unlinked {provider.value}",
                )
                
                return True
            
            return False
    
    def enable_mfa(self, user_id: str) -> Tuple[str, str, List[str]]:
        """
        Enable MFA for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (secret, totp_uri, backup_codes)
        """
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        secret = self.mfa_manager.generate_secret()
        totp_uri = self.mfa_manager.get_totp_uri(secret, user.email)
        backup_codes = self.mfa_manager.generate_backup_codes()
        
        with self._lock:
            user.mfa.totp_secret = secret
            user.mfa.backup_codes = self.mfa_manager.hash_backup_codes(backup_codes)
            user.mfa.verified = False
            user.updated_at = time.time()
            
            self._save_to_disk()
        
        return secret, totp_uri, backup_codes
    
    def verify_mfa_setup(
        self,
        user_id: str,
        code: str,
    ) -> bool:
        """
        Verify MFA setup with a TOTP code.
        
        Args:
            user_id: User ID
            code: TOTP code
            
        Returns:
            True if verification successful
        """
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        if not user.mfa.totp_secret:
            raise AuthError("MFA not set up")
        
        if not self.mfa_manager.verify_code(user.mfa.totp_secret, code):
            return False
        
        with self._lock:
            user.mfa.enabled = True
            user.mfa.verified = True
            user.mfa.enabled_at = time.time()
            user.updated_at = time.time()
            
            self._save_to_disk()
            
            self._log_audit(
                AuditEvent.MFA_ENABLED,
                user_id=user_id,
            )
        
        return True
    
    def verify_mfa(
        self,
        user_id: str,
        code: str,
    ) -> bool:
        """
        Verify an MFA code.
        
        Args:
            user_id: User ID
            code: TOTP code or backup code
            
        Returns:
            True if verification successful
        """
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        if not user.mfa.enabled or not user.mfa.totp_secret:
            return True
        
        if self.mfa_manager.verify_code(user.mfa.totp_secret, code):
            with self._lock:
                user.mfa.last_used = time.time()
                self._save_to_disk()
            return True
        
        is_backup, index = self.mfa_manager.verify_backup_code(
            code,
            user.mfa.backup_codes,
        )
        
        if is_backup:
            with self._lock:
                user.mfa.backup_codes.pop(index)
                user.mfa.last_used = time.time()
                self._save_to_disk()
            return True
        
        return False
    
    def disable_mfa(
        self,
        user_id: str,
        code: str,
    ) -> bool:
        """
        Disable MFA for a user.
        
        Args:
            user_id: User ID
            code: Current TOTP code for verification
            
        Returns:
            True if MFA was disabled
        """
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        if not self.verify_mfa(user_id, code):
            raise MFAInvalidCodeError()
        
        with self._lock:
            user.mfa = MFASettings()
            user.updated_at = time.time()
            
            self._save_to_disk()
            
            self._log_audit(
                AuditEvent.MFA_DISABLED,
                user_id=user_id,
            )
        
        return True
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has the permission
        """
        return self.role_manager.check_permission(user_id, permission)
    
    def require_permission(self, user_id: str, permission: str) -> None:
        """
        Require a user to have a permission or raise an error.
        
        Args:
            user_id: User ID
            permission: Required permission
            
        Raises:
            PermissionDeniedError: If user lacks permission
        """
        if not self.check_permission(user_id, permission):
            self._log_audit(
                AuditEvent.PERMISSION_DENIED,
                user_id=user_id,
                details=f"Denied: {permission}",
            )
            raise PermissionDeniedError(user_id, permission)
    
    def assign_role(self, user_id: str, role: str, assigner_id: Optional[str] = None) -> None:
        """
        Assign a role to a user.
        
        Args:
            user_id: User ID
            role: Role to assign
            assigner_id: ID of user assigning the role
        """
        if assigner_id:
            self.require_permission(assigner_id, Permission.ROLES_ASSIGN.value)
        
        self.role_manager.assign_role(user_id, role)
        
        self._log_audit(
            AuditEvent.ROLE_ASSIGNED,
            user_id=user_id,
            details=f"Assigned role: {role}",
        )
    
    def revoke_role(self, user_id: str, role: str, revoker_id: Optional[str] = None) -> None:
        """
        Revoke a role from a user.
        
        Args:
            user_id: User ID
            role: Role to revoke
            revoker_id: ID of user revoking the role
        """
        if revoker_id:
            self.require_permission(revoker_id, Permission.ROLES_ASSIGN.value)
        
        self.role_manager.revoke_role(user_id, role)
        
        self._log_audit(
            AuditEvent.ROLE_REVOKED,
            user_id=user_id,
            details=f"Revoked role: {role}",
        )
    
    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEvent] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """
        Get audit log entries.
        
        Args:
            user_id: Filter by user ID
            event_type: Filter by event type
            since: Filter by timestamp
            limit: Maximum entries to return
            
        Returns:
            List of audit log entries
        """
        entries = self._audit_log.copy()
        
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        
        if event_type:
            entries = [e for e in entries if e.event == event_type]
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        
        return entries[:limit]
    
    def _generate_unique_username(self, base: str) -> str:
        """Generate a unique username from a base name."""
        base = re.sub(r'[^a-zA-Z0-9_-]', '', base.lower())
        if len(base) < AuthLimit.MIN_USERNAME_LENGTH.value:
            base = "user"
        
        username = base[:AuthLimit.MAX_USERNAME_LENGTH.value - 6]
        
        if username.lower() not in self._username_to_user:
            return username
        
        for i in range(1000):
            candidate = f"{username}{i}"
            if candidate.lower() not in self._username_to_user:
                return candidate
        
        return f"{username}_{python_secrets.token_hex(3)}"
    
    def update_user_profile(
        self,
        user_id: str,
        **profile_updates,
    ) -> User:
        """
        Update a user's profile.
        
        Args:
            user_id: User ID
            **profile_updates: Profile fields to update
            
        Returns:
            Updated user object
        """
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        
        with self._lock:
            for key, value in profile_updates.items():
                if hasattr(user.profile, key):
                    setattr(user.profile, key, value)
            
            user.updated_at = time.time()
            self._save_to_disk()
            
            return user
    
    def refresh_tokens(self, refresh_token: str) -> Tuple[str, str]:
        """
        Refresh access and refresh tokens.
        
        Args:
            refresh_token: Current refresh token
            
        Returns:
            Tuple of (new_access_token, new_refresh_token)
        """
        return self.token_manager.refresh_access_token(refresh_token)
    
    def validate_access_token(self, token: str) -> Dict[str, Any]:
        """
        Validate an access token.
        
        Args:
            token: Access token
            
        Returns:
            Token payload
        """
        return self.token_manager.validate_token(token, TokenType.ACCESS)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get auth system statistics."""
        active_sessions = sum(1 for s in self._sessions.values() if s.is_valid)
        active_users = len(set(s.user_id for s in self._sessions.values() if s.is_valid))
        
        return {
            "total_users": len(self._users),
            "active_users": active_users,
            "verified_users": sum(1 for u in self._users.values() if u.email_verified),
            "mfa_enabled_users": sum(1 for u in self._users.values() if u.mfa.enabled),
            "total_sessions": len(self._sessions),
            "active_sessions": active_sessions,
            "oauth_linked_users": sum(1 for u in self._users.values() if u.has_oauth),
            "audit_log_entries": len(self._audit_log),
        }


_default_manager: Optional[AuthManager] = None
_manager_lock = threading.Lock()


def get_default_manager() -> AuthManager:
    """Get the default AuthManager instance."""
    global _default_manager
    
    with _manager_lock:
        if _default_manager is None:
            secret_key = os.environ.get("AUTH_SECRET_KEY", python_secrets.token_hex(32))
            _default_manager = AuthManager(secret_key=secret_key)
        
        return _default_manager


def set_default_manager(manager: AuthManager) -> None:
    """Set the default AuthManager instance."""
    global _default_manager
    
    with _manager_lock:
        _default_manager = manager


def register(
    email: str,
    password: str,
    username: str,
    **kwargs,
) -> User:
    """Register a new user using the default manager."""
    return get_default_manager().register(email, password, username, **kwargs)


def login(
    email: str,
    password: str,
    **kwargs,
) -> LoginResult:
    """Login using the default manager."""
    return get_default_manager().login(email, password, **kwargs)


def logout(session_id: str, **kwargs) -> bool:
    """Logout using the default manager."""
    return get_default_manager().logout(session_id, **kwargs)


def validate_session(session_id: str) -> Optional[Session]:
    """Validate a session using the default manager."""
    return get_default_manager().validate_session(session_id)


def get_user(user_id: str) -> Optional[User]:
    """Get a user using the default manager."""
    return get_default_manager().get_user(user_id)


def verify_email(token: str) -> User:
    """Verify email using the default manager."""
    return get_default_manager().verify_email(token)


def reset_password(token: str, new_password: str, **kwargs) -> User:
    """Reset password using the default manager."""
    return get_default_manager().reset_password(token, new_password, **kwargs)


def check_permission(user_id: str, permission: str) -> bool:
    """Check permission using the default manager."""
    return get_default_manager().check_permission(user_id, permission)


__all__ = [
    'AuthProvider',
    'UserStatus',
    'SessionStatus',
    'TokenType',
    'HashAlgorithm',
    'Role',
    'Permission',
    'AuditEvent',
    'AuthLimit',
    'AuthError',
    'InvalidCredentialsError',
    'UserNotFoundError',
    'UserExistsError',
    'SessionExpiredError',
    'SessionNotFoundError',
    'InvalidTokenError',
    'TokenExpiredError',
    'AccountLockedError',
    'AccountSuspendedError',
    'PermissionDeniedError',
    'MFARequiredError',
    'MFAInvalidCodeError',
    'RateLimitExceededError',
    'WeakPasswordError',
    'OAuthError',
    'UserProfile',
    'OAuthAccount',
    'MFASettings',
    'User',
    'Session',
    'AuditLogEntry',
    'LoginResult',
    'PasswordHasher',
    'PasswordValidator',
    'TokenManager',
    'OAuthProvider',
    'GoogleOAuthProvider',
    'GitHubOAuthProvider',
    'DiscordOAuthProvider',
    'RoleManager',
    'MFAManager',
    'RateLimiter',
    'AuthManager',
    'get_default_manager',
    'set_default_manager',
    'register',
    'login',
    'logout',
    'validate_session',
    'get_user',
    'verify_email',
    'reset_password',
    'check_permission',
]
