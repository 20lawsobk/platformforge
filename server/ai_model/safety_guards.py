"""
Comprehensive Safety Guards System for Platform Forge AI Model

This module provides safety mechanisms to prevent destructive actions without
explicit user confirmation. It includes detection of dangerous operations,
validation against user intent, and risk assessment.

Key Components:
- DestructiveActionDetector: Detects dangerous operations like database drops,
  file deletions, production deployments
- ActionValidator: Validates proposed actions against user instructions
- SafetyConfig: Configurable safety levels and operation whitelists/blacklists
- RiskAssessment: Calculates risk scores with detailed breakdowns

Usage:
    from server.ai_model.safety_guards import (
        DestructiveActionDetector,
        ActionValidator,
        SafetyConfig,
        assess_risk,
        is_destructive,
        requires_confirmation,
        get_safe_alternative,
        validate_against_instructions,
    )
    
    # Quick check if action is destructive
    if is_destructive("DROP TABLE users"):
        confirmation = requires_confirmation("DROP TABLE users")
        print(f"Warning: {confirmation.description}")
    
    # Full risk assessment
    risk = assess_risk("rm -rf /var/log/*", context="production")
    print(f"Risk score: {risk.score}/100")
    
    # Validate against user instructions
    result = validate_against_instructions(
        action="git push --force",
        user_instructions="never force push to main branch"
    )
    if result.conflicts_detected:
        print(f"Conflict: {result.conflict_details}")
"""

import re
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class Severity(Enum):
    """Severity levels for detected risks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def __lt__(self, other):
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) < order.index(other)
    
    def __le__(self, other):
        return self == other or self < other
    
    @property
    def numeric_value(self) -> int:
        """Return numeric value for calculations."""
        mapping = {
            Severity.LOW: 10,
            Severity.MEDIUM: 25,
            Severity.HIGH: 50,
            Severity.CRITICAL: 100
        }
        return mapping[self]


class SafetyLevel(Enum):
    """Safety configuration levels."""
    PERMISSIVE = "permissive"
    NORMAL = "normal"
    STRICT = "strict"


class ActionCategory(Enum):
    """Categories of potentially dangerous actions."""
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    GIT = "git"
    CLOUD_INFRA = "cloud_infrastructure"
    DEPLOYMENT = "deployment"
    CREDENTIALS = "credentials"
    NETWORK = "network"
    SYSTEM = "system"
    CODE_EXECUTION = "code_execution"


@dataclass
class DetectedRisk:
    """Represents a single detected risk in an action."""
    category: ActionCategory
    severity: Severity
    pattern_matched: str
    description: str
    impact: str
    mitigation: str
    confidence: float = 1.0
    line_number: Optional[int] = None
    code_snippet: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "pattern_matched": self.pattern_matched,
            "description": self.description,
            "impact": self.impact,
            "mitigation": self.mitigation,
            "confidence": self.confidence,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
        }


@dataclass
class ConfirmationRequired:
    """Information about an action requiring user confirmation."""
    action_type: str
    severity: Severity
    description: str
    impact: str
    rollback_possible: bool
    suggested_alternatives: List[str] = field(default_factory=list)
    requires_explicit_yes: bool = True
    backup_recommended: bool = False
    estimated_data_loss: str = ""
    affected_resources: List[str] = field(default_factory=list)
    confirmation_prompt: str = ""
    
    def __post_init__(self):
        if not self.confirmation_prompt:
            self.confirmation_prompt = self._generate_prompt()
    
    def _generate_prompt(self) -> str:
        """Generate a user-friendly confirmation prompt."""
        severity_emoji = {
            Severity.LOW: "[LOW]",
            Severity.MEDIUM: "[MEDIUM]",
            Severity.HIGH: "[HIGH]", 
            Severity.CRITICAL: "[CRITICAL]"
        }
        
        prompt_parts = [
            f"{severity_emoji.get(self.severity, '')} {self.description}",
            f"Impact: {self.impact}",
        ]
        
        if self.affected_resources:
            prompt_parts.append(f"Affected resources: {', '.join(self.affected_resources)}")
        
        if self.estimated_data_loss:
            prompt_parts.append(f"Estimated data loss: {self.estimated_data_loss}")
        
        if self.backup_recommended:
            prompt_parts.append("RECOMMENDATION: Create a backup before proceeding")
        
        rollback_msg = "This action CAN be rolled back" if self.rollback_possible else "This action CANNOT be undone"
        prompt_parts.append(rollback_msg)
        
        if self.suggested_alternatives:
            prompt_parts.append(f"Alternatives: {', '.join(self.suggested_alternatives)}")
        
        if self.requires_explicit_yes:
            prompt_parts.append("Type 'yes' to confirm this action")
        
        return "\n".join(prompt_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "severity": self.severity.value,
            "description": self.description,
            "impact": self.impact,
            "rollback_possible": self.rollback_possible,
            "suggested_alternatives": self.suggested_alternatives,
            "requires_explicit_yes": self.requires_explicit_yes,
            "backup_recommended": self.backup_recommended,
            "estimated_data_loss": self.estimated_data_loss,
            "affected_resources": self.affected_resources,
            "confirmation_prompt": self.confirmation_prompt,
        }


@dataclass
class ValidationResult:
    """Result of validating an action against user instructions."""
    is_valid: bool
    conflicts_detected: bool = False
    conflict_details: List[str] = field(default_factory=list)
    matched_restrictions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    suggested_modifications: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "conflicts_detected": self.conflicts_detected,
            "conflict_details": self.conflict_details,
            "matched_restrictions": self.matched_restrictions,
            "confidence": self.confidence,
            "suggested_modifications": self.suggested_modifications,
        }


@dataclass
class RiskAssessmentResult:
    """Complete risk assessment for an action."""
    action: str
    score: int
    severity: Severity
    risks: List[DetectedRisk] = field(default_factory=list)
    is_destructive: bool = False
    requires_confirmation: bool = False
    confirmation_info: Optional[ConfirmationRequired] = None
    mitigations: List[str] = field(default_factory=list)
    safe_alternative: Optional[str] = None
    context_factors: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._calculate_severity()
    
    def _calculate_severity(self):
        """Calculate overall severity based on individual risks."""
        if not self.risks:
            self.severity = Severity.LOW
            return
        
        max_severity = max(risk.severity for risk in self.risks)
        self.severity = max_severity
        
        critical_count = sum(1 for r in self.risks if r.severity == Severity.CRITICAL)
        high_count = sum(1 for r in self.risks if r.severity == Severity.HIGH)
        
        if critical_count >= 1 or (high_count >= 2 and self.score >= 70):
            self.severity = Severity.CRITICAL
        elif high_count >= 1 or self.score >= 50:
            self.severity = Severity.HIGH
        elif self.score >= 25:
            self.severity = Severity.MEDIUM
    
    def get_summary(self) -> str:
        """Generate a human-readable summary."""
        risk_counts = {}
        for risk in self.risks:
            category = risk.category.value
            risk_counts[category] = risk_counts.get(category, 0) + 1
        
        category_summary = ", ".join(f"{count} {cat}" for cat, count in risk_counts.items())
        
        return (
            f"Risk Score: {self.score}/100 ({self.severity.value.upper()})\n"
            f"Destructive: {'Yes' if self.is_destructive else 'No'}\n"
            f"Risks detected: {len(self.risks)} ({category_summary})\n"
            f"Requires confirmation: {'Yes' if self.requires_confirmation else 'No'}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "score": self.score,
            "severity": self.severity.value,
            "risks": [r.to_dict() for r in self.risks],
            "is_destructive": self.is_destructive,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_info": self.confirmation_info.to_dict() if self.confirmation_info else None,
            "mitigations": self.mitigations,
            "safe_alternative": self.safe_alternative,
            "context_factors": self.context_factors,
            "summary": self.get_summary(),
        }


class DestructivePatterns:
    """Comprehensive patterns for detecting destructive actions."""
    
    DATABASE_PATTERNS: List[Tuple[str, str, Severity, str, str]] = [
        (r'\bDROP\s+DATABASE\b', "DROP DATABASE", Severity.CRITICAL,
         "Permanently deletes entire database", "All data in database will be lost permanently"),
        (r'\bDROP\s+TABLE\s+\w+', "DROP TABLE", Severity.CRITICAL,
         "Permanently deletes table and all its data", "All rows in table will be lost"),
        (r'\bDROP\s+SCHEMA\b', "DROP SCHEMA", Severity.CRITICAL,
         "Deletes entire schema and all contained objects", "All tables, views, functions in schema will be lost"),
        (r'\bTRUNCATE\s+TABLE\b', "TRUNCATE TABLE", Severity.HIGH,
         "Removes all rows from table", "All data will be deleted but table structure preserved"),
        (r'\bDELETE\s+FROM\s+\w+\s*(?:;|\s*$|WHERE\s+1\s*=\s*1)', "DELETE all rows", Severity.HIGH,
         "Deletes all rows from table without WHERE clause", "Complete data loss for table"),
        (r'\bDELETE\s+FROM\s+\w+', "DELETE statement", Severity.MEDIUM,
         "Deletes rows from table", "Data matching WHERE clause will be deleted"),
        (r'\bALTER\s+TABLE\s+\w+\s+DROP\s+COLUMN', "DROP COLUMN", Severity.HIGH,
         "Removes column from table", "Column data will be permanently lost"),
        (r'\bALTER\s+TABLE\s+\w+\s+DROP\s+CONSTRAINT', "DROP CONSTRAINT", Severity.MEDIUM,
         "Removes constraint from table", "Data integrity may be compromised"),
        (r'\bUPDATE\s+\w+\s+SET\s+.*(?:;|\s*$)(?!\s*WHERE)', "UPDATE without WHERE", Severity.HIGH,
         "Updates all rows in table without WHERE clause", "All rows will be modified"),
        (r'\bDROP\s+INDEX\b', "DROP INDEX", Severity.MEDIUM,
         "Removes index from table", "Query performance may degrade"),
        (r'\bDROP\s+VIEW\b', "DROP VIEW", Severity.MEDIUM,
         "Removes view definition", "Dependent queries may fail"),
        (r'\bDROP\s+FUNCTION\b', "DROP FUNCTION", Severity.MEDIUM,
         "Removes stored function", "Dependent code may fail"),
        (r'\bDROP\s+PROCEDURE\b', "DROP PROCEDURE", Severity.MEDIUM,
         "Removes stored procedure", "Dependent code may fail"),
        (r'\bDROP\s+TRIGGER\b', "DROP TRIGGER", Severity.MEDIUM,
         "Removes trigger", "Automated actions will no longer execute"),
        (r'\bDROP\s+USER\b', "DROP USER", Severity.HIGH,
         "Removes database user", "User access will be revoked"),
        (r'\bREVOKE\s+ALL\b', "REVOKE ALL privileges", Severity.HIGH,
         "Removes all privileges from user/role", "Access will be completely revoked"),
        (r'\bGRANT\s+.*\bWITH\s+GRANT\s+OPTION', "GRANT with GRANT OPTION", Severity.MEDIUM,
         "Grants privileges with ability to further grant", "Privilege escalation risk"),
    ]
    
    FILESYSTEM_PATTERNS: List[Tuple[str, str, Severity, str, str]] = [
        (r'\brm\s+-rf\s+/', "rm -rf /", Severity.CRITICAL,
         "Recursively removes files from root", "Complete system destruction possible"),
        (r'\brm\s+-rf\s+\*', "rm -rf *", Severity.CRITICAL,
         "Recursively removes all files in directory", "All files in current directory will be deleted"),
        (r'\brm\s+-rf\s+~', "rm -rf ~", Severity.CRITICAL,
         "Recursively removes home directory", "All user files will be deleted"),
        (r'\brm\s+-[rR]f?\s+', "rm -r[f]", Severity.HIGH,
         "Recursive file deletion", "Directory tree will be deleted"),
        (r'\brm\s+-f\s+', "rm -f", Severity.MEDIUM,
         "Force file deletion without confirmation", "Files deleted without prompts"),
        (r'\brmdir\s+', "rmdir", Severity.MEDIUM,
         "Remove directory", "Directory will be deleted"),
        (r'\bunlink\s*\(', "unlink()", Severity.MEDIUM,
         "Remove file programmatically", "File will be deleted"),
        (r'shutil\.rmtree\s*\(', "shutil.rmtree()", Severity.HIGH,
         "Recursively remove directory tree", "Directory and all contents deleted"),
        (r'os\.remove\s*\(', "os.remove()", Severity.MEDIUM,
         "Remove file", "File will be deleted"),
        (r'os\.unlink\s*\(', "os.unlink()", Severity.MEDIUM,
         "Remove file", "File will be deleted"),
        (r'os\.rmdir\s*\(', "os.rmdir()", Severity.MEDIUM,
         "Remove directory", "Directory will be deleted"),
        (r'pathlib\.Path.*\.unlink\s*\(', "Path.unlink()", Severity.MEDIUM,
         "Remove file using pathlib", "File will be deleted"),
        (r'pathlib\.Path.*\.rmdir\s*\(', "Path.rmdir()", Severity.MEDIUM,
         "Remove directory using pathlib", "Directory will be deleted"),
        (r'\bdd\s+if=.*of=/dev/', "dd to device", Severity.CRITICAL,
         "Direct disk write", "Disk contents may be overwritten"),
        (r'\bmkfs\b', "mkfs (format)", Severity.CRITICAL,
         "Create/format filesystem", "All data on device will be lost"),
        (r'\bformat\s+[a-zA-Z]:', "format drive", Severity.CRITICAL,
         "Format disk drive", "All data on drive will be lost"),
        (r'\bdel\s+/[sS]\s+', "del /s (recursive)", Severity.HIGH,
         "Recursive file deletion on Windows", "Directory tree will be deleted"),
        (r'>[\s/]*[a-zA-Z]+.*\.(conf|config|cfg|ini|env)', "Overwrite config file", Severity.HIGH,
         "Redirecting output to config file", "Configuration may be corrupted"),
        (r'>\s*/dev/sd[a-z]', "Write to disk device", Severity.CRITICAL,
         "Direct write to disk device", "Disk contents will be overwritten"),
    ]
    
    GIT_PATTERNS: List[Tuple[str, str, Severity, str, str]] = [
        (r'git\s+push\s+.*--force(?:-with-lease)?', "git push --force", Severity.HIGH,
         "Force push overwrites remote history", "Remote commits may be lost"),
        (r'git\s+push\s+-f\b', "git push -f", Severity.HIGH,
         "Force push (short flag)", "Remote commits may be lost"),
        (r'git\s+reset\s+--hard', "git reset --hard", Severity.HIGH,
         "Hard reset discards all changes", "Uncommitted changes will be lost"),
        (r'git\s+clean\s+-[fxd]+', "git clean -f[xd]", Severity.HIGH,
         "Remove untracked files", "Untracked files will be deleted"),
        (r'git\s+checkout\s+--\s+\.', "git checkout -- .", Severity.MEDIUM,
         "Discard all local changes", "Working directory changes will be lost"),
        (r'git\s+branch\s+-[dD]\s+', "git branch -d/-D", Severity.MEDIUM,
         "Delete branch", "Branch and its unique commits may be lost"),
        (r'git\s+stash\s+drop', "git stash drop", Severity.MEDIUM,
         "Drop stash entry", "Stashed changes will be lost"),
        (r'git\s+stash\s+clear', "git stash clear", Severity.HIGH,
         "Clear all stashes", "All stashed changes will be lost"),
        (r'git\s+reflog\s+expire', "git reflog expire", Severity.HIGH,
         "Expire reflog entries", "Recovery history will be reduced"),
        (r'git\s+gc\s+--prune=now', "git gc --prune=now", Severity.HIGH,
         "Aggressive garbage collection", "Unreachable objects immediately deleted"),
        (r'git\s+filter-branch', "git filter-branch", Severity.CRITICAL,
         "Rewrite repository history", "All commit hashes will change"),
        (r'git\s+rebase\s+-i.*main', "git rebase -i main", Severity.HIGH,
         "Interactive rebase on main", "History of main branch modified"),
        (r'git\s+rebase\s+-i.*master', "git rebase -i master", Severity.HIGH,
         "Interactive rebase on master", "History of master branch modified"),
    ]
    
    CLOUD_INFRA_PATTERNS: List[Tuple[str, str, Severity, str, str]] = [
        (r'terraform\s+destroy', "terraform destroy", Severity.CRITICAL,
         "Destroy all Terraform-managed infrastructure", "All managed resources will be deleted"),
        (r'terraform\s+apply\s+.*-auto-approve', "terraform apply -auto-approve", Severity.HIGH,
         "Apply changes without confirmation", "Infrastructure changes applied immediately"),
        (r'kubectl\s+delete\s+(?:namespace|ns)\s+', "kubectl delete namespace", Severity.CRITICAL,
         "Delete Kubernetes namespace", "All resources in namespace will be deleted"),
        (r'kubectl\s+delete\s+.*--all\b', "kubectl delete --all", Severity.CRITICAL,
         "Delete all resources of a type", "Multiple resources will be deleted"),
        (r'kubectl\s+delete\s+(?:deployment|deploy|pod|svc|service)\s+', "kubectl delete resource", Severity.HIGH,
         "Delete Kubernetes resource", "Resource will be terminated"),
        (r'kubectl\s+drain\s+', "kubectl drain", Severity.HIGH,
         "Drain node", "Pods will be evicted from node"),
        (r'kubectl\s+cordon\s+', "kubectl cordon", Severity.MEDIUM,
         "Cordon node", "No new pods will be scheduled"),
        (r'docker\s+rm\s+-f', "docker rm -f", Severity.MEDIUM,
         "Force remove container", "Running container will be killed and removed"),
        (r'docker\s+system\s+prune\s+-a', "docker system prune -a", Severity.HIGH,
         "Remove all unused Docker resources", "Images, containers, volumes may be removed"),
        (r'docker\s+volume\s+rm', "docker volume rm", Severity.HIGH,
         "Remove Docker volume", "Volume data will be lost"),
        (r'docker\s+image\s+prune\s+-a', "docker image prune -a", Severity.MEDIUM,
         "Remove all unused images", "Unused images will be deleted"),
        (r'docker-compose\s+down\s+-v', "docker-compose down -v", Severity.HIGH,
         "Stop containers and remove volumes", "Volume data will be lost"),
        (r'aws\s+s3\s+rm\s+.*--recursive', "aws s3 rm --recursive", Severity.CRITICAL,
         "Recursively delete S3 objects", "All objects in prefix will be deleted"),
        (r'aws\s+s3\s+rb\s+', "aws s3 rb", Severity.CRITICAL,
         "Remove S3 bucket", "Bucket and contents will be deleted"),
        (r'aws\s+ec2\s+terminate-instances', "aws ec2 terminate-instances", Severity.CRITICAL,
         "Terminate EC2 instances", "Instances will be permanently deleted"),
        (r'aws\s+rds\s+delete-db-instance', "aws rds delete-db-instance", Severity.CRITICAL,
         "Delete RDS instance", "Database instance will be deleted"),
        (r'gcloud\s+.*delete\b', "gcloud delete", Severity.HIGH,
         "Delete Google Cloud resource", "Resource will be removed"),
        (r'az\s+.*delete\b', "az delete", Severity.HIGH,
         "Delete Azure resource", "Resource will be removed"),
        (r'helm\s+uninstall\b', "helm uninstall", Severity.HIGH,
         "Uninstall Helm release", "All release resources will be removed"),
        (r'pulumi\s+destroy', "pulumi destroy", Severity.CRITICAL,
         "Destroy Pulumi stack", "All managed resources will be deleted"),
    ]
    
    DEPLOYMENT_PATTERNS: List[Tuple[str, str, Severity, str, str]] = [
        (r'deploy.*(?:prod|production)', "Deploy to production", Severity.HIGH,
         "Deployment to production environment", "Production service will be modified"),
        (r'kubectl\s+apply.*-n\s+prod', "kubectl apply to prod", Severity.HIGH,
         "Apply to production namespace", "Production workloads affected"),
        (r'kubectl\s+set\s+image.*prod', "Update production image", Severity.HIGH,
         "Change container image in production", "Production pods will restart"),
        (r'--env[= ]prod', "Production environment flag", Severity.HIGH,
         "Operation targeting production", "Production environment affected"),
        (r'RAILS_ENV=production', "Rails production mode", Severity.HIGH,
         "Running in production mode", "Production database may be affected"),
        (r'NODE_ENV=production', "Node production mode", Severity.MEDIUM,
         "Running in production mode", "Production configuration used"),
        (r'--release', "Release deployment", Severity.MEDIUM,
         "Creating release deployment", "New version will be deployed"),
        (r'rollout\s+restart', "Restart rollout", Severity.MEDIUM,
         "Restart deployment rollout", "Pods will be restarted"),
    ]
    
    CREDENTIALS_PATTERNS: List[Tuple[str, str, Severity, str, str]] = [
        (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', "Hardcoded password", Severity.HIGH,
         "Password appears to be hardcoded", "Credentials may be exposed"),
        (r'(?:api[_-]?key|apikey)\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", Severity.HIGH,
         "API key appears to be hardcoded", "API key may be exposed"),
        (r'(?:secret|token)\s*=\s*["\'][^"\']+["\']', "Hardcoded secret/token", Severity.HIGH,
         "Secret or token appears to be hardcoded", "Secret may be exposed"),
        (r'(?:ssh|private)[_-]?key\s*=', "Private key reference", Severity.HIGH,
         "Private key being assigned", "Private key may be exposed"),
        (r'echo\s+.*(?:password|secret|key|token)', "Echo credentials", Severity.HIGH,
         "Credentials echoed to output", "Credentials may be logged"),
        (r'curl.*-u\s+[^$]', "curl with credentials", Severity.MEDIUM,
         "Credentials in curl command", "Credentials may be in history"),
        (r'(?:DELETE|REVOKE)\s+.*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)', "Delete credentials", Severity.HIGH,
         "Deleting security credentials", "Access may be revoked"),
        (r'(?:ROTATE|REGENERATE)\s+.*(?:KEY|SECRET|TOKEN)', "Rotate credentials", Severity.MEDIUM,
         "Rotating security credentials", "Old credentials will be invalidated"),
        (r'chmod\s+777\s+', "chmod 777", Severity.HIGH,
         "Setting world-readable/writable permissions", "Security vulnerability"),
        (r'chmod\s+666\s+', "chmod 666", Severity.MEDIUM,
         "Setting world-readable/writable permissions", "Security vulnerability"),
    ]
    
    SYSTEM_PATTERNS: List[Tuple[str, str, Severity, str, str]] = [
        (r'\bkill\s+-9\s+', "kill -9", Severity.MEDIUM,
         "Force kill process", "Process terminated immediately"),
        (r'\bkillall\s+', "killall", Severity.HIGH,
         "Kill all processes by name", "Multiple processes may be killed"),
        (r'\bpkill\s+', "pkill", Severity.HIGH,
         "Kill processes by pattern", "Multiple processes may be killed"),
        (r'\bshutdown\b', "shutdown", Severity.CRITICAL,
         "System shutdown", "System will be powered off"),
        (r'\breboot\b', "reboot", Severity.CRITICAL,
         "System reboot", "System will restart"),
        (r'\binit\s+0\b', "init 0", Severity.CRITICAL,
         "System halt", "System will be halted"),
        (r'\bsystemctl\s+stop\s+', "systemctl stop", Severity.HIGH,
         "Stop system service", "Service will be stopped"),
        (r'\bservice\s+\w+\s+stop', "service stop", Severity.HIGH,
         "Stop service", "Service will be stopped"),
        (r'iptables\s+-F', "iptables flush", Severity.CRITICAL,
         "Flush all firewall rules", "All firewall rules will be removed"),
        (r'ufw\s+disable', "ufw disable", Severity.HIGH,
         "Disable firewall", "Firewall protection disabled"),
        (r'setenforce\s+0', "Disable SELinux", Severity.HIGH,
         "Disable SELinux enforcement", "Security policy disabled"),
    ]
    
    CODE_EXECUTION_PATTERNS: List[Tuple[str, str, Severity, str, str]] = [
        (r'\beval\s*\(', "eval()", Severity.HIGH,
         "Dynamic code execution with eval", "Arbitrary code may be executed"),
        (r'\bexec\s*\(', "exec()", Severity.HIGH,
         "Dynamic code execution with exec", "Arbitrary code may be executed"),
        (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "subprocess shell=True", Severity.HIGH,
         "Shell command execution", "Command injection risk"),
        (r'os\.system\s*\(', "os.system()", Severity.HIGH,
         "System command execution", "Command injection risk"),
        (r'os\.popen\s*\(', "os.popen()", Severity.HIGH,
         "Process open", "Command injection risk"),
        (r'__import__\s*\(', "__import__()", Severity.MEDIUM,
         "Dynamic import", "Arbitrary module may be loaded"),
        (r'compile\s*\(.*exec', "compile() with exec", Severity.HIGH,
         "Compile and execute code", "Arbitrary code may be executed"),
    ]
    
    @classmethod
    def get_all_patterns(cls) -> Dict[ActionCategory, List[Tuple[str, str, Severity, str, str]]]:
        """Get all patterns organized by category."""
        return {
            ActionCategory.DATABASE: cls.DATABASE_PATTERNS,
            ActionCategory.FILESYSTEM: cls.FILESYSTEM_PATTERNS,
            ActionCategory.GIT: cls.GIT_PATTERNS,
            ActionCategory.CLOUD_INFRA: cls.CLOUD_INFRA_PATTERNS,
            ActionCategory.DEPLOYMENT: cls.DEPLOYMENT_PATTERNS,
            ActionCategory.CREDENTIALS: cls.CREDENTIALS_PATTERNS,
            ActionCategory.SYSTEM: cls.SYSTEM_PATTERNS,
            ActionCategory.CODE_EXECUTION: cls.CODE_EXECUTION_PATTERNS,
        }


class NegativeInstructionPatterns:
    """Patterns for detecting negative instructions from users."""
    
    NEGATIVE_PREFIXES = [
        r"don'?t",
        r"do\s+not",
        r"never",
        r"avoid",
        r"stop",
        r"don'?t\s+ever",
        r"refrain\s+from",
        r"please\s+don'?t",
        r"please\s+do\s+not",
        r"must\s+not",
        r"mustn'?t",
        r"shouldn'?t",
        r"should\s+not",
        r"cannot",
        r"can'?t",
        r"won'?t",
        r"will\s+not",
        r"no\s+longer",
        r"not\s+allowed\s+to",
        r"forbidden\s+to",
        r"prohibited\s+from",
        r"stay\s+away\s+from",
        r"keep\s+away\s+from",
    ]
    
    NEGATIVE_SUFFIXES = [
        r"is\s+not\s+allowed",
        r"is\s+prohibited",
        r"is\s+forbidden",
        r"should\s+be\s+avoided",
        r"must\s+be\s+avoided",
        r"is\s+off\s+limits",
        r"is\s+out\s+of\s+bounds",
    ]
    
    @classmethod
    def compile_patterns(cls) -> List[re.Pattern]:
        """Compile all negative instruction patterns."""
        patterns = []
        
        for prefix in cls.NEGATIVE_PREFIXES:
            pattern = re.compile(rf'\b{prefix}\s+(.+?)(?:[.!?]|$)', re.IGNORECASE)
            patterns.append(pattern)
        
        for suffix in cls.NEGATIVE_SUFFIXES:
            pattern = re.compile(rf'(.+?)\s+{suffix}', re.IGNORECASE)
            patterns.append(pattern)
        
        return patterns


@dataclass
class SafetyConfig:
    """Configuration for safety behavior."""
    level: SafetyLevel = SafetyLevel.NORMAL
    auto_backup: bool = True
    require_confirmation_above: Severity = Severity.MEDIUM
    whitelist: Set[str] = field(default_factory=set)
    blacklist: Set[str] = field(default_factory=set)
    allowed_categories: Set[ActionCategory] = field(default_factory=lambda: set(ActionCategory))
    blocked_categories: Set[ActionCategory] = field(default_factory=set)
    max_risk_score: int = 70
    enable_alternatives: bool = True
    log_all_checks: bool = False
    production_patterns: List[str] = field(default_factory=lambda: ["prod", "production", "live", "main", "master"])
    
    @classmethod
    def strict(cls) -> "SafetyConfig":
        """Create strict safety configuration."""
        return cls(
            level=SafetyLevel.STRICT,
            auto_backup=True,
            require_confirmation_above=Severity.LOW,
            max_risk_score=30,
            enable_alternatives=True,
            log_all_checks=True,
        )
    
    @classmethod
    def normal(cls) -> "SafetyConfig":
        """Create normal safety configuration."""
        return cls(
            level=SafetyLevel.NORMAL,
            auto_backup=True,
            require_confirmation_above=Severity.MEDIUM,
            max_risk_score=70,
            enable_alternatives=True,
            log_all_checks=False,
        )
    
    @classmethod
    def permissive(cls) -> "SafetyConfig":
        """Create permissive safety configuration."""
        return cls(
            level=SafetyLevel.PERMISSIVE,
            auto_backup=False,
            require_confirmation_above=Severity.CRITICAL,
            max_risk_score=100,
            enable_alternatives=False,
            log_all_checks=False,
        )
    
    def is_whitelisted(self, action: str) -> bool:
        """Check if action matches a whitelist pattern."""
        for pattern in self.whitelist:
            if re.search(pattern, action, re.IGNORECASE):
                return True
        return False
    
    def is_blacklisted(self, action: str) -> bool:
        """Check if action matches a blacklist pattern."""
        for pattern in self.blacklist:
            if re.search(pattern, action, re.IGNORECASE):
                return True
        return False
    
    def is_category_allowed(self, category: ActionCategory) -> bool:
        """Check if a category is allowed."""
        if category in self.blocked_categories:
            return False
        if self.allowed_categories and category not in self.allowed_categories:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "auto_backup": self.auto_backup,
            "require_confirmation_above": self.require_confirmation_above.value,
            "whitelist": list(self.whitelist),
            "blacklist": list(self.blacklist),
            "allowed_categories": [c.value for c in self.allowed_categories],
            "blocked_categories": [c.value for c in self.blocked_categories],
            "max_risk_score": self.max_risk_score,
            "enable_alternatives": self.enable_alternatives,
            "log_all_checks": self.log_all_checks,
            "production_patterns": self.production_patterns,
        }


class DestructiveActionDetector:
    """
    Detects dangerous operations that could cause data loss or system damage.
    
    Usage:
        detector = DestructiveActionDetector()
        risks = detector.detect("DROP TABLE users CASCADE")
        for risk in risks:
            print(f"{risk.severity.value}: {risk.description}")
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig.normal()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficient matching."""
        self._compiled_patterns: Dict[ActionCategory, List[Tuple[re.Pattern, str, Severity, str, str]]] = {}
        
        for category, patterns in DestructivePatterns.get_all_patterns().items():
            self._compiled_patterns[category] = [
                (re.compile(pattern, re.IGNORECASE | re.MULTILINE), name, severity, desc, impact)
                for pattern, name, severity, desc, impact in patterns
            ]
    
    def detect(self, action: str, context: Optional[Dict[str, Any]] = None) -> List[DetectedRisk]:
        """
        Detect destructive patterns in an action.
        
        Args:
            action: The action/command/code to analyze
            context: Optional context (e.g., environment, user, etc.)
            
        Returns:
            List of detected risks
        """
        if self.config.is_whitelisted(action):
            return []
        
        context = context or {}
        risks: List[DetectedRisk] = []
        
        lines = action.split('\n')
        
        for category, patterns in self._compiled_patterns.items():
            if not self.config.is_category_allowed(category):
                continue
            
            for compiled_pattern, name, severity, description, impact in patterns:
                for line_num, line in enumerate(lines, 1):
                    matches = compiled_pattern.finditer(line)
                    for match in matches:
                        adjusted_severity = self._adjust_severity(severity, context)
                        mitigation = self._get_mitigation(category, name)
                        
                        risks.append(DetectedRisk(
                            category=category,
                            severity=adjusted_severity,
                            pattern_matched=name,
                            description=description,
                            impact=impact,
                            mitigation=mitigation,
                            line_number=line_num,
                            code_snippet=line.strip(),
                        ))
        
        if self.config.is_blacklisted(action):
            risks.append(DetectedRisk(
                category=ActionCategory.SYSTEM,
                severity=Severity.CRITICAL,
                pattern_matched="BLACKLISTED",
                description="Action matches a blacklisted pattern",
                impact="Action has been explicitly blocked by configuration",
                mitigation="Contact administrator to modify blacklist if needed",
            ))
        
        return risks
    
    def _adjust_severity(self, base_severity: Severity, context: Dict[str, Any]) -> Severity:
        """Adjust severity based on context."""
        if context.get("environment") == "production":
            if base_severity == Severity.MEDIUM:
                return Severity.HIGH
            elif base_severity == Severity.HIGH:
                return Severity.CRITICAL
        
        if context.get("environment") == "development":
            if base_severity == Severity.LOW:
                return Severity.LOW
            elif base_severity == Severity.MEDIUM:
                return Severity.LOW
        
        if self.config.level == SafetyLevel.STRICT:
            if base_severity == Severity.LOW:
                return Severity.MEDIUM
            elif base_severity == Severity.MEDIUM:
                return Severity.HIGH
        
        return base_severity
    
    def _get_mitigation(self, category: ActionCategory, pattern_name: str) -> str:
        """Get mitigation suggestions for a detected pattern."""
        mitigations = {
            ActionCategory.DATABASE: {
                "DROP DATABASE": "Create a backup first with pg_dump or equivalent",
                "DROP TABLE": "Use DROP TABLE IF EXISTS with backup, or consider soft delete",
                "TRUNCATE TABLE": "Consider backing up data or using DELETE with WHERE clause",
                "DELETE all rows": "Add a WHERE clause to limit scope, or backup first",
                "DELETE statement": "Verify WHERE clause and consider transaction wrapper",
                "DROP COLUMN": "Create migration with rollback capability",
                "UPDATE without WHERE": "Always include a WHERE clause to limit scope",
                "default": "Create a database backup before proceeding",
            },
            ActionCategory.FILESYSTEM: {
                "rm -rf /": "Never run this command - it will destroy the system",
                "rm -rf *": "Use ls first to verify files, consider moving to trash",
                "shutil.rmtree()": "Add confirmation prompt and backup important files",
                "default": "Verify path and create backup before deletion",
            },
            ActionCategory.GIT: {
                "git push --force": "Use --force-with-lease for safer force push",
                "git push -f": "Use --force-with-lease instead",
                "git reset --hard": "Create a backup branch first: git branch backup-YYYYMMDD",
                "git clean -f": "Use git clean -n first to preview what will be deleted",
                "default": "Create a backup branch before proceeding",
            },
            ActionCategory.CLOUD_INFRA: {
                "terraform destroy": "Use terraform plan -destroy first to review changes",
                "kubectl delete namespace": "Backup namespace resources with kubectl get all -n <ns> -o yaml",
                "kubectl delete --all": "Target specific resources instead of using --all",
                "docker system prune -a": "Use without -a to keep tagged images",
                "default": "Review and backup affected resources before proceeding",
            },
            ActionCategory.DEPLOYMENT: {
                "Deploy to production": "Test in staging first, have rollback plan ready",
                "default": "Ensure proper testing and rollback procedures are in place",
            },
            ActionCategory.CREDENTIALS: {
                "Hardcoded password": "Use environment variables or secret manager",
                "Hardcoded API key": "Use environment variables or secret manager",
                "chmod 777": "Use minimal required permissions (e.g., 755 or 644)",
                "default": "Store credentials in environment variables or secret manager",
            },
            ActionCategory.SYSTEM: {
                "kill -9": "Try graceful termination first (SIGTERM)",
                "killall": "Target specific process IDs instead",
                "iptables flush": "Save existing rules first: iptables-save > backup.rules",
                "default": "Verify impact and have recovery plan ready",
            },
            ActionCategory.CODE_EXECUTION: {
                "eval()": "Use safer alternatives like ast.literal_eval() for data",
                "exec()": "Validate and sanitize input, consider alternatives",
                "subprocess shell=True": "Use shell=False with command list",
                "os.system()": "Use subprocess.run() with shell=False",
                "default": "Validate all input and use safer alternatives where possible",
            },
        }
        
        category_mitigations = mitigations.get(category, {})
        return category_mitigations.get(pattern_name, category_mitigations.get("default", "Review carefully before proceeding"))
    
    def get_impact_assessment(self, risks: List[DetectedRisk]) -> Dict[str, Any]:
        """Generate an impact assessment from detected risks."""
        if not risks:
            return {
                "overall_severity": Severity.LOW.value,
                "total_risks": 0,
                "categories_affected": [],
                "critical_count": 0,
                "requires_immediate_attention": False,
                "summary": "No significant risks detected",
            }
        
        categories = set(r.category for r in risks)
        severity_counts = {s: 0 for s in Severity}
        
        for risk in risks:
            severity_counts[risk.severity] += 1
        
        max_severity = max(r.severity for r in risks)
        
        return {
            "overall_severity": max_severity.value,
            "total_risks": len(risks),
            "categories_affected": [c.value for c in categories],
            "critical_count": severity_counts[Severity.CRITICAL],
            "high_count": severity_counts[Severity.HIGH],
            "medium_count": severity_counts[Severity.MEDIUM],
            "low_count": severity_counts[Severity.LOW],
            "requires_immediate_attention": severity_counts[Severity.CRITICAL] > 0,
            "summary": self._generate_impact_summary(risks, max_severity),
        }
    
    def _generate_impact_summary(self, risks: List[DetectedRisk], max_severity: Severity) -> str:
        """Generate a human-readable impact summary."""
        critical = [r for r in risks if r.severity == Severity.CRITICAL]
        high = [r for r in risks if r.severity == Severity.HIGH]
        
        parts = []
        
        if critical:
            patterns = set(r.pattern_matched for r in critical)
            parts.append(f"CRITICAL: {len(critical)} critical risk(s) including {', '.join(list(patterns)[:3])}")
        
        if high:
            patterns = set(r.pattern_matched for r in high)
            parts.append(f"HIGH: {len(high)} high-severity risk(s) including {', '.join(list(patterns)[:3])}")
        
        if not parts:
            return f"Detected {len(risks)} risk(s) at {max_severity.value} severity or below"
        
        return "; ".join(parts)


class ActionValidator:
    """
    Validates proposed actions against user intent and instructions.
    
    Detects when AI is about to do something the user explicitly said NOT to do.
    
    Usage:
        validator = ActionValidator()
        result = validator.validate(
            action="git push --force origin main",
            user_instructions="never force push to main branch"
        )
        if result.conflicts_detected:
            print(f"Conflict: {result.conflict_details}")
    """
    
    def __init__(self):
        self._negative_patterns = NegativeInstructionPatterns.compile_patterns()
        self._action_keywords = self._build_action_keywords()
    
    def _build_action_keywords(self) -> Dict[str, List[str]]:
        """Build a mapping of action types to related keywords."""
        return {
            "delete": ["delete", "remove", "drop", "rm", "del", "unlink", "erase", "wipe", "destroy"],
            "modify": ["modify", "change", "update", "alter", "edit", "patch", "transform"],
            "create": ["create", "add", "new", "insert", "make", "generate", "build"],
            "deploy": ["deploy", "push", "publish", "release", "ship", "rollout"],
            "force": ["force", "override", "bypass", "ignore", "skip"],
            "production": ["prod", "production", "live", "main", "master", "release"],
            "database": ["database", "db", "table", "schema", "sql", "query"],
            "git": ["git", "commit", "push", "pull", "merge", "rebase", "branch"],
            "file": ["file", "folder", "directory", "path", "fs"],
            "permission": ["permission", "chmod", "chown", "access", "privilege", "grant"],
            "backup": ["backup", "dump", "export", "snapshot", "archive"],
            "restore": ["restore", "import", "recover", "rollback"],
        }
    
    def validate(
        self,
        action: str,
        user_instructions: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate an action against user instructions.
        
        Args:
            action: The proposed action/command/code
            user_instructions: User's instructions including any restrictions
            context: Optional additional context
            
        Returns:
            ValidationResult indicating if the action conflicts with instructions
        """
        restrictions = self._extract_restrictions(user_instructions)
        
        if not restrictions:
            return ValidationResult(is_valid=True)
        
        conflicts = []
        matched_restrictions = []
        
        for restriction in restrictions:
            conflict = self._check_conflict(action, restriction)
            if conflict:
                conflicts.append(conflict)
                matched_restrictions.append(restriction)
        
        if conflicts:
            suggestions = self._generate_suggestions(action, conflicts)
            confidence = self._calculate_confidence(conflicts)
            
            return ValidationResult(
                is_valid=False,
                conflicts_detected=True,
                conflict_details=conflicts,
                matched_restrictions=matched_restrictions,
                confidence=confidence,
                suggested_modifications=suggestions,
            )
        
        return ValidationResult(is_valid=True)
    
    def _extract_restrictions(self, instructions: str) -> List[str]:
        """Extract negative instructions/restrictions from user instructions."""
        restrictions = []
        
        for pattern in self._negative_patterns:
            matches = pattern.findall(instructions)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if match:
                    restrictions.append(match.strip())
        
        return restrictions
    
    def _check_conflict(self, action: str, restriction: str) -> Optional[str]:
        """Check if an action conflicts with a restriction."""
        action_lower = action.lower()
        restriction_lower = restriction.lower()
        
        restriction_keywords = set(re.findall(r'\b\w+\b', restriction_lower))
        action_keywords = set(re.findall(r'\b\w+\b', action_lower))
        
        common_keywords = restriction_keywords & action_keywords
        if common_keywords:
            significant = [kw for kw in common_keywords if len(kw) > 3]
            if significant:
                return f"Action contains restricted keywords: {', '.join(significant)}"
        
        for action_type, keywords in self._action_keywords.items():
            restriction_has_type = any(kw in restriction_lower for kw in keywords)
            action_has_type = any(kw in action_lower for kw in keywords)
            
            if restriction_has_type and action_has_type:
                return f"Action involves '{action_type}' which user restricted"
        
        targets = re.findall(r'\b(?:main|master|production?|prod|live|users?|data|config|\.env)\b', restriction_lower)
        for target in targets:
            if target in action_lower:
                return f"Action targets restricted resource: {target}"
        
        return None
    
    def _calculate_confidence(self, conflicts: List[str]) -> float:
        """Calculate confidence level of conflict detection."""
        if not conflicts:
            return 0.0
        
        base_confidence = 0.5
        
        base_confidence += min(0.3, len(conflicts) * 0.1)
        
        for conflict in conflicts:
            if "restricted resource" in conflict:
                base_confidence += 0.15
            elif "restricted keywords" in conflict:
                base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _generate_suggestions(self, action: str, conflicts: List[str]) -> List[str]:
        """Generate suggestions for modifying the action to avoid conflicts."""
        suggestions = []
        
        if any("force" in c.lower() for c in conflicts):
            suggestions.append("Remove --force flag or use --force-with-lease for git operations")
        
        if any("production" in c.lower() or "prod" in c.lower() for c in conflicts):
            suggestions.append("Target a non-production environment (development, staging)")
        
        if any("main" in c.lower() or "master" in c.lower() for c in conflicts):
            suggestions.append("Use a feature branch instead of main/master")
        
        if any("delete" in c.lower() or "remove" in c.lower() for c in conflicts):
            suggestions.append("Consider soft delete or archiving instead of permanent deletion")
        
        if not suggestions:
            suggestions.append("Review user instructions and modify action to comply")
        
        return suggestions


class SafeAlternatives:
    """Database of safe alternatives for dangerous operations."""
    
    ALTERNATIVES: Dict[str, Dict[str, Any]] = {
        "rm -rf": {
            "alternative": "mv to trash or use trash-cli",
            "command": "trash-put or mv <files> ~/.local/share/Trash/files/",
            "explanation": "Moving to trash allows recovery if needed",
        },
        "DROP TABLE": {
            "alternative": "Rename table instead of dropping",
            "command": "ALTER TABLE <table> RENAME TO <table>_archived_YYYYMMDD",
            "explanation": "Renaming preserves data while removing from active use",
        },
        "TRUNCATE TABLE": {
            "alternative": "Create backup table before truncating",
            "command": "CREATE TABLE <table>_backup AS SELECT * FROM <table>; TRUNCATE TABLE <table>;",
            "explanation": "Backup table allows data recovery",
        },
        "DELETE FROM": {
            "alternative": "Use soft delete with status column",
            "command": "UPDATE <table> SET deleted_at = NOW() WHERE <condition>",
            "explanation": "Soft delete preserves data for potential recovery",
        },
        "git push --force": {
            "alternative": "Use force-with-lease",
            "command": "git push --force-with-lease",
            "explanation": "Prevents overwriting commits others have pushed",
        },
        "git reset --hard": {
            "alternative": "Create backup branch first",
            "command": "git branch backup-$(date +%Y%m%d) && git reset --hard",
            "explanation": "Backup branch allows recovery of reset commits",
        },
        "terraform destroy": {
            "alternative": "Target specific resources",
            "command": "terraform destroy -target=<resource>",
            "explanation": "Targeted destroy limits blast radius",
        },
        "kubectl delete namespace": {
            "alternative": "Scale down instead of deleting",
            "command": "kubectl scale --replicas=0 deployment --all -n <namespace>",
            "explanation": "Scaling to zero stops resources without deleting them",
        },
        "docker system prune -a": {
            "alternative": "Prune without removing tagged images",
            "command": "docker system prune (without -a)",
            "explanation": "Preserves tagged images that might be needed",
        },
        "chmod 777": {
            "alternative": "Use minimal required permissions",
            "command": "chmod 755 for directories, chmod 644 for files",
            "explanation": "Minimal permissions reduce security risk",
        },
        "eval()": {
            "alternative": "Use ast.literal_eval() for data parsing",
            "command": "import ast; ast.literal_eval(data)",
            "explanation": "literal_eval only parses literal Python structures",
        },
        "os.system()": {
            "alternative": "Use subprocess with shell=False",
            "command": "subprocess.run(['cmd', 'arg1', 'arg2'], shell=False)",
            "explanation": "Avoids shell injection vulnerabilities",
        },
    }
    
    @classmethod
    def get_alternative(cls, action: str) -> Optional[Dict[str, Any]]:
        """Get a safe alternative for a dangerous action."""
        action_lower = action.lower()
        
        for pattern, alternative in cls.ALTERNATIVES.items():
            if pattern.lower() in action_lower:
                return alternative
        
        return None
    
    @classmethod
    def get_all_alternatives(cls) -> Dict[str, Dict[str, Any]]:
        """Get all registered alternatives."""
        return cls.ALTERNATIVES.copy()


def assess_risk(
    action: str,
    context: Optional[str] = None,
    config: Optional[SafetyConfig] = None
) -> RiskAssessmentResult:
    """
    Perform comprehensive risk assessment on an action.
    
    Args:
        action: The action/command/code to assess
        context: Optional context string (e.g., "production", "development")
        config: Optional safety configuration
        
    Returns:
        RiskAssessmentResult with score, risks, and recommendations
    """
    config = config or SafetyConfig.normal()
    detector = DestructiveActionDetector(config)
    
    context_dict = {}
    if context:
        context_dict["environment"] = context
        if any(p in context.lower() for p in config.production_patterns):
            context_dict["is_production"] = True
    
    risks = detector.detect(action, context_dict)
    
    base_score = 0
    for risk in risks:
        base_score += risk.severity.numeric_value * risk.confidence
    
    score = min(100, int(base_score))
    
    is_destructive = any(r.severity >= Severity.HIGH for r in risks)
    
    severity = Severity.LOW
    if score >= 70:
        severity = Severity.CRITICAL
    elif score >= 50:
        severity = Severity.HIGH
    elif score >= 25:
        severity = Severity.MEDIUM
    
    requires_confirm = severity >= config.require_confirmation_above
    
    confirmation_info = None
    if requires_confirm:
        confirmation_info = _build_confirmation_required(action, risks, severity)
    
    mitigations = list(set(r.mitigation for r in risks))
    
    safe_alt = None
    if config.enable_alternatives:
        alt_info = SafeAlternatives.get_alternative(action)
        if alt_info:
            safe_alt = alt_info.get("command")
    
    return RiskAssessmentResult(
        action=action,
        score=score,
        severity=severity,
        risks=risks,
        is_destructive=is_destructive,
        requires_confirmation=requires_confirm,
        confirmation_info=confirmation_info,
        mitigations=mitigations,
        safe_alternative=safe_alt,
        context_factors=context_dict,
    )


def _build_confirmation_required(
    action: str,
    risks: List[DetectedRisk],
    severity: Severity
) -> ConfirmationRequired:
    """Build a ConfirmationRequired object from detected risks."""
    if not risks:
        return ConfirmationRequired(
            action_type="unknown",
            severity=severity,
            description=f"Action requires confirmation: {action[:50]}...",
            impact="Unknown impact",
            rollback_possible=True,
        )
    
    primary_risk = max(risks, key=lambda r: r.severity.numeric_value)
    
    affected_resources = []
    resource_patterns = [
        (r'\b([\w-]+)\.([\w-]+)\b', "table or file"),  # table.column or file.ext
        (r'\b([a-z0-9-]+)(?:-pod|-deploy|-svc)\b', "k8s resource"),  # k8s naming
        (r'/[\w/.-]+', "file path"),  # file paths
    ]
    
    for pattern, _ in resource_patterns:
        matches = re.findall(pattern, action)
        for match in matches:
            if isinstance(match, tuple):
                affected_resources.append('.'.join(match))
            else:
                affected_resources.append(match)
    
    rollback_possible = primary_risk.category in [
        ActionCategory.GIT,
        ActionCategory.DEPLOYMENT,
    ]
    
    alternatives = []
    for risk in risks:
        alt = SafeAlternatives.get_alternative(risk.pattern_matched)
        if alt:
            alternatives.append(alt.get("alternative", ""))
    
    return ConfirmationRequired(
        action_type=primary_risk.pattern_matched,
        severity=severity,
        description=primary_risk.description,
        impact=primary_risk.impact,
        rollback_possible=rollback_possible,
        suggested_alternatives=list(set(alternatives))[:3],
        requires_explicit_yes=severity >= Severity.HIGH,
        backup_recommended=primary_risk.category in [
            ActionCategory.DATABASE,
            ActionCategory.FILESYSTEM,
        ],
        estimated_data_loss=_estimate_data_loss(risks),
        affected_resources=affected_resources[:5],
    )


def _estimate_data_loss(risks: List[DetectedRisk]) -> str:
    """Estimate potential data loss from risks."""
    critical = [r for r in risks if r.severity == Severity.CRITICAL]
    high = [r for r in risks if r.severity == Severity.HIGH]
    
    if critical:
        if any("DATABASE" in r.category.value.upper() for r in critical):
            return "Complete database or table loss possible"
        if any("FILESYSTEM" in r.category.value.upper() for r in critical):
            return "Complete directory tree loss possible"
        return "Significant data loss possible"
    
    if high:
        if any("DATABASE" in r.category.value.upper() for r in high):
            return "Partial table data may be lost"
        if any("FILESYSTEM" in r.category.value.upper() for r in high):
            return "Multiple files may be deleted"
        return "Some data may be affected"
    
    return "Minimal data impact expected"


def is_destructive(action: str, config: Optional[SafetyConfig] = None) -> bool:
    """
    Quick check if an action is potentially destructive.
    
    Args:
        action: The action/command/code to check
        config: Optional safety configuration
        
    Returns:
        True if action contains destructive patterns
    """
    config = config or SafetyConfig.normal()
    detector = DestructiveActionDetector(config)
    risks = detector.detect(action)
    
    return any(r.severity >= Severity.HIGH for r in risks)


def requires_confirmation(
    action: str,
    config: Optional[SafetyConfig] = None
) -> Optional[ConfirmationRequired]:
    """
    Check if an action requires user confirmation.
    
    Args:
        action: The action/command/code to check
        config: Optional safety configuration
        
    Returns:
        ConfirmationRequired object if confirmation needed, None otherwise
    """
    config = config or SafetyConfig.normal()
    assessment = assess_risk(action, config=config)
    
    if assessment.requires_confirmation:
        return assessment.confirmation_info
    
    return None


def get_safe_alternative(action: str) -> Optional[str]:
    """
    Get a safer alternative for a dangerous action.
    
    Args:
        action: The action/command/code
        
    Returns:
        A safer alternative command/approach, or None if no alternative found
    """
    alt_info = SafeAlternatives.get_alternative(action)
    if alt_info:
        return alt_info.get("command")
    return None


def validate_against_instructions(
    action: str,
    user_instructions: str,
    context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    Validate an action against user instructions.
    
    Args:
        action: The proposed action/command/code
        user_instructions: User's instructions including any restrictions
        context: Optional additional context
        
    Returns:
        ValidationResult indicating if the action conflicts with instructions
    """
    validator = ActionValidator()
    return validator.validate(action, user_instructions, context)


__all__ = [
    "Severity",
    "SafetyLevel", 
    "ActionCategory",
    "DetectedRisk",
    "ConfirmationRequired",
    "ValidationResult",
    "RiskAssessmentResult",
    "DestructivePatterns",
    "NegativeInstructionPatterns",
    "SafetyConfig",
    "DestructiveActionDetector",
    "ActionValidator",
    "SafeAlternatives",
    "assess_risk",
    "is_destructive",
    "requires_confirmation",
    "get_safe_alternative",
    "validate_against_instructions",
]
