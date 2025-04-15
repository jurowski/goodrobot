from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Union, Callable, Any, Tuple
from enum import Enum, auto
from datetime import datetime
import hashlib
import json
from semantic_version import Version
import difflib
from collections import defaultdict

class VersionStatus(Enum):
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class ChangeType(Enum):
    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features, backward compatible
    PATCH = "patch"  # Bug fixes, backward compatible
    HOTFIX = "hotfix"  # Emergency fixes

@dataclass
class VersionMetadata:
    """Metadata for version control"""
    created_by: str
    created_at: datetime
    status: VersionStatus
    change_type: ChangeType
    description: str
    dependencies: Set[str] = field(default_factory=set)
    reviewers: Set[str] = field(default_factory=set)
    review_comments: List[Dict] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

@dataclass
class VersionDiff:
    """Represents differences between versions"""
    added: List[str]
    removed: List[str]
    modified: List[str]
    unchanged: List[str]
    change_type: ChangeType
    compatibility_issues: List[str]

class VersionControl:
    """Advanced version control system"""
    
    def __init__(self):
        self.versions: Dict[Version, Dict] = {}
        self.metadata: Dict[Version, VersionMetadata] = {}
        self.branches: Dict[str, List[Version]] = defaultdict(list)
        self.tags: Dict[str, Version] = {}
        self.locks: Set[Version] = set()
        self.version_graph: Dict[Version, Set[Version]] = defaultdict(set)
        self.merge_strategies: Dict[str, Callable] = {}
        
    async def create_version(self,
                           content: Dict,
                           version: Version,
                           metadata: VersionMetadata) -> Version:
        """Create a new version"""
        if version in self.versions:
            raise ValueError(f"Version {version} already exists")
            
        # Generate version hash
        version_hash = self._generate_hash(content)
        
        # Store version
        self.versions[version] = {
            'content': content,
            'hash': version_hash,
            'created_at': metadata.created_at
        }
        
        # Store metadata
        self.metadata[version] = metadata
        
        # Update version graph
        if self.versions:
            parent_version = max(v for v in self.versions.keys() if v < version)
            self.version_graph[parent_version].add(version)
            
        return version
        
    def create_branch(self,
                     name: str,
                     base_version: Version,
                     author: str) -> Version:
        """Create a new branch"""
        if base_version not in self.versions:
            raise ValueError(f"Base version {base_version} not found")
            
        # Create branch version
        branch_version = Version(
            f"{base_version.major}.{base_version.minor}.{base_version.patch}-{name}.1"
        )
        
        # Copy content and create metadata
        base_content = self.versions[base_version]['content']
        metadata = VersionMetadata(
            created_by=author,
            created_at=datetime.now(),
            status=VersionStatus.DRAFT,
            change_type=ChangeType.MINOR,
            description=f"Branched from {base_version}"
        )
        
        # Create branch version
        self.versions[branch_version] = {
            'content': base_content.copy(),
            'hash': self._generate_hash(base_content),
            'created_at': datetime.now()
        }
        
        self.metadata[branch_version] = metadata
        self.branches[name].append(branch_version)
        
        return branch_version
        
    async def merge_versions(self,
                           source: Version,
                           target: Version,
                           strategy: str = "recursive") -> Optional[Version]:
        """Merge two versions"""
        if source not in self.versions or target not in self.versions:
            raise ValueError("Invalid source or target version")
            
        # Check for conflicts
        conflicts = self._detect_conflicts(source, target)
        if conflicts:
            return None
            
        # Apply merge strategy
        if strategy in self.merge_strategies:
            merged_content = self.merge_strategies[strategy](
                self.versions[source]['content'],
                self.versions[target]['content']
            )
        else:
            merged_content = self._default_merge(
                self.versions[source]['content'],
                self.versions[target]['content']
            )
            
        # Create new version for merge result
        new_version = Version(
            f"{max(source.major, target.major)}."
            f"{max(source.minor, target.minor)}."
            f"{max(source.patch, target.patch) + 1}"
        )
        
        metadata = VersionMetadata(
            created_by="system",
            created_at=datetime.now(),
            status=VersionStatus.REVIEW,
            change_type=ChangeType.MINOR,
            description=f"Merged {source} into {target}"
        )
        
        await self.create_version(merged_content, new_version, metadata)
        return new_version
        
    def compare_versions(self,
                        version1: Version,
                        version2: Version) -> VersionDiff:
        """Compare two versions"""
        if version1 not in self.versions or version2 not in self.versions:
            raise ValueError("Invalid versions")
            
        content1 = self.versions[version1]['content']
        content2 = self.versions[version2]['content']
        
        # Compare content
        added = []
        removed = []
        modified = []
        unchanged = []
        
        all_keys = set(content1.keys()) | set(content2.keys())
        
        for key in all_keys:
            if key not in content1:
                added.append(key)
            elif key not in content2:
                removed.append(key)
            elif content1[key] != content2[key]:
                modified.append(key)
            else:
                unchanged.append(key)
                
        # Determine change type
        change_type = self._determine_change_type(added, removed, modified)
        
        # Check compatibility
        compatibility_issues = self._check_compatibility(
            version1, version2, added, removed, modified
        )
        
        return VersionDiff(
            added=added,
            removed=removed,
            modified=modified,
            unchanged=unchanged,
            change_type=change_type,
            compatibility_issues=compatibility_issues
        )
        
    def tag_version(self, version: Version, tag: str, message: str = ""):
        """Tag a specific version"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
            
        self.tags[tag] = version
        self.metadata[version].tags.add(tag)
        
    def lock_version(self, version: Version):
        """Lock a version to prevent modifications"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
            
        self.locks.add(version)
        
    def unlock_version(self, version: Version):
        """Unlock a version"""
        self.locks.discard(version)
        
    def get_version_history(self,
                          version: Version,
                          include_branches: bool = True) -> List[Dict]:
        """Get complete history of a version"""
        history = []
        current = version
        
        while current:
            metadata = self.metadata[current]
            history.append({
                'version': str(current),
                'created_at': metadata.created_at,
                'created_by': metadata.created_by,
                'status': metadata.status.value,
                'change_type': metadata.change_type.value,
                'description': metadata.description,
                'tags': list(metadata.tags)
            })
            
            if include_branches:
                # Include branch information
                branch_versions = [
                    v for v in self.version_graph[current]
                    if v.prerelease
                ]
                for branch_version in branch_versions:
                    history.extend(
                        self.get_version_history(branch_version, False)
                    )
                    
            # Move to parent version
            parents = [v for v in self.versions.keys() if v < current]
            current = max(parents) if parents else None
            
        return history
        
    def _generate_hash(self, content: Dict) -> str:
        """Generate a unique hash for content"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
        
    def _detect_conflicts(self,
                         source: Version,
                         target: Version) -> List[str]:
        """Detect conflicts between versions"""
        conflicts = []
        source_content = self.versions[source]['content']
        target_content = self.versions[target]['content']
        
        # Check for direct conflicts
        for key in set(source_content.keys()) & set(target_content.keys()):
            if source_content[key] != target_content[key]:
                conflicts.append(f"Conflict in {key}")
                
        return conflicts
        
    def _default_merge(self,
                      source_content: Dict,
                      target_content: Dict) -> Dict:
        """Default merge strategy"""
        merged = target_content.copy()
        merged.update(source_content)
        return merged
        
    def _determine_change_type(self,
                             added: List[str],
                             removed: List[str],
                             modified: List[str]) -> ChangeType:
        """Determine type of change"""
        if removed:
            return ChangeType.MAJOR
        elif added:
            return ChangeType.MINOR
        elif modified:
            return ChangeType.PATCH
        return ChangeType.PATCH
        
    def _check_compatibility(self,
                           version1: Version,
                           version2: Version,
                           added: List[str],
                           removed: List[str],
                           modified: List[str]) -> List[str]:
        """Check for compatibility issues"""
        issues = []
        
        if removed:
            issues.append(f"Removed features: {', '.join(removed)}")
            
        if modified:
            # Check for breaking changes in modified features
            for feature in modified:
                if self._is_breaking_change(version1, version2, feature):
                    issues.append(f"Breaking change in {feature}")
                    
        return issues
        
    def _is_breaking_change(self,
                           version1: Version,
                           version2: Version,
                           feature: str) -> bool:
        """Check if a change is breaking"""
        # Implementation depends on specific compatibility rules
        return False
