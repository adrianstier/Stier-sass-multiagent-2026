"""Shared Artifact Registry for Data Science Multi-Agent Framework.

This module provides artifact management capabilities:
- Artifact registration and tracking
- Version control for artifacts
- Lineage tracking (data provenance)
- Artifact discovery and retrieval
- Metadata management
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from pathlib import Path
import uuid
import json
import hashlib
import os


class ArtifactType(str, Enum):
    """Types of artifacts in the data science workflow."""
    # Data artifacts
    RAW_DATA = "raw_data"
    CLEANED_DATA = "cleaned_data"
    TRANSFORMED_DATA = "transformed_data"
    FEATURE_SET = "feature_set"
    TRAIN_SET = "train_set"
    TEST_SET = "test_set"
    VALIDATION_SET = "validation_set"

    # Model artifacts
    MODEL = "model"
    TRAINED_MODEL = "trained_model"  # Alias for MODEL
    MODEL_CHECKPOINT = "model_checkpoint"
    PREPROCESSOR = "preprocessor"
    ENCODER = "encoder"
    FEATURE_SELECTOR = "feature_selector"

    # Report artifacts
    REPORT = "report"  # Generic report
    VISUALIZATION = "visualization"  # Generic visualization
    EDA_REPORT = "eda_report"
    EVALUATION_REPORT = "evaluation_report"
    STATISTICAL_REPORT = "statistical_report"
    FAIRNESS_REPORT = "fairness_report"

    # Visualization artifacts
    PLOT = "plot"
    DASHBOARD = "dashboard"
    CHART = "chart"

    # Pipeline artifacts
    PIPELINE_CONFIG = "pipeline_config"
    EXPERIMENT_CONFIG = "experiment_config"

    # Metadata artifacts
    DATA_SCHEMA = "data_schema"
    FEATURE_SCHEMA = "feature_schema"
    MODEL_METADATA = "model_metadata"

    # Documentation
    DOCUMENTATION = "documentation"
    README = "readme"


class ArtifactStatus(str, Enum):
    """Status of an artifact."""
    DRAFT = "draft"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ArtifactVersion:
    """Version information for an artifact."""
    version: str = "1.0.0"  # Semantic version: major.minor.patch
    path: str = ""  # Storage path for this version
    created_by: str = ""  # Agent that created this version
    changelog: str = ""
    is_latest: bool = True
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ArtifactLineage:
    """Lineage information for an artifact."""
    parent_artifacts: List[str] = field(default_factory=list)  # Artifact IDs
    source_task: str = ""  # Task ID that produced this artifact
    source_agent: str = ""  # Agent that produced this artifact
    transformation_description: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ArtifactMetadata:
    """Extended metadata for artifacts."""
    # Basic metadata
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Technical metadata
    format: str = ""  # parquet, csv, pkl, json, png, etc.
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None  # SHA-256 hash
    encoding: Optional[str] = None

    # Data-specific metadata (if applicable)
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    schema: Optional[Dict[str, str]] = None

    # Model-specific metadata (if applicable)
    model_type: Optional[str] = None
    framework: Optional[str] = None  # sklearn, pytorch, tensorflow, etc.
    metrics: Dict[str, float] = field(default_factory=dict)

    # Custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """Representation of a data science artifact."""
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: ArtifactType = ArtifactType.RAW_DATA
    status: ArtifactStatus = ArtifactStatus.DRAFT

    # Location
    storage_path: str = ""
    relative_path: str = ""

    # Versioning
    version_info: ArtifactVersion = field(default_factory=ArtifactVersion)
    previous_versions: List[str] = field(default_factory=list)  # Version strings

    # Lineage
    lineage: ArtifactLineage = field(default_factory=ArtifactLineage)

    # Metadata
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "status": self.status.value,
            "storage_path": self.storage_path,
            "relative_path": self.relative_path,
            "version_info": {
                "version": self.version_info.version,
                "path": self.version_info.path,
                "created_by": self.version_info.created_by,
                "changelog": self.version_info.changelog,
                "is_latest": self.version_info.is_latest,
                "timestamp": self.version_info.timestamp,
            },
            "lineage": {
                "parent_artifacts": self.lineage.parent_artifacts,
                "source_task": self.lineage.source_task,
                "source_agent": self.lineage.source_agent,
                "transformation_description": self.lineage.transformation_description,
                "created_at": self.lineage.created_at,
            },
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "tags": self.metadata.tags,
                "format": self.metadata.format,
                "size_bytes": self.metadata.size_bytes,
                "checksum": self.metadata.checksum,
                "row_count": self.metadata.row_count,
                "column_count": self.metadata.column_count,
                "schema": self.metadata.schema,
                "model_type": self.metadata.model_type,
                "framework": self.metadata.framework,
                "metrics": self.metadata.metrics,
                "custom": self.metadata.custom,
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create artifact from dictionary."""
        artifact = cls()
        artifact.artifact_id = data.get("artifact_id", artifact.artifact_id)
        artifact.artifact_type = ArtifactType(data.get("artifact_type", "raw_data"))
        artifact.status = ArtifactStatus(data.get("status", "draft"))
        artifact.storage_path = data.get("storage_path", "")
        artifact.relative_path = data.get("relative_path", "")
        artifact.created_at = data.get("created_at", artifact.created_at)
        artifact.updated_at = data.get("updated_at", artifact.updated_at)

        if "version_info" in data:
            vi = data["version_info"]
            artifact.version_info = ArtifactVersion(
                version=vi.get("version", "1.0.0"),
                path=vi.get("path", ""),
                created_by=vi.get("created_by", ""),
                changelog=vi.get("changelog", ""),
                is_latest=vi.get("is_latest", True),
                timestamp=vi.get("timestamp", ""),
            )

        if "lineage" in data:
            lin = data["lineage"]
            artifact.lineage = ArtifactLineage(
                parent_artifacts=lin.get("parent_artifacts", []),
                source_task=lin.get("source_task", ""),
                source_agent=lin.get("source_agent", ""),
                transformation_description=lin.get("transformation_description", ""),
                created_at=lin.get("created_at", ""),
            )

        if "metadata" in data:
            meta = data["metadata"]
            artifact.metadata = ArtifactMetadata(
                name=meta.get("name", ""),
                description=meta.get("description", ""),
                tags=meta.get("tags", []),
                format=meta.get("format", ""),
                size_bytes=meta.get("size_bytes"),
                checksum=meta.get("checksum"),
                row_count=meta.get("row_count"),
                column_count=meta.get("column_count"),
                schema=meta.get("schema"),
                model_type=meta.get("model_type"),
                framework=meta.get("framework"),
                metrics=meta.get("metrics", {}),
                custom=meta.get("custom", {}),
            )

        return artifact


class ArtifactRegistry:
    """Registry for managing data science artifacts."""

    def __init__(self, base_path: str = "./artifacts"):
        self.base_path = Path(base_path)
        self._artifacts: Dict[str, Artifact] = {}
        self._index_by_type: Dict[ArtifactType, List[str]] = {}
        self._index_by_agent: Dict[str, List[str]] = {}
        self._index_by_task: Dict[str, List[str]] = {}
        self._index_by_tag: Dict[str, List[str]] = {}

        # Ensure base directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create artifact directory structure."""
        directories = [
            "data/raw",
            "data/cleaned",
            "data/features",
            "models/experiments",
            "models/production",
            "reports/eda",
            "reports/evaluation",
            "reports/statistical",
            "visualizations",
            "pipelines",
            "metadata",
        ]
        for dir_path in directories:
            (self.base_path / dir_path).mkdir(parents=True, exist_ok=True)

    def register(self, artifact: Artifact) -> str:
        """Register a new artifact."""
        # Generate ID if not provided
        if not artifact.artifact_id:
            artifact.artifact_id = str(uuid.uuid4())

        # Update timestamps
        artifact.created_at = datetime.utcnow().isoformat()
        artifact.updated_at = artifact.created_at

        # Store artifact
        self._artifacts[artifact.artifact_id] = artifact

        # Update indices
        self._update_indices(artifact)

        # Persist metadata
        self._persist_metadata(artifact)

        return artifact.artifact_id

    def _update_indices(self, artifact: Artifact) -> None:
        """Update lookup indices for an artifact."""
        # Type index
        if artifact.artifact_type not in self._index_by_type:
            self._index_by_type[artifact.artifact_type] = []
        if artifact.artifact_id not in self._index_by_type[artifact.artifact_type]:
            self._index_by_type[artifact.artifact_type].append(artifact.artifact_id)

        # Agent index
        agent = artifact.lineage.source_agent
        if agent:
            if agent not in self._index_by_agent:
                self._index_by_agent[agent] = []
            if artifact.artifact_id not in self._index_by_agent[agent]:
                self._index_by_agent[agent].append(artifact.artifact_id)

        # Task index
        task = artifact.lineage.source_task
        if task:
            if task not in self._index_by_task:
                self._index_by_task[task] = []
            if artifact.artifact_id not in self._index_by_task[task]:
                self._index_by_task[task].append(artifact.artifact_id)

        # Tag index
        for tag in artifact.metadata.tags:
            if tag not in self._index_by_tag:
                self._index_by_tag[tag] = []
            if artifact.artifact_id not in self._index_by_tag[tag]:
                self._index_by_tag[tag].append(artifact.artifact_id)

    def _persist_metadata(self, artifact: Artifact) -> None:
        """Persist artifact metadata to disk."""
        metadata_dir = self.base_path / "metadata"
        metadata_file = metadata_dir / f"{artifact.artifact_id}.json"

        with open(metadata_file, "w") as f:
            json.dump(artifact.to_dict(), f, indent=2)

    def get(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID."""
        return self._artifacts.get(artifact_id)

    def update(self, artifact: Artifact) -> None:
        """Update an existing artifact."""
        if artifact.artifact_id not in self._artifacts:
            raise ValueError(f"Artifact {artifact.artifact_id} not found")

        artifact.updated_at = datetime.utcnow().isoformat()
        self._artifacts[artifact.artifact_id] = artifact
        self._persist_metadata(artifact)

    def delete(self, artifact_id: str, soft_delete: bool = True) -> None:
        """Delete an artifact."""
        if artifact_id not in self._artifacts:
            raise ValueError(f"Artifact {artifact_id} not found")

        if soft_delete:
            # Mark as archived
            artifact = self._artifacts[artifact_id]
            artifact.status = ArtifactStatus.ARCHIVED
            artifact.updated_at = datetime.utcnow().isoformat()
            self._persist_metadata(artifact)
        else:
            # Remove from memory and disk
            del self._artifacts[artifact_id]
            metadata_file = self.base_path / "metadata" / f"{artifact_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()

    def find_by_type(self, artifact_type: ArtifactType) -> List[Artifact]:
        """Find all artifacts of a specific type."""
        artifact_ids = self._index_by_type.get(artifact_type, [])
        return [self._artifacts[aid] for aid in artifact_ids if aid in self._artifacts]

    def find_by_agent(self, agent_name: str) -> List[Artifact]:
        """Find all artifacts produced by a specific agent."""
        artifact_ids = self._index_by_agent.get(agent_name, [])
        return [self._artifacts[aid] for aid in artifact_ids if aid in self._artifacts]

    def find_by_task(self, task_id: str) -> List[Artifact]:
        """Find all artifacts produced by a specific task."""
        artifact_ids = self._index_by_task.get(task_id, [])
        return [self._artifacts[aid] for aid in artifact_ids if aid in self._artifacts]

    def find_by_tag(self, tag: str) -> List[Artifact]:
        """Find all artifacts with a specific tag."""
        artifact_ids = self._index_by_tag.get(tag, [])
        return [self._artifacts[aid] for aid in artifact_ids if aid in self._artifacts]

    def find_by_status(self, status: ArtifactStatus) -> List[Artifact]:
        """Find all artifacts with a specific status."""
        return [a for a in self._artifacts.values() if a.status == status]

    def search(
        self,
        artifact_type: Optional[ArtifactType] = None,
        status: Optional[ArtifactStatus] = None,
        agent: Optional[str] = None,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
        name_contains: Optional[str] = None,
    ) -> List[Artifact]:
        """Search artifacts with multiple criteria."""
        results = list(self._artifacts.values())

        if artifact_type:
            results = [a for a in results if a.artifact_type == artifact_type]

        if status:
            results = [a for a in results if a.status == status]

        if agent:
            results = [a for a in results if a.lineage.source_agent == agent]

        if task:
            results = [a for a in results if a.lineage.source_task == task]

        if tags:
            results = [
                a for a in results
                if any(t in a.metadata.tags for t in tags)
            ]

        if name_contains:
            results = [
                a for a in results
                if name_contains.lower() in a.metadata.name.lower()
            ]

        return results

    def get_lineage(self, artifact_id: str) -> Dict[str, Any]:
        """Get complete lineage for an artifact."""
        artifact = self.get(artifact_id)
        if not artifact:
            return {}

        lineage = {
            "artifact": artifact.to_dict(),
            "parents": [],
            "children": [],
        }

        # Get parent artifacts
        for parent_id in artifact.lineage.parent_artifacts:
            parent = self.get(parent_id)
            if parent:
                lineage["parents"].append(parent.to_dict())

        # Find children (artifacts that have this as parent)
        for other_artifact in self._artifacts.values():
            if artifact_id in other_artifact.lineage.parent_artifacts:
                lineage["children"].append(other_artifact.to_dict())

        return lineage

    def get_full_lineage_graph(self, artifact_id: str) -> Dict[str, Any]:
        """Get the full lineage graph (all ancestors and descendants)."""
        visited = set()
        nodes = []
        edges = []

        def traverse_up(aid: str):
            if aid in visited:
                return
            visited.add(aid)

            artifact = self.get(aid)
            if not artifact:
                return

            nodes.append({
                "id": aid,
                "type": artifact.artifact_type.value,
                "name": artifact.metadata.name,
                "agent": artifact.lineage.source_agent,
            })

            for parent_id in artifact.lineage.parent_artifacts:
                edges.append({"source": parent_id, "target": aid})
                traverse_up(parent_id)

        def traverse_down(aid: str):
            for other_artifact in self._artifacts.values():
                if aid in other_artifact.lineage.parent_artifacts:
                    if other_artifact.artifact_id not in visited:
                        visited.add(other_artifact.artifact_id)
                        nodes.append({
                            "id": other_artifact.artifact_id,
                            "type": other_artifact.artifact_type.value,
                            "name": other_artifact.metadata.name,
                            "agent": other_artifact.lineage.source_agent,
                        })
                        edges.append({
                            "source": aid,
                            "target": other_artifact.artifact_id
                        })
                        traverse_down(other_artifact.artifact_id)

        traverse_up(artifact_id)
        traverse_down(artifact_id)

        return {"nodes": nodes, "edges": edges}

    def create_version(
        self,
        artifact_id: str,
        new_path: str,
        changelog: str,
        agent: str,
    ) -> Artifact:
        """Create a new version of an existing artifact."""
        original = self.get(artifact_id)
        if not original:
            raise ValueError(f"Artifact {artifact_id} not found")

        # Parse and increment version
        parts = original.version_info.version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_version = ".".join(parts)

        # Mark old version as not latest
        original.version_info.is_latest = False
        self.update(original)

        # Create new artifact
        new_artifact = Artifact(
            artifact_type=original.artifact_type,
            status=ArtifactStatus.DRAFT,
            storage_path=new_path,
            relative_path=str(Path(new_path).relative_to(self.base_path)),
            version_info=ArtifactVersion(
                version=new_version,
                created_by=agent,
                changelog=changelog,
                is_latest=True,
            ),
            lineage=ArtifactLineage(
                parent_artifacts=[artifact_id],
                source_agent=agent,
                transformation_description=f"New version: {changelog}",
            ),
            metadata=ArtifactMetadata(
                name=original.metadata.name,
                description=original.metadata.description,
                tags=original.metadata.tags.copy(),
                format=original.metadata.format,
            ),
        )
        new_artifact.previous_versions = original.previous_versions + [
            original.version_info.version
        ]

        self.register(new_artifact)
        return new_artifact

    def promote_to_production(self, artifact_id: str) -> None:
        """Promote an artifact to production status."""
        artifact = self.get(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact {artifact_id} not found")

        if artifact.status == ArtifactStatus.PRODUCTION:
            return

        # Demote any existing production artifact of the same type and name
        for other in self.find_by_type(artifact.artifact_type):
            if (
                other.metadata.name == artifact.metadata.name
                and other.status == ArtifactStatus.PRODUCTION
                and other.artifact_id != artifact_id
            ):
                other.status = ArtifactStatus.DEPRECATED
                self.update(other)

        # Promote this artifact
        artifact.status = ArtifactStatus.PRODUCTION
        self.update(artifact)

    def load_from_disk(self) -> None:
        """Load all artifacts from disk metadata files."""
        metadata_dir = self.base_path / "metadata"
        if not metadata_dir.exists():
            return

        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    content = f.read().strip()
                    if not content:
                        continue  # Skip empty files
                    data = json.loads(content)
                    artifact = Artifact.from_dict(data)
                    self._artifacts[artifact.artifact_id] = artifact
                    self._update_indices(artifact)
            except (json.JSONDecodeError, KeyError) as e:
                # Skip corrupt or invalid files
                continue

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = {
            "total_artifacts": len(self._artifacts),
            "by_type": {},
            "by_status": {},
            "by_agent": {},
            "total_size_bytes": 0,
        }

        for artifact in self._artifacts.values():
            # By type
            type_key = artifact.artifact_type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1

            # By status
            status_key = artifact.status.value
            stats["by_status"][status_key] = stats["by_status"].get(status_key, 0) + 1

            # By agent
            agent = artifact.lineage.source_agent
            if agent:
                stats["by_agent"][agent] = stats["by_agent"].get(agent, 0) + 1

            # Size
            if artifact.metadata.size_bytes:
                stats["total_size_bytes"] += artifact.metadata.size_bytes

        return stats


# Helper functions
def compute_file_checksum(file_path: str) -> str:
    """Compute SHA-256 checksum for a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def create_artifact_from_file(
    file_path: str,
    artifact_type: ArtifactType,
    name: str,
    description: str,
    agent: str,
    task_id: str,
    tags: Optional[List[str]] = None,
    parent_artifacts: Optional[List[str]] = None,
) -> Artifact:
    """Create an artifact from an existing file."""
    path = Path(file_path)

    artifact = Artifact(
        artifact_type=artifact_type,
        status=ArtifactStatus.DRAFT,
        storage_path=str(path.absolute()),
        relative_path=str(path),
        version_info=ArtifactVersion(
            version="1.0.0",
            created_by=agent,
            changelog="Initial version",
        ),
        lineage=ArtifactLineage(
            parent_artifacts=parent_artifacts or [],
            source_task=task_id,
            source_agent=agent,
        ),
        metadata=ArtifactMetadata(
            name=name,
            description=description,
            tags=tags or [],
            format=path.suffix.lstrip("."),
            size_bytes=get_file_size(file_path) if path.exists() else None,
            checksum=compute_file_checksum(file_path) if path.exists() else None,
        ),
    )

    return artifact


# Global registry instance
_registry: Optional[ArtifactRegistry] = None


def get_artifact_registry(base_path: str = "./artifacts") -> ArtifactRegistry:
    """Get the global artifact registry instance."""
    global _registry
    if _registry is None:
        _registry = ArtifactRegistry(base_path)
        _registry.load_from_disk()
    return _registry


def reset_artifact_registry() -> None:
    """Reset the global artifact registry (useful for testing)."""
    global _registry
    _registry = None
