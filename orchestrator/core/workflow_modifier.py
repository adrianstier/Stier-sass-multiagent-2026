"""Dynamic Workflow Modification.

Enables runtime workflow changes:
- Add/remove/modify tasks mid-execution
- Support conditional branching based on gate outcomes
- Enable parallel branch spawning for independent features
- API endpoints for human intervention
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from uuid import UUID
import uuid as uuid_module

from orchestrator.core.database import get_db
from orchestrator.core.models import (
    Run, Task, Event, Artifact,
    RunStatus, TaskStatus, GateStatus
)
from orchestrator.core.task_dsl import TaskSpec, ValidationMethod
from orchestrator.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Workflow Branch
# =============================================================================

class WorkflowBranch:
    """Represents a parallel branch in the workflow."""

    def __init__(
        self,
        name: str,
        tasks: List[TaskSpec],
        condition: Optional[str] = None,
        merge_strategy: str = "wait_all",
    ):
        """
        Args:
            name: Branch name for identification
            tasks: List of TaskSpecs for this branch
            condition: Optional condition expression for branch execution
            merge_strategy: How to merge back (wait_all, wait_any, no_wait)
        """
        self.name = name
        self.tasks = tasks
        self.condition = condition
        self.merge_strategy = merge_strategy
        self.created_at = datetime.utcnow()


# =============================================================================
# Workflow Modifier
# =============================================================================

class WorkflowModifier:
    """Modifies workflows at runtime."""

    def __init__(self, run_id: str):
        self.run_id = run_id

    def add_task(
        self,
        task_spec: TaskSpec,
        after_task_type: Optional[str] = None,
        added_by: str = "system",
    ) -> Optional[Task]:
        """Add a new task to the running workflow.

        Args:
            task_spec: The task specification
            after_task_type: Insert after this task type (adds as dependency)
            added_by: Who added this task

        Returns:
            The created Task or None if failed
        """
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            if not run:
                logger.error("add_task_failed", run_id=self.run_id, reason="run_not_found")
                return None

            if run.status not in [RunStatus.RUNNING, RunStatus.NEEDS_INPUT]:
                logger.error(
                    "add_task_failed",
                    run_id=self.run_id,
                    reason=f"invalid_status_{run.status.value}"
                )
                return None

            # Build dependencies
            dependencies = []
            if after_task_type:
                # Find the task to depend on
                dep_task = db.query(Task).filter(
                    Task.run_id == UUID(self.run_id),
                    Task.task_type == after_task_type,
                ).first()
                if dep_task:
                    dependencies.append(str(dep_task.id))

            # Add any explicitly specified dependencies
            for dep_type in task_spec.dependencies:
                dep_task = db.query(Task).filter(
                    Task.run_id == UUID(self.run_id),
                    Task.task_type == dep_type,
                ).first()
                if dep_task and str(dep_task.id) not in dependencies:
                    dependencies.append(str(dep_task.id))

            # Generate idempotency key
            idempotency_key = task_spec.generate_idempotency_key(self.run_id)

            # Check for existing task with same idempotency key
            existing = db.query(Task).filter(
                Task.run_id == UUID(self.run_id),
                Task.idempotency_key == idempotency_key,
            ).first()

            if existing:
                logger.warning(
                    "task_already_exists",
                    run_id=self.run_id,
                    task_type=task_spec.task_type,
                )
                return existing

            # Create the task
            task = Task(
                id=uuid_module.uuid4(),
                run_id=UUID(self.run_id),
                task_type=task_spec.task_type,
                assigned_role=task_spec.assigned_role,
                description=task_spec.description,
                dependencies=dependencies,
                expected_artifacts=task_spec.expected_artifacts,
                validation_method=task_spec.validation_method.value,
                priority=task_spec.priority,
                idempotency_key=idempotency_key,
                max_retries=task_spec.max_retries,
                task_metadata={
                    "is_gate": task_spec.is_gate,
                    "gate_blocks": task_spec.gate_blocks,
                    "added_dynamically": True,
                    "added_by": added_by,
                    "added_at": datetime.utcnow().isoformat(),
                },
            )
            db.add(task)

            # Record event
            event = Event(
                run_id=UUID(self.run_id),
                event_type="task_added_dynamically",
                actor=added_by,
                data={
                    "task_id": str(task.id),
                    "task_type": task_spec.task_type,
                    "assigned_role": task_spec.assigned_role,
                    "after_task_type": after_task_type,
                    "dependencies": dependencies,
                },
            )
            db.add(event)
            db.commit()

            logger.info(
                "task_added_dynamically",
                run_id=self.run_id,
                task_id=str(task.id),
                task_type=task_spec.task_type,
                added_by=added_by,
            )

            return task

    def remove_task(
        self,
        task_id: str,
        removed_by: str = "system",
        reason: str = "",
    ) -> bool:
        """Remove a pending task from the workflow.

        Only pending tasks can be removed. Running/completed tasks cannot be removed.

        Args:
            task_id: The task ID to remove
            removed_by: Who removed this task
            reason: Reason for removal

        Returns:
            True if removed, False otherwise
        """
        with get_db() as db:
            task = db.query(Task).filter(
                Task.id == UUID(task_id),
                Task.run_id == UUID(self.run_id),
            ).first()

            if not task:
                return False

            if task.status != TaskStatus.PENDING:
                logger.warning(
                    "cannot_remove_task",
                    task_id=task_id,
                    status=task.status.value,
                    reason="only_pending_tasks_can_be_removed",
                )
                return False

            # Update tasks that depend on this one
            dependent_tasks = db.query(Task).filter(
                Task.run_id == UUID(self.run_id),
            ).all()

            for dep_task in dependent_tasks:
                if task_id in (dep_task.dependencies or []):
                    # Remove this task from dependencies
                    new_deps = [d for d in dep_task.dependencies if d != task_id]
                    dep_task.dependencies = new_deps

            # Mark task as skipped instead of deleting (preserve history)
            task.status = TaskStatus.SKIPPED
            task.error_message = f"Removed: {reason}"
            task.completed_at = datetime.utcnow()

            # Record event
            event = Event(
                run_id=UUID(self.run_id),
                task_id=UUID(task_id),
                event_type="task_removed",
                actor=removed_by,
                data={
                    "reason": reason,
                    "task_type": task.task_type,
                },
            )
            db.add(event)
            db.commit()

            logger.info(
                "task_removed",
                run_id=self.run_id,
                task_id=task_id,
                removed_by=removed_by,
                reason=reason,
            )

            return True

    def modify_task(
        self,
        task_id: str,
        modifications: Dict[str, Any],
        modified_by: str = "system",
    ) -> bool:
        """Modify a pending task's properties.

        Args:
            task_id: The task ID to modify
            modifications: Dict of properties to modify
            modified_by: Who modified this task

        Returns:
            True if modified, False otherwise
        """
        allowed_modifications = {
            "description", "priority", "max_retries", "assigned_role",
            "expected_artifacts", "input_data"
        }

        with get_db() as db:
            task = db.query(Task).filter(
                Task.id == UUID(task_id),
                Task.run_id == UUID(self.run_id),
            ).first()

            if not task:
                return False

            if task.status != TaskStatus.PENDING:
                logger.warning(
                    "cannot_modify_task",
                    task_id=task_id,
                    status=task.status.value,
                )
                return False

            # Apply modifications
            applied = {}
            for key, value in modifications.items():
                if key in allowed_modifications:
                    old_value = getattr(task, key, None)
                    setattr(task, key, value)
                    applied[key] = {"old": old_value, "new": value}

            if not applied:
                return False

            # Record event
            event = Event(
                run_id=UUID(self.run_id),
                task_id=UUID(task_id),
                event_type="task_modified",
                actor=modified_by,
                data={
                    "modifications": applied,
                },
            )
            db.add(event)
            db.commit()

            logger.info(
                "task_modified",
                run_id=self.run_id,
                task_id=task_id,
                modified_by=modified_by,
                modifications=list(applied.keys()),
            )

            return True

    def add_dependency(
        self,
        task_id: str,
        depends_on_task_id: str,
        added_by: str = "system",
    ) -> bool:
        """Add a dependency between tasks.

        Args:
            task_id: The task that should wait
            depends_on_task_id: The task to depend on
            added_by: Who added this dependency

        Returns:
            True if added, False otherwise
        """
        with get_db() as db:
            task = db.query(Task).filter(
                Task.id == UUID(task_id),
                Task.run_id == UUID(self.run_id),
            ).first()

            dep_task = db.query(Task).filter(
                Task.id == UUID(depends_on_task_id),
                Task.run_id == UUID(self.run_id),
            ).first()

            if not task or not dep_task:
                return False

            if task.status != TaskStatus.PENDING:
                return False

            # Check for circular dependency
            if self._would_create_cycle(db, task_id, depends_on_task_id):
                logger.warning(
                    "circular_dependency_prevented",
                    task_id=task_id,
                    depends_on=depends_on_task_id,
                )
                return False

            # Add dependency
            current_deps = task.dependencies or []
            if depends_on_task_id not in current_deps:
                task.dependencies = current_deps + [depends_on_task_id]

                event = Event(
                    run_id=UUID(self.run_id),
                    task_id=UUID(task_id),
                    event_type="dependency_added",
                    actor=added_by,
                    data={
                        "depends_on": depends_on_task_id,
                        "depends_on_type": dep_task.task_type,
                    },
                )
                db.add(event)
                db.commit()

            return True

    def _would_create_cycle(
        self, db, task_id: str, new_dep_id: str
    ) -> bool:
        """Check if adding a dependency would create a cycle."""
        visited: Set[str] = set()

        def dfs(current_id: str) -> bool:
            if current_id == task_id:
                return True
            if current_id in visited:
                return False

            visited.add(current_id)

            task = db.query(Task).filter(
                Task.id == UUID(current_id),
                Task.run_id == UUID(self.run_id),
            ).first()

            if task:
                for dep_id in (task.dependencies or []):
                    if dfs(dep_id):
                        return True

            return False

        return dfs(new_dep_id)

    def create_branch(
        self,
        branch: WorkflowBranch,
        starts_after_task_id: Optional[str] = None,
        created_by: str = "system",
    ) -> List[Task]:
        """Create a parallel branch in the workflow.

        Args:
            branch: The WorkflowBranch definition
            starts_after_task_id: Optional task ID that all branch tasks depend on
            created_by: Who created this branch

        Returns:
            List of created Tasks
        """
        created_tasks = []
        task_id_map: Dict[str, str] = {}

        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            if not run or run.status not in [RunStatus.RUNNING, RunStatus.NEEDS_INPUT]:
                return []

            # Create all tasks in the branch
            for task_spec in branch.tasks:
                # Build dependencies
                dependencies = []

                # Add dependency on starting task if specified
                if starts_after_task_id and not task_spec.dependencies:
                    dependencies.append(starts_after_task_id)

                # Add internal branch dependencies
                for dep_type in task_spec.dependencies:
                    if dep_type in task_id_map:
                        dependencies.append(task_id_map[dep_type])
                    else:
                        # Try to find existing task with this type
                        existing = db.query(Task).filter(
                            Task.run_id == UUID(self.run_id),
                            Task.task_type == dep_type,
                        ).first()
                        if existing:
                            dependencies.append(str(existing.id))

                # Generate idempotency key with branch name
                idempotency_key = f"{self.run_id}_{branch.name}_{task_spec.task_type}"
                import hashlib
                idempotency_key = hashlib.sha256(idempotency_key.encode()).hexdigest()[:32]

                task = Task(
                    id=uuid_module.uuid4(),
                    run_id=UUID(self.run_id),
                    task_type=f"{branch.name}_{task_spec.task_type}",
                    assigned_role=task_spec.assigned_role,
                    description=task_spec.description,
                    dependencies=dependencies,
                    expected_artifacts=task_spec.expected_artifacts,
                    validation_method=task_spec.validation_method.value,
                    priority=task_spec.priority,
                    idempotency_key=idempotency_key,
                    max_retries=task_spec.max_retries,
                    task_metadata={
                        "is_gate": task_spec.is_gate,
                        "gate_blocks": task_spec.gate_blocks,
                        "branch_name": branch.name,
                        "branch_merge_strategy": branch.merge_strategy,
                        "added_dynamically": True,
                        "added_by": created_by,
                    },
                )
                db.add(task)
                task_id_map[task_spec.task_type] = str(task.id)
                created_tasks.append(task)

            # Record event
            event = Event(
                run_id=UUID(self.run_id),
                event_type="branch_created",
                actor=created_by,
                data={
                    "branch_name": branch.name,
                    "task_count": len(created_tasks),
                    "merge_strategy": branch.merge_strategy,
                    "condition": branch.condition,
                    "task_types": [t.task_type for t in created_tasks],
                },
            )
            db.add(event)
            db.commit()

            logger.info(
                "branch_created",
                run_id=self.run_id,
                branch_name=branch.name,
                task_count=len(created_tasks),
                created_by=created_by,
            )

            return created_tasks

    def insert_gate(
        self,
        gate_task_type: str,
        gate_role: str,
        after_task_types: List[str],
        blocks_task_types: List[str],
        description: str = "Quality gate",
        added_by: str = "system",
    ) -> Optional[Task]:
        """Insert a new quality gate into the workflow.

        Args:
            gate_task_type: Type name for the gate task
            gate_role: Role that performs the gate (e.g., code_reviewer)
            after_task_types: Task types this gate depends on
            blocks_task_types: Task types that this gate blocks if failed
            description: Gate description
            added_by: Who added this gate

        Returns:
            The created gate Task or None
        """
        gate_spec = TaskSpec(
            task_type=gate_task_type,
            assigned_role=gate_role,
            description=description,
            dependencies=after_task_types,
            expected_artifacts=[f"{gate_task_type}_report"],
            validation_method=ValidationMethod.GATE_APPROVAL,
            priority=100,  # High priority for gates
            is_gate=True,
            gate_blocks=blocks_task_types,
        )

        return self.add_task(gate_spec, added_by=added_by)

    def get_workflow_graph(self) -> Dict[str, Any]:
        """Get the current workflow as a graph structure.

        Returns:
            Dict with nodes (tasks) and edges (dependencies)
        """
        with get_db() as db:
            tasks = db.query(Task).filter(
                Task.run_id == UUID(self.run_id)
            ).all()

            nodes = []
            edges = []

            for task in tasks:
                nodes.append({
                    "id": str(task.id),
                    "type": task.task_type,
                    "role": task.assigned_role,
                    "status": task.status.value,
                    "is_gate": (task.task_metadata or {}).get("is_gate", False),
                    "branch": (task.task_metadata or {}).get("branch_name"),
                    "priority": task.priority,
                })

                for dep_id in (task.dependencies or []):
                    edges.append({
                        "from": dep_id,
                        "to": str(task.id),
                    })

            return {
                "run_id": self.run_id,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }


# =============================================================================
# Conditional Branching
# =============================================================================

class ConditionalBranchEvaluator:
    """Evaluates conditions for conditional branching."""

    def __init__(self, run_id: str):
        self.run_id = run_id

    def evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition expression.

        Supported conditions:
        - gate:code_review:passed
        - gate:code_review:failed
        - artifact:requirements_document:exists
        - task:backend_development:completed
        - context:feature_flags:mobile_enabled

        Args:
            condition: The condition expression

        Returns:
            True if condition is met
        """
        parts = condition.split(":")

        if len(parts) < 3:
            return False

        condition_type = parts[0]

        if condition_type == "gate":
            return self._evaluate_gate_condition(parts[1], parts[2])
        elif condition_type == "artifact":
            return self._evaluate_artifact_condition(parts[1], parts[2])
        elif condition_type == "task":
            return self._evaluate_task_condition(parts[1], parts[2])
        elif condition_type == "context":
            return self._evaluate_context_condition(parts[1], parts[2])

        return False

    def _evaluate_gate_condition(self, gate_type: str, expected: str) -> bool:
        """Evaluate a gate condition."""
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            if not run:
                return False

            if gate_type == "code_review":
                status = run.code_review_status
            elif gate_type == "security_review":
                status = run.security_review_status
            else:
                return False

            if expected == "passed":
                return status in [GateStatus.PASSED, GateStatus.WAIVED]
            elif expected == "failed":
                return status == GateStatus.FAILED
            elif expected == "pending":
                return status == GateStatus.PENDING

        return False

    def _evaluate_artifact_condition(self, artifact_type: str, check: str) -> bool:
        """Evaluate an artifact condition."""
        with get_db() as db:
            artifact = db.query(Artifact).filter(
                Artifact.run_id == UUID(self.run_id),
                Artifact.artifact_type == artifact_type,
            ).first()

            if check == "exists":
                return artifact is not None
            elif check == "valid":
                return artifact is not None and artifact.is_valid

        return False

    def _evaluate_task_condition(self, task_type: str, expected: str) -> bool:
        """Evaluate a task condition."""
        with get_db() as db:
            task = db.query(Task).filter(
                Task.run_id == UUID(self.run_id),
                Task.task_type == task_type,
            ).first()

            if not task:
                return expected == "not_exists"

            if expected == "completed":
                return task.status == TaskStatus.COMPLETED
            elif expected == "failed":
                return task.status == TaskStatus.FAILED
            elif expected == "pending":
                return task.status == TaskStatus.PENDING
            elif expected == "exists":
                return True

        return False

    def _evaluate_context_condition(self, context_key: str, check: str) -> bool:
        """Evaluate a context condition."""
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            if not run or not run.context:
                return False

            value = run.context.get(context_key)

            if check == "true":
                return bool(value)
            elif check == "false":
                return not value
            elif check == "exists":
                return value is not None

        return False


def get_workflow_modifier(run_id: str) -> WorkflowModifier:
    """Get a workflow modifier for a run."""
    return WorkflowModifier(run_id)


def get_condition_evaluator(run_id: str) -> ConditionalBranchEvaluator:
    """Get a condition evaluator for a run."""
    return ConditionalBranchEvaluator(run_id)
