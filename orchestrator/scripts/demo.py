#!/usr/bin/env python3
"""Demo script for the multi-agent orchestration system.

This script demonstrates the full workflow by:
1. Creating a new run with a CRUD app goal
2. Monitoring the workflow progress
3. Displaying artifacts as they're created
"""

import time
import uuid
from datetime import datetime

from orchestrator.core.database import init_db, get_db
from orchestrator.core.models import Run, Task, Artifact, RunStatus, TaskStatus
from orchestrator.agents.orchestrator import OrchestratorAgent


def print_banner():
    """Print demo banner."""
    print("\n" + "=" * 70)
    print("  MULTI-AGENT ORCHESTRATOR DEMO")
    print("  'Hello, Multi-Agent' - Building a CRUD App")
    print("=" * 70 + "\n")


def print_status(run_id: str):
    """Print current run status."""
    orchestrator = OrchestratorAgent(run_id)
    status = orchestrator.get_status()

    print(f"\n{'─' * 50}")
    print(f"Run Status: {status['status']}")
    print(f"Iteration: {status['iteration']}/{status['max_iterations']}")
    print(f"Gates: CR={status['gates']['code_review']}, SEC={status['gates']['security_review']}")
    print(f"Tasks: {status['tasks']}")
    print(f"Artifacts: {status['artifacts_count']}")
    print(f"{'─' * 50}")


def print_tasks(run_id: str):
    """Print task progress."""
    with get_db() as db:
        tasks = db.query(Task).filter(
            Task.run_id == uuid.UUID(run_id)
        ).order_by(Task.priority.desc()).all()

        print("\nTask Progress:")
        for task in tasks:
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.QUEUED: "📤",
                TaskStatus.RUNNING: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
            }.get(task.status, "❓")

            print(f"  {status_icon} {task.task_type:<30} ({task.assigned_role})")


def print_new_artifacts(run_id: str, seen_ids: set):
    """Print newly created artifacts."""
    with get_db() as db:
        artifacts = db.query(Artifact).filter(
            Artifact.run_id == uuid.UUID(run_id)
        ).all()

        for artifact in artifacts:
            if str(artifact.id) not in seen_ids:
                seen_ids.add(str(artifact.id))
                print(f"\n📄 New Artifact: {artifact.artifact_type}")
                print(f"   Name: {artifact.name}")
                print(f"   By: {artifact.produced_by}")
                print(f"   Preview: {artifact.content[:200]}...")


def run_demo_sync():
    """Run the demo in synchronous mode (for testing without Celery)."""
    print_banner()

    # Initialize database
    print("Initializing database...")
    init_db()

    # Create run
    run_id = str(uuid.uuid4())
    goal = "Build a simple CRUD app for managing TODO items with a REST API"

    print(f"\nCreating new run: {run_id}")
    print(f"Goal: {goal}\n")

    with get_db() as db:
        run = Run(
            id=uuid.UUID(run_id),
            goal=goal,
            status=RunStatus.PENDING,
            context={},
        )
        db.add(run)
        db.commit()

    # Initialize the run
    print("Initializing workflow...")
    orchestrator = OrchestratorAgent(run_id)
    orchestrator.initialize_run(goal, {})

    print_status(run_id)
    print_tasks(run_id)

    # Track seen artifacts
    seen_artifacts = set()

    # Run orchestration loop
    print("\n" + "=" * 70)
    print("  STARTING ORCHESTRATION LOOP")
    print("=" * 70)

    max_ticks = 20
    for tick in range(max_ticks):
        print(f"\n>>> Tick {tick + 1}/{max_ticks}")

        # Get ready tasks
        ready_tasks = orchestrator.get_ready_tasks()
        if not ready_tasks:
            print("No tasks ready to execute.")

            if orchestrator.is_complete():
                print("\n✅ WORKFLOW COMPLETE!")
                break

            # Check if stuck
            with get_db() as db:
                pending = db.query(Task).filter(
                    Task.run_id == uuid.UUID(run_id),
                    Task.status.in_([TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING])
                ).count()

                if pending == 0:
                    print("No pending tasks and workflow not complete.")
                    break

            time.sleep(1)
            continue

        # Execute ready tasks (synchronously for demo)
        for task in ready_tasks[:1]:  # Execute one at a time
            print(f"\nExecuting: {task.task_type} ({task.assigned_role})")

            # Import here to avoid circular import
            from orchestrator.agents.tasks import execute_agent_task

            try:
                result = execute_agent_task(run_id, str(task.id))
                if result.get("success"):
                    print(f"  ✅ Completed successfully")
                else:
                    print(f"  ❌ Failed: {result.get('error')}")
            except Exception as e:
                print(f"  ❌ Error: {e}")

        # Print progress
        print_tasks(run_id)
        print_new_artifacts(run_id, seen_artifacts)
        print_status(run_id)

        time.sleep(0.5)

    # Final summary
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)

    print_status(run_id)

    with get_db() as db:
        artifacts = db.query(Artifact).filter(
            Artifact.run_id == uuid.UUID(run_id)
        ).all()

        print(f"\nGenerated {len(artifacts)} artifacts:")
        for artifact in artifacts:
            print(f"  - {artifact.artifact_type}: {artifact.name}")

    print(f"\nRun ID: {run_id}")
    print("Use 'orchestrator artifacts <run_id> --content' to view full artifacts")


def main():
    """Main entry point."""
    import sys

    if "--async" in sys.argv:
        print("Async mode requires running Celery workers.")
        print("Use 'make docker-up' to start all services.")
        print("Then use 'orchestrator run \"<goal>\"' to start a run.")
    else:
        run_demo_sync()


if __name__ == "__main__":
    main()
