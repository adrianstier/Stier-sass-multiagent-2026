#!/usr/bin/env python3
"""CLI for the multi-agent orchestration system.

Usage:
    orchestrator init-db          Initialize the database
    orchestrator run <goal>       Start a new run with the given goal
    orchestrator status <run_id>  Check the status of a run
    orchestrator list             List all runs
    orchestrator events <run_id>  Show events for a run
    orchestrator artifacts <run_id>  Show artifacts for a run
    orchestrator tick <run_id>    Manually trigger orchestrator tick
"""

import argparse
import json
import sys
import uuid
from datetime import datetime
from typing import Optional

from orchestrator.core.database import init_db, get_db
from orchestrator.core.models import Run, Task, Event, Artifact, RunStatus


def cmd_init_db(args):
    """Initialize the database."""
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")


def cmd_run(args):
    """Start a new orchestration run."""
    goal = args.goal
    run_id = str(uuid.uuid4())

    print(f"Creating new run: {run_id}")
    print(f"Goal: {goal}")

    # Create run in database
    with get_db() as db:
        run = Run(
            id=uuid.UUID(run_id),
            goal=goal,
            status=RunStatus.PENDING,
            context=json.loads(args.context) if args.context else {},
        )
        db.add(run)
        db.commit()

    # Start the run via Celery
    if not args.sync:
        from orchestrator.agents.tasks import start_run
        result = start_run.delay(run_id, goal, run.context)
        print(f"Run started asynchronously. Celery task ID: {result.id}")
    else:
        # Synchronous execution for testing
        from orchestrator.agents.orchestrator import OrchestratorAgent
        orchestrator = OrchestratorAgent(run_id)
        orchestrator.initialize_run(goal, {})
        print("Run initialized (sync mode). Use 'tick' command to progress.")

    print(f"\nRun ID: {run_id}")
    print(f"Check status with: orchestrator status {run_id}")


def cmd_status(args):
    """Check the status of a run."""
    run_id = args.run_id

    with get_db() as db:
        run = db.query(Run).filter(Run.id == uuid.UUID(run_id)).first()
        if not run:
            print(f"Run {run_id} not found")
            sys.exit(1)

        tasks = db.query(Task).filter(Task.run_id == uuid.UUID(run_id)).all()

        print(f"\n{'='*60}")
        print(f"Run: {run_id}")
        print(f"{'='*60}")
        print(f"Goal: {run.goal}")
        print(f"Status: {run.status.value}")
        print(f"Iteration: {run.current_iteration}/{run.max_iterations}")
        print(f"Created: {run.created_at}")
        print(f"Started: {run.started_at or 'Not started'}")
        print(f"Completed: {run.completed_at or 'Not completed'}")

        if run.blocked_reason:
            print(f"Blocked: {run.blocked_reason}")

        print(f"\nQuality Gates:")
        print(f"  Code Review: {run.code_review_status.value}")
        print(f"  Security Review: {run.security_review_status.value}")

        print(f"\nTasks ({len(tasks)} total):")
        print(f"  {'Task Type':<30} {'Role':<20} {'Status':<15}")
        print(f"  {'-'*30} {'-'*20} {'-'*15}")
        for task in sorted(tasks, key=lambda t: t.priority, reverse=True):
            print(f"  {task.task_type:<30} {task.assigned_role:<20} {task.status.value:<15}")

        # Count artifacts
        artifact_count = db.query(Artifact).filter(
            Artifact.run_id == uuid.UUID(run_id)
        ).count()
        print(f"\nArtifacts: {artifact_count}")


def cmd_list(args):
    """List all runs."""
    with get_db() as db:
        runs = db.query(Run).order_by(Run.created_at.desc()).limit(args.limit).all()

        if not runs:
            print("No runs found.")
            return

        print(f"\n{'Run ID':<40} {'Status':<15} {'Goal':<40} {'Created':<20}")
        print(f"{'-'*40} {'-'*15} {'-'*40} {'-'*20}")

        for run in runs:
            goal_short = run.goal[:37] + "..." if len(run.goal) > 40 else run.goal
            created = run.created_at.strftime("%Y-%m-%d %H:%M")
            print(f"{str(run.id):<40} {run.status.value:<15} {goal_short:<40} {created:<20}")


def cmd_events(args):
    """Show events for a run."""
    run_id = args.run_id

    with get_db() as db:
        events = db.query(Event).filter(
            Event.run_id == uuid.UUID(run_id)
        ).order_by(Event.timestamp.desc()).limit(args.limit).all()

        if not events:
            print(f"No events found for run {run_id}")
            return

        print(f"\nEvents for run {run_id}:")
        print(f"{'='*80}")

        for event in reversed(events):
            timestamp = event.timestamp.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] {event.event_type} ({event.actor})")
            if args.verbose:
                print(f"  Data: {json.dumps(event.data, indent=4)}")


def cmd_artifacts(args):
    """Show artifacts for a run."""
    run_id = args.run_id

    with get_db() as db:
        artifacts = db.query(Artifact).filter(
            Artifact.run_id == uuid.UUID(run_id)
        ).order_by(Artifact.created_at).all()

        if not artifacts:
            print(f"No artifacts found for run {run_id}")
            return

        print(f"\nArtifacts for run {run_id}:")
        print(f"{'='*80}")

        for artifact in artifacts:
            print(f"\n--- {artifact.artifact_type}: {artifact.name} ---")
            print(f"Produced by: {artifact.produced_by}")
            print(f"Created: {artifact.created_at}")

            if args.content:
                print(f"\nContent:")
                print(artifact.content[:2000])
                if len(artifact.content) > 2000:
                    print("\n... [truncated]")


def cmd_tick(args):
    """Manually trigger orchestrator tick."""
    run_id = args.run_id

    from orchestrator.agents.tasks import orchestrator_tick

    if args.sync:
        # Synchronous execution
        result = orchestrator_tick(run_id)
    else:
        result = orchestrator_tick.delay(run_id)
        result = result.get(timeout=60)

    print(f"Tick result: {json.dumps(result, indent=2)}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Orchestration System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init-db
    subparsers.add_parser("init-db", help="Initialize the database")

    # run
    run_parser = subparsers.add_parser("run", help="Start a new run")
    run_parser.add_argument("goal", help="The goal for the orchestration")
    run_parser.add_argument("--context", help="JSON context/parameters")
    run_parser.add_argument("--sync", action="store_true", help="Run synchronously (for testing)")

    # status
    status_parser = subparsers.add_parser("status", help="Check run status")
    status_parser.add_argument("run_id", help="The run ID to check")

    # list
    list_parser = subparsers.add_parser("list", help="List all runs")
    list_parser.add_argument("--limit", type=int, default=20, help="Limit number of results")

    # events
    events_parser = subparsers.add_parser("events", help="Show events for a run")
    events_parser.add_argument("run_id", help="The run ID")
    events_parser.add_argument("--limit", type=int, default=50, help="Limit number of events")
    events_parser.add_argument("-v", "--verbose", action="store_true", help="Show full event data")

    # artifacts
    artifacts_parser = subparsers.add_parser("artifacts", help="Show artifacts for a run")
    artifacts_parser.add_argument("run_id", help="The run ID")
    artifacts_parser.add_argument("-c", "--content", action="store_true", help="Show artifact content")

    # tick
    tick_parser = subparsers.add_parser("tick", help="Manually trigger orchestrator tick")
    tick_parser.add_argument("run_id", help="The run ID")
    tick_parser.add_argument("--sync", action="store_true", help="Run synchronously")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    commands = {
        "init-db": cmd_init_db,
        "run": cmd_run,
        "status": cmd_status,
        "list": cmd_list,
        "events": cmd_events,
        "artifacts": cmd_artifacts,
        "tick": cmd_tick,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
