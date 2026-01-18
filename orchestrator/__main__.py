#!/usr/bin/env python3
"""CLI entry point for the multi-agent orchestrator.

Usage:
    python -m orchestrator "Build a REST API for user management"
    python -m orchestrator --existing "Add dark mode to the settings page"
    python -m orchestrator --server  # Start the API server
"""

import sys
import asyncio
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Orchestrator - Describe what you want, agents build it"
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Natural language description of what to build"
    )
    parser.add_argument(
        "--existing", "-e",
        action="store_true",
        help="Working on an existing codebase (vs greenfield)"
    )
    parser.add_argument(
        "--server", "-s",
        action="store_true",
        help="Start the API server instead"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the workflow plan without executing"
    )

    args = parser.parse_args()

    if args.server:
        # Start the FastAPI server
        import uvicorn
        print(f"Starting orchestrator API on http://localhost:{args.port}")
        print("Docs available at http://localhost:{args.port}/docs")
        uvicorn.run(
            "orchestrator.api.main:app",
            host="0.0.0.0",
            port=args.port,
            reload=True
        )
        return

    if not args.task:
        parser.print_help()
        print("\nExamples:")
        print('  python -m orchestrator "Build a todo app with React"')
        print('  python -m orchestrator --existing "Add user authentication"')
        print('  python -m orchestrator --server')
        sys.exit(1)

    # Run the orchestrator
    project_type = "existing" if args.existing else "greenfield"

    print(f"\n{'='*60}")
    print(f"Multi-Agent Orchestrator")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Mode: {project_type}")
    print(f"{'='*60}\n")

    asyncio.run(run_workflow(args.task, project_type, args.dry_run))


async def run_workflow(task: str, project_type: str, dry_run: bool = False):
    """Execute the multi-agent workflow."""
    from orchestrator.core.orchestrator import MultiAgentOrchestrator
    from orchestrator.core.config import Settings

    try:
        settings = Settings()
        orchestrator = MultiAgentOrchestrator(settings)

        if dry_run:
            print("DRY RUN - Planning only, no execution\n")
            # Just show what would happen
            plan = await orchestrator.plan_workflow(task, project_type)
            print("Planned workflow:")
            for i, step in enumerate(plan.get("steps", []), 1):
                print(f"  {i}. {step.get('description', step)}")
            return

        print("Starting workflow...\n")
        result = await orchestrator.execute_workflow(
            task=task,
            project_type=project_type
        )

        print(f"\n{'='*60}")
        print("Workflow Complete!")
        print(f"{'='*60}")

        if result.get("success"):
            print(f"Status: SUCCESS")
            if result.get("artifacts"):
                print(f"\nArtifacts created:")
                for artifact in result["artifacts"]:
                    print(f"  - {artifact}")
        else:
            print(f"Status: FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Set ANTHROPIC_API_KEY environment variable")
        print("  2. Redis running (for Celery task queue)")
        print("  3. Installed dependencies: pip install -e .")
        sys.exit(1)


if __name__ == "__main__":
    main()
