"""FastAPI application for the multi-agent orchestration system."""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator.core.database import get_db, init_db
from orchestrator.core.models import Run, Task, Event, Artifact, RunStatus, TaskStatus

app = FastAPI(
    title="Multi-Agent Orchestrator API",
    description="API for managing multi-agent development workflows",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class RunCreate(BaseModel):
    goal: str = Field(..., description="The goal for the orchestration run")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class RunResponse(BaseModel):
    id: str
    goal: str
    status: str
    current_phase: Optional[str]
    iteration: int
    max_iterations: int
    code_review_status: str
    security_review_status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    blocked_reason: Optional[str]

    class Config:
        from_attributes = True


class TaskResponse(BaseModel):
    id: str
    task_type: str
    assigned_role: str
    description: str
    status: str
    priority: int
    retry_count: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class EventResponse(BaseModel):
    id: str
    event_type: str
    actor: str
    data: Dict[str, Any]
    timestamp: datetime

    class Config:
        from_attributes = True


class ArtifactResponse(BaseModel):
    id: str
    artifact_type: str
    name: str
    content_type: str
    produced_by: str
    created_at: datetime

    class Config:
        from_attributes = True


class ArtifactDetailResponse(ArtifactResponse):
    content: str
    metadata: Dict[str, Any]


class StatusResponse(BaseModel):
    run_id: str
    status: str
    current_phase: Optional[str]
    iteration: int
    max_iterations: int
    gates: Dict[str, str]
    tasks: Dict[str, int]
    artifacts_count: int
    blocked_reason: Optional[str]


# Startup event
@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    init_db()


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Multi-Agent Orchestrator API",
        "version": "1.0.0",
        "status": "running",
    }


@app.post("/runs", response_model=RunResponse)
async def create_run(run_data: RunCreate):
    """Create and start a new orchestration run."""
    run_id = str(uuid.uuid4())

    with get_db() as db:
        run = Run(
            id=uuid.UUID(run_id),
            goal=run_data.goal,
            status=RunStatus.PENDING,
            context=run_data.context,
        )
        db.add(run)
        db.commit()
        db.refresh(run)

    # Start the run asynchronously
    from orchestrator.agents.tasks import start_run
    start_run.delay(run_id, run_data.goal, run_data.context)

    with get_db() as db:
        run = db.query(Run).filter(Run.id == uuid.UUID(run_id)).first()
        return _run_to_response(run)


@app.get("/runs", response_model=List[RunResponse])
async def list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
):
    """List all runs with optional filtering."""
    with get_db() as db:
        query = db.query(Run)

        if status:
            query = query.filter(Run.status == RunStatus(status))

        runs = query.order_by(Run.created_at.desc()).offset(offset).limit(limit).all()
        return [_run_to_response(run) for run in runs]


@app.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str):
    """Get details of a specific run."""
    with get_db() as db:
        run = db.query(Run).filter(Run.id == uuid.UUID(run_id)).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return _run_to_response(run)


@app.get("/runs/{run_id}/status", response_model=StatusResponse)
async def get_run_status(run_id: str):
    """Get detailed status of a run including task summary."""
    from orchestrator.agents.orchestrator import OrchestratorAgent

    orchestrator = OrchestratorAgent(run_id)
    return orchestrator.get_status()


@app.get("/runs/{run_id}/tasks", response_model=List[TaskResponse])
async def get_run_tasks(run_id: str):
    """Get all tasks for a run."""
    with get_db() as db:
        tasks = db.query(Task).filter(
            Task.run_id == uuid.UUID(run_id)
        ).order_by(Task.priority.desc()).all()

        return [_task_to_response(task) for task in tasks]


@app.get("/runs/{run_id}/events", response_model=List[EventResponse])
async def get_run_events(
    run_id: str,
    limit: int = Query(50, ge=1, le=500),
    event_type: Optional[str] = None,
):
    """Get events for a run."""
    with get_db() as db:
        query = db.query(Event).filter(Event.run_id == uuid.UUID(run_id))

        if event_type:
            query = query.filter(Event.event_type == event_type)

        events = query.order_by(Event.timestamp.desc()).limit(limit).all()
        return [_event_to_response(event) for event in events]


@app.get("/runs/{run_id}/artifacts", response_model=List[ArtifactResponse])
async def get_run_artifacts(
    run_id: str,
    artifact_type: Optional[str] = None,
):
    """Get artifacts for a run."""
    with get_db() as db:
        query = db.query(Artifact).filter(Artifact.run_id == uuid.UUID(run_id))

        if artifact_type:
            query = query.filter(Artifact.artifact_type == artifact_type)

        artifacts = query.order_by(Artifact.created_at).all()
        return [_artifact_to_response(artifact) for artifact in artifacts]


@app.get("/runs/{run_id}/artifacts/{artifact_id}", response_model=ArtifactDetailResponse)
async def get_artifact(run_id: str, artifact_id: str):
    """Get a specific artifact with content."""
    with get_db() as db:
        artifact = db.query(Artifact).filter(
            Artifact.id == uuid.UUID(artifact_id),
            Artifact.run_id == uuid.UUID(run_id),
        ).first()

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        return _artifact_detail_to_response(artifact)


@app.post("/runs/{run_id}/tick")
async def trigger_tick(run_id: str):
    """Manually trigger an orchestrator tick."""
    from orchestrator.agents.tasks import orchestrator_tick

    result = orchestrator_tick.delay(run_id)
    return {"task_id": result.id, "status": "dispatched"}


@app.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Cancel a running orchestration."""
    with get_db() as db:
        run = db.query(Run).filter(Run.id == uuid.UUID(run_id)).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.utcnow()
        db.commit()

        return {"status": "cancelled", "run_id": run_id}


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Helper functions
def _run_to_response(run: Run) -> RunResponse:
    return RunResponse(
        id=str(run.id),
        goal=run.goal,
        status=run.status.value,
        current_phase=run.current_phase,
        iteration=run.current_iteration,
        max_iterations=run.max_iterations,
        code_review_status=run.code_review_status.value,
        security_review_status=run.security_review_status.value,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        blocked_reason=run.blocked_reason,
    )


def _task_to_response(task: Task) -> TaskResponse:
    return TaskResponse(
        id=str(task.id),
        task_type=task.task_type,
        assigned_role=task.assigned_role,
        description=task.description,
        status=task.status.value,
        priority=task.priority,
        retry_count=task.retry_count,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error_message=task.error_message,
    )


def _event_to_response(event: Event) -> EventResponse:
    return EventResponse(
        id=str(event.id),
        event_type=event.event_type,
        actor=event.actor,
        data=event.data,
        timestamp=event.timestamp,
    )


def _artifact_to_response(artifact: Artifact) -> ArtifactResponse:
    return ArtifactResponse(
        id=str(artifact.id),
        artifact_type=artifact.artifact_type,
        name=artifact.name,
        content_type=artifact.content_type,
        produced_by=artifact.produced_by,
        created_at=artifact.created_at,
    )


def _artifact_detail_to_response(artifact: Artifact) -> ArtifactDetailResponse:
    return ArtifactDetailResponse(
        id=str(artifact.id),
        artifact_type=artifact.artifact_type,
        name=artifact.name,
        content_type=artifact.content_type,
        produced_by=artifact.produced_by,
        created_at=artifact.created_at,
        content=artifact.content,
        metadata=artifact.metadata,
    )
