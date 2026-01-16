"""FastAPI application for the multi-agent orchestration system.

Production-ready with:
- JWT and API key authentication
- Multi-tenant organization isolation
- Rate limiting
- Input validation
- Structured logging
- Error handling
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware

from orchestrator.core.config import settings
from orchestrator.core.database import get_db, init_db
from orchestrator.core.models import Run, Task, Event, Artifact, RunStatus, TaskStatus, GateStatus
from orchestrator.core.auth import (
    AuthContext,
    get_current_auth,
    get_optional_auth,
    require_scope,
    Organization,
    User,
    APIKey,
    create_access_token,
    create_refresh_token,
    generate_api_key,
)
from orchestrator.core.rate_limit import RateLimitMiddleware
from orchestrator.core.logging import (
    configure_logging,
    get_logger,
    RequestIDMiddleware,
    org_id_var,
    user_id_var,
)

# Configure structured logging
configure_logging()
logger = get_logger(__name__)

# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="Multi-Agent Orchestrator API",
    description="Production-ready API for managing multi-agent development workflows",
    version="2.0.0",
    docs_url="/docs" if settings.log_level == "DEBUG" else None,  # Hide docs in production
    redoc_url="/redoc" if settings.log_level == "DEBUG" else None,
)

# Add middlewares
app.add_middleware(RequestIDMiddleware)

if settings.rate_limit_enabled:
    app.add_middleware(RateLimitMiddleware, use_redis=True)

# CORS middleware
origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions."""
    logger.warning(
        "http_exception",
        path=request.url.path,
        status_code=exc.status_code,
        detail=exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


# =============================================================================
# Pydantic Models with Validation
# =============================================================================

class RunCreate(BaseModel):
    goal: str = Field(..., min_length=10, max_length=10000, description="The goal for the orchestration run")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    budget_tokens: Optional[int] = Field(None, ge=1000, le=10000000, description="Token budget for this run")

    @validator("goal")
    def goal_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Goal cannot be empty or whitespace only")
        return v.strip()


class RunResponse(BaseModel):
    id: str
    organization_id: str
    goal: str
    status: str
    current_phase: Optional[str]
    iteration: int
    max_iterations: int
    tokens_used: int
    budget_tokens: Optional[int]
    total_cost_usd: str
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
    tokens_used: int
    budget_tokens: Optional[int]
    total_cost_usd: str
    gates: Dict[str, str]
    tasks: Dict[str, int]
    artifacts_count: int
    blocked_reason: Optional[str]


class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    scopes: str = Field("*", description="Comma-separated scopes")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: str
    created_at: datetime
    expires_at: Optional[datetime]
    # Note: full key is only returned on creation


class APIKeyCreatedResponse(APIKeyResponse):
    key: str  # Full key - only returned once on creation


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


# =============================================================================
# Startup & Health
# =============================================================================

@app.on_event("startup")
async def startup():
    """Initialize database and logging on startup."""
    logger.info("application_starting", version="2.0.0")
    init_db()
    logger.info("application_started")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Multi-Agent Orchestrator API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs" if settings.log_level == "DEBUG" else "disabled",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint (unauthenticated)."""
    # Check database connectivity
    try:
        with get_db() as db:
            db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
    }


# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.post("/auth/api-keys", response_model=APIKeyCreatedResponse)
async def create_api_key(
    key_data: APIKeyCreate,
    auth: AuthContext = Depends(get_current_auth),
):
    """Create a new API key for the organization."""
    full_key, prefix, key_hash = generate_api_key()

    expires_at = None
    if key_data.expires_in_days:
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)

    with get_db() as db:
        api_key = APIKey(
            organization_id=uuid.UUID(auth.organization_id),
            created_by_user_id=uuid.UUID(auth.user_id) if auth.user_id else None,
            name=key_data.name,
            key_prefix=prefix,
            key_hash=key_hash,
            scopes=key_data.scopes,
            expires_at=expires_at,
        )
        db.add(api_key)
        db.commit()
        db.refresh(api_key)

        logger.info(
            "api_key_created",
            key_id=str(api_key.id),
            name=key_data.name,
        )

        return APIKeyCreatedResponse(
            id=str(api_key.id),
            name=api_key.name,
            key_prefix=api_key.key_prefix,
            scopes=api_key.scopes,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
            key=full_key,  # Only returned once!
        )


@app.get("/auth/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(auth: AuthContext = Depends(get_current_auth)):
    """List API keys for the organization."""
    with get_db() as db:
        keys = db.query(APIKey).filter(
            APIKey.organization_id == uuid.UUID(auth.organization_id),
            APIKey.is_active == True,
        ).all()

        return [
            APIKeyResponse(
                id=str(k.id),
                name=k.name,
                key_prefix=k.key_prefix,
                scopes=k.scopes,
                created_at=k.created_at,
                expires_at=k.expires_at,
            )
            for k in keys
        ]


@app.delete("/auth/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    auth: AuthContext = Depends(get_current_auth),
):
    """Revoke an API key."""
    with get_db() as db:
        api_key = db.query(APIKey).filter(
            APIKey.id == uuid.UUID(key_id),
            APIKey.organization_id == uuid.UUID(auth.organization_id),
        ).first()

        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")

        api_key.is_active = False
        db.commit()

        logger.info("api_key_revoked", key_id=key_id)

        return {"status": "revoked", "key_id": key_id}


# =============================================================================
# Run Endpoints
# =============================================================================

@app.post("/runs", response_model=RunResponse)
async def create_run(
    run_data: RunCreate,
    auth: AuthContext = Depends(require_scope("runs:write")),
):
    """Create and start a new orchestration run."""
    run_id = str(uuid.uuid4())

    # Set default token budget if not provided
    budget = run_data.budget_tokens or settings.default_run_token_budget

    with get_db() as db:
        run = Run(
            id=uuid.UUID(run_id),
            organization_id=uuid.UUID(auth.organization_id),
            created_by_user_id=uuid.UUID(auth.user_id) if auth.user_id else None,
            goal=run_data.goal,
            status=RunStatus.PENDING,
            context=run_data.context,
            budget_tokens=budget,
        )
        db.add(run)
        db.commit()
        db.refresh(run)

    logger.info(
        "run_created",
        run_id=run_id,
        goal_length=len(run_data.goal),
        budget_tokens=budget,
    )

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
    auth: AuthContext = Depends(require_scope("runs:read")),
):
    """List runs for the authenticated organization."""
    with get_db() as db:
        # Filter by organization (multi-tenancy)
        query = db.query(Run).filter(Run.organization_id == uuid.UUID(auth.organization_id))

        if status:
            # Validate status enum
            try:
                status_enum = RunStatus(status.lower())
                query = query.filter(Run.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Valid values: {[s.value for s in RunStatus]}",
                )

        runs = query.order_by(Run.created_at.desc()).offset(offset).limit(limit).all()
        return [_run_to_response(run) for run in runs]


@app.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    auth: AuthContext = Depends(require_scope("runs:read")),
):
    """Get details of a specific run."""
    # Validate UUID format
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    with get_db() as db:
        run = db.query(Run).filter(
            Run.id == run_uuid,
            Run.organization_id == uuid.UUID(auth.organization_id),
        ).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        return _run_to_response(run)


@app.get("/runs/{run_id}/status", response_model=StatusResponse)
async def get_run_status(
    run_id: str,
    auth: AuthContext = Depends(require_scope("runs:read")),
):
    """Get detailed status of a run including task summary."""
    # Validate UUID and ownership
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    with get_db() as db:
        run = db.query(Run).filter(
            Run.id == run_uuid,
            Run.organization_id == uuid.UUID(auth.organization_id),
        ).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

    from orchestrator.agents.orchestrator import OrchestratorAgent
    orchestrator = OrchestratorAgent(run_id)
    status = orchestrator.get_status()

    # Add cost info
    status["tokens_used"] = run.tokens_used
    status["budget_tokens"] = run.budget_tokens
    status["total_cost_usd"] = run.total_cost_usd

    return status


@app.get("/runs/{run_id}/tasks", response_model=List[TaskResponse])
async def get_run_tasks(
    run_id: str,
    auth: AuthContext = Depends(require_scope("runs:read")),
):
    """Get all tasks for a run."""
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    with get_db() as db:
        # Verify run belongs to org
        run = db.query(Run).filter(
            Run.id == run_uuid,
            Run.organization_id == uuid.UUID(auth.organization_id),
        ).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        tasks = db.query(Task).filter(
            Task.run_id == run_uuid
        ).order_by(Task.priority.desc()).all()

        return [_task_to_response(task) for task in tasks]


@app.get("/runs/{run_id}/events", response_model=List[EventResponse])
async def get_run_events(
    run_id: str,
    limit: int = Query(50, ge=1, le=500),
    event_type: Optional[str] = None,
    auth: AuthContext = Depends(require_scope("runs:read")),
):
    """Get events for a run."""
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    with get_db() as db:
        # Verify run belongs to org
        run = db.query(Run).filter(
            Run.id == run_uuid,
            Run.organization_id == uuid.UUID(auth.organization_id),
        ).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        query = db.query(Event).filter(Event.run_id == run_uuid)

        if event_type:
            query = query.filter(Event.event_type == event_type)

        events = query.order_by(Event.timestamp.desc()).limit(limit).all()
        return [_event_to_response(event) for event in events]


@app.get("/runs/{run_id}/artifacts", response_model=List[ArtifactResponse])
async def get_run_artifacts(
    run_id: str,
    artifact_type: Optional[str] = None,
    auth: AuthContext = Depends(require_scope("artifacts:read")),
):
    """Get artifacts for a run."""
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    with get_db() as db:
        # Verify run belongs to org
        run = db.query(Run).filter(
            Run.id == run_uuid,
            Run.organization_id == uuid.UUID(auth.organization_id),
        ).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        query = db.query(Artifact).filter(Artifact.run_id == run_uuid)

        if artifact_type:
            query = query.filter(Artifact.artifact_type == artifact_type)

        artifacts = query.order_by(Artifact.created_at).all()
        return [_artifact_to_response(artifact) for artifact in artifacts]


@app.get("/runs/{run_id}/artifacts/{artifact_id}", response_model=ArtifactDetailResponse)
async def get_artifact(
    run_id: str,
    artifact_id: str,
    auth: AuthContext = Depends(require_scope("artifacts:read")),
):
    """Get a specific artifact with content."""
    try:
        run_uuid = uuid.UUID(run_id)
        artifact_uuid = uuid.UUID(artifact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")

    with get_db() as db:
        # Verify run belongs to org
        run = db.query(Run).filter(
            Run.id == run_uuid,
            Run.organization_id == uuid.UUID(auth.organization_id),
        ).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        artifact = db.query(Artifact).filter(
            Artifact.id == artifact_uuid,
            Artifact.run_id == run_uuid,
        ).first()

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        return _artifact_detail_to_response(artifact)


@app.post("/runs/{run_id}/tick")
async def trigger_tick(
    run_id: str,
    auth: AuthContext = Depends(require_scope("runs:write")),
):
    """Manually trigger an orchestrator tick."""
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    with get_db() as db:
        run = db.query(Run).filter(
            Run.id == run_uuid,
            Run.organization_id == uuid.UUID(auth.organization_id),
        ).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

    from orchestrator.agents.tasks import orchestrator_tick
    result = orchestrator_tick.delay(run_id)

    logger.info("tick_triggered", run_id=run_id, celery_task_id=result.id)

    return {"task_id": result.id, "status": "dispatched"}


@app.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    auth: AuthContext = Depends(require_scope("runs:write")),
):
    """Cancel a running orchestration."""
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    with get_db() as db:
        run = db.query(Run).filter(
            Run.id == run_uuid,
            Run.organization_id == uuid.UUID(auth.organization_id),
        ).first()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        if run.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel run with status: {run.status.value}",
            )

        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.utcnow()
        db.commit()

        logger.info("run_cancelled", run_id=run_id)

        return {"status": "cancelled", "run_id": run_id}


# =============================================================================
# Organization Endpoints
# =============================================================================

@app.get("/organization")
async def get_organization(auth: AuthContext = Depends(get_current_auth)):
    """Get current organization details."""
    with get_db() as db:
        org = db.query(Organization).filter(
            Organization.id == uuid.UUID(auth.organization_id)
        ).first()

        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")

        return {
            "id": str(org.id),
            "name": org.name,
            "slug": org.slug,
            "plan": org.plan,
            "monthly_token_limit": org.monthly_token_limit,
            "tokens_used_this_month": org.tokens_used_this_month,
            "is_active": org.is_active,
        }


@app.get("/organization/usage")
async def get_organization_usage(auth: AuthContext = Depends(get_current_auth)):
    """Get organization usage statistics."""
    with get_db() as db:
        org = db.query(Organization).filter(
            Organization.id == uuid.UUID(auth.organization_id)
        ).first()

        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")

        # Get run statistics
        from sqlalchemy import func

        run_stats = db.query(
            func.count(Run.id).label("total_runs"),
            func.sum(Run.tokens_used).label("total_tokens"),
        ).filter(
            Run.organization_id == uuid.UUID(auth.organization_id)
        ).first()

        return {
            "organization_id": str(org.id),
            "plan": org.plan,
            "monthly_token_limit": org.monthly_token_limit,
            "tokens_used_this_month": org.tokens_used_this_month,
            "total_runs": run_stats.total_runs or 0,
            "total_tokens_all_time": run_stats.total_tokens or 0,
            "billing_cycle_start": org.billing_cycle_start.isoformat() if org.billing_cycle_start else None,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def _run_to_response(run: Run) -> RunResponse:
    return RunResponse(
        id=str(run.id),
        organization_id=str(run.organization_id),
        goal=run.goal,
        status=run.status.value,
        current_phase=run.current_phase,
        iteration=run.current_iteration,
        max_iterations=run.max_iterations,
        tokens_used=run.tokens_used or 0,
        budget_tokens=run.budget_tokens,
        total_cost_usd=run.total_cost_usd or "0.00",
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
