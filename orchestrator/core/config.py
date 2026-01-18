"""Configuration settings for the orchestration system."""

import secrets
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/orchestrator"

    # Redis/Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Authentication
    jwt_secret_key: str = secrets.token_urlsafe(32)  # Override in production!
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7

    # LLM Configuration
    anthropic_api_key: Optional[str] = None
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096

    # Orchestrator settings
    max_iterations: int = 50
    task_timeout_seconds: int = 300
    max_retries: int = 3

    # Quality gate settings
    require_code_review: bool = True
    require_security_review: bool = True

    # Rate limiting
    rate_limit_enabled: bool = True
    default_rate_limit: int = 120  # requests per minute
    run_create_rate_limit: int = 10  # new runs per minute

    # Cost controls
    default_run_token_budget: int = 100000  # 100k tokens per run
    default_org_monthly_limit: int = 1000000  # 1M tokens per month

    # Observability
    log_level: str = "INFO"
    log_format: str = "json"  # json or text
    redact_sensitive_data: bool = True

    # CORS
    cors_origins: str = "*"  # Comma-separated origins or * for all

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()


# Role queue mapping
ROLE_QUEUES = {
    "orchestrator": "q_orch",
    "business_analyst": "q_ba",
    "project_manager": "q_pm",
    "ux_engineer": "q_ux",
    "tech_lead": "q_tl",
    "database_engineer": "q_db",
    "backend_engineer": "q_be",
    "frontend_engineer": "q_fe",
    "code_reviewer": "q_cr",
    "security_reviewer": "q_sec",
    "cleanup_agent": "q_cleanup",
    "data_scientist": "q_ds",
    "design_reviewer": "q_dr",
}

# All queues for Celery configuration
ALL_QUEUES = list(ROLE_QUEUES.values())


# Role descriptions for quick reference
ROLE_DESCRIPTIONS = {
    "orchestrator": "Coordinates multi-agent workflows and task dispatch",
    "business_analyst": "Requirements gathering, stakeholder analysis, success criteria",
    "project_manager": "Project planning, timeline, resource allocation, risk assessment",
    "ux_engineer": "User experience design, wireframes, accessibility, design systems",
    "tech_lead": "Technical architecture, technology stack, API design, implementation guidelines",
    "database_engineer": "Database schema design, optimization, migrations, data security",
    "backend_engineer": "API implementation, business logic, authentication, testing",
    "frontend_engineer": "UI components, state management, accessibility, performance",
    "code_reviewer": "Code quality gate - standards, bugs, test coverage, documentation",
    "security_reviewer": "Security gate - vulnerabilities, OWASP, compliance, threat assessment",
    "cleanup_agent": "Repository hygiene, dead code removal, AI artifact cleanup",
    "data_scientist": "Data analysis, ML pipelines, feature engineering, model design",
    "design_reviewer": "Design quality gate - UI/UX consistency, accessibility, responsive behavior",
}
