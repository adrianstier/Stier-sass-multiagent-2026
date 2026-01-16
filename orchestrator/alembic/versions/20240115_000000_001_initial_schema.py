"""Initial schema with auth and multi-tenancy

Revision ID: 001
Revises:
Create Date: 2024-01-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create organizations table
    op.create_table('organizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('slug', sa.String(length=100), nullable=False),
        sa.Column('plan', sa.String(length=50), nullable=True),
        sa.Column('monthly_token_limit', sa.Integer(), nullable=True),
        sa.Column('tokens_used_this_month', sa.Integer(), nullable=True),
        sa.Column('billing_cycle_start', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('slug')
    )

    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )

    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_by_user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('key_prefix', sa.String(length=10), nullable=False),
        sa.Column('key_hash', sa.String(length=255), nullable=False),
        sa.Column('scopes', sa.Text(), nullable=True),
        sa.Column('rate_limit', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create runs table with multi-tenancy
    op.create_table('runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_by_user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('goal', sa.Text(), nullable=False),
        sa.Column('context', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('status', sa.Enum('pending', 'planning', 'running', 'needs_input', 'paused', 'completed', 'failed', 'cancelled', name='runstatus'), nullable=False),
        sa.Column('current_phase', sa.String(length=100), nullable=True),
        sa.Column('success_criteria', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('acceptance_criteria', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('max_iterations', sa.Integer(), nullable=True),
        sa.Column('current_iteration', sa.Integer(), nullable=True),
        sa.Column('budget_tokens', sa.Integer(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('total_cost_usd', sa.String(length=20), nullable=True),
        sa.Column('code_review_status', sa.Enum('pending', 'passed', 'failed', 'waived', name='gatestatus'), nullable=True),
        sa.Column('security_review_status', sa.Enum('pending', 'passed', 'failed', 'waived', name='gatestatus'), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('blocked_reason', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_runs_created_at', 'runs', ['created_at'], unique=False)
    op.create_index('ix_runs_organization_id', 'runs', ['organization_id'], unique=False)
    op.create_index('ix_runs_status', 'runs', ['status'], unique=False)

    # Create tasks table
    op.create_table('tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('task_type', sa.String(length=100), nullable=False),
        sa.Column('assigned_role', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('input_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('expected_artifacts', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('validation_method', sa.String(length=100), nullable=True),
        sa.Column('dependencies', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('status', sa.Enum('pending', 'queued', 'running', 'waiting_dependency', 'needs_input', 'completed', 'failed', 'skipped', name='taskstatus'), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=True),
        sa.Column('idempotency_key', sa.String(length=255), nullable=False),
        sa.Column('retry_count', sa.Integer(), nullable=True),
        sa.Column('max_retries', sa.Integer(), nullable=True),
        sa.Column('result', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('queued_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id', 'idempotency_key', name='uq_task_idempotency')
    )
    op.create_index('ix_tasks_assigned_role', 'tasks', ['assigned_role'], unique=False)
    op.create_index('ix_tasks_run_id', 'tasks', ['run_id'], unique=False)
    op.create_index('ix_tasks_status', 'tasks', ['status'], unique=False)

    # Create events table
    op.create_table('events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('actor', sa.String(length=100), nullable=False),
        sa.Column('data', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=True),
        sa.Column('response', sa.Text(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_events_event_type', 'events', ['event_type'], unique=False)
    op.create_index('ix_events_run_id', 'events', ['run_id'], unique=False)
    op.create_index('ix_events_task_id', 'events', ['task_id'], unique=False)
    op.create_index('ix_events_timestamp', 'events', ['timestamp'], unique=False)

    # Create artifacts table
    op.create_table('artifacts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('artifact_type', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('version', sa.Integer(), nullable=True),
        sa.Column('content_type', sa.String(length=100), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('is_valid', sa.Boolean(), nullable=True),
        sa.Column('validation_errors', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('produced_by', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_artifacts_artifact_type', 'artifacts', ['artifact_type'], unique=False)
    op.create_index('ix_artifacts_produced_by', 'artifacts', ['produced_by'], unique=False)
    op.create_index('ix_artifacts_run_id', 'artifacts', ['run_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_artifacts_run_id', table_name='artifacts')
    op.drop_index('ix_artifacts_produced_by', table_name='artifacts')
    op.drop_index('ix_artifacts_artifact_type', table_name='artifacts')
    op.drop_table('artifacts')
    op.drop_index('ix_events_timestamp', table_name='events')
    op.drop_index('ix_events_task_id', table_name='events')
    op.drop_index('ix_events_run_id', table_name='events')
    op.drop_index('ix_events_event_type', table_name='events')
    op.drop_table('events')
    op.drop_index('ix_tasks_status', table_name='tasks')
    op.drop_index('ix_tasks_run_id', table_name='tasks')
    op.drop_index('ix_tasks_assigned_role', table_name='tasks')
    op.drop_table('tasks')
    op.drop_index('ix_runs_status', table_name='runs')
    op.drop_index('ix_runs_organization_id', table_name='runs')
    op.drop_index('ix_runs_created_at', table_name='runs')
    op.drop_table('runs')
    op.drop_table('api_keys')
    op.drop_table('users')
    op.drop_table('organizations')

    # Drop enums
    op.execute('DROP TYPE IF EXISTS runstatus')
    op.execute('DROP TYPE IF EXISTS taskstatus')
    op.execute('DROP TYPE IF EXISTS gatestatus')
