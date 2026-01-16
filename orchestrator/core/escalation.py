"""
Human-in-the-Loop Escalation System

Implements tiered escalation with parallel work continuation:
- Agent → Lead Agent → Human escalation tiers
- Timeout-based auto-decisions with conservative defaults
- Partial approvals for gates (approve with conditions)
- Slack/email notifications for urgent escalations
- Non-blocking escalation that allows work to continue
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import httpx
from sqlalchemy import Column, DateTime, Enum as SQLEnum, ForeignKey, Integer, String, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Session

from orchestrator.core.models import Base, Event, Run, Task, get_db

logger = logging.getLogger(__name__)


class EscalationTier(str, Enum):
    """Escalation hierarchy tiers."""
    AGENT = "agent"
    LEAD_AGENT = "lead_agent"
    HUMAN = "human"


class EscalationStatus(str, Enum):
    """Status of an escalation request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPROVED_WITH_CONDITIONS = "approved_with_conditions"
    AUTO_DECIDED = "auto_decided"
    TIMED_OUT = "timed_out"
    ESCALATED = "escalated"


class EscalationPriority(str, Enum):
    """Priority levels for escalations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationType(str, Enum):
    """Types of escalation requests."""
    GATE_APPROVAL = "gate_approval"
    BUDGET_OVERRIDE = "budget_override"
    SECURITY_REVIEW = "security_review"
    ERROR_RESOLUTION = "error_resolution"
    AMBIGUOUS_REQUIREMENT = "ambiguous_requirement"
    QUALITY_CONCERN = "quality_concern"
    SENSITIVE_DATA = "sensitive_data"


# Database model for escalation requests
class EscalationRequest(Base):
    """Persistent escalation request."""
    __tablename__ = "escalation_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False, index=True)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True, index=True)

    escalation_type = Column(SQLEnum(EscalationType), nullable=False)
    current_tier = Column(SQLEnum(EscalationTier), default=EscalationTier.AGENT)
    status = Column(SQLEnum(EscalationStatus), default=EscalationStatus.PENDING)
    priority = Column(SQLEnum(EscalationPriority), default=EscalationPriority.MEDIUM)

    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    context = Column(JSONB, default=dict)

    # Decision tracking
    decision = Column(Text, nullable=True)
    conditions = Column(JSONB, default=list)  # Conditions for partial approval
    decided_by = Column(String(255), nullable=True)
    decided_at = Column(DateTime, nullable=True)

    # Auto-decision configuration
    auto_decision_enabled = Column(Boolean, default=True)
    auto_decision_timeout = Column(Integer, default=3600)  # seconds
    auto_decision_default = Column(String(50), default="reject")  # Conservative default

    # Notification tracking
    notifications_sent = Column(JSONB, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)


@dataclass
class EscalationConfig:
    """Configuration for escalation behavior."""
    # Timeouts per tier (seconds)
    tier_timeouts: dict[EscalationTier, int] = field(default_factory=lambda: {
        EscalationTier.AGENT: 300,       # 5 minutes
        EscalationTier.LEAD_AGENT: 1800,  # 30 minutes
        EscalationTier.HUMAN: 3600,       # 1 hour
    })

    # Auto-decision defaults per type (conservative)
    auto_decisions: dict[EscalationType, str] = field(default_factory=lambda: {
        EscalationType.GATE_APPROVAL: "reject",
        EscalationType.BUDGET_OVERRIDE: "reject",
        EscalationType.SECURITY_REVIEW: "reject",
        EscalationType.ERROR_RESOLUTION: "retry",
        EscalationType.AMBIGUOUS_REQUIREMENT: "pause",
        EscalationType.QUALITY_CONCERN: "reject",
        EscalationType.SENSITIVE_DATA: "reject",
    })

    # Notification channels
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#orchestrator-escalations"
    email_recipients: list[str] = field(default_factory=list)
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None

    # Priority thresholds for auto-escalation
    auto_escalate_on_timeout: bool = True
    critical_always_human: bool = True


@dataclass
class EscalationDecision:
    """Result of an escalation decision."""
    status: EscalationStatus
    decision: str
    conditions: list[str] = field(default_factory=list)
    decided_by: str = ""
    tier: EscalationTier = EscalationTier.AGENT
    metadata: dict[str, Any] = field(default_factory=dict)


class NotificationService:
    """Handles sending notifications for escalations."""

    def __init__(self, config: EscalationConfig):
        self.config = config

    async def send_slack_notification(
        self,
        escalation: EscalationRequest,
        message: str
    ) -> bool:
        """Send Slack notification for escalation."""
        if not self.config.slack_webhook_url:
            logger.warning("Slack webhook not configured")
            return False

        priority_emoji = {
            EscalationPriority.LOW: "🔵",
            EscalationPriority.MEDIUM: "🟡",
            EscalationPriority.HIGH: "🟠",
            EscalationPriority.CRITICAL: "🔴",
        }

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{priority_emoji.get(escalation.priority, '⚪')} Escalation: {escalation.title}",
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Type:*\n{escalation.escalation_type.value}"},
                    {"type": "mrkdwn", "text": f"*Priority:*\n{escalation.priority.value}"},
                    {"type": "mrkdwn", "text": f"*Tier:*\n{escalation.current_tier.value}"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n`{escalation.run_id}`"},
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Description:*\n{escalation.description[:500]}"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Approve"},
                        "style": "primary",
                        "action_id": f"approve_{escalation.id}",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Reject"},
                        "style": "danger",
                        "action_id": f"reject_{escalation.id}",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "⚠️ Approve with Conditions"},
                        "action_id": f"conditional_{escalation.id}",
                    },
                ]
            }
        ]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.slack_webhook_url,
                    json={
                        "channel": self.config.slack_channel,
                        "blocks": blocks,
                        "text": message,
                    },
                    timeout=10.0
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    async def send_email_notification(
        self,
        escalation: EscalationRequest,
        subject: str,
        body: str
    ) -> bool:
        """Send email notification for escalation."""
        if not self.config.smtp_host or not self.config.email_recipients:
            logger.warning("Email not configured")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg["Subject"] = f"[{escalation.priority.value.upper()}] {subject}"
            msg["From"] = self.config.smtp_user
            msg["To"] = ", ".join(self.config.email_recipients)

            html_body = f"""
            <html>
            <body>
                <h2>Escalation Request: {escalation.title}</h2>
                <table>
                    <tr><td><strong>Type:</strong></td><td>{escalation.escalation_type.value}</td></tr>
                    <tr><td><strong>Priority:</strong></td><td>{escalation.priority.value}</td></tr>
                    <tr><td><strong>Current Tier:</strong></td><td>{escalation.current_tier.value}</td></tr>
                    <tr><td><strong>Run ID:</strong></td><td>{escalation.run_id}</td></tr>
                </table>
                <h3>Description</h3>
                <p>{escalation.description}</p>
                <h3>Context</h3>
                <pre>{json.dumps(escalation.context, indent=2)}</pre>
                <hr>
                <p>Please respond via the orchestrator dashboard or API.</p>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                if self.config.smtp_user and self.config.smtp_password:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)

            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    async def notify_escalation(
        self,
        escalation: EscalationRequest,
        db: Session
    ) -> list[str]:
        """Send all configured notifications for an escalation."""
        notifications_sent = []

        message = f"New escalation: {escalation.title} ({escalation.priority.value} priority)"

        # Send Slack notification
        if await self.send_slack_notification(escalation, message):
            notifications_sent.append("slack")

        # Send email for high/critical priority
        if escalation.priority in [EscalationPriority.HIGH, EscalationPriority.CRITICAL]:
            if await self.send_email_notification(
                escalation,
                f"Escalation: {escalation.title}",
                escalation.description
            ):
                notifications_sent.append("email")

        # Update notification tracking
        escalation.notifications_sent = escalation.notifications_sent + [
            {"channel": channel, "sent_at": datetime.utcnow().isoformat()}
            for channel in notifications_sent
        ]
        db.commit()

        return notifications_sent


class EscalationManager:
    """
    Manages human-in-the-loop escalation with tiered hierarchy.

    Features:
    - Agent → Lead Agent → Human escalation
    - Parallel work continuation while awaiting decisions
    - Timeout-based auto-decisions with conservative defaults
    - Partial approvals with conditions
    - Multi-channel notifications
    """

    def __init__(self, config: Optional[EscalationConfig] = None):
        self.config = config or EscalationConfig()
        self.notification_service = NotificationService(self.config)
        self._pending_callbacks: dict[uuid.UUID, Callable] = {}

    async def create_escalation(
        self,
        db: Session,
        run_id: uuid.UUID,
        escalation_type: EscalationType,
        title: str,
        description: str,
        priority: EscalationPriority = EscalationPriority.MEDIUM,
        task_id: Optional[uuid.UUID] = None,
        context: Optional[dict] = None,
        auto_decision_timeout: Optional[int] = None,
    ) -> EscalationRequest:
        """Create a new escalation request."""

        # Critical priority always goes to human
        initial_tier = EscalationTier.HUMAN if (
            priority == EscalationPriority.CRITICAL and
            self.config.critical_always_human
        ) else EscalationTier.AGENT

        timeout = auto_decision_timeout or self.config.tier_timeouts.get(
            initial_tier, 3600
        )

        escalation = EscalationRequest(
            run_id=run_id,
            task_id=task_id,
            escalation_type=escalation_type,
            current_tier=initial_tier,
            priority=priority,
            title=title,
            description=description,
            context=context or {},
            auto_decision_timeout=timeout,
            auto_decision_default=self.config.auto_decisions.get(
                escalation_type, "reject"
            ),
            expires_at=datetime.utcnow() + timedelta(seconds=timeout),
        )

        db.add(escalation)
        db.commit()
        db.refresh(escalation)

        # Record event
        event = Event(
            run_id=run_id,
            task_id=task_id,
            event_type="escalation_created",
            payload={
                "escalation_id": str(escalation.id),
                "type": escalation_type.value,
                "tier": initial_tier.value,
                "priority": priority.value,
                "title": title,
            }
        )
        db.add(event)
        db.commit()

        # Send notifications
        await self.notification_service.notify_escalation(escalation, db)

        logger.info(
            f"Created escalation {escalation.id}: {title} "
            f"(tier={initial_tier.value}, priority={priority.value})"
        )

        return escalation

    async def escalate_to_next_tier(
        self,
        db: Session,
        escalation_id: uuid.UUID,
        reason: str = "timeout"
    ) -> Optional[EscalationRequest]:
        """Escalate to the next tier in the hierarchy."""
        escalation = db.query(EscalationRequest).filter(
            EscalationRequest.id == escalation_id
        ).first()

        if not escalation or escalation.status != EscalationStatus.PENDING:
            return None

        # Determine next tier
        tier_order = [EscalationTier.AGENT, EscalationTier.LEAD_AGENT, EscalationTier.HUMAN]
        current_index = tier_order.index(escalation.current_tier)

        if current_index >= len(tier_order) - 1:
            # Already at highest tier, apply auto-decision
            return await self._apply_auto_decision(db, escalation)

        next_tier = tier_order[current_index + 1]
        escalation.current_tier = next_tier
        escalation.status = EscalationStatus.ESCALATED
        escalation.expires_at = datetime.utcnow() + timedelta(
            seconds=self.config.tier_timeouts.get(next_tier, 3600)
        )

        # Reset status for new tier
        escalation.status = EscalationStatus.PENDING

        db.commit()

        # Record event
        event = Event(
            run_id=escalation.run_id,
            task_id=escalation.task_id,
            event_type="escalation_tier_change",
            payload={
                "escalation_id": str(escalation.id),
                "previous_tier": tier_order[current_index].value,
                "new_tier": next_tier.value,
                "reason": reason,
            }
        )
        db.add(event)
        db.commit()

        # Send notifications for new tier
        await self.notification_service.notify_escalation(escalation, db)

        logger.info(
            f"Escalated {escalation.id} to {next_tier.value} (reason: {reason})"
        )

        return escalation

    async def _apply_auto_decision(
        self,
        db: Session,
        escalation: EscalationRequest
    ) -> EscalationRequest:
        """Apply auto-decision when escalation times out at highest tier."""
        default_decision = escalation.auto_decision_default

        escalation.status = EscalationStatus.AUTO_DECIDED
        escalation.decision = default_decision
        escalation.decided_by = "system_auto"
        escalation.decided_at = datetime.utcnow()

        db.commit()

        # Record event
        event = Event(
            run_id=escalation.run_id,
            task_id=escalation.task_id,
            event_type="escalation_auto_decided",
            payload={
                "escalation_id": str(escalation.id),
                "decision": default_decision,
                "reason": "timeout_at_highest_tier",
            }
        )
        db.add(event)
        db.commit()

        logger.warning(
            f"Auto-decided escalation {escalation.id}: {default_decision} "
            "(timed out at human tier)"
        )

        return escalation

    async def decide_escalation(
        self,
        db: Session,
        escalation_id: uuid.UUID,
        status: EscalationStatus,
        decision: str,
        decided_by: str,
        conditions: Optional[list[str]] = None,
    ) -> EscalationDecision:
        """Record a decision for an escalation."""
        escalation = db.query(EscalationRequest).filter(
            EscalationRequest.id == escalation_id
        ).first()

        if not escalation:
            raise ValueError(f"Escalation {escalation_id} not found")

        if escalation.status != EscalationStatus.PENDING:
            raise ValueError(
                f"Escalation {escalation_id} already decided: {escalation.status}"
            )

        escalation.status = status
        escalation.decision = decision
        escalation.decided_by = decided_by
        escalation.decided_at = datetime.utcnow()
        escalation.conditions = conditions or []

        db.commit()

        # Record event
        event = Event(
            run_id=escalation.run_id,
            task_id=escalation.task_id,
            event_type="escalation_decided",
            payload={
                "escalation_id": str(escalation.id),
                "status": status.value,
                "decision": decision,
                "decided_by": decided_by,
                "conditions": conditions or [],
            }
        )
        db.add(event)
        db.commit()

        logger.info(
            f"Escalation {escalation.id} decided: {status.value} by {decided_by}"
        )

        # Execute callback if registered
        if escalation.id in self._pending_callbacks:
            callback = self._pending_callbacks.pop(escalation.id)
            try:
                await callback(EscalationDecision(
                    status=status,
                    decision=decision,
                    conditions=conditions or [],
                    decided_by=decided_by,
                    tier=escalation.current_tier,
                ))
            except Exception as e:
                logger.error(f"Escalation callback failed: {e}")

        return EscalationDecision(
            status=status,
            decision=decision,
            conditions=conditions or [],
            decided_by=decided_by,
            tier=escalation.current_tier,
        )

    def register_callback(
        self,
        escalation_id: uuid.UUID,
        callback: Callable[[EscalationDecision], Any]
    ):
        """Register a callback for when an escalation is decided."""
        self._pending_callbacks[escalation_id] = callback

    async def check_expired_escalations(self, db: Session) -> list[EscalationRequest]:
        """Check and process expired escalations."""
        now = datetime.utcnow()

        expired = db.query(EscalationRequest).filter(
            EscalationRequest.status == EscalationStatus.PENDING,
            EscalationRequest.expires_at <= now,
        ).all()

        processed = []
        for escalation in expired:
            if self.config.auto_escalate_on_timeout:
                updated = await self.escalate_to_next_tier(
                    db, escalation.id, reason="timeout"
                )
            else:
                updated = await self._apply_auto_decision(db, escalation)

            if updated:
                processed.append(updated)

        return processed

    async def get_pending_escalations(
        self,
        db: Session,
        run_id: Optional[uuid.UUID] = None,
        tier: Optional[EscalationTier] = None,
        priority: Optional[EscalationPriority] = None,
    ) -> list[EscalationRequest]:
        """Get pending escalations with optional filters."""
        query = db.query(EscalationRequest).filter(
            EscalationRequest.status == EscalationStatus.PENDING
        )

        if run_id:
            query = query.filter(EscalationRequest.run_id == run_id)
        if tier:
            query = query.filter(EscalationRequest.current_tier == tier)
        if priority:
            query = query.filter(EscalationRequest.priority == priority)

        return query.order_by(
            EscalationRequest.priority.desc(),
            EscalationRequest.created_at.asc()
        ).all()

    async def get_escalation_stats(self, db: Session) -> dict[str, Any]:
        """Get escalation statistics."""
        from sqlalchemy import func

        total = db.query(func.count(EscalationRequest.id)).scalar() or 0
        pending = db.query(func.count(EscalationRequest.id)).filter(
            EscalationRequest.status == EscalationStatus.PENDING
        ).scalar() or 0

        by_status = dict(
            db.query(
                EscalationRequest.status,
                func.count(EscalationRequest.id)
            ).group_by(EscalationRequest.status).all()
        )

        by_tier = dict(
            db.query(
                EscalationRequest.current_tier,
                func.count(EscalationRequest.id)
            ).filter(
                EscalationRequest.status == EscalationStatus.PENDING
            ).group_by(EscalationRequest.current_tier).all()
        )

        by_priority = dict(
            db.query(
                EscalationRequest.priority,
                func.count(EscalationRequest.id)
            ).filter(
                EscalationRequest.status == EscalationStatus.PENDING
            ).group_by(EscalationRequest.priority).all()
        )

        # Average resolution time
        resolved = db.query(
            func.avg(
                func.extract('epoch', EscalationRequest.decided_at) -
                func.extract('epoch', EscalationRequest.created_at)
            )
        ).filter(
            EscalationRequest.decided_at.isnot(None)
        ).scalar()

        return {
            "total": total,
            "pending": pending,
            "by_status": {k.value if k else "unknown": v for k, v in by_status.items()},
            "by_tier": {k.value if k else "unknown": v for k, v in by_tier.items()},
            "by_priority": {k.value if k else "unknown": v for k, v in by_priority.items()},
            "avg_resolution_seconds": resolved or 0,
        }


class NonBlockingEscalation:
    """
    Wrapper for creating escalations that don't block workflow execution.

    Allows parallel work to continue while awaiting human decisions.
    """

    def __init__(self, manager: EscalationManager):
        self.manager = manager
        self._tasks: dict[uuid.UUID, asyncio.Task] = {}

    async def create_non_blocking(
        self,
        db: Session,
        run_id: uuid.UUID,
        escalation_type: EscalationType,
        title: str,
        description: str,
        on_decision: Callable[[EscalationDecision], Any],
        priority: EscalationPriority = EscalationPriority.MEDIUM,
        task_id: Optional[uuid.UUID] = None,
        context: Optional[dict] = None,
    ) -> uuid.UUID:
        """
        Create an escalation that doesn't block execution.

        Returns the escalation ID immediately, calls on_decision when resolved.
        """
        escalation = await self.manager.create_escalation(
            db=db,
            run_id=run_id,
            escalation_type=escalation_type,
            title=title,
            description=description,
            priority=priority,
            task_id=task_id,
            context=context,
        )

        # Register callback
        self.manager.register_callback(escalation.id, on_decision)

        # Start monitoring task
        self._tasks[escalation.id] = asyncio.create_task(
            self._monitor_escalation(escalation.id, db)
        )

        return escalation.id

    async def _monitor_escalation(
        self,
        escalation_id: uuid.UUID,
        db: Session
    ):
        """Monitor an escalation for timeout."""
        while True:
            await asyncio.sleep(60)  # Check every minute

            escalation = db.query(EscalationRequest).filter(
                EscalationRequest.id == escalation_id
            ).first()

            if not escalation or escalation.status != EscalationStatus.PENDING:
                break

            if escalation.expires_at and datetime.utcnow() >= escalation.expires_at:
                await self.manager.escalate_to_next_tier(
                    db, escalation_id, reason="timeout"
                )


# Singleton instance
_escalation_manager: Optional[EscalationManager] = None


def get_escalation_manager(
    config: Optional[EscalationConfig] = None
) -> EscalationManager:
    """Get or create the escalation manager singleton."""
    global _escalation_manager
    if _escalation_manager is None:
        _escalation_manager = EscalationManager(config)
    return _escalation_manager


# Convenience functions for common escalation patterns
async def request_gate_approval(
    db: Session,
    run_id: uuid.UUID,
    task_id: uuid.UUID,
    gate_type: str,
    gate_result: dict,
    priority: EscalationPriority = EscalationPriority.MEDIUM,
) -> EscalationRequest:
    """Request human approval for a failed quality gate."""
    manager = get_escalation_manager()

    return await manager.create_escalation(
        db=db,
        run_id=run_id,
        task_id=task_id,
        escalation_type=EscalationType.GATE_APPROVAL,
        title=f"Gate Approval Required: {gate_type}",
        description=f"Quality gate '{gate_type}' requires human review.\n\n"
                    f"Gate Result: {json.dumps(gate_result, indent=2)}",
        priority=priority,
        context={"gate_type": gate_type, "gate_result": gate_result},
    )


async def request_budget_override(
    db: Session,
    run_id: uuid.UUID,
    current_spend: float,
    requested_amount: float,
    budget_limit: float,
    reason: str,
) -> EscalationRequest:
    """Request human approval for budget override."""
    manager = get_escalation_manager()

    return await manager.create_escalation(
        db=db,
        run_id=run_id,
        escalation_type=EscalationType.BUDGET_OVERRIDE,
        title=f"Budget Override Request: ${requested_amount:.2f}",
        description=f"Run has exceeded its budget and requires additional funding.\n\n"
                    f"Current Spend: ${current_spend:.2f}\n"
                    f"Budget Limit: ${budget_limit:.2f}\n"
                    f"Requested Additional: ${requested_amount:.2f}\n\n"
                    f"Reason: {reason}",
        priority=EscalationPriority.HIGH,
        context={
            "current_spend": current_spend,
            "requested_amount": requested_amount,
            "budget_limit": budget_limit,
            "reason": reason,
        },
    )


async def request_security_review(
    db: Session,
    run_id: uuid.UUID,
    task_id: uuid.UUID,
    security_findings: list[dict],
) -> EscalationRequest:
    """Request human review of security findings."""
    manager = get_escalation_manager()

    severity_counts = {}
    for finding in security_findings:
        sev = finding.get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    priority = EscalationPriority.CRITICAL if severity_counts.get("critical", 0) > 0 else (
        EscalationPriority.HIGH if severity_counts.get("high", 0) > 0 else
        EscalationPriority.MEDIUM
    )

    return await manager.create_escalation(
        db=db,
        run_id=run_id,
        task_id=task_id,
        escalation_type=EscalationType.SECURITY_REVIEW,
        title=f"Security Review Required: {len(security_findings)} findings",
        description=f"Security scan found potential vulnerabilities.\n\n"
                    f"Findings by severity: {json.dumps(severity_counts)}\n\n"
                    f"Please review and approve or reject the code.",
        priority=priority,
        context={"findings": security_findings, "severity_counts": severity_counts},
    )
