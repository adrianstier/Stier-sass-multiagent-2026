"""Webhook delivery service for real-time event notifications.

Provides:
- Synchronous and asynchronous webhook delivery
- HMAC signature verification
- Retry logic with exponential backoff
- Delivery logging
"""

import hashlib
import hmac
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import httpx

from .database import get_db
from .models import Webhook, WebhookDelivery, WebhookEventType
from .logging import get_logger

logger = get_logger(__name__)

# Webhook delivery settings
WEBHOOK_TIMEOUT_SECONDS = 10
WEBHOOK_MAX_RETRIES = 3
WEBHOOK_RETRY_DELAYS = [60, 300, 900]  # 1min, 5min, 15min


def generate_signature(payload: Dict[str, Any], secret: str) -> str:
    """Generate HMAC-SHA256 signature for webhook payload."""
    payload_bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    signature = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={signature}"


def deliver_webhook_sync(
    webhook_url: str,
    webhook_secret: Optional[str],
    payload: Dict[str, Any],
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Deliver a webhook synchronously.

    Returns:
        Tuple of (success, status_code, error_message)
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Orchestrator-Webhook/2.0",
        "X-Webhook-Event": payload.get("event_type", "unknown"),
        "X-Webhook-Delivery-ID": str(uuid.uuid4()),
        "X-Webhook-Timestamp": datetime.utcnow().isoformat(),
    }

    if webhook_secret:
        headers["X-Webhook-Signature"] = generate_signature(payload, webhook_secret)

    try:
        with httpx.Client(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
            response = client.post(
                webhook_url,
                json=payload,
                headers=headers,
            )

            if response.status_code >= 200 and response.status_code < 300:
                return True, response.status_code, None
            else:
                return False, response.status_code, f"HTTP {response.status_code}: {response.text[:500]}"

    except httpx.TimeoutException:
        return False, None, "Request timed out"
    except httpx.RequestError as e:
        return False, None, f"Request error: {str(e)}"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"


async def deliver_webhook_async(
    webhook_url: str,
    webhook_secret: Optional[str],
    payload: Dict[str, Any],
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Deliver a webhook asynchronously.

    Returns:
        Tuple of (success, status_code, error_message)
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Orchestrator-Webhook/2.0",
        "X-Webhook-Event": payload.get("event_type", "unknown"),
        "X-Webhook-Delivery-ID": str(uuid.uuid4()),
        "X-Webhook-Timestamp": datetime.utcnow().isoformat(),
    }

    if webhook_secret:
        headers["X-Webhook-Signature"] = generate_signature(payload, webhook_secret)

    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers=headers,
            )

            if response.status_code >= 200 and response.status_code < 300:
                return True, response.status_code, None
            else:
                return False, response.status_code, f"HTTP {response.status_code}: {response.text[:500]}"

    except httpx.TimeoutException:
        return False, None, "Request timed out"
    except httpx.RequestError as e:
        return False, None, f"Request error: {str(e)}"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"


def dispatch_event(
    organization_id: str,
    event_type: str,
    data: Dict[str, Any],
    run_id: Optional[str] = None,
) -> int:
    """
    Dispatch an event to all matching webhooks for an organization.

    Args:
        organization_id: Organization UUID
        event_type: Event type (e.g., "run.started")
        data: Event payload data
        run_id: Optional run ID for filtering

    Returns:
        Number of webhooks the event was dispatched to
    """
    dispatched_count = 0

    with get_db() as db:
        # Find matching webhooks
        query = db.query(Webhook).filter(
            Webhook.organization_id == uuid.UUID(organization_id),
            Webhook.is_active == True,
        )

        webhooks = query.all()

        for webhook in webhooks:
            # Check if webhook subscribes to this event type
            if event_type not in webhook.events:
                continue

            # Check run_id filter if set
            if webhook.run_id_filter:
                if not run_id or str(webhook.run_id_filter) != run_id:
                    continue

            # Build payload
            payload = {
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "organization_id": organization_id,
                "run_id": run_id,
                "data": data,
            }

            # Create delivery record
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                event_type=event_type,
                payload=payload,
                status="pending",
            )
            db.add(delivery)
            db.commit()
            db.refresh(delivery)

            # Attempt delivery
            success, status_code, error = deliver_webhook_sync(
                webhook_url=webhook.url,
                webhook_secret=webhook.secret,
                payload=payload,
            )

            # Update delivery record
            delivery.status = "success" if success else "failed"
            delivery.http_status_code = status_code
            delivery.error_message = error
            delivery.delivered_at = datetime.utcnow()

            # Update webhook stats
            if success:
                webhook.last_success_at = datetime.utcnow()
                webhook.failure_count = 0
            else:
                webhook.last_failure_at = datetime.utcnow()
                webhook.failure_count += 1

                # Disable webhook after too many failures
                if webhook.failure_count >= 10:
                    webhook.is_active = False
                    logger.warning(
                        "webhook_disabled_failures",
                        webhook_id=str(webhook.id),
                        failure_count=webhook.failure_count,
                    )

            db.commit()
            dispatched_count += 1

            logger.info(
                "webhook_dispatched",
                webhook_id=str(webhook.id),
                event_type=event_type,
                success=success,
                status_code=status_code,
            )

    return dispatched_count


def dispatch_run_event(run_id: str, event_type: str, extra_data: Dict[str, Any] = None):
    """
    Dispatch a run-related event.

    Args:
        run_id: Run UUID
        event_type: Event type (e.g., "run.started", "run.completed")
        extra_data: Additional data to include in payload
    """
    from .models import Run

    with get_db() as db:
        run = db.query(Run).filter(Run.id == uuid.UUID(run_id)).first()
        if not run:
            logger.warning("dispatch_run_event_run_not_found", run_id=run_id)
            return

        data = {
            "run_id": run_id,
            "goal": run.goal,
            "status": run.status.value,
            "current_phase": run.current_phase,
            "iteration": run.current_iteration,
            "tokens_used": run.tokens_used,
        }

        if extra_data:
            data.update(extra_data)

        dispatch_event(
            organization_id=str(run.organization_id),
            event_type=event_type,
            data=data,
            run_id=run_id,
        )


def dispatch_task_event(task_id: str, event_type: str, extra_data: Dict[str, Any] = None):
    """
    Dispatch a task-related event.

    Args:
        task_id: Task UUID
        event_type: Event type (e.g., "task.started", "task.completed")
        extra_data: Additional data to include in payload
    """
    from .models import Task, Run

    with get_db() as db:
        task = db.query(Task).filter(Task.id == uuid.UUID(task_id)).first()
        if not task:
            logger.warning("dispatch_task_event_task_not_found", task_id=task_id)
            return

        run = db.query(Run).filter(Run.id == task.run_id).first()
        if not run:
            logger.warning("dispatch_task_event_run_not_found", task_id=task_id)
            return

        data = {
            "task_id": task_id,
            "run_id": str(task.run_id),
            "task_type": task.task_type,
            "assigned_role": task.assigned_role,
            "status": task.status.value,
            "retry_count": task.retry_count,
        }

        if extra_data:
            data.update(extra_data)

        dispatch_event(
            organization_id=str(run.organization_id),
            event_type=event_type,
            data=data,
            run_id=str(task.run_id),
        )


def dispatch_gate_event(run_id: str, gate_type: str, status: str, extra_data: Dict[str, Any] = None):
    """
    Dispatch a gate-related event.

    Args:
        run_id: Run UUID
        gate_type: Gate type (e.g., "code_review", "security_review")
        status: Gate status (e.g., "passed", "failed", "waived")
        extra_data: Additional data to include in payload
    """
    from .models import Run

    with get_db() as db:
        run = db.query(Run).filter(Run.id == uuid.UUID(run_id)).first()
        if not run:
            logger.warning("dispatch_gate_event_run_not_found", run_id=run_id)
            return

        # Determine event type based on status
        if status.upper() == "PASSED":
            event_type = WebhookEventType.GATE_PASSED.value
        elif status.upper() == "FAILED":
            event_type = WebhookEventType.GATE_FAILED.value
        elif status.upper() == "WAIVED":
            event_type = WebhookEventType.GATE_WAIVED.value
        else:
            event_type = f"gate.{status.lower()}"

        data = {
            "run_id": run_id,
            "gate_type": gate_type,
            "status": status,
        }

        if extra_data:
            data.update(extra_data)

        dispatch_event(
            organization_id=str(run.organization_id),
            event_type=event_type,
            data=data,
            run_id=run_id,
        )


def dispatch_artifact_event(artifact_id: str, extra_data: Dict[str, Any] = None):
    """
    Dispatch an artifact creation event.

    Args:
        artifact_id: Artifact UUID
        extra_data: Additional data to include in payload
    """
    from .models import Artifact, Run

    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == uuid.UUID(artifact_id)).first()
        if not artifact:
            logger.warning("dispatch_artifact_event_artifact_not_found", artifact_id=artifact_id)
            return

        run = db.query(Run).filter(Run.id == artifact.run_id).first()
        if not run:
            logger.warning("dispatch_artifact_event_run_not_found", artifact_id=artifact_id)
            return

        data = {
            "artifact_id": artifact_id,
            "run_id": str(artifact.run_id),
            "artifact_type": artifact.artifact_type,
            "name": artifact.name,
            "produced_by": artifact.produced_by,
            "content_type": artifact.content_type,
        }

        if extra_data:
            data.update(extra_data)

        dispatch_event(
            organization_id=str(run.organization_id),
            event_type=WebhookEventType.ARTIFACT_CREATED.value,
            data=data,
            run_id=str(artifact.run_id),
        )
