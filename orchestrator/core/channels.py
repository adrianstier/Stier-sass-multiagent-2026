"""Agent Collaboration Channels.

Provides real-time communication between agents:
- Redis pub/sub for agent-to-agent messaging
- Synchronous handoffs for tightly coupled tasks
- Multi-agent "meetings" for complex decisions
- Message history and audit trail
"""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set
from uuid import UUID
import uuid as uuid_module
import threading
import time

from orchestrator.core.config import settings
from orchestrator.core.database import get_db
from orchestrator.core.models import Event
from orchestrator.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Message Types
# =============================================================================

@dataclass
class AgentMessage:
    """A message sent between agents."""

    id: str
    channel_id: str
    sender_role: str
    sender_task_id: Optional[str]
    message_type: str  # request, response, notification, handoff, meeting_invite
    content: Dict[str, Any]
    priority: int = 0  # Higher = more urgent
    requires_response: bool = False
    response_timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "sender_role": self.sender_role,
            "sender_task_id": self.sender_task_id,
            "message_type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "requires_response": self.requires_response,
            "response_timeout_seconds": self.response_timeout_seconds,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        return cls(
            id=data["id"],
            channel_id=data["channel_id"],
            sender_role=data["sender_role"],
            sender_task_id=data.get("sender_task_id"),
            message_type=data["message_type"],
            content=data["content"],
            priority=data.get("priority", 0),
            requires_response=data.get("requires_response", False),
            response_timeout_seconds=data.get("response_timeout_seconds", 300),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.utcnow(),
        )


@dataclass
class HandoffRequest:
    """A synchronous handoff request between agents."""

    id: str
    from_role: str
    to_role: str
    task_id: str
    run_id: str
    handoff_type: str  # review_request, clarification, approval, delegation
    context: Dict[str, Any]
    artifacts: List[str]  # Artifact IDs to include
    instructions: str
    deadline: Optional[datetime] = None
    status: str = "pending"  # pending, accepted, completed, rejected, timeout
    response: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_role": self.from_role,
            "to_role": self.to_role,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "handoff_type": self.handoff_type,
            "context": self.context,
            "artifacts": self.artifacts,
            "instructions": self.instructions,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status,
            "response": self.response,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AgentMeeting:
    """A multi-agent meeting for complex decisions."""

    id: str
    run_id: str
    topic: str
    initiated_by: str
    participants: List[str]  # Role names
    agenda: List[str]
    context: Dict[str, Any]
    status: str = "scheduled"  # scheduled, in_progress, concluded, cancelled
    messages: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    concluded_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "topic": self.topic,
            "initiated_by": self.initiated_by,
            "participants": self.participants,
            "agenda": self.agenda,
            "context": self.context,
            "status": self.status,
            "messages": self.messages,
            "decisions": self.decisions,
            "created_at": self.created_at.isoformat(),
            "concluded_at": self.concluded_at.isoformat() if self.concluded_at else None,
        }


# =============================================================================
# Channel Manager
# =============================================================================

class ChannelManager:
    """Manages communication channels between agents."""

    def __init__(self):
        self._redis = None
        self._pubsub = None
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_handlers: Dict[str, Callable] = {}
        self._pending_handoffs: Dict[str, HandoffRequest] = {}
        self._active_meetings: Dict[str, AgentMeeting] = {}
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def redis(self):
        """Lazy initialization of Redis client."""
        if self._redis is None:
            import redis
            self._redis = redis.from_url(settings.redis_url)
        return self._redis

    def _get_channel_key(self, run_id: str, channel_type: str = "general") -> str:
        """Get Redis channel key."""
        return f"agent_channel:{run_id}:{channel_type}"

    def _get_handoff_key(self, handoff_id: str) -> str:
        """Get Redis key for a handoff."""
        return f"handoff:{handoff_id}"

    def _get_meeting_key(self, meeting_id: str) -> str:
        """Get Redis key for a meeting."""
        return f"meeting:{meeting_id}"

    # =========================================================================
    # Message Publishing
    # =========================================================================

    def send_message(
        self,
        run_id: str,
        sender_role: str,
        message_type: str,
        content: Dict[str, Any],
        sender_task_id: Optional[str] = None,
        target_role: Optional[str] = None,
        priority: int = 0,
        requires_response: bool = False,
    ) -> AgentMessage:
        """Send a message to the agent channel.

        Args:
            run_id: The run ID
            sender_role: Role of the sending agent
            message_type: Type of message
            content: Message content
            sender_task_id: Optional task ID of sender
            target_role: Optional specific recipient role
            priority: Message priority
            requires_response: Whether response is required

        Returns:
            The sent AgentMessage
        """
        channel_type = f"role:{target_role}" if target_role else "general"
        channel_key = self._get_channel_key(run_id, channel_type)

        message = AgentMessage(
            id=str(uuid_module.uuid4()),
            channel_id=channel_key,
            sender_role=sender_role,
            sender_task_id=sender_task_id,
            message_type=message_type,
            content=content,
            priority=priority,
            requires_response=requires_response,
        )

        # Publish to Redis
        self.redis.publish(channel_key, json.dumps(message.to_dict()))

        # Store in message history
        history_key = f"message_history:{run_id}"
        self.redis.lpush(history_key, json.dumps(message.to_dict()))
        self.redis.ltrim(history_key, 0, 999)  # Keep last 1000 messages
        self.redis.expire(history_key, 86400)  # 24 hour TTL

        # Record event
        self._record_event(run_id, "agent_message_sent", message.to_dict())

        logger.debug(
            "message_sent",
            run_id=run_id,
            sender=sender_role,
            message_type=message_type,
            target=target_role,
        )

        return message

    def broadcast_notification(
        self,
        run_id: str,
        sender_role: str,
        notification_type: str,
        content: Dict[str, Any],
    ) -> AgentMessage:
        """Broadcast a notification to all agents in a run.

        Args:
            run_id: The run ID
            sender_role: Role of the sender
            notification_type: Type of notification
            content: Notification content

        Returns:
            The sent message
        """
        return self.send_message(
            run_id=run_id,
            sender_role=sender_role,
            message_type="notification",
            content={
                "notification_type": notification_type,
                **content,
            },
        )

    # =========================================================================
    # Handoff Management
    # =========================================================================

    def create_handoff(
        self,
        run_id: str,
        from_role: str,
        to_role: str,
        task_id: str,
        handoff_type: str,
        instructions: str,
        context: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[str]] = None,
        timeout_seconds: int = 300,
    ) -> HandoffRequest:
        """Create a synchronous handoff request.

        Args:
            run_id: The run ID
            from_role: Role initiating the handoff
            to_role: Role receiving the handoff
            task_id: Related task ID
            handoff_type: Type of handoff
            instructions: Instructions for the recipient
            context: Additional context
            artifacts: Artifact IDs to include
            timeout_seconds: Timeout for response

        Returns:
            The created HandoffRequest
        """
        handoff = HandoffRequest(
            id=str(uuid_module.uuid4()),
            from_role=from_role,
            to_role=to_role,
            task_id=task_id,
            run_id=run_id,
            handoff_type=handoff_type,
            context=context or {},
            artifacts=artifacts or [],
            instructions=instructions,
            deadline=datetime.utcnow() if timeout_seconds else None,
        )

        # Store in Redis
        handoff_key = self._get_handoff_key(handoff.id)
        self.redis.set(handoff_key, json.dumps(handoff.to_dict()))
        self.redis.expire(handoff_key, timeout_seconds + 60)

        # Add to pending
        self._pending_handoffs[handoff.id] = handoff

        # Notify target role
        self.send_message(
            run_id=run_id,
            sender_role=from_role,
            message_type="handoff",
            content={
                "handoff_id": handoff.id,
                "handoff_type": handoff_type,
                "instructions": instructions,
            },
            target_role=to_role,
            requires_response=True,
            priority=5,  # High priority
        )

        # Record event
        self._record_event(run_id, "handoff_created", handoff.to_dict(), task_id)

        logger.info(
            "handoff_created",
            run_id=run_id,
            handoff_id=handoff.id,
            from_role=from_role,
            to_role=to_role,
            handoff_type=handoff_type,
        )

        return handoff

    def respond_to_handoff(
        self,
        handoff_id: str,
        status: str,
        response: Dict[str, Any],
        responder_role: str,
    ) -> bool:
        """Respond to a handoff request.

        Args:
            handoff_id: The handoff ID
            status: Response status (accepted, completed, rejected)
            response: Response content
            responder_role: Role responding

        Returns:
            True if successful
        """
        handoff_key = self._get_handoff_key(handoff_id)
        handoff_data = self.redis.get(handoff_key)

        if not handoff_data:
            return False

        handoff_dict = json.loads(handoff_data)
        handoff_dict["status"] = status
        handoff_dict["response"] = response

        # Update in Redis
        self.redis.set(handoff_key, json.dumps(handoff_dict))

        # Notify originator
        self.send_message(
            run_id=handoff_dict["run_id"],
            sender_role=responder_role,
            message_type="handoff_response",
            content={
                "handoff_id": handoff_id,
                "status": status,
                "response": response,
            },
            target_role=handoff_dict["from_role"],
        )

        # Record event
        self._record_event(
            handoff_dict["run_id"],
            "handoff_responded",
            {"handoff_id": handoff_id, "status": status},
            handoff_dict["task_id"],
        )

        logger.info(
            "handoff_responded",
            handoff_id=handoff_id,
            status=status,
            responder=responder_role,
        )

        return True

    def wait_for_handoff_response(
        self,
        handoff_id: str,
        timeout_seconds: int = 300,
    ) -> Optional[Dict[str, Any]]:
        """Wait for a handoff response (blocking).

        Args:
            handoff_id: The handoff ID
            timeout_seconds: Maximum wait time

        Returns:
            The response or None if timeout
        """
        start_time = time.time()
        handoff_key = self._get_handoff_key(handoff_id)

        while time.time() - start_time < timeout_seconds:
            handoff_data = self.redis.get(handoff_key)
            if handoff_data:
                handoff_dict = json.loads(handoff_data)
                if handoff_dict.get("status") not in ["pending", None]:
                    return handoff_dict
            time.sleep(1)

        # Timeout - mark handoff as timed out
        handoff_data = self.redis.get(handoff_key)
        if handoff_data:
            handoff_dict = json.loads(handoff_data)
            handoff_dict["status"] = "timeout"
            self.redis.set(handoff_key, json.dumps(handoff_dict))

        return None

    # =========================================================================
    # Agent Meetings
    # =========================================================================

    def create_meeting(
        self,
        run_id: str,
        initiated_by: str,
        topic: str,
        participants: List[str],
        agenda: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentMeeting:
        """Create a multi-agent meeting.

        Args:
            run_id: The run ID
            initiated_by: Role initiating the meeting
            topic: Meeting topic
            participants: List of participating roles
            agenda: Meeting agenda items
            context: Additional context

        Returns:
            The created AgentMeeting
        """
        meeting = AgentMeeting(
            id=str(uuid_module.uuid4()),
            run_id=run_id,
            topic=topic,
            initiated_by=initiated_by,
            participants=participants,
            agenda=agenda,
            context=context or {},
        )

        # Store in Redis
        meeting_key = self._get_meeting_key(meeting.id)
        self.redis.set(meeting_key, json.dumps(meeting.to_dict()))
        self.redis.expire(meeting_key, 3600)  # 1 hour TTL

        self._active_meetings[meeting.id] = meeting

        # Notify all participants
        for role in participants:
            self.send_message(
                run_id=run_id,
                sender_role=initiated_by,
                message_type="meeting_invite",
                content={
                    "meeting_id": meeting.id,
                    "topic": topic,
                    "agenda": agenda,
                    "participants": participants,
                },
                target_role=role,
            )

        # Record event
        self._record_event(run_id, "meeting_created", meeting.to_dict())

        logger.info(
            "meeting_created",
            run_id=run_id,
            meeting_id=meeting.id,
            topic=topic,
            participants=participants,
        )

        return meeting

    def add_meeting_message(
        self,
        meeting_id: str,
        sender_role: str,
        content: str,
        message_type: str = "discussion",
    ) -> bool:
        """Add a message to a meeting.

        Args:
            meeting_id: The meeting ID
            sender_role: Role sending the message
            content: Message content
            message_type: Type (discussion, proposal, vote, decision)

        Returns:
            True if successful
        """
        meeting_key = self._get_meeting_key(meeting_id)
        meeting_data = self.redis.get(meeting_key)

        if not meeting_data:
            return False

        meeting_dict = json.loads(meeting_data)

        message = {
            "id": str(uuid_module.uuid4()),
            "sender_role": sender_role,
            "content": content,
            "message_type": message_type,
            "timestamp": datetime.utcnow().isoformat(),
        }

        meeting_dict["messages"].append(message)
        self.redis.set(meeting_key, json.dumps(meeting_dict))

        # Broadcast to all participants
        for role in meeting_dict["participants"]:
            if role != sender_role:
                self.send_message(
                    run_id=meeting_dict["run_id"],
                    sender_role=sender_role,
                    message_type="meeting_message",
                    content={
                        "meeting_id": meeting_id,
                        "message": message,
                    },
                    target_role=role,
                )

        return True

    def record_meeting_decision(
        self,
        meeting_id: str,
        decision: str,
        decided_by: str,
        rationale: str,
        votes: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Record a decision made in a meeting.

        Args:
            meeting_id: The meeting ID
            decision: The decision made
            decided_by: Role that made the decision
            rationale: Reasoning behind the decision
            votes: Optional votes from participants

        Returns:
            True if successful
        """
        meeting_key = self._get_meeting_key(meeting_id)
        meeting_data = self.redis.get(meeting_key)

        if not meeting_data:
            return False

        meeting_dict = json.loads(meeting_data)

        decision_record = {
            "id": str(uuid_module.uuid4()),
            "decision": decision,
            "decided_by": decided_by,
            "rationale": rationale,
            "votes": votes or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        meeting_dict["decisions"].append(decision_record)
        self.redis.set(meeting_key, json.dumps(meeting_dict))

        # Record event
        self._record_event(
            meeting_dict["run_id"],
            "meeting_decision",
            {"meeting_id": meeting_id, "decision": decision_record},
        )

        return True

    def conclude_meeting(
        self,
        meeting_id: str,
        summary: str,
        concluded_by: str,
    ) -> Optional[Dict[str, Any]]:
        """Conclude a meeting.

        Args:
            meeting_id: The meeting ID
            summary: Meeting summary
            concluded_by: Role concluding the meeting

        Returns:
            The concluded meeting data or None
        """
        meeting_key = self._get_meeting_key(meeting_id)
        meeting_data = self.redis.get(meeting_key)

        if not meeting_data:
            return None

        meeting_dict = json.loads(meeting_data)
        meeting_dict["status"] = "concluded"
        meeting_dict["concluded_at"] = datetime.utcnow().isoformat()
        meeting_dict["summary"] = summary

        self.redis.set(meeting_key, json.dumps(meeting_dict))

        # Notify all participants
        for role in meeting_dict["participants"]:
            self.send_message(
                run_id=meeting_dict["run_id"],
                sender_role=concluded_by,
                message_type="meeting_concluded",
                content={
                    "meeting_id": meeting_id,
                    "summary": summary,
                    "decisions": meeting_dict["decisions"],
                },
                target_role=role,
            )

        # Record event
        self._record_event(meeting_dict["run_id"], "meeting_concluded", meeting_dict)

        logger.info(
            "meeting_concluded",
            meeting_id=meeting_id,
            decisions_count=len(meeting_dict["decisions"]),
        )

        return meeting_dict

    # =========================================================================
    # Message History
    # =========================================================================

    def get_message_history(
        self,
        run_id: str,
        limit: int = 100,
        role_filter: Optional[str] = None,
    ) -> List[AgentMessage]:
        """Get message history for a run.

        Args:
            run_id: The run ID
            limit: Maximum messages to return
            role_filter: Optional filter by sender role

        Returns:
            List of messages
        """
        history_key = f"message_history:{run_id}"
        messages_data = self.redis.lrange(history_key, 0, limit - 1)

        messages = []
        for msg_data in messages_data:
            try:
                msg_dict = json.loads(msg_data)
                if role_filter and msg_dict.get("sender_role") != role_filter:
                    continue
                messages.append(AgentMessage.from_dict(msg_dict))
            except (json.JSONDecodeError, KeyError):
                continue

        return messages

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def subscribe(
        self,
        run_id: str,
        role: str,
        callback: Callable[[AgentMessage], None],
    ) -> str:
        """Subscribe to messages for a role.

        Args:
            run_id: The run ID
            role: The role to subscribe as
            callback: Callback function for messages

        Returns:
            Subscription ID
        """
        channel_key = self._get_channel_key(run_id, f"role:{role}")
        general_key = self._get_channel_key(run_id, "general")

        subscription_id = f"{run_id}:{role}:{uuid_module.uuid4()}"

        # Store callback
        if channel_key not in self._subscribers:
            self._subscribers[channel_key] = []
        if general_key not in self._subscribers:
            self._subscribers[general_key] = []

        self._subscribers[channel_key].append((subscription_id, callback))
        self._subscribers[general_key].append((subscription_id, callback))

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from messages."""
        for channel, subs in self._subscribers.items():
            self._subscribers[channel] = [
                (sid, cb) for sid, cb in subs if sid != subscription_id
            ]

    # =========================================================================
    # Helpers
    # =========================================================================

    def _record_event(
        self,
        run_id: str,
        event_type: str,
        data: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> None:
        """Record a channel event."""
        with get_db() as db:
            event = Event(
                run_id=UUID(run_id),
                task_id=UUID(task_id) if task_id else None,
                event_type=event_type,
                actor="channel_manager",
                data=data,
            )
            db.add(event)
            db.commit()


# Global channel manager instance
_channel_manager: Optional[ChannelManager] = None


def get_channel_manager() -> ChannelManager:
    """Get the global channel manager instance."""
    global _channel_manager
    if _channel_manager is None:
        _channel_manager = ChannelManager()
    return _channel_manager
