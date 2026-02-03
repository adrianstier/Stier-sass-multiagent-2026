"""Agent Memory and Context Window Management.

Provides intelligent context management to:
- Summarize large artifacts before passing to agents
- Implement sliding window for long conversations
- Add semantic chunking for code artifacts
- Track token usage per artifact and compress when nearing limits
"""

import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from orchestrator.core.config import settings
from orchestrator.core.database import get_db
from orchestrator.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation).

    Uses a simple heuristic: ~4 characters per token for English text.
    For code, uses ~3.5 characters per token.
    """
    if not text:
        return 0

    # Check if text looks like code
    code_indicators = ["{", "}", "def ", "function ", "class ", "import ", "const ", "let "]
    is_code = any(indicator in text for indicator in code_indicators)

    chars_per_token = 3.5 if is_code else 4.0
    return int(len(text) / chars_per_token)


def estimate_tokens_for_messages(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        if isinstance(msg.get("content"), str):
            total += estimate_tokens(msg["content"])
        elif isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    total += estimate_tokens(str(block.get("content", "")))
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    total += estimate_tokens(str(block.get("input", "")))
    return total


def estimate_request_tokens(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None
) -> int:
    """Estimate total tokens for an API request.

    Args:
        system_prompt: The system prompt text
        messages: List of conversation messages
        tools: Optional list of tool definitions

    Returns:
        Estimated total token count for the request
    """
    total = estimate_tokens(system_prompt)
    total += estimate_tokens_for_messages(messages)
    if tools:
        # Tools add ~100-150 tokens each for JSON schema
        total += len(tools) * 120
    # Add overhead for message formatting
    total += len(messages) * 10
    return total


def truncate_messages_to_fit(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    system_prompt_tokens: int,
    tool_tokens: int = 0,
    reserved_output_tokens: int = 8192
) -> List[Dict[str, Any]]:
    """Truncate messages to fit within token limit, preserving most recent.

    Strategy:
    1. Calculate available tokens for messages
    2. Keep the most recent messages that fit
    3. Summarize removed messages into a context note

    Args:
        messages: List of conversation messages
        max_tokens: Maximum context window tokens
        system_prompt_tokens: Tokens used by system prompt
        tool_tokens: Tokens used by tool definitions
        reserved_output_tokens: Tokens to reserve for response

    Returns:
        Truncated list of messages that fits within budget
    """
    available = max_tokens - system_prompt_tokens - tool_tokens - reserved_output_tokens

    if available <= 0:
        logger.error(
            "context_budget_exhausted",
            max_tokens=max_tokens,
            system_prompt_tokens=system_prompt_tokens,
            tool_tokens=tool_tokens,
            reserved_output_tokens=reserved_output_tokens
        )
        # Return just the last message as emergency fallback
        return messages[-1:] if messages else []

    # Calculate tokens for each message
    message_tokens = []
    for msg in messages:
        tokens = estimate_tokens(str(msg.get("content", "")))
        message_tokens.append(tokens)

    # Keep messages from the end until we hit the limit
    kept_messages = []
    total_kept = 0
    removed_count = 0

    for i in range(len(messages) - 1, -1, -1):
        if total_kept + message_tokens[i] <= available:
            kept_messages.insert(0, messages[i])
            total_kept += message_tokens[i]
        else:
            removed_count = i + 1
            break

    if removed_count > 0:
        # Add a summary note about removed context
        summary_note = {
            "role": "user",
            "content": f"[Note: {removed_count} earlier messages were summarized to fit context window. Continuing from recent context.]"
        }
        kept_messages.insert(0, summary_note)

        logger.warning(
            "messages_truncated",
            removed_count=removed_count,
            kept_count=len(kept_messages),
            tokens_before=sum(message_tokens),
            tokens_after=total_kept
        )

    return kept_messages


def check_context_limits(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None
) -> Tuple[int, float, bool]:
    """Check if request is within context limits.

    Args:
        system_prompt: The system prompt text
        messages: List of conversation messages
        tools: Optional list of tool definitions

    Returns:
        Tuple of (estimated_tokens, usage_percentage, needs_truncation)
    """
    estimated = estimate_request_tokens(system_prompt, messages, tools)

    # Use default values if settings not available
    try:
        max_tokens = settings.context_window_max_tokens
        critical_threshold = settings.context_critical_threshold
    except Exception:
        max_tokens = 180000
        critical_threshold = 0.85

    usage_pct = estimated / max_tokens
    needs_truncation = usage_pct >= critical_threshold

    return estimated, usage_pct, needs_truncation


# =============================================================================
# Content Summarization
# =============================================================================

class ContentSummarizer:
    """Summarizes content intelligently based on type."""

    # Maximum tokens before summarization kicks in
    SUMMARY_THRESHOLD = 2000

    # Target tokens after summarization
    SUMMARY_TARGET = 500

    def __init__(self):
        self._summary_cache: Dict[str, str] = {}

    def should_summarize(self, content: str) -> bool:
        """Check if content should be summarized."""
        return estimate_tokens(content) > self.SUMMARY_THRESHOLD

    def get_cache_key(self, content: str) -> str:
        """Generate cache key for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def summarize(
        self,
        content: str,
        content_type: str = "text/markdown",
        max_tokens: int = None,
    ) -> str:
        """Summarize content based on its type.

        Args:
            content: The content to summarize
            content_type: MIME type of the content
            max_tokens: Maximum tokens for summary (default: SUMMARY_TARGET)

        Returns:
            Summarized content
        """
        if not self.should_summarize(content):
            return content

        # Check cache
        cache_key = self.get_cache_key(content)
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        max_tokens = max_tokens or self.SUMMARY_TARGET

        # Choose summarization strategy based on content type
        if "json" in content_type:
            summary = self._summarize_json(content, max_tokens)
        elif "code" in content_type or self._looks_like_code(content):
            summary = self._summarize_code(content, max_tokens)
        elif "markdown" in content_type:
            summary = self._summarize_markdown(content, max_tokens)
        else:
            summary = self._summarize_text(content, max_tokens)

        # Cache the result
        self._summary_cache[cache_key] = summary

        logger.debug(
            "content_summarized",
            original_tokens=estimate_tokens(content),
            summary_tokens=estimate_tokens(summary),
            content_type=content_type,
        )

        return summary

    def _looks_like_code(self, content: str) -> bool:
        """Heuristically determine if content is code."""
        code_patterns = [
            r"def\s+\w+\s*\(",  # Python function
            r"function\s+\w+\s*\(",  # JavaScript function
            r"class\s+\w+",  # Class definition
            r"import\s+\w+",  # Import statement
            r"from\s+\w+\s+import",  # Python import
            r"const\s+\w+\s*=",  # JavaScript const
            r"let\s+\w+\s*=",  # JavaScript let
        ]
        return any(re.search(pattern, content) for pattern in code_patterns)

    def _summarize_markdown(self, content: str, max_tokens: int) -> str:
        """Summarize markdown content by extracting structure."""
        lines = content.split("\n")
        summary_parts = []

        # Extract headers and their first paragraph
        current_header = None
        header_content = []

        for line in lines:
            if line.startswith("#"):
                if current_header and header_content:
                    # Add previous header with brief content
                    brief = " ".join(header_content)[:200]
                    summary_parts.append(f"{current_header}\n{brief}...")
                current_header = line
                header_content = []
            elif current_header and line.strip():
                header_content.append(line.strip())

        # Add last header
        if current_header and header_content:
            brief = " ".join(header_content)[:200]
            summary_parts.append(f"{current_header}\n{brief}...")

        # If no headers, use first and last portions
        if not summary_parts:
            return self._summarize_text(content, max_tokens)

        summary = "\n\n".join(summary_parts)

        # Truncate if still too long
        target_chars = max_tokens * 4
        if len(summary) > target_chars:
            summary = summary[:target_chars] + "\n\n[... content truncated ...]"

        return summary

    def _summarize_code(self, content: str, max_tokens: int) -> str:
        """Summarize code by extracting signatures and structure."""
        lines = content.split("\n")
        summary_parts = []

        # Extract imports, class definitions, and function signatures
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Import statements
            if stripped.startswith(("import ", "from ", "const ", "let ", "var ")):
                summary_parts.append(stripped)

            # Class definitions
            elif stripped.startswith("class "):
                summary_parts.append(f"\n{stripped}")
                # Include docstring if present
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    summary_parts.append(lines[i + 1].strip())

            # Function definitions
            elif re.match(r"^\s*(def|function|async def|async function)\s+", line):
                summary_parts.append(f"\n{stripped}")
                # Include docstring if present
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    summary_parts.append(lines[i + 1].strip())

            # Decorator
            elif stripped.startswith("@"):
                summary_parts.append(stripped)

        summary = "\n".join(summary_parts)

        # Truncate if still too long
        target_chars = max_tokens * 4
        if len(summary) > target_chars:
            summary = summary[:target_chars] + "\n# ... code truncated ..."

        if not summary.strip():
            # Fall back to text summarization
            return self._summarize_text(content, max_tokens)

        return f"[Code Summary]\n{summary}"

    def _summarize_json(self, content: str, max_tokens: int) -> str:
        """Summarize JSON by extracting structure."""
        import json

        try:
            data = json.loads(content)
            return self._summarize_json_structure(data, max_tokens)
        except json.JSONDecodeError:
            return self._summarize_text(content, max_tokens)

    def _summarize_json_structure(
        self, data: Any, max_tokens: int, depth: int = 0
    ) -> str:
        """Recursively summarize JSON structure."""
        indent = "  " * depth

        if isinstance(data, dict):
            if not data:
                return "{}"
            parts = ["{"]
            for key in list(data.keys())[:10]:  # Limit keys
                value = data[key]
                if isinstance(value, (dict, list)):
                    value_str = self._summarize_json_structure(value, max_tokens // 2, depth + 1)
                else:
                    value_str = repr(value)[:50]
                parts.append(f'{indent}  "{key}": {value_str},')
            if len(data) > 10:
                parts.append(f"{indent}  ... ({len(data) - 10} more keys)")
            parts.append(f"{indent}}}")
            return "\n".join(parts)

        elif isinstance(data, list):
            if not data:
                return "[]"
            if len(data) > 3:
                items = [self._summarize_json_structure(data[0], max_tokens // 4, depth + 1)]
                return f"[{items[0]}, ... ({len(data)} items total)]"
            return f"[{len(data)} items]"

        else:
            return repr(data)[:50]

    def _summarize_text(self, content: str, max_tokens: int) -> str:
        """Summarize plain text by taking beginning and end."""
        target_chars = max_tokens * 4
        half_chars = target_chars // 2

        if len(content) <= target_chars:
            return content

        # Take beginning and end
        beginning = content[:half_chars]
        ending = content[-half_chars:]

        return f"{beginning}\n\n[... {len(content) - target_chars} characters omitted ...]\n\n{ending}"


# =============================================================================
# Context Window Manager
# =============================================================================

@dataclass
class ContextWindow:
    """Represents a managed context window for an agent."""

    max_tokens: int = 100000  # Claude's context window
    reserved_output_tokens: int = 4096  # Reserve for output
    system_prompt_tokens: int = 0
    current_tokens: int = 0

    # Sliding window of messages
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # Artifact references with summaries
    artifact_summaries: Dict[str, str] = field(default_factory=dict)

    @property
    def available_tokens(self) -> int:
        """Tokens available for context."""
        return (
            self.max_tokens
            - self.reserved_output_tokens
            - self.system_prompt_tokens
            - self.current_tokens
        )

    @property
    def usage_percentage(self) -> float:
        """Current usage as percentage."""
        used = self.system_prompt_tokens + self.current_tokens
        available = self.max_tokens - self.reserved_output_tokens
        return (used / available) * 100 if available > 0 else 100


class ContextWindowManager:
    """Manages context windows for agent conversations.

    Features:
    - Automatic summarization of large artifacts
    - Sliding window for conversation history
    - Token budget tracking
    - Semantic chunking for code
    """

    # Thresholds
    CONTEXT_WARNING_THRESHOLD = 0.7  # 70% usage
    CONTEXT_CRITICAL_THRESHOLD = 0.9  # 90% usage

    def __init__(self):
        self.summarizer = ContentSummarizer()
        self._windows: Dict[str, ContextWindow] = {}

    def create_window(
        self,
        session_id: str,
        system_prompt: str,
        max_tokens: int = 100000,
    ) -> ContextWindow:
        """Create a new context window for a session.

        Args:
            session_id: Unique identifier for this context window
            system_prompt: The system prompt to use
            max_tokens: Maximum context tokens (default: 100k)

        Returns:
            The created ContextWindow
        """
        window = ContextWindow(
            max_tokens=max_tokens,
            system_prompt_tokens=estimate_tokens(system_prompt),
        )
        self._windows[session_id] = window
        return window

    def get_window(self, session_id: str) -> Optional[ContextWindow]:
        """Get an existing context window."""
        return self._windows.get(session_id)

    def add_message(
        self,
        session_id: str,
        message: Dict[str, Any],
    ) -> bool:
        """Add a message to the context window.

        Applies sliding window if necessary to stay within limits.

        Args:
            session_id: The context window session ID
            message: The message to add

        Returns:
            True if message was added, False if window is full
        """
        window = self._windows.get(session_id)
        if not window:
            return False

        msg_tokens = estimate_tokens(str(message.get("content", "")))

        # Check if we need to apply sliding window
        if window.usage_percentage > self.CONTEXT_CRITICAL_THRESHOLD:
            self._apply_sliding_window(window)

        # Check if message fits
        if msg_tokens > window.available_tokens:
            logger.warning(
                "context_window_full",
                session_id=session_id,
                msg_tokens=msg_tokens,
                available_tokens=window.available_tokens,
            )
            return False

        window.messages.append(message)
        window.current_tokens += msg_tokens

        return True

    def _apply_sliding_window(self, window: ContextWindow) -> None:
        """Apply sliding window to reduce context size.

        Strategy:
        1. Keep system prompt (never removed)
        2. Keep the most recent messages
        3. Summarize and remove older messages
        """
        if len(window.messages) <= 2:
            return

        # Calculate how much we need to free
        target_usage = self.CONTEXT_WARNING_THRESHOLD
        target_tokens = int((window.max_tokens - window.reserved_output_tokens) * target_usage)
        tokens_to_free = window.current_tokens - target_tokens

        if tokens_to_free <= 0:
            return

        freed = 0
        messages_to_remove = 0

        # Remove oldest messages first (skip most recent 2)
        for i, msg in enumerate(window.messages[:-2]):
            msg_tokens = estimate_tokens(str(msg.get("content", "")))
            freed += msg_tokens
            messages_to_remove = i + 1
            if freed >= tokens_to_free:
                break

        if messages_to_remove > 0:
            # Create a summary of removed messages
            removed = window.messages[:messages_to_remove]
            summary = self._summarize_conversation(removed)

            # Replace removed messages with summary
            window.messages = [
                {"role": "user", "content": f"[Previous conversation summary]\n{summary}"}
            ] + window.messages[messages_to_remove:]

            window.current_tokens -= freed
            window.current_tokens += estimate_tokens(summary)

            logger.info(
                "sliding_window_applied",
                messages_removed=messages_to_remove,
                tokens_freed=freed,
                new_usage=window.usage_percentage,
            )

    def _summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize a list of conversation messages."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                # Truncate long content
                brief = content[:200] + "..." if len(content) > 200 else content
                parts.append(f"{role}: {brief}")

        return "\n".join(parts)

    def prepare_artifact_for_context(
        self,
        session_id: str,
        artifact_id: str,
        content: str,
        content_type: str,
    ) -> str:
        """Prepare an artifact for inclusion in context.

        Summarizes if necessary and caches the result.

        Args:
            session_id: The context window session ID
            artifact_id: Unique artifact identifier
            content: The artifact content
            content_type: MIME type of the content

        Returns:
            The prepared content (possibly summarized)
        """
        window = self._windows.get(session_id)
        if not window:
            return self.summarizer.summarize(content, content_type)

        # Check if we have a cached summary
        if artifact_id in window.artifact_summaries:
            return window.artifact_summaries[artifact_id]

        # Determine if summarization is needed based on available space
        content_tokens = estimate_tokens(content)

        if content_tokens > window.available_tokens * 0.3:
            # Summarize if content would use more than 30% of available space
            max_tokens = int(window.available_tokens * 0.15)
            prepared = self.summarizer.summarize(content, content_type, max_tokens)
        else:
            prepared = content

        # Cache the result
        window.artifact_summaries[artifact_id] = prepared

        return prepared

    def prepare_dependencies_output(
        self,
        session_id: str,
        dependencies_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare dependencies output for context inclusion.

        Summarizes large artifacts while preserving structure.

        Args:
            session_id: The context window session ID
            dependencies_output: The raw dependencies output

        Returns:
            Prepared dependencies output with summarized content
        """
        prepared = {}

        for dep_name, dep_content in dependencies_output.items():
            if isinstance(dep_content, str):
                # Single artifact content
                prepared[dep_name] = self.prepare_artifact_for_context(
                    session_id,
                    f"{dep_name}_primary",
                    dep_content,
                    "text/markdown",
                )
            elif isinstance(dep_content, dict):
                # Multiple artifacts
                prepared_dep = {"primary": None, "artifacts": []}

                if "primary" in dep_content:
                    prepared_dep["primary"] = self.prepare_artifact_for_context(
                        session_id,
                        f"{dep_name}_primary",
                        dep_content["primary"],
                        "text/markdown",
                    )

                if "artifacts" in dep_content:
                    for i, art in enumerate(dep_content["artifacts"]):
                        if isinstance(art, dict) and "content" in art:
                            prepared_art = {
                                "name": art.get("name", f"artifact_{i}"),
                                "type": art.get("type", "unknown"),
                                "content": self.prepare_artifact_for_context(
                                    session_id,
                                    f"{dep_name}_artifact_{i}",
                                    art["content"],
                                    art.get("content_type", "text/markdown"),
                                ),
                            }
                            prepared_dep["artifacts"].append(prepared_art)

                prepared[dep_name] = prepared_dep
            else:
                prepared[dep_name] = dep_content

        return prepared

    def get_context_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current context window status."""
        window = self._windows.get(session_id)
        if not window:
            return {"error": "Window not found"}

        return {
            "max_tokens": window.max_tokens,
            "current_tokens": window.current_tokens,
            "system_prompt_tokens": window.system_prompt_tokens,
            "available_tokens": window.available_tokens,
            "usage_percentage": round(window.usage_percentage, 1),
            "message_count": len(window.messages),
            "artifact_count": len(window.artifact_summaries),
            "status": self._get_status_level(window.usage_percentage),
        }

    def _get_status_level(self, usage: float) -> str:
        """Get status level based on usage percentage."""
        if usage >= self.CONTEXT_CRITICAL_THRESHOLD * 100:
            return "critical"
        elif usage >= self.CONTEXT_WARNING_THRESHOLD * 100:
            return "warning"
        return "healthy"

    def cleanup_session(self, session_id: str) -> None:
        """Clean up a context window session."""
        if session_id in self._windows:
            del self._windows[session_id]


# =============================================================================
# Semantic Code Chunker
# =============================================================================

class SemanticCodeChunker:
    """Chunks code into semantic units for better context management."""

    def chunk_code(
        self,
        code: str,
        language: str = "python",
        max_chunk_tokens: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Chunk code into semantic units.

        Args:
            code: The source code to chunk
            language: Programming language (python, javascript, etc.)
            max_chunk_tokens: Maximum tokens per chunk

        Returns:
            List of chunks with metadata
        """
        if language == "python":
            return self._chunk_python(code, max_chunk_tokens)
        elif language in ("javascript", "typescript"):
            return self._chunk_javascript(code, max_chunk_tokens)
        else:
            return self._chunk_generic(code, max_chunk_tokens)

    def _chunk_python(
        self, code: str, max_chunk_tokens: int
    ) -> List[Dict[str, Any]]:
        """Chunk Python code by classes and functions."""
        chunks = []
        lines = code.split("\n")

        current_chunk = []
        current_type = "module"
        current_name = "module"
        indent_level = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            # Detect class or function start
            if stripped.startswith("class "):
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, current_type, current_name
                    ))
                current_chunk = [line]
                current_type = "class"
                match = re.match(r"class\s+(\w+)", stripped)
                current_name = match.group(1) if match else "unknown"
                indent_level = len(line) - len(stripped)

            elif stripped.startswith("def ") or stripped.startswith("async def "):
                if current_chunk and indent_level == 0:
                    chunks.append(self._create_chunk(
                        current_chunk, current_type, current_name
                    ))
                    current_chunk = [line]
                    current_type = "function"
                    match = re.match(r"(?:async\s+)?def\s+(\w+)", stripped)
                    current_name = match.group(1) if match else "unknown"
                    indent_level = len(line) - len(stripped)
                else:
                    current_chunk.append(line)

            else:
                current_chunk.append(line)

            # Check chunk size
            if estimate_tokens("\n".join(current_chunk)) > max_chunk_tokens:
                # Split current chunk
                chunks.append(self._create_chunk(
                    current_chunk, current_type, current_name
                ))
                current_chunk = []
                current_type = "continuation"
                current_name = f"{current_name}_cont"

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, current_type, current_name
            ))

        return chunks

    def _chunk_javascript(
        self, code: str, max_chunk_tokens: int
    ) -> List[Dict[str, Any]]:
        """Chunk JavaScript/TypeScript code."""
        # Similar logic to Python but with JS patterns
        chunks = []
        lines = code.split("\n")

        current_chunk = []
        current_type = "module"
        current_name = "module"

        for line in lines:
            stripped = line.strip()

            # Detect function or class
            if re.match(r"(export\s+)?(class|function|const\s+\w+\s*=\s*(?:async\s*)?\()", stripped):
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, current_type, current_name
                    ))
                current_chunk = [line]
                current_type = "function" if "function" in stripped else "class"
                match = re.search(r"(class|function)\s+(\w+)|const\s+(\w+)", stripped)
                if match:
                    current_name = match.group(2) or match.group(3) or "unknown"
            else:
                current_chunk.append(line)

            if estimate_tokens("\n".join(current_chunk)) > max_chunk_tokens:
                chunks.append(self._create_chunk(
                    current_chunk, current_type, current_name
                ))
                current_chunk = []

        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, current_type, current_name
            ))

        return chunks

    def _chunk_generic(
        self, code: str, max_chunk_tokens: int
    ) -> List[Dict[str, Any]]:
        """Generic chunking by lines."""
        chunks = []
        lines = code.split("\n")
        current_chunk = []

        for line in lines:
            current_chunk.append(line)
            if estimate_tokens("\n".join(current_chunk)) > max_chunk_tokens:
                chunks.append(self._create_chunk(
                    current_chunk, "block", f"block_{len(chunks)}"
                ))
                current_chunk = []

        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, "block", f"block_{len(chunks)}"
            ))

        return chunks

    def _create_chunk(
        self,
        lines: List[str],
        chunk_type: str,
        name: str,
    ) -> Dict[str, Any]:
        """Create a chunk dictionary."""
        content = "\n".join(lines)
        return {
            "type": chunk_type,
            "name": name,
            "content": content,
            "tokens": estimate_tokens(content),
            "lines": len(lines),
        }


# Global instances
_context_manager: Optional[ContextWindowManager] = None
_code_chunker: Optional[SemanticCodeChunker] = None


def get_context_manager() -> ContextWindowManager:
    """Get the global context window manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextWindowManager()
    return _context_manager


def get_code_chunker() -> SemanticCodeChunker:
    """Get the global code chunker."""
    global _code_chunker
    if _code_chunker is None:
        _code_chunker = SemanticCodeChunker()
    return _code_chunker
