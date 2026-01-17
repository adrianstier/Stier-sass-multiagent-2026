"""Expanded Observability Dashboard Metrics.

Provides:
- Agent execution time histograms (p50, p95, p99)
- Token efficiency metrics
- Gate pass/fail rates
- Workflow bottleneck identification
- OpenTelemetry tracing support
"""

import time
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from orchestrator.core.database import get_db
from orchestrator.core.models import Run, Task, Event, Artifact, TaskStatus, GateStatus, RunStatus
from orchestrator.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Metrics Data Classes
# =============================================================================

@dataclass
class HistogramMetric:
    """Histogram metric with percentiles."""

    name: str
    unit: str
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    values: List[float] = field(default_factory=list)

    def record(self, value: float) -> None:
        """Record a value."""
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.values.append(value)

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0

    def percentile(self, p: float) -> float:
        """Get percentile value (0-100)."""
        if not self.values:
            return 0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * p / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "count": self.count,
            "sum": round(self.sum, 2),
            "mean": round(self.mean, 2),
            "min": round(self.min, 2) if self.min != float('inf') else 0,
            "max": round(self.max, 2) if self.max != float('-inf') else 0,
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
        }


@dataclass
class GaugeMetric:
    """Gauge metric (current value)."""

    name: str
    unit: str
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def set(self, value: float) -> None:
        self.value = value
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CounterMetric:
    """Counter metric (monotonically increasing)."""

    name: str
    unit: str
    value: int = 0
    labels: Dict[str, str] = field(default_factory=dict)

    def increment(self, amount: int = 1) -> None:
        self.value += amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
            "labels": self.labels,
        }


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self):
        # Execution time histograms by role
        self._execution_times: Dict[str, HistogramMetric] = defaultdict(
            lambda: HistogramMetric(name="execution_time", unit="seconds")
        )

        # Token usage histograms by role
        self._token_usage: Dict[str, HistogramMetric] = defaultdict(
            lambda: HistogramMetric(name="token_usage", unit="tokens")
        )

        # Gate metrics
        self._gate_metrics: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"passed": 0, "failed": 0, "waived": 0}
        )

        # Counters
        self._counters: Dict[str, CounterMetric] = {}

        # LLM request tracking
        self._llm_latencies: HistogramMetric = HistogramMetric(
            name="llm_latency", unit="milliseconds"
        )

    def record_task_execution(
        self,
        role: str,
        duration_seconds: float,
        tokens_used: int,
        success: bool,
    ) -> None:
        """Record task execution metrics."""
        self._execution_times[role].record(duration_seconds)
        self._token_usage[role].record(tokens_used)

        # Update counters
        status = "success" if success else "failure"
        counter_key = f"tasks_{status}_{role}"
        if counter_key not in self._counters:
            self._counters[counter_key] = CounterMetric(
                name="tasks_total",
                unit="count",
                labels={"role": role, "status": status},
            )
        self._counters[counter_key].increment()

    def record_gate_result(
        self,
        gate_type: str,
        result: str,  # passed, failed, waived
    ) -> None:
        """Record gate result."""
        if result in self._gate_metrics[gate_type]:
            self._gate_metrics[gate_type][result] += 1

    def record_llm_latency(self, latency_ms: float) -> None:
        """Record LLM request latency."""
        self._llm_latencies.record(latency_ms)

    def get_execution_time_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get execution time metrics by role."""
        return {role: hist.to_dict() for role, hist in self._execution_times.items()}

    def get_token_efficiency_metrics(self) -> Dict[str, Any]:
        """Get token efficiency metrics."""
        efficiency = {}
        for role, hist in self._token_usage.items():
            exec_time = self._execution_times.get(role)
            if exec_time and exec_time.count > 0:
                efficiency[role] = {
                    "avg_tokens_per_task": round(hist.mean, 0),
                    "tokens_per_second": round(
                        hist.sum / exec_time.sum, 2
                    ) if exec_time.sum > 0 else 0,
                    "total_tokens": int(hist.sum),
                    "task_count": hist.count,
                }
        return efficiency

    def get_gate_metrics(self) -> Dict[str, Any]:
        """Get gate pass/fail metrics."""
        result = {}
        for gate_type, counts in self._gate_metrics.items():
            total = sum(counts.values())
            result[gate_type] = {
                **counts,
                "total": total,
                "pass_rate": round(
                    (counts["passed"] + counts["waived"]) / total * 100, 1
                ) if total > 0 else 0,
            }
        return result

    def get_llm_metrics(self) -> Dict[str, Any]:
        """Get LLM latency metrics."""
        return {
            "latency": self._llm_latencies.to_dict(),
            "requests_total": self._llm_latencies.count,
        }


# =============================================================================
# Bottleneck Analyzer
# =============================================================================

class BottleneckAnalyzer:
    """Analyzes workflow bottlenecks."""

    def analyze_run(self, run_id: str) -> Dict[str, Any]:
        """Analyze a run for bottlenecks.

        Args:
            run_id: The run ID

        Returns:
            Dict with bottleneck analysis
        """
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(run_id)).first()
            if not run:
                return {"error": "Run not found"}

            tasks = db.query(Task).filter(Task.run_id == UUID(run_id)).all()

            # Calculate task durations
            task_durations = []
            for task in tasks:
                if task.started_at and task.completed_at:
                    duration = (task.completed_at - task.started_at).total_seconds()
                    task_durations.append({
                        "task_type": task.task_type,
                        "role": task.assigned_role,
                        "duration_seconds": duration,
                        "status": task.status.value,
                    })

            # Find bottlenecks
            bottlenecks = self._identify_bottlenecks(task_durations)

            # Calculate wait times (time between task completion and next task start)
            wait_times = self._calculate_wait_times(tasks)

            # Get critical path
            critical_path = self._find_critical_path(tasks)

            return {
                "run_id": run_id,
                "total_duration_seconds": self._calculate_total_duration(run),
                "task_durations": sorted(
                    task_durations, key=lambda x: x["duration_seconds"], reverse=True
                )[:10],
                "bottlenecks": bottlenecks,
                "wait_times": wait_times,
                "critical_path": critical_path,
                "recommendations": self._generate_recommendations(bottlenecks, wait_times),
            }

    def _identify_bottlenecks(
        self, task_durations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify bottleneck tasks."""
        if not task_durations:
            return []

        durations = [t["duration_seconds"] for t in task_durations]
        if not durations:
            return []

        mean_duration = statistics.mean(durations)
        threshold = mean_duration * 2  # Tasks taking 2x average are bottlenecks

        bottlenecks = []
        for task in task_durations:
            if task["duration_seconds"] > threshold:
                bottlenecks.append({
                    "task_type": task["task_type"],
                    "role": task["role"],
                    "duration_seconds": task["duration_seconds"],
                    "over_average_by": round(
                        task["duration_seconds"] / mean_duration, 1
                    ),
                    "severity": "high" if task["duration_seconds"] > threshold * 2 else "medium",
                })

        return sorted(bottlenecks, key=lambda x: x["duration_seconds"], reverse=True)

    def _calculate_wait_times(self, tasks: List[Task]) -> Dict[str, Any]:
        """Calculate wait times between tasks."""
        completed_tasks = [
            t for t in tasks
            if t.status == TaskStatus.COMPLETED and t.completed_at
        ]
        completed_tasks.sort(key=lambda t: t.completed_at)

        wait_times = []
        for i in range(1, len(completed_tasks)):
            prev = completed_tasks[i - 1]
            curr = completed_tasks[i]

            if curr.started_at and prev.completed_at:
                wait = (curr.started_at - prev.completed_at).total_seconds()
                if wait > 0:
                    wait_times.append({
                        "from_task": prev.task_type,
                        "to_task": curr.task_type,
                        "wait_seconds": wait,
                    })

        total_wait = sum(w["wait_seconds"] for w in wait_times)

        return {
            "total_wait_seconds": round(total_wait, 1),
            "longest_waits": sorted(
                wait_times, key=lambda x: x["wait_seconds"], reverse=True
            )[:5],
            "average_wait_seconds": round(
                total_wait / len(wait_times), 1
            ) if wait_times else 0,
        }

    def _find_critical_path(self, tasks: List[Task]) -> List[str]:
        """Find the critical path through the workflow."""
        # Build dependency graph
        task_by_id = {str(t.id): t for t in tasks}
        task_by_type = {t.task_type: t for t in tasks}

        # Find tasks with no dependents (end tasks)
        all_deps = set()
        for t in tasks:
            for dep in (t.dependencies or []):
                all_deps.add(dep)

        end_tasks = [t for t in tasks if str(t.id) not in all_deps]

        # Find longest path using DFS
        def get_path_duration(task_id: str, visited: set) -> Tuple[float, List[str]]:
            if task_id in visited:
                return 0, []

            visited.add(task_id)
            task = task_by_id.get(task_id)
            if not task:
                return 0, []

            # Task's own duration
            duration = 0
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()

            # Find longest dependency path
            max_dep_duration = 0
            max_dep_path = []

            for dep_id in (task.dependencies or []):
                dep_duration, dep_path = get_path_duration(dep_id, visited.copy())
                if dep_duration > max_dep_duration:
                    max_dep_duration = dep_duration
                    max_dep_path = dep_path

            return duration + max_dep_duration, max_dep_path + [task.task_type]

        # Find critical path from each end task
        longest_duration = 0
        critical_path = []

        for end_task in end_tasks:
            duration, path = get_path_duration(str(end_task.id), set())
            if duration > longest_duration:
                longest_duration = duration
                critical_path = path

        return critical_path

    def _calculate_total_duration(self, run: Run) -> float:
        """Calculate total run duration."""
        if run.started_at and run.completed_at:
            return (run.completed_at - run.started_at).total_seconds()
        elif run.started_at:
            return (datetime.utcnow() - run.started_at).total_seconds()
        return 0

    def _generate_recommendations(
        self,
        bottlenecks: List[Dict[str, Any]],
        wait_times: Dict[str, Any],
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Bottleneck recommendations
        for bn in bottlenecks[:3]:
            if bn["severity"] == "high":
                recommendations.append(
                    f"🔴 High-priority: {bn['task_type']} takes {bn['over_average_by']}x average. "
                    f"Consider optimizing {bn['role']} agent prompts or splitting the task."
                )
            else:
                recommendations.append(
                    f"🟡 Medium-priority: {bn['task_type']} is slower than average. "
                    f"Review {bn['role']} agent efficiency."
                )

        # Wait time recommendations
        if wait_times["total_wait_seconds"] > 60:
            recommendations.append(
                f"⏳ Total wait time is {wait_times['total_wait_seconds']:.0f}s. "
                f"Consider optimizing task scheduling or parallelization."
            )

        for wait in wait_times["longest_waits"][:2]:
            if wait["wait_seconds"] > 30:
                recommendations.append(
                    f"⏳ {wait['wait_seconds']:.0f}s wait between {wait['from_task']} and {wait['to_task']}. "
                    f"Check if these can run in parallel or if queuing is optimal."
                )

        if not recommendations:
            recommendations.append("✅ No significant bottlenecks detected.")

        return recommendations


# =============================================================================
# Dashboard Data Provider
# =============================================================================

class DashboardDataProvider:
    """Provides data for observability dashboard."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.bottleneck_analyzer = BottleneckAnalyzer()

    def get_organization_metrics(
        self,
        organization_id: str,
        time_range_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get metrics for an organization.

        Args:
            organization_id: The organization ID
            time_range_hours: Hours to look back

        Returns:
            Dict with dashboard metrics
        """
        cutoff = datetime.utcnow() - timedelta(hours=time_range_hours)

        with get_db() as db:
            # Get runs in time range
            runs = db.query(Run).filter(
                Run.organization_id == UUID(organization_id),
                Run.created_at >= cutoff,
            ).all()

            run_ids = [r.id for r in runs]

            # Aggregate metrics from events
            events = db.query(Event).filter(
                Event.run_id.in_(run_ids),
                Event.event_type.in_([
                    "llm_response", "task_completed", "task_failed",
                    "gate_passed", "gate_failed",
                ]),
            ).all()

            # Process events
            for event in events:
                data = event.data or {}

                if event.event_type == "llm_response":
                    # Extract latency if available
                    if "latency_ms" in data:
                        self.metrics_collector.record_llm_latency(data["latency_ms"])

                elif event.event_type in ["task_completed", "task_failed"]:
                    role = event.actor
                    tokens = data.get("tokens_used", 0)
                    duration = data.get("duration_seconds", 0)
                    success = event.event_type == "task_completed"

                    if duration > 0:
                        self.metrics_collector.record_task_execution(
                            role, duration, tokens, success
                        )

                elif event.event_type in ["gate_passed", "gate_failed"]:
                    gate_type = data.get("gate_type", "unknown")
                    result = "passed" if "passed" in event.event_type else "failed"
                    self.metrics_collector.record_gate_result(gate_type, result)

            # Build dashboard data
            return {
                "organization_id": organization_id,
                "time_range_hours": time_range_hours,
                "summary": self._get_summary(runs),
                "execution_times": self.metrics_collector.get_execution_time_metrics(),
                "token_efficiency": self.metrics_collector.get_token_efficiency_metrics(),
                "gate_metrics": self.metrics_collector.get_gate_metrics(),
                "llm_metrics": self.metrics_collector.get_llm_metrics(),
                "recent_runs": self._get_recent_runs_summary(runs[:10]),
            }

    def _get_summary(self, runs: List[Run]) -> Dict[str, Any]:
        """Get summary statistics."""
        if not runs:
            return {
                "total_runs": 0,
                "completed": 0,
                "failed": 0,
                "running": 0,
            }

        by_status = defaultdict(int)
        total_tokens = 0
        total_cost = 0.0

        for run in runs:
            by_status[run.status.value] += 1
            total_tokens += run.tokens_used or 0
            total_cost += float(run.total_cost_usd or "0")

        return {
            "total_runs": len(runs),
            "completed": by_status.get("completed", 0),
            "failed": by_status.get("failed", 0),
            "running": by_status.get("running", 0),
            "needs_input": by_status.get("needs_input", 0),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 2),
            "avg_tokens_per_run": int(total_tokens / len(runs)) if runs else 0,
        }

    def _get_recent_runs_summary(self, runs: List[Run]) -> List[Dict[str, Any]]:
        """Get summary of recent runs."""
        return [
            {
                "id": str(r.id),
                "goal": r.goal[:100] + "..." if len(r.goal) > 100 else r.goal,
                "status": r.status.value,
                "tokens_used": r.tokens_used,
                "cost_usd": r.total_cost_usd,
                "created_at": r.created_at.isoformat(),
            }
            for r in runs
        ]

    def analyze_run_performance(self, run_id: str) -> Dict[str, Any]:
        """Get detailed performance analysis for a run."""
        return self.bottleneck_analyzer.analyze_run(run_id)

    def get_prometheus_metrics(self, organization_id: str) -> str:
        """Get metrics in Prometheus format.

        Args:
            organization_id: The organization ID

        Returns:
            String in Prometheus exposition format
        """
        lines = []

        # Get fresh metrics
        metrics = self.get_organization_metrics(organization_id, time_range_hours=1)

        # Summary metrics
        summary = metrics.get("summary", {})
        lines.extend([
            "# HELP orchestrator_runs_total Total number of runs",
            "# TYPE orchestrator_runs_total counter",
            f'orchestrator_runs_total{{status="completed"}} {summary.get("completed", 0)}',
            f'orchestrator_runs_total{{status="failed"}} {summary.get("failed", 0)}',
            f'orchestrator_runs_total{{status="running"}} {summary.get("running", 0)}',
        ])

        # Token metrics
        lines.extend([
            "# HELP orchestrator_tokens_total Total tokens used",
            "# TYPE orchestrator_tokens_total counter",
            f"orchestrator_tokens_total {summary.get('total_tokens', 0)}",
        ])

        # Execution time metrics
        lines.extend([
            "# HELP orchestrator_task_duration_seconds Task execution duration",
            "# TYPE orchestrator_task_duration_seconds histogram",
        ])

        exec_times = metrics.get("execution_times", {})
        for role, data in exec_times.items():
            lines.append(
                f'orchestrator_task_duration_seconds_sum{{role="{role}"}} {data.get("sum", 0)}'
            )
            lines.append(
                f'orchestrator_task_duration_seconds_count{{role="{role}"}} {data.get("count", 0)}'
            )

        # Gate metrics
        lines.extend([
            "# HELP orchestrator_gate_results Gate check results",
            "# TYPE orchestrator_gate_results counter",
        ])

        gate_metrics = metrics.get("gate_metrics", {})
        for gate_type, data in gate_metrics.items():
            lines.append(
                f'orchestrator_gate_results{{gate="{gate_type}",result="passed"}} {data.get("passed", 0)}'
            )
            lines.append(
                f'orchestrator_gate_results{{gate="{gate_type}",result="failed"}} {data.get("failed", 0)}'
            )

        # LLM latency
        llm = metrics.get("llm_metrics", {}).get("latency", {})
        lines.extend([
            "# HELP orchestrator_llm_latency_ms LLM request latency",
            "# TYPE orchestrator_llm_latency_ms summary",
            f'orchestrator_llm_latency_ms{{quantile="0.5"}} {llm.get("p50", 0)}',
            f'orchestrator_llm_latency_ms{{quantile="0.95"}} {llm.get("p95", 0)}',
            f'orchestrator_llm_latency_ms{{quantile="0.99"}} {llm.get("p99", 0)}',
        ])

        return "\n".join(lines)


# Global instances
_dashboard_provider: Optional[DashboardDataProvider] = None
_metrics_collector: Optional[MetricsCollector] = None


def get_dashboard_provider() -> DashboardDataProvider:
    """Get the global dashboard data provider instance."""
    global _dashboard_provider
    if _dashboard_provider is None:
        _dashboard_provider = DashboardDataProvider()
    return _dashboard_provider


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
