"""Production health monitoring for Social-Media-Radar.

Implements HealthMonitor which extends the lightweight checks in
app/core/health.py with intelligence-layer checks:

- LLM provider circuit-breaker state
- HNSW vector index initialisation
- Database connectivity
- p99 latency assertion against Prometheus Histogram metrics

Wired into GET /api/v1/health via app/api/main.py.
"""

import logging
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PerformanceViolationError(Exception):
    """Raised when recorded p99 latency exceeds the configured threshold.

    Args:
        component: Name of the component that violated the threshold.
        observed_ms: Observed p99 latency in milliseconds.
        threshold_ms: Configured threshold in milliseconds.
    """

    def __init__(self, component: str, observed_ms: float, threshold_ms: float) -> None:
        self.component = component
        self.observed_ms = observed_ms
        self.threshold_ms = threshold_ms
        super().__init__(
            f"p99 latency violation for '{component}': "
            f"{observed_ms:.1f}ms > threshold {threshold_ms:.1f}ms"
        )


class ComponentStatus(BaseModel):
    """Health status of one system component."""

    name: str
    healthy: bool
    detail: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthReport(BaseModel):
    """Aggregated health report returned by HealthMonitor.check()."""

    healthy: bool = Field(..., description="True only when ALL critical components are healthy.")
    components: List[ComponentStatus] = Field(default_factory=list)
    checked_at: float = Field(default_factory=time.time)


class HealthMonitor:
    """Checks infrastructure and intelligence-layer component health.

    Args:
        llm_router: Optional pre-built LLMRouter; created lazily when None.
        hnsw_index: Optional pre-built HNSWIndex; skips check when None.
    """

    def __init__(self, llm_router=None, hnsw_index=None) -> None:
        self._llm_router = llm_router
        self._hnsw_index = hnsw_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(self) -> HealthReport:
        """Run all health checks and return an aggregated HealthReport.

        Returns:
            :class:`HealthReport` where ``healthy`` is ``True`` only when
            every critical component passes its check.

        Raises:
            Never raises; all exceptions are caught and surfaced as unhealthy
            component entries in the report.
        """
        statuses: List[ComponentStatus] = []

        statuses.append(await self._check_db())
        statuses.append(self._check_circuit_breakers())
        statuses.append(self._check_hnsw())

        overall_healthy = all(s.healthy for s in statuses)
        return HealthReport(healthy=overall_healthy, components=statuses)

    async def assert_p99_latency(
        self,
        component: str,
        threshold_ms: float,
    ) -> None:
        """Assert that the recorded p99 latency for ``component`` is within threshold.

        Reads from the Prometheus ``llm_request_duration_seconds`` Histogram
        (defined in app/llm/monitoring.py).  The p99 is estimated from the
        Histogram's ``_sum / _count`` ratio (mean) with a 3× headroom factor as
        an approximation — replace with a proper p99 quantile from Prometheus
        remote-read in production.

        Args:
            component: Provider label to filter by (e.g. ``"openai"``).
            threshold_ms: Maximum acceptable p99 latency in milliseconds.

        Raises:
            PerformanceViolationError: If estimated p99 exceeds ``threshold_ms``.
        """
        try:
            from app.llm.monitoring import llm_request_duration_seconds

            # Collect samples across all label combinations for this component
            total_sum = 0.0
            total_count = 0.0
            for metric in llm_request_duration_seconds.collect():
                for sample in metric.samples:
                    if sample.name.endswith("_sum") and sample.labels.get("provider") == component:
                        total_sum += sample.value
                    if sample.name.endswith("_count") and sample.labels.get("provider") == component:
                        total_count += sample.value

            if total_count == 0:
                logger.debug("No latency samples for component '%s' — skipping p99 check", component)
                return

            mean_s = total_sum / total_count
            # Conservative p99 ≈ 3× mean for long-tail distributions (replace with real quantile)
            estimated_p99_ms = mean_s * 3 * 1000.0

            if estimated_p99_ms > threshold_ms:
                raise PerformanceViolationError(component, estimated_p99_ms, threshold_ms)

            logger.debug(
                "p99 latency for '%s': %.1fms (threshold=%.1fms) — OK",
                component, estimated_p99_ms, threshold_ms,
            )
        except PerformanceViolationError:
            raise
        except Exception as exc:
            logger.warning("Could not read latency metrics for '%s': %s", component, exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _check_db(self) -> ComponentStatus:
        t0 = time.monotonic()
        try:
            from app.core.db import async_engine
            from sqlalchemy import text
            async with async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return ComponentStatus(
                name="database",
                healthy=True,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return ComponentStatus(name="database", healthy=False, detail=str(exc))

    def _check_circuit_breakers(self) -> ComponentStatus:
        try:
            router = self._llm_router
            if router is None:
                from app.llm.router import get_router
                router = get_router()

            open_breakers: List[str] = []
            for provider_name, client in getattr(router, "_clients", {}).items():
                cb = getattr(client, "circuit_breaker", None)
                if cb is not None and getattr(cb, "is_open", False):
                    open_breakers.append(provider_name)

            if open_breakers:
                return ComponentStatus(
                    name="llm_circuit_breakers",
                    healthy=False,
                    detail=f"Open breakers: {', '.join(open_breakers)}",
                )
            return ComponentStatus(name="llm_circuit_breakers", healthy=True)
        except Exception as exc:
            return ComponentStatus(name="llm_circuit_breakers", healthy=False, detail=str(exc))

    def _check_hnsw(self) -> ComponentStatus:
        try:
            index = self._hnsw_index
            if index is None:
                # No index injected — treat as degraded but not critical
                return ComponentStatus(
                    name="hnsw_index",
                    healthy=True,
                    detail="No index injected (candidate retriever uses platform priors only)",
                )
            element_count = getattr(index, "element_count", None)
            if element_count is not None and element_count == 0:
                return ComponentStatus(
                    name="hnsw_index",
                    healthy=False,
                    detail="HNSW index is empty — no exemplars loaded",
                )
            return ComponentStatus(
                name="hnsw_index",
                healthy=True,
                detail=f"element_count={element_count}",
            )
        except Exception as exc:
            return ComponentStatus(name="hnsw_index", healthy=False, detail=str(exc))

