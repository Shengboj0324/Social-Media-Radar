"""Human-in-the-loop feedback collection for response draft approval/rejection.

Implements the FeedbackCollector class which records human approval or rejection
of LLM-generated response drafts and exposes approval rate metrics for use by
the calibration system (app/intelligence/calibration.py).

All records are kept in-memory by default.  A SQLAlchemy session may be injected
for durable persistence once a ``FeedbackRecordDB`` migration is applied.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from uuid import UUID

from app.domain.inference_models import SignalType

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Immutable record of a single human feedback event.

    Attributes:
        signal_id: UUID of the parent :class:`~app.domain.inference_models.SignalInference`.
        draft_id: Identifier of the specific response draft (e.g. ``"v1_professional"``).
        approved: ``True`` if the human approved the draft; ``False`` if rejected.
        correction: Optional free-text correction supplied by the reviewer.
        signal_type: The :class:`~app.domain.inference_models.SignalType` of the inference.
        recorded_at: UTC timestamp when the feedback was recorded.
    """

    signal_id: UUID
    draft_id: str
    approved: bool
    signal_type: SignalType
    correction: Optional[str] = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackCollector:
    """Collects and aggregates human feedback on generated response drafts.

    Thread-safe.  All mutations are protected by a :class:`threading.Lock`.

    Usage::

        collector = FeedbackCollector()
        collector.record_feedback(
            signal_id=uuid,
            draft_id="v1_professional",
            approved=True,
            signal_type=SignalType.CHURN_RISK,
        )
        rate = collector.get_approval_rate(SignalType.CHURN_RISK, window_days=30)

    Args:
        session: Optional SQLAlchemy async session for durable persistence.
            When ``None`` (default) records are kept in memory only.
    """

    def __init__(self, session=None) -> None:
        self._session = session
        self._records: List[FeedbackRecord] = []
        self._lock = threading.Lock()
        logger.info("FeedbackCollector initialised (backend=%s)", "db" if session else "memory")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        signal_id: UUID,
        draft_id: str,
        approved: bool,
        signal_type: SignalType,
        correction: Optional[str] = None,
    ) -> FeedbackRecord:
        """Record a human approval or rejection of a response draft.

        Args:
            signal_id: UUID of the parent SignalInference.
            draft_id: Opaque string identifying which draft variant was reviewed.
            approved: ``True`` = approved, ``False`` = rejected.
            signal_type: Signal type of the associated inference.
            correction: Optional corrected text supplied by the reviewer.

        Returns:
            The newly created :class:`FeedbackRecord`.
        """
        record = FeedbackRecord(
            signal_id=signal_id,
            draft_id=draft_id,
            approved=approved,
            signal_type=signal_type,
            correction=correction,
        )
        with self._lock:
            self._records.append(record)

        logger.debug(
            "Feedback recorded: signal=%s draft=%s approved=%s",
            signal_id, draft_id, approved,
        )
        return record

    def get_approval_rate(
        self,
        signal_type: SignalType,
        window_days: int = 30,
    ) -> float:
        """Compute the approval rate for a given signal type over a time window.

        Args:
            signal_type: The signal type to filter by.
            window_days: Number of past days to include (default 30).

        Returns:
            Approval rate in [0.0, 1.0].  Returns ``0.0`` when no records exist
            for the given filters.

        Raises:
            ValueError: If ``window_days`` is less than 1.
        """
        if window_days < 1:
            raise ValueError(f"window_days must be >= 1, got {window_days}")

        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

        with self._lock:
            relevant = [
                r for r in self._records
                if r.signal_type == signal_type and r.recorded_at >= cutoff
            ]

        if not relevant:
            logger.debug(
                "No feedback records for signal_type=%s in past %d days",
                signal_type.value, window_days,
            )
            return 0.0

        approval_rate = sum(1 for r in relevant if r.approved) / len(relevant)
        logger.debug(
            "Approval rate for %s (window=%dd): %.3f (%d records)",
            signal_type.value, window_days, approval_rate, len(relevant),
        )
        return approval_rate

    def get_all_records(self) -> List[FeedbackRecord]:
        """Return a snapshot of all stored feedback records (defensive copy).

        Returns:
            List of :class:`FeedbackRecord` instances.
        """
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        """Remove all stored records.  Primarily used in tests."""
        with self._lock:
            self._records.clear()
        logger.debug("FeedbackCollector cleared")

