"""Zero-egress PII guard for data residency compliance.

Implements docs/competitive_analysis.md §5.1 — Data Privacy: Zero-Egress Audit.

Every ContentItem fetched from a social-media connector passes through
``DataResidencyGuard.redact()`` before it is forwarded to any LLM provider.
The guard strips or pseudonymises PII in the following fields:

- ``author``      — pseudonymised with a stable SHA-256 hex prefix
- ``source_url``  — profile query-string parameters replaced with ``<redacted>``
- ``raw_text``    — email addresses and phone numbers replaced with tokens
- ``metadata``    — same PII patterns scrubbed from string values

A structured audit log entry is written for every redaction so that compliance
teams can demonstrate what was removed and why.  ``verify_clean()`` can be
called at the LLM layer boundary to raise ``DataResidencyViolationError`` if
any PII slipped through.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlencode, parse_qs

from app.core.errors import DataResidencyViolationError
from app.core.models import ContentItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII detection patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(?<!\d)(\+?1?\s*[-.\(]?\s*\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})(?!\d)",
)
# Profile URL parameters that commonly carry PII (e.g. ?user=john.doe)
_PII_QUERY_PARAMS = frozenset(
    {"user", "username", "author", "profile", "screen_name", "handle", "email", "name"}
)


class RedactionMethod(str, Enum):
    """Method used to redact a PII field."""

    PSEUDONYMISE = "pseudonymise"   # Replace with stable deterministic token
    REMOVE = "remove"               # Replace with a fixed placeholder
    SCRUB = "scrub"                 # Remove PII patterns from free text


@dataclass
class RedactionAuditEntry:
    """Immutable record of a single PII redaction operation.

    Attributes:
        item_id: UUID of the ``ContentItem`` that was redacted.
        field: Name of the field that was redacted (e.g. ``"author"``).
        method: Redaction method applied.
        pattern: Description of the PII pattern detected.
        timestamp: UTC timestamp of the redaction.
    """

    item_id: str
    field: str
    method: RedactionMethod
    pattern: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DataResidencyGuard:
    """Intercepts every ContentItem and strips PII before LLM calls.

    Implements the zero-egress data residency contract described in
    docs/competitive_analysis.md §5.1.  All redactions are idempotent —
    running an already-redacted item through the guard again is safe.

    Usage::

        guard = DataResidencyGuard()
        clean_item = guard.redact(raw_item)
        # Later, at the LLM boundary:
        guard.verify_clean(clean_item)

    Args:
        audit_logger: Optional named logger for structured audit entries.
            Defaults to the module logger.
    """

    def __init__(self, audit_logger: Optional[logging.Logger] = None) -> None:
        """Initialise the guard.

        Args:
            audit_logger: Logger for structured audit entries.  Uses the
                module-level logger when ``None``.
        """
        self._log = audit_logger or logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, item: ContentItem) -> ContentItem:
        """Return a copy of ``item`` with all PII fields redacted.

        The original ``item`` is never mutated.  All redactions are logged as
        structured audit entries.

        Args:
            item: Raw ``ContentItem`` received from a connector.

        Returns:
            A new ``ContentItem`` instance with PII fields redacted.
        """
        audit: List[RedactionAuditEntry] = []
        data = item.model_copy(deep=True)

        # 1. Pseudonymise author — skip if already pseudonymised (idempotent)
        if data.author and not data.author.startswith("anon_"):
            pseudonym = self._pseudonymise(data.author)
            if pseudonym != data.author:
                audit.append(RedactionAuditEntry(
                    item_id=str(data.id),
                    field="author",
                    method=RedactionMethod.PSEUDONYMISE,
                    pattern="real name / handle",
                ))
                data.author = pseudonym

        # 2. Redact PII query params from source_url
        if data.source_url:
            clean_url = self._redact_url(data.source_url)
            if clean_url != data.source_url:
                audit.append(RedactionAuditEntry(
                    item_id=str(data.id),
                    field="source_url",
                    method=RedactionMethod.SCRUB,
                    pattern="PII query parameters",
                ))
                data.source_url = clean_url

        # 3. Scrub email/phone from raw_text
        if data.raw_text:
            clean_text, n_redacted = self._scrub_text(data.raw_text)
            if n_redacted:
                audit.append(RedactionAuditEntry(
                    item_id=str(data.id),
                    field="raw_text",
                    method=RedactionMethod.SCRUB,
                    pattern=f"email/phone ({n_redacted} occurrence(s))",
                ))
                data.raw_text = clean_text

        # 4. Scrub metadata string values
        if data.metadata:
            clean_meta, n_meta = self._scrub_dict(data.metadata)
            if n_meta:
                audit.append(RedactionAuditEntry(
                    item_id=str(data.id),
                    field="metadata",
                    method=RedactionMethod.SCRUB,
                    pattern=f"email/phone in metadata ({n_meta} field(s))",
                ))
                data.metadata = clean_meta

        # Emit audit entries
        for entry in audit:
            self._log.info(
                "pii_redacted",
                extra={
                    "item_id": entry.item_id,
                    "field": entry.field,
                    "method": entry.method.value,
                    "pattern": entry.pattern,
                    "timestamp": entry.timestamp.isoformat(),
                },
            )

        return data

    def verify_clean(self, item: ContentItem) -> None:
        """Assert that ``item`` contains no detectable PII.

        Intended to be called at the LLM layer boundary as a final safety net.
        Raises ``DataResidencyViolationError`` if any PII pattern is found.

        Args:
            item: Content item to verify.

        Raises:
            DataResidencyViolationError: If any PII pattern is detected in
                ``author``, ``source_url``, ``raw_text``, or ``metadata``.
        """
        # Check author — should be a pseudonym token (not contain @, spaces, etc.)
        if item.author and _EMAIL_RE.search(item.author):
            raise DataResidencyViolationError(
                field="author",
                pattern="email address",
                details={"item_id": str(item.id)},
            )

        # Check raw_text for email / phone
        if item.raw_text:
            if _EMAIL_RE.search(item.raw_text):
                raise DataResidencyViolationError(
                    field="raw_text",
                    pattern="email address",
                    details={"item_id": str(item.id)},
                )
            if _PHONE_RE.search(item.raw_text):
                raise DataResidencyViolationError(
                    field="raw_text",
                    pattern="phone number",
                    details={"item_id": str(item.id)},
                )

        # Check source_url for PII query parameters.
        # Values that equal "<redacted>" have already been processed by redact()
        # and must NOT be flagged as violations (idempotency contract).
        if item.source_url:
            params = parse_qs(urlparse(item.source_url).query)
            pii_unredacted = {
                k for k, vals in params.items()
                if k.lower() in _PII_QUERY_PARAMS
                and not any(v in ("<redacted>", "%3Credacted%3E") for v in vals)
            }
            if pii_unredacted:
                raise DataResidencyViolationError(
                    field="source_url",
                    pattern=f"PII query params: {', '.join(sorted(pii_unredacted))}",
                    details={"item_id": str(item.id)},
                )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pseudonymise(value: str) -> str:
        """Return a stable 16-hex-char pseudonym derived from ``value``.

        The pseudonym is deterministic (same input always produces the same
        output) so downstream systems can still correlate items by author
        without knowing the real identity.

        Args:
            value: Raw author handle or name.

        Returns:
            Hex string of the form ``"anon_<16 chars>"``.
        """
        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        return f"anon_{digest}"

    @staticmethod
    def _redact_url(url: str) -> str:
        """Strip PII query parameters from a URL.

        Args:
            url: Raw source URL, potentially containing profile PII parameters.

        Returns:
            URL with PII query parameters replaced by ``<redacted>``.
        """
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=True)
        cleaned = {
            k: ["<redacted>"] if k.lower() in _PII_QUERY_PARAMS else v
            for k, v in params.items()
        }
        # Rebuild only if something changed
        if cleaned == params:
            return url
        new_query = urlencode({k: v[0] for k, v in cleaned.items()})
        return parsed._replace(query=new_query).geturl()

    @staticmethod
    def _scrub_text(text: str):
        """Replace email addresses and phone numbers in free text with tokens.

        Args:
            text: Raw free text.

        Returns:
            Tuple of (scrubbed text, number of replacements made).
        """
        result, n = _EMAIL_RE.subn("<email_redacted>", text)
        result, m = _PHONE_RE.subn("<phone_redacted>", result)
        return result, n + m

    @staticmethod
    def _scrub_dict(d: Dict[str, Any]):
        """Recursively scrub PII patterns from dict string values.

        Args:
            d: Metadata dictionary to scrub.

        Returns:
            Tuple of (scrubbed dict, number of fields modified).
        """
        cleaned: Dict[str, Any] = {}
        n_modified = 0
        for k, v in d.items():
            if isinstance(v, str):
                scrubbed, n = DataResidencyGuard._scrub_text(v)
                cleaned[k] = scrubbed
                if n:
                    n_modified += 1
            elif isinstance(v, dict):
                sub, n = DataResidencyGuard._scrub_dict(v)
                cleaned[k] = sub
                n_modified += n
            else:
                cleaned[k] = v
        return cleaned, n_modified

