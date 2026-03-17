"""Cluster summarisation routed through LLMRouter with DataResidencyGuard enforcement.

Every ``ContentItem`` injected into a prompt is first passed through the
module-level ``_GUARD`` singleton (a ``DataResidencyGuard``), enforcing the
zero-egress PII contract defined in ``app/core/data_residency.py``.

All LLM calls are routed through ``LLMRouter.generate_for_signal()`` with
``signal_type=None`` so the frontier model is selected unconditionally, the
circuit-breaker and rate-limiter middleware apply, and cost metrics are
recorded automatically via the existing monitoring stack.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from app.core.data_residency import DataResidencyGuard
from app.core.models import Cluster, ContentItem
from app.llm.models import LLMMessage
from app.llm.router import LLMRouter, get_router

logger = logging.getLogger(__name__)

# Module-level singleton — one DataResidencyGuard shared across all ClusterSummarizer
# instances so compiled regex patterns and the audit logger are constructed once.
_GUARD: DataResidencyGuard = DataResidencyGuard()


class ClusterSummarizer:
    """Generate summaries for content clusters via LLMRouter.

    Key design decisions:

    * **PII safety** — every ``ContentItem`` field injected into a prompt is
      first passed through ``_GUARD.redact()``, pseudonymising ``author`` values
      and stripping emails/phone numbers from ``raw_text`` before any text
      reaches an LLM provider.

    * **Unified LLM routing** — all generation calls go through
      ``LLMRouter.generate_for_signal(signal_type=None, ...)`` so the frontier
      model is selected, the circuit breaker fires on provider failures, and
      request cost is tracked via the existing monitoring stack.

    * **No hardcoded outputs** — fallback paths return factually-derived content
      from cluster metadata; no pre-written editorial sentences are ever returned
      to callers.
    """

    def __init__(
        self,
        llm_client: Optional[object] = None,
        use_ensemble: bool = True,
        enable_quality_validation: bool = True,
    ) -> None:
        """Initialise the cluster summariser.

        Args:
            llm_client: **Deprecated — ignored.** Accepted for backward
                compatibility with existing call-sites (e.g. ``DigestEngine``).
                All LLM calls now go through ``LLMRouter``; this parameter
                will be removed in a future release.
            use_ensemble: **Deprecated — ignored.** Ensemble selection has
                been superseded by ``LLMRouter``'s tiered model dispatch.
            enable_quality_validation: **Deprecated — ignored.** Quality
                validation is handled downstream by the router's provider stack.
        """
        self._router: LLMRouter = get_router()

    async def summarize_cluster(self, cluster: Cluster) -> Dict[str, Any]:
        """Generate a PII-safe summary for a content cluster via LLMRouter.

        Every ``ContentItem`` in the cluster is redacted by ``_GUARD`` before
        any field is injected into the prompt.  The LLM call routes through
        ``LLMRouter.generate_for_signal(signal_type=None, ...)`` so the frontier
        model is always selected and full reliability middleware applies.

        Args:
            cluster: Content cluster to summarise.

        Returns:
            Mapping with the following keys:

            * ``topic`` (``str``) — brief topic title produced by the LLM.
            * ``summary`` (``str``) — comprehensive narrative summary.
            * ``key_points`` (``List[str]``) — top bullet points.
            * ``platforms`` (``List[str]``) — platform names represented.
            * ``perspective_notes`` (``Optional[str]``) — cross-platform
              framing differences, or ``None`` when unavailable.
            * ``quality_score`` (``float``) — ``1.0`` on success; ``0.0``
              when the fallback path was taken.

        Note:
            This method never raises.  All LLM and parse errors fall back to
            ``_create_fallback_summary``.
        """
        prompt: str = self._build_enhanced_cluster_prompt(cluster)
        response_content: str = ""

        try:
            response_content = await self._router.generate_for_signal(
                signal_type=None,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=800,
            )
            summary_data: Dict[str, Any] = json.loads(response_content)
            summary_data["quality_score"] = 1.0
            return summary_data

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM response as JSON: %s", exc)
            logger.error(
                "Response content (first 500 chars): %s", response_content[:500]
            )
            return self._create_fallback_summary(cluster)

        except Exception as exc:
            logger.error("Error generating cluster summary: %s", exc, exc_info=True)
            return self._create_fallback_summary(cluster)

    def _build_enhanced_cluster_prompt(self, cluster: Cluster) -> str:
        """Build an enhanced, PII-safe prompt with chain-of-thought reasoning.

        Each ``ContentItem`` in the cluster is passed through ``_GUARD.redact()``
        at the top of the loop before any field (``author``, ``raw_text``, media
        labels, engagement metrics, or timestamps) is appended to the prompt,
        enforcing the zero-egress data residency contract at the call site.

        Args:
            cluster: Content cluster whose items will be summarised.

        Returns:
            Fully assembled prompt string ready to pass to the LLM.
        """
        prompt_parts: List[str] = []

        # System context with chain-of-thought
        prompt_parts.append(
            "You are an expert intelligence analyst creating high-quality summaries of related content.\n"
        )
        prompt_parts.append(
            "Your task is to analyze content items that discuss the same topic or event, "
            "and create a comprehensive, factual summary.\n"
        )
        prompt_parts.append(
            "\nAPPROACH:\n"
            "1. First, identify the core topic/event\n"
            "2. Extract key facts from each source\n"
            "3. Synthesize information across sources\n"
            "4. Note any conflicting information or different perspectives\n"
            "5. Highlight the most newsworthy aspects\n"
            "6. Create a clear, concise summary\n"
        )

        # Add content items with media awareness
        prompt_parts.append("\nCONTENT ITEMS:\n")

        video_count: int = 0
        image_count: int = 0

        for i, item in enumerate(cluster.items[:10], 1):  # Limit to 10 items
            # Enforce zero-egress PII contract before any field touches the prompt.
            # redact() pseudonymises author and scrubs emails/phones from raw_text.
            safe_item: ContentItem = _GUARD.redact(item)

            prompt_parts.append(
                f"\n{i}. [{safe_item.source_platform.value}] {safe_item.title}"
            )

            if safe_item.author:
                prompt_parts.append(f"   Author: {safe_item.author}")

            if safe_item.raw_text:
                # Truncate long text but keep more context
                text: str = safe_item.raw_text[:800]
                prompt_parts.append(f"   Content: {text}...")

            # Enhanced media information
            if safe_item.media_urls:
                media_types: List[str] = []
                for url in safe_item.media_urls:
                    if any(
                        ext in url.lower()
                        for ext in [".mp4", ".mov", ".avi", "youtube.com", "youtu.be"]
                    ):
                        media_types.append("video")
                        video_count += 1
                    elif any(
                        ext in url.lower()
                        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
                    ):
                        media_types.append("image")
                        image_count += 1

                if media_types:
                    prompt_parts.append(
                        f"   Media: {', '.join(set(media_types))}"
                        f" ({len(safe_item.media_urls)} items)"
                    )

            # Add engagement metrics if available
            if safe_item.metadata:
                metrics: List[str] = []
                if "view_count" in safe_item.metadata:
                    metrics.append(f"{safe_item.metadata['view_count']:,} views")
                if "like_count" in safe_item.metadata:
                    metrics.append(f"{safe_item.metadata['like_count']:,} likes")
                if "score" in safe_item.metadata:
                    metrics.append(f"{safe_item.metadata['score']} score")
                if metrics:
                    prompt_parts.append(f"   Engagement: {', '.join(metrics)}")

            prompt_parts.append(
                f"   Published: {safe_item.published_at.strftime('%Y-%m-%d %H:%M UTC')}"
            )

        # Add media context
        if video_count > 0 or image_count > 0:
            prompt_parts.append(f"\nMEDIA CONTEXT: This topic includes {video_count} videos and {image_count} images.")

        # Enhanced instructions
        prompt_parts.append("\n\nINSTRUCTIONS:")
        prompt_parts.append("1. Identify the main topic or event with precision")
        prompt_parts.append("2. Synthesize ONLY factual information from the sources")
        prompt_parts.append("3. Note platform-specific perspectives or framing differences")
        prompt_parts.append("4. If videos/images are present, mention their relevance to the story")
        prompt_parts.append("5. Highlight the most important developments or revelations")
        prompt_parts.append("6. Maintain objectivity - avoid speculation or editorializing")
        prompt_parts.append("7. If sources conflict, note the discrepancy")
        prompt_parts.append("8. Keep summary concise but comprehensive (200-400 words)")

        # Add output format
        prompt_parts.append("\n\nOUTPUT FORMAT:")
        prompt_parts.append("Provide your response as a JSON object with the following structure:")
        prompt_parts.append('{')
        prompt_parts.append('  "topic": "Brief topic title (max 100 chars)",')
        prompt_parts.append('  "summary": "Comprehensive summary (200-400 words)",')
        prompt_parts.append('  "key_points": ["point 1", "point 2", "point 3"],')
        prompt_parts.append('  "platforms": ["platform1", "platform2"],')
        prompt_parts.append(
            '  "perspective_notes": "Any notable differences in coverage or perspective across sources"'
        )
        prompt_parts.append('}')

        return "\n".join(prompt_parts)

    def _create_fallback_summary(self, cluster: Cluster) -> Dict[str, Any]:
        """Produce a factually-derived fallback summary when the LLM call fails.

        All values are derived directly from cluster metadata — no pre-written
        editorial sentences are returned.  ``perspective_notes`` is explicitly
        set to ``None`` rather than a fabricated cross-platform conclusion.
        ``quality_score`` is set to ``0.0`` to signal to callers that this is
        a degraded, non-LLM response.

        Args:
            cluster: Cluster to produce fallback content for.

        Returns:
            Mapping with the same keys as ``summarize_cluster``'s return value.
        """
        platforms: List[str] = sorted(
            {item.source_platform.value for item in cluster.items}
        )
        key_points: List[str] = [item.title for item in cluster.items[:3]]
        platform_str: str = ", ".join(platforms) if platforms else "unknown platform(s)"
        item_count: int = len(cluster.items)

        title_fragment: str = (
            " Titles: "
            + "; ".join(item.title[:60] for item in cluster.items[:3])
            + "."
            if cluster.items
            else ""
        )

        return {
            "topic": cluster.topic,
            "summary": (
                f"{item_count} item(s) from {platform_str}"
                f" on topic: {cluster.topic}.{title_fragment}"
            ),
            "key_points": key_points,
            "platforms": platforms,
            "perspective_notes": None,
            "quality_score": 0.0,
        }


    async def generate_cross_platform_analysis(
        self, clusters: List[Cluster]
    ) -> Optional[str]:
        """Generate a cross-platform perspective analysis across multiple clusters.

        Returns ``None`` rather than a hardcoded string when the cluster list is
        empty or when the LLM call fails, leaving the decision of how to handle
        the absence of analysis to the caller.

        Args:
            clusters: List of content clusters to analyse.

        Returns:
            Cross-platform analysis string produced by the LLM, or ``None``
            if no clusters were supplied or if the LLM call fails.
        """
        if not clusters:
            return None

        prompt: str = self._build_cross_platform_prompt(clusters)

        try:
            return await self._router.generate_for_signal(
                signal_type=None,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.7,
                max_tokens=600,
            )
        except Exception as exc:
            logger.error(
                "Error generating cross-platform analysis: %s", exc, exc_info=True
            )
            return None

    def _build_cross_platform_prompt(self, clusters: List[Cluster]) -> str:
        """Build a PII-safe prompt for cross-platform perspective analysis.

        ``cluster.summary`` for each cluster is passed through
        ``DataResidencyGuard._scrub_text`` before injection to remove any email
        addresses or phone numbers that may have been echoed back by a prior LLM
        call and stored in the summary field.

        Args:
            clusters: List of clusters to include; only the first five are used.

        Returns:
            Fully assembled prompt string ready to pass to the LLM.
        """
        prompt_parts: List[str] = []

        prompt_parts.append(
            "Analyze how different platforms are covering the following topics. "
            "Identify any notable differences in perspective, emphasis, or framing:\n"
        )

        for cluster in clusters[:5]:  # Top 5 clusters
            platform_str: str = ", ".join(
                [p.value for p in cluster.platforms_represented]
            )
            # Scrub the pre-existing summary before injecting it into the prompt —
            # a prior LLM call may have echoed PII from item content back into it.
            safe_summary, _ = DataResidencyGuard._scrub_text(cluster.summary[:200])
            prompt_parts.append(f"\n- {cluster.topic} (covered by: {platform_str})")
            prompt_parts.append(f"  Summary: {safe_summary}")

        prompt_parts.append(
            "\n\nProvide a brief analysis (3-5 sentences) of how different platforms "
            "are approaching these topics differently."
        )

        return "\n".join(prompt_parts)
