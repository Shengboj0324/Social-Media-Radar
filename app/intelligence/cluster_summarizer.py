"""Industrial-grade cluster summarization with ensemble LLMs and quality validation."""

import json
import logging
from typing import Any, Dict, List, Optional

from app.core.models import Cluster
from app.llm.client_base import BaseLLMClient

logger = logging.getLogger(__name__)


class ClusterSummarizer:
    """Generate industrial-grade summaries for content clusters.

    Features:
    - Multi-provider LLM ensemble for peak quality
    - Enhanced prompt engineering with chain-of-thought
    - Quality validation and scoring
    - Media-aware summarization (videos, images)
    - Multi-language support
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        use_ensemble: bool = True,
        enable_quality_validation: bool = True,
    ):
        """Initialize cluster summarizer.

        Args:
            llm_client: LLM client for generating summaries (legacy)
            use_ensemble: Use LLM ensemble for better quality
            enable_quality_validation: Enable quality scoring
        """
        self.llm_client = llm_client
        self.use_ensemble = use_ensemble
        self.enable_quality_validation = enable_quality_validation

        # Initialize ensemble if enabled
        if use_ensemble:
            from app.llm.ensemble import LLMEnsemble, EnsembleStrategy
            self.ensemble = LLMEnsemble(
                strategy=EnsembleStrategy.BEST_OF_N,
                enable_quality_validation=enable_quality_validation,
            )
        else:
            self.ensemble = None

    async def summarize_cluster(self, cluster: Cluster) -> Dict[str, Any]:
        """Generate industrial-grade summary for a content cluster.

        Args:
            cluster: Content cluster to summarize

        Returns:
            Dictionary with summary data including:
            - topic: Brief topic title
            - summary: Comprehensive summary
            - key_points: List of key points
            - platforms: Platforms represented
            - perspective_notes: Cross-platform perspective analysis
            - quality_score: Summary quality score (0-1)
        """
        # Build enhanced prompt with media awareness
        prompt = self._build_enhanced_cluster_prompt(cluster)

        try:
            # Generate summary using ensemble or single client
            if self.use_ensemble and self.ensemble:
                ensemble_summary = await self.ensemble.generate_summary(
                    prompt=prompt,
                    max_tokens=800,
                    temperature=0.3,  # Lower for factual content
                )
                response_content = ensemble_summary.content
                quality_score = ensemble_summary.quality.overall_score
                logger.info(
                    f"Ensemble summary quality: {quality_score:.2f} "
                    f"(provider: {ensemble_summary.provider.value})"
                )
            else:
                # Fallback to single client
                response = await self.llm_client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800,
                )
                response_content = response.content
                quality_score = 0.8  # Default score

            # Parse JSON response
            summary_data = json.loads(response_content)
            summary_data["quality_score"] = quality_score
            return summary_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response content: {response_content[:500]}")
            # Return fallback structure
            return self._create_fallback_summary(cluster)

        except Exception as e:
            logger.error(f"Error generating cluster summary: {e}")
            return self._create_fallback_summary(cluster)

    def _build_enhanced_cluster_prompt(self, cluster: Cluster) -> str:
        """Build enhanced prompt with chain-of-thought and media awareness."""
        prompt_parts = []

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

        video_count = 0
        image_count = 0

        for i, item in enumerate(cluster.items[:10], 1):  # Limit to 10 items
            prompt_parts.append(f"\n{i}. [{item.source_platform.value}] {item.title}")

            if item.author:
                prompt_parts.append(f"   Author: {item.author}")

            if item.raw_text:
                # Truncate long text but keep more context
                text = item.raw_text[:800]
                prompt_parts.append(f"   Content: {text}...")

            # Enhanced media information
            if item.media_urls:
                media_types = []
                for url in item.media_urls:
                    if any(ext in url.lower() for ext in ['.mp4', '.mov', '.avi', 'youtube.com', 'youtu.be']):
                        media_types.append("video")
                        video_count += 1
                    elif any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        media_types.append("image")
                        image_count += 1

                if media_types:
                    prompt_parts.append(f"   Media: {', '.join(set(media_types))} ({len(item.media_urls)} items)")

            # Add engagement metrics if available
            if item.metadata:
                metrics = []
                if "view_count" in item.metadata:
                    metrics.append(f"{item.metadata['view_count']:,} views")
                if "like_count" in item.metadata:
                    metrics.append(f"{item.metadata['like_count']:,} likes")
                if "score" in item.metadata:
                    metrics.append(f"{item.metadata['score']} score")
                if metrics:
                    prompt_parts.append(f"   Engagement: {', '.join(metrics)}")

            prompt_parts.append(f"   Published: {item.published_at.strftime('%Y-%m-%d %H:%M UTC')}")

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
        """Create fallback summary when LLM fails."""
        platforms = list({item.source_platform.value for item in cluster.items})

        # Extract key points from titles
        key_points = [item.title for item in cluster.items[:3]]

        return {
            "topic": cluster.topic,
            "summary": f"This cluster contains {len(cluster.items)} items from {', '.join(platforms)} "
            f"discussing {cluster.topic}. Key items include: {', '.join([item.title[:50] for item in cluster.items[:3]])}.",
            "key_points": key_points,
            "platforms": platforms,
            "perspective_notes": "Multiple platforms covering this topic with varying perspectives.",
        }


    async def generate_cross_platform_analysis(
        self, clusters: List[Cluster]
    ) -> str:
        """Generate cross-platform perspective analysis across multiple clusters.

        Args:
            clusters: List of content clusters

        Returns:
            Cross-platform analysis summary
        """
        if not clusters:
            return "No content available for cross-platform analysis."

        # Build analysis prompt
        prompt = self._build_cross_platform_prompt(clusters)

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=600,
            )
            return response.content

        except Exception as e:
            logger.error(f"Error generating cross-platform analysis: {e}")
            return "Cross-platform analysis unavailable."

    def _build_cross_platform_prompt(self, clusters: List[Cluster]) -> str:
        """Build prompt for cross-platform analysis."""
        prompt_parts = []

        prompt_parts.append(
            "Analyze how different platforms are covering the following topics. "
            "Identify any notable differences in perspective, emphasis, or framing:\n"
        )

        for cluster in clusters[:5]:  # Top 5 clusters
            platforms = ", ".join([p.value for p in cluster.platforms_represented])
            prompt_parts.append(f"\n- {cluster.topic} (covered by: {platforms})")
            prompt_parts.append(f"  Summary: {cluster.summary[:200]}")

        prompt_parts.append(
            "\n\nProvide a brief analysis (3-5 sentences) of how different platforms "
            "are approaching these topics differently."
        )

        return "\n".join(prompt_parts)
