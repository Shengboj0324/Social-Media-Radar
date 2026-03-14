"""Competitor Intelligence Workflow - Advanced competitive analysis and positioning.

This module implements the complete Competitor Intelligence workflow for
competitor weakness signals. It includes sentiment analysis, weakness categorization,
and strategic positioning content generation.

Design principles:
- Ethical competitive intelligence (no bashing)
- Helpful positioning (solve their problem, don't attack)
- Multi-format content generation (response, blog, comparison)
- Trend tracking for recurring weaknesses
"""

import logging
import re
from typing import Any, Dict, List

from app.core.signal_models import ActionableSignal
from app.intelligence.response_generator import ResponseGenerator
from app.intelligence.response_playbook import ResponseChannel
from app.workflows.workflow_models import WorkflowExecution, WorkflowStep

logger = logging.getLogger(__name__)


class CompetitorIntelligenceWorkflow:
    """Advanced Competitor Intelligence workflow implementation.

    This class provides specialized handlers for competitor weakness analysis,
    including sentiment analysis, weakness categorization, and positioning
    content generation.
    """

    def __init__(self, response_generator: ResponseGenerator):
        """Initialize Competitor Intelligence workflow.

        Args:
            response_generator: Response generator for content creation
        """
        self.response_generator = response_generator
        # In-memory occurrence counter for recurring-pattern detection.
        # Key: (competitor_name_lower, complaint_type_lower) → count
        self._pattern_counts: Dict[tuple, int] = {}

    async def analyze_competitor_complaint(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Analyze competitor complaint in detail.

        This step performs deep analysis of the complaint to extract:
        - Competitor name
        - Complaint type/category
        - Specific pain points
        - Severity level
        - Sentiment intensity
        - Recurring pattern indicators

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Complaint analysis results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        logger.info(f"Analyzing competitor complaint for signal {signal.id}")

        # Extract competitor name
        competitor_name = self._extract_competitor_name(signal)

        # Categorize complaint type
        complaint_type = self._categorize_complaint(signal)

        # Extract specific pain points
        pain_points = self._extract_complaint_pain_points(signal)

        # Assess severity
        severity = self._assess_complaint_severity(signal)

        # Analyze sentiment
        sentiment = self._analyze_sentiment(signal)

        # Check if this is a recurring pattern
        is_recurring = self._check_recurring_pattern(competitor_name, complaint_type)

        # Extract impact indicators
        impact_indicators = self._extract_impact_indicators(signal)

        analysis = {
            "competitor_name": competitor_name,
            "complaint_type": complaint_type,
            "pain_points": pain_points,
            "severity": severity,
            "sentiment": sentiment,
            "is_recurring": is_recurring,
            "impact_indicators": impact_indicators,
        }

        # Store in context
        execution.context["competitor_analysis"] = analysis

        logger.info(
            f"Competitor analysis complete: {competitor_name} - {complaint_type} "
            f"(severity={severity}, sentiment={sentiment['polarity']})"
        )

        return analysis

    def _extract_competitor_name(self, signal: ActionableSignal) -> str:
        """Extract competitor name from signal."""
        # Check metadata first
        if "competitor_name" in signal.metadata:
            return signal.metadata["competitor_name"]

        # Extract from text
        text = f"{signal.title} {signal.description}"

        # Common patterns
        patterns = [
            r"(\w+) is terrible",
            r"(\w+) sucks",
            r"(\w+)'s (?:pricing|support|service)",
            r"switching from (\w+)",
            r"leaving (\w+)",
            r"(\w+) doesn't",
            r"(\w+) can't",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Filter out common words
                if name.lower() not in ["the", "this", "that", "they", "it"]:
                    return name

        return "unknown"

    def _categorize_complaint(self, signal: ActionableSignal) -> str:
        """Categorize the type of complaint."""
        text = f"{signal.title} {signal.description}".lower()

        # Complaint categories with patterns
        categories = {
            "pricing": r"(expensive|overpriced|pricing|cost|too much money|can't afford)",
            "support": r"(support|customer service|help|response time|no reply|ignored)",
            "reliability": r"(crashes|bugs|broken|downtime|outage|doesn't work|unreliable)",
            "performance": r"(slow|sluggish|lag|performance|takes forever|timeout)",
            "features": r"(missing|lacks|doesn't have|no support for|limited features)",
            "usability": r"(complicated|confusing|hard to use|difficult|ux|ui|interface)",
            "integration": r"(doesn't integrate|no api|can't connect|integration)",
            "security": r"(security|breach|unsafe|vulnerable|privacy)",
            "scalability": r"(doesn't scale|can't handle|limitations|outgrew)",
        }

        # Find matching categories
        matches = []
        for category, pattern in categories.items():
            if re.search(pattern, text):
                matches.append(category)

        # Return primary category or general
        return matches[0] if matches else "general"

    def _extract_complaint_pain_points(self, signal: ActionableSignal) -> List[str]:
        """Extract specific pain points from complaint."""
        text = f"{signal.title} {signal.description}".lower()
        pain_points = []

        # Specific pain point patterns
        pain_patterns = {
            "no_response": r"(no response|haven't heard|waiting for|days? without)",
            "data_loss": r"(lost data|data loss|deleted|disappeared)",
            "poor_documentation": r"(no docs|poor documentation|unclear|confusing docs)",
            "forced_upgrade": r"(forced upgrade|breaking changes|deprecated)",
            "vendor_lock": r"(locked in|can't export|can't migrate|stuck with)",
            "hidden_fees": r"(hidden fees|surprise charges|unexpected cost)",
            "poor_onboarding": r"(hard to set up|difficult onboarding|no guidance)",
        }

        for pain_type, pattern in pain_patterns.items():
            if re.search(pattern, text):
                pain_points.append(pain_type)

        return pain_points if pain_points else ["general_complaint"]

    def _assess_complaint_severity(self, signal: ActionableSignal) -> str:
        """Assess severity of the complaint."""
        text = f"{signal.title} {signal.description}".lower()

        # Critical severity indicators
        critical_indicators = [
            "data loss", "security breach", "lawsuit", "legal",
            "going out of business", "shutting down", "bankruptcy"
        ]

        # High severity indicators
        high_indicators = [
            "terrible", "worst", "awful", "horrible", "disaster",
            "never again", "warning", "avoid", "scam"
        ]

        # Check for critical
        if any(indicator in text for indicator in critical_indicators):
            return "critical"

        # Check for high
        if any(indicator in text for indicator in high_indicators):
            return "high"

        # Check signal scores
        if signal.urgency_score >= 0.7 and signal.impact_score >= 0.7:
            return "high"
        elif signal.urgency_score >= 0.5 or signal.impact_score >= 0.5:
            return "medium"
        else:
            return "low"

    def _analyze_sentiment(self, signal: ActionableSignal) -> Dict[str, Any]:
        """Analyze sentiment of the complaint."""
        text = f"{signal.title} {signal.description}".lower()

        # Simple sentiment analysis (in production, use ML model)
        negative_words = [
            "terrible", "awful", "horrible", "worst", "hate", "frustrated",
            "angry", "disappointed", "useless", "waste", "regret"
        ]

        positive_words = [
            "good", "great", "excellent", "love", "happy", "satisfied",
            "recommend", "best", "amazing"
        ]

        neg_count = sum(1 for word in negative_words if word in text)
        pos_count = sum(1 for word in positive_words if word in text)

        # Calculate polarity (-1 to 1)
        total = neg_count + pos_count
        if total == 0:
            polarity = 0.0
        else:
            polarity = (pos_count - neg_count) / total

        # Intensity (0 to 1)
        intensity = min(total / 10.0, 1.0)

        return {
            "polarity": polarity,
            "intensity": intensity,
            "negative_count": neg_count,
            "positive_count": pos_count,
        }

    def _resolve_channel(
        self,
        signal: ActionableSignal,
        config: Dict[str, Any],
    ) -> ResponseChannel:
        """Resolve target ResponseChannel from step config or signal, defaulting to REDDIT."""
        config_channel = config.get("channel")
        if config_channel:
            try:
                return ResponseChannel(config_channel)
            except ValueError:
                pass
        if signal.suggested_channel:
            try:
                return ResponseChannel(signal.suggested_channel.lower())
            except ValueError:
                pass
        return ResponseChannel.REDDIT

    def _check_recurring_pattern(
        self,
        competitor_name: str,
        complaint_type: str,
    ) -> bool:
        """Check if this is a recurring complaint pattern.

        Uses an in-memory counter keyed by (competitor_name, complaint_type).
        A pattern is considered recurring once it has been seen ≥ 2 times
        in this session, which is a conservative but meaningful threshold.
        """
        key = (competitor_name.lower(), complaint_type.lower())
        self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1
        is_recurring = self._pattern_counts[key] >= 2

        if is_recurring:
            logger.info(
                f"Recurring pattern detected: competitor={competitor_name!r} "
                f"complaint={complaint_type!r} count={self._pattern_counts[key]}"
            )
        return is_recurring

    def _extract_impact_indicators(self, signal: ActionableSignal) -> Dict[str, Any]:
        """Extract indicators of complaint impact."""
        text = f"{signal.title} {signal.description}".lower()

        impact = {
            "public_visibility": "high" if signal.source_platform in ["twitter", "reddit"] else "medium",
            "has_audience": False,
            "viral_potential": False,
            "influencer": False,
        }

        # Check for audience indicators
        if any(word in text for word in ["team", "company", "organization", "users"]):
            impact["has_audience"] = True

        # Check for viral potential
        if any(word in text for word in ["everyone", "all", "warning", "psa"]):
            impact["viral_potential"] = True

        # Check metadata for follower count (if available)
        if signal.metadata.get("author_followers", 0) > 1000:
            impact["influencer"] = True

        return impact

    async def identify_positioning_angle(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Identify how our product addresses these pain points.

        This step determines:
        - Which features solve their problems
        - Key differentiators vs competitor
        - Positioning angle (helpful, not salesy)
        - Content opportunities

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Positioning analysis
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)
        analysis = execution.context.get("competitor_analysis", {})

        logger.info(f"Identifying positioning angle for signal {signal.id}")

        complaint_type = analysis.get("complaint_type", "general")
        pain_points = analysis.get("pain_points", [])

        # Map complaint types to our strengths
        positioning = self._map_to_strengths(complaint_type, pain_points)

        # Identify key differentiators
        differentiators = self._identify_differentiators(complaint_type, analysis)

        # Determine positioning angle
        angle = self._determine_positioning_angle(analysis)

        # Suggest content types
        content_opportunities = self._suggest_content_types(analysis, positioning)

        result = {
            "positioning": positioning,
            "differentiators": differentiators,
            "angle": angle,
            "content_opportunities": content_opportunities,
        }

        # Store in context
        execution.context["positioning"] = result

        logger.info(
            f"Positioning identified: {angle} angle with "
            f"{len(differentiators)} differentiators"
        )

        return result

    def _map_to_strengths(
        self,
        complaint_type: str,
        pain_points: List[str],
    ) -> Dict[str, Any]:
        """Map competitor weaknesses to our strengths."""
        # Strength mapping (customize for your product)
        strength_map = {
            "pricing": {
                "our_advantage": "transparent_pricing",
                "message": "Simple, transparent pricing with no hidden fees",
                "features": ["free_tier", "flexible_plans", "no_contracts"],
            },
            "support": {
                "our_advantage": "responsive_support",
                "message": "24/7 support with <1hr response time",
                "features": ["live_chat", "dedicated_support", "community"],
            },
            "reliability": {
                "our_advantage": "99.9_uptime",
                "message": "Enterprise-grade reliability with 99.9% uptime SLA",
                "features": ["redundancy", "monitoring", "auto_failover"],
            },
            "performance": {
                "our_advantage": "fast_performance",
                "message": "Built for speed with sub-second response times",
                "features": ["optimized_infrastructure", "caching", "cdn"],
            },
            "usability": {
                "our_advantage": "intuitive_ux",
                "message": "Designed for ease of use, no training required",
                "features": ["clean_ui", "guided_onboarding", "templates"],
            },
        }

        return strength_map.get(complaint_type, {
            "our_advantage": "better_alternative",
            "message": "A better alternative that solves your problems",
            "features": ["reliable", "affordable", "easy_to_use"],
        })

    def _identify_differentiators(
        self,
        complaint_type: str,
        analysis: Dict[str, Any],
    ) -> List[str]:
        """Identify key differentiators vs competitor."""
        differentiators = []

        # Based on complaint type
        diff_map = {
            "pricing": ["transparent_pricing", "free_tier", "no_hidden_fees"],
            "support": ["fast_response", "dedicated_support", "community"],
            "reliability": ["high_uptime", "redundancy", "monitoring"],
            "performance": ["fast", "optimized", "scalable"],
            "usability": ["intuitive", "easy_setup", "no_training"],
        }

        differentiators.extend(diff_map.get(complaint_type, ["better_alternative"]))

        return differentiators

    def _determine_positioning_angle(self, analysis: Dict[str, Any]) -> str:
        """Determine the best positioning angle."""
        severity = analysis.get("severity", "medium")
        sentiment = analysis.get("sentiment", {})

        # High severity + very negative = empathetic helper
        if severity in ["high", "critical"] and sentiment.get("polarity", 0) < -0.5:
            return "empathetic_helper"

        # Recurring pattern = thought_leader
        if analysis.get("is_recurring"):
            return "thought_leader"

        # Default = helpful_alternative
        return "helpful_alternative"

    def _suggest_content_types(
        self,
        analysis: Dict[str, Any],
        positioning: Dict[str, Any],
    ) -> List[str]:
        """Suggest content types to create."""
        content_types = ["public_response"]  # Always include response

        severity = analysis.get("severity", "medium")
        is_recurring = analysis.get("is_recurring", False)

        # High severity = blog post
        if severity in ["high", "critical"]:
            content_types.append("blog_post")

        # Recurring = comparison chart
        if is_recurring:
            content_types.append("comparison_chart")

        # Always useful
        content_types.append("helpful_guide")

        return content_types

    async def generate_positioning_content(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Generate helpful positioning content.

        This step creates content that:
        - Helps the user solve their problem
        - Subtly positions our solution
        - Avoids direct competitor bashing
        - Provides genuine value

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Generated content data
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)
        analysis = execution.context.get("competitor_analysis", {})
        positioning = execution.context.get("positioning", {})

        logger.info(f"Generating positioning content for signal {signal.id}")

        # Enhance signal metadata with analysis
        enhanced_signal = signal.model_copy()
        enhanced_signal.metadata.update({
            "competitor_name": analysis.get("competitor_name", "unknown"),
            "complaint_type": analysis.get("complaint_type", "general"),
            "pain_points": analysis.get("pain_points", []),
            "positioning_angle": positioning.get("angle", "helpful_alternative"),
            "differentiators": positioning.get("differentiators", []),
        })

        # Generate content variants
        content_types = positioning.get("content_opportunities", ["public_response"])

        generated_content = {}

        # Generate public response
        if "public_response" in content_types:
            channel = self._resolve_channel(enhanced_signal, step.config)
            variants = await self.response_generator.generate_variants(
                signal=enhanced_signal,
                num_variants=3,
                channel=channel,
            )

            if variants:
                generated_content["public_response"] = {
                    "content": variants[0].content,
                    "tone": variants[0].tone.value,
                    "score": variants[0].overall_score,
                }

        # Store in context
        execution.context["generated_content"] = generated_content

        logger.info(
            f"Content generation complete: {len(generated_content)} content types"
        )

        return {
            "content_types_generated": list(generated_content.keys()),
            "primary_content": generated_content.get("public_response", {}).get("content", ""),
            "positioning_applied": True,
        }
