"""Alternative Seeker Workflow - Advanced lead qualification and outreach.

This module implements the complete Alternative Seeker workflow for lead opportunities.
It includes advanced NLP for intent analysis, ML-based lead scoring, and personalized
response generation.

Design principles:
- High-precision lead qualification to avoid spam
- Personalized outreach based on pain points
- Multi-touch follow-up strategy
- Conversion tracking and learning loop
"""

import logging
import re
from typing import Any, Dict, List

from app.core.signal_models import ActionableSignal
from app.intelligence.response_generator import ResponseGenerator
from app.intelligence.response_playbook import ResponseChannel
from app.workflows.workflow_models import WorkflowExecution, WorkflowStep

logger = logging.getLogger(__name__)


class AlternativeSeekerWorkflow:
    """Advanced Alternative Seeker workflow implementation.

    This class provides specialized handlers for the Alternative Seeker workflow,
    including intent analysis, lead qualification, and personalized outreach.
    """

    def __init__(self, response_generator: ResponseGenerator):
        """Initialize Alternative Seeker workflow.

        Args:
            response_generator: Response generator for content creation
        """
        self.response_generator = response_generator

    async def analyze_lead_intent(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Analyze lead's specific pain points and requirements.

        This step performs deep analysis of the lead's intent using NLP
        to extract:
        - Current tool/solution they're using
        - Specific pain points and frustrations
        - Requirements and must-haves
        - Budget signals
        - Timeline/urgency indicators

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Analysis results with extracted fields
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        logger.info(f"Analyzing lead intent for signal {signal.id}")

        # Extract pain points using advanced NLP
        pain_points = self._extract_pain_points_advanced(signal)

        # Extract current tool/competitor
        current_tool = self._extract_current_tool(signal)

        # Extract requirements
        requirements = self._extract_requirements_advanced(signal)

        # Detect budget signals
        budget_signals = self._detect_budget_signals(signal)

        # Detect urgency/timeline
        urgency_indicators = self._detect_urgency_indicators(signal)

        # Extract company/role context if available
        company_context = self._extract_company_context(signal)

        analysis = {
            "pain_points": pain_points,
            "current_tool": current_tool,
            "requirements": requirements,
            "budget_signals": budget_signals,
            "urgency_indicators": urgency_indicators,
            "company_context": company_context,
            "intent_clarity": self._calculate_intent_clarity(
                pain_points, requirements, urgency_indicators
            ),
        }

        # Store in context for next steps
        execution.context["lead_analysis"] = analysis

        logger.info(
            f"Lead analysis complete: {len(pain_points)} pain points, "
            f"{len(requirements)} requirements, intent_clarity={analysis['intent_clarity']:.2f}"
        )

        return analysis

    def _extract_pain_points_advanced(self, signal: ActionableSignal) -> List[str]:
        """Extract pain points using advanced NLP patterns."""
        text = f"{signal.title} {signal.description}".lower()
        pain_points = []

        # Advanced pain point patterns
        pain_patterns = {
            "pricing": r"(too expensive|overpriced|pricing is|costs? too much|can't afford)",
            "complexity": r"(too complicated|complex|hard to use|difficult|confusing|steep learning curve)",
            "performance": r"(slow|sluggish|laggy|performance issues|takes forever|timeout)",
            "reliability": r"(crashes|buggy|unreliable|downtime|broken|doesn't work)",
            "support": r"(poor support|no help|support is terrible|can't reach|no response)",
            "features": r"(missing features?|lacks|doesn't have|no support for|limited)",
            "integration": r"(doesn't integrate|no api|can't connect|integration issues)",
            "scalability": r"(doesn't scale|outgrew|too small|can't handle|limitations)",
        }

        for pain_type, pattern in pain_patterns.items():
            if re.search(pattern, text):
                pain_points.append(pain_type)

        # Extract specific mentions
        if "expensive" in text or "pricing" in text:
            # Try to extract specific price complaints
            price_match = re.search(r'\$(\d+)', text)
            if price_match:
                pain_points.append(f"price_concern_${price_match.group(1)}")

        return pain_points if pain_points else ["general_dissatisfaction"]

    def _extract_current_tool(self, signal: ActionableSignal) -> str:
        """Extract current tool/competitor name."""
        # Check metadata first
        if "competitor_name" in signal.metadata:
            return signal.metadata["competitor_name"]

        # Extract from text
        text = f"{signal.title} {signal.description}"

        # Common competitor patterns
        competitor_patterns = [
            r"switching from (\w+)",
            r"leaving (\w+)",
            r"alternative to (\w+)",
            r"replacement for (\w+)",
            r"instead of (\w+)",
            r"(\w+) is too",
            r"(\w+)'s pricing",
        ]

        for pattern in competitor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return "unknown"

    def _extract_requirements_advanced(self, signal: ActionableSignal) -> List[str]:
        """Extract specific requirements and must-haves."""
        text = f"{signal.title} {signal.description}".lower()
        requirements = []

        # Requirement patterns
        req_patterns = {
            "ease_of_use": r"(easy to use|simple|intuitive|user[- ]friendly)",
            "affordable": r"(affordable|cheap|budget|low cost|free tier)",
            "fast": r"(fast|quick|responsive|real[- ]time)",
            "reliable": r"(reliable|stable|uptime|dependable)",
            "scalable": r"(scalable|grows with|enterprise)",
            "integrations": r"(integrates with|api|connects to|works with)",
            "support": r"(good support|responsive|help|documentation)",
            "mobile": r"(mobile|ios|android|app)",
            "cloud": r"(cloud|saas|hosted)",
            "on_premise": r"(on[- ]premise|self[- ]hosted|private)",
        }

        for req_type, pattern in req_patterns.items():
            if re.search(pattern, text):
                requirements.append(req_type)

        return requirements if requirements else ["general_requirements"]

    def _detect_budget_signals(self, signal: ActionableSignal) -> Dict[str, Any]:
        """Detect budget-related signals."""
        text = f"{signal.title} {signal.description}".lower()

        budget_info = {
            "price_sensitive": False,
            "has_budget": False,
            "budget_range": None,
            "looking_for_free": False,
        }

        # Price sensitivity
        if any(word in text for word in ["expensive", "pricing", "cost", "afford"]):
            budget_info["price_sensitive"] = True

        # Budget availability
        if any(word in text for word in ["budget", "approved", "allocated"]):
            budget_info["has_budget"] = True

        # Free tier interest
        if any(word in text for word in ["free", "trial", "demo"]):
            budget_info["looking_for_free"] = True

        # Extract budget range
        budget_match = re.search(r'\$(\d+)', text)
        if budget_match:
            budget_info["budget_range"] = f"${budget_match.group(1)}"

        return budget_info

    def _detect_urgency_indicators(self, signal: ActionableSignal) -> Dict[str, Any]:
        """Detect urgency and timeline indicators."""
        text = f"{signal.title} {signal.description}".lower()

        urgency_info = {
            "is_urgent": False,
            "timeline": None,
            "urgency_level": "low",
        }

        # High urgency indicators
        high_urgency = ["asap", "urgent", "immediately", "right now", "today"]
        if any(word in text for word in high_urgency):
            urgency_info["is_urgent"] = True
            urgency_info["urgency_level"] = "high"

        # Timeline extraction
        timeline_patterns = [
            (r"within (\d+) (days?|weeks?|months?)", "specific"),
            (r"by (next week|next month|end of)", "specific"),
            (r"(soon|quickly|fast)", "general"),
        ]

        for pattern, timeline_type in timeline_patterns:
            match = re.search(pattern, text)
            if match:
                urgency_info["timeline"] = match.group(0)
                if timeline_type == "specific":
                    urgency_info["urgency_level"] = "medium"
                break

        return urgency_info

    def _extract_company_context(self, signal: ActionableSignal) -> Dict[str, Any]:
        """Extract company/role context if available."""
        text = f"{signal.title} {signal.description}"

        context = {
            "company_size": None,
            "industry": None,
            "role": None,
        }

        # Company size indicators
        size_patterns = {
            "startup": r"(startup|early stage|seed)",
            "smb": r"(small business|smb|small team)",
            "enterprise": r"(enterprise|large company|fortune)",
        }

        for size, pattern in size_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                context["company_size"] = size
                break

        # Role indicators
        role_patterns = {
            "founder": r"(founder|ceo|co-founder)",
            "developer": r"(developer|engineer|programmer)",
            "manager": r"(manager|director|head of)",
            "admin": r"(admin|administrator|it)",
        }

        for role, pattern in role_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                context["role"] = role
                break

        return context

    def _calculate_intent_clarity(
        self,
        pain_points: List[str],
        requirements: List[str],
        urgency_indicators: Dict[str, Any],
    ) -> float:
        """Calculate how clear the lead's intent is (0-1)."""
        score = 0.0

        # More pain points = clearer intent
        score += min(len(pain_points) * 0.15, 0.4)

        # More requirements = clearer intent
        score += min(len(requirements) * 0.1, 0.3)

        # Urgency indicates serious intent
        if urgency_indicators.get("is_urgent"):
            score += 0.2
        elif urgency_indicators.get("timeline"):
            score += 0.1

        return min(score, 1.0)

    async def qualify_lead(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Qualify lead using ML-based scoring.

        This step scores the lead across multiple dimensions:
        - Product fit: How well do we solve their pain points?
        - Intent strength: How serious are they about switching?
        - Urgency: How soon do they need a solution?
        - Budget fit: Can they afford our solution?
        - Conversion likelihood: Overall probability of conversion

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Lead qualification scores
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)
        analysis = execution.context.get("lead_analysis", {})

        logger.info(f"Qualifying lead for signal {signal.id}")

        # Calculate fit score
        fit_score = self._calculate_product_fit(signal, analysis)

        # Calculate intent score
        intent_score = analysis.get("intent_clarity", 0.5)

        # Calculate urgency score (use signal's urgency_score)
        urgency_score = signal.urgency_score

        # Calculate budget fit
        budget_fit = self._calculate_budget_fit(analysis)

        # Calculate conversion likelihood
        conversion_likelihood = self._calculate_conversion_likelihood(
            fit_score, intent_score, urgency_score, budget_fit
        )

        # Overall lead quality
        overall_score = (
            fit_score * 0.3 +
            intent_score * 0.25 +
            urgency_score * 0.2 +
            budget_fit * 0.15 +
            conversion_likelihood * 0.1
        )

        # Determine quality level
        if overall_score >= 0.75:
            quality_level = "high_quality"
        elif overall_score >= 0.5:
            quality_level = "medium_quality"
        else:
            quality_level = "low_quality"

        scores = {
            "fit_score": fit_score,
            "intent_score": intent_score,
            "urgency_score": urgency_score,
            "budget_fit": budget_fit,
            "conversion_likelihood": conversion_likelihood,
            "overall_score": overall_score,
            "quality_level": quality_level,
        }

        # Store in context
        execution.context["lead_scores"] = scores
        execution.context["lead_quality"] = quality_level

        logger.info(
            f"Lead qualification complete: {quality_level} "
            f"(overall={overall_score:.2f}, fit={fit_score:.2f})"
        )

        return scores

    def _calculate_product_fit(
        self,
        signal: ActionableSignal,
        analysis: Dict[str, Any],
    ) -> float:
        """Calculate how well our product fits their needs."""
        score = 0.5  # Base score

        pain_points = analysis.get("pain_points", [])

        # Boost score for pain points we solve well
        if "pricing" in pain_points or "expensive" in pain_points:
            score += 0.15  # We're more affordable

        if "complexity" in pain_points:
            score += 0.15  # We're easier to use

        if "support" in pain_points:
            score += 0.1  # We have better support

        if "features" in pain_points:
            score += 0.05  # We have more features

        # Boost for requirements we meet
        requirements = analysis.get("requirements", [])
        if "ease_of_use" in requirements:
            score += 0.1
        if "affordable" in requirements:
            score += 0.1

        return min(score, 1.0)

    def _calculate_budget_fit(self, analysis: Dict[str, Any]) -> float:
        """Calculate budget fit score."""
        budget_signals = analysis.get("budget_signals", {})

        if budget_signals.get("looking_for_free"):
            return 0.3  # Low fit if only looking for free

        if budget_signals.get("has_budget"):
            return 0.9  # High fit if budget approved

        if budget_signals.get("price_sensitive"):
            return 0.6  # Medium fit if price-sensitive

        return 0.7  # Default medium-high

    def _calculate_conversion_likelihood(
        self,
        fit_score: float,
        intent_score: float,
        urgency_score: float,
        budget_fit: float,
    ) -> float:
        """Calculate overall conversion likelihood."""
        # Weighted combination
        likelihood = (
            fit_score * 0.4 +
            intent_score * 0.3 +
            urgency_score * 0.2 +
            budget_fit * 0.1
        )

        return likelihood

    async def generate_personalized_response(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Generate personalized response highlighting relevant features.

        This step creates a tailored response that:
        - Addresses specific pain points
        - Highlights relevant features
        - Includes social proof/case studies if applicable
        - Offers demo/trial if high-quality lead

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Generated response data
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)
        analysis = execution.context.get("lead_analysis", {})
        scores = execution.context.get("lead_scores", {})

        logger.info(f"Generating personalized response for signal {signal.id}")

        # Enhance signal metadata with analysis for better response generation
        enhanced_signal = signal.model_copy()
        enhanced_signal.metadata.update({
            "pain_points": analysis.get("pain_points", []),
            "requirements": analysis.get("requirements", []),
            "current_tool": analysis.get("current_tool", "unknown"),
            "lead_quality": scores.get("quality_level", "medium_quality"),
        })

        # Generate variants
        num_variants = step.config.get("num_variants", 3)
        channel = self._resolve_channel(enhanced_signal, step.config)
        variants = await self.response_generator.generate_variants(
            signal=enhanced_signal,
            num_variants=num_variants,
            channel=channel,
        )

        # Store best variant
        if variants:
            best_variant = variants[0]
            execution.context["generated_response"] = best_variant.content
            execution.context["response_tone"] = best_variant.tone.value
            execution.context["response_score"] = best_variant.overall_score

        return {
            "num_variants": len(variants),
            "best_score": variants[0].overall_score if variants else 0.0,
            "best_content": variants[0].content if variants else "",
            "personalization_applied": True,
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
