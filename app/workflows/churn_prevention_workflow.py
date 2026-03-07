"""Churn Prevention workflow for customer retention.

This workflow handles churn risk signals by:
- Assessing churn risk severity based on multiple factors
- Analyzing customer sentiment and complaint history
- Selecting optimal retention strategy
- Generating personalized retention content
- Escalating high-risk cases to customer success team

Key Features:
- Multi-factor risk assessment (sentiment, tenure, value, history)
- 4-level severity scoring (low, medium, high, critical)
- Strategic retention approach selection
- Empathetic, solution-focused content generation
- Automatic escalation for high-risk accounts
"""

import logging
from typing import Any, Dict, List

from app.core.signal_models import ActionableSignal
from app.intelligence.response_generator import ResponseGenerator
from app.intelligence.response_playbook import ResponseChannel
from app.workflows.workflow_models import WorkflowExecution, WorkflowStep

logger = logging.getLogger(__name__)


class ChurnPreventionWorkflow:
    """Specialized workflow for churn prevention and customer retention."""

    def __init__(self, response_generator: ResponseGenerator):
        """Initialize churn prevention workflow.

        Args:
            response_generator: Response generator for content creation
        """
        self.response_generator = response_generator

    async def assess_churn_risk(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Assess churn risk severity and urgency.

        This method evaluates multiple risk factors:
        - Sentiment analysis (negative emotion intensity)
        - Account value (revenue impact)
        - Customer tenure (relationship length)
        - Complaint history (recurring issues)
        - Escalation indicators (public complaints, legal threats)

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Risk assessment with severity level and risk factors
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        logger.info(f"Assessing churn risk for signal {signal.id}")

        # Extract risk factors
        sentiment = self._analyze_churn_sentiment(signal)
        account_value = self._estimate_account_value(signal)
        tenure = self._estimate_tenure(signal)
        complaint_history = self._analyze_complaint_history(signal)
        escalation_risk = self._assess_escalation_risk(signal)

        # Calculate severity score (0-1)
        severity_score = self._calculate_severity_score(
            sentiment=sentiment,
            account_value=account_value,
            tenure=tenure,
            complaint_history=complaint_history,
            escalation_risk=escalation_risk,
        )

        # Determine severity level
        severity_level = self._determine_severity_level(severity_score)

        # Identify primary churn drivers
        churn_drivers = self._identify_churn_drivers(signal, sentiment)

        # Assess urgency
        urgency = self._assess_urgency(signal, escalation_risk)

        result = {
            "severity_score": severity_score,
            "severity_level": severity_level,
            "sentiment": sentiment,
            "account_value": account_value,
            "tenure": tenure,
            "complaint_history": complaint_history,
            "escalation_risk": escalation_risk,
            "churn_drivers": churn_drivers,
            "urgency": urgency,
            "requires_escalation": severity_level in ["high", "critical"],
        }

        # Store in context
        execution.context["churn_assessment"] = result
        execution.context["severity"] = severity_level

        logger.info(
            f"Churn risk assessed: {severity_level} severity "
            f"(score: {severity_score:.2f}, drivers: {churn_drivers})"
        )

        return result

    def _analyze_churn_sentiment(self, signal: ActionableSignal) -> Dict[str, Any]:
        """Analyze sentiment specific to churn risk.

        Focuses on:
        - Frustration and anger
        - Disappointment and regret
        - Threat indicators (cancellation, switching)
        """
        text = f"{signal.title} {signal.description}".lower()

        # Churn-specific negative indicators
        frustration_words = [
            "frustrated", "angry", "furious", "fed up", "sick of",
            "tired of", "done with", "had enough", "can't take",
        ]

        disappointment_words = [
            "disappointed", "let down", "expected better", "regret",
            "waste", "mistake", "should have", "wish i",
        ]

        threat_words = [
            "cancel", "canceling", "cancelling", "switch", "switching",
            "leaving", "moving to", "alternative", "competitor",
            "refund", "money back", "unsubscribe",
        ]

        # Count occurrences
        frustration_count = sum(1 for word in frustration_words if word in text)
        disappointment_count = sum(1 for word in disappointment_words if word in text)
        threat_count = sum(1 for word in threat_words if word in text)

        # Calculate intensity (0-1)
        total_negative = frustration_count + disappointment_count + threat_count
        intensity = min(total_negative / 5.0, 1.0)

        # Determine dominant emotion
        if threat_count > 0:
            dominant_emotion = "threat"
        elif frustration_count >= disappointment_count:
            dominant_emotion = "frustration"
        else:
            dominant_emotion = "disappointment"

        return {
            "intensity": intensity,
            "dominant_emotion": dominant_emotion,
            "frustration_count": frustration_count,
            "disappointment_count": disappointment_count,
            "threat_count": threat_count,
            "has_explicit_threat": threat_count > 0,
        }

    def _estimate_account_value(self, signal: ActionableSignal) -> str:
        """Estimate account value tier from signal metadata.

        Tiers: enterprise, business, professional, basic
        """
        # Check metadata for account info
        metadata = signal.metadata
        account_type = metadata.get("account_type", "").lower()
        plan_tier = metadata.get("plan_tier", "").lower()
        monthly_value = metadata.get("monthly_value", 0)

        # Determine value tier
        if account_type == "enterprise" or monthly_value >= 1000:
            return "enterprise"
        elif account_type == "business" or monthly_value >= 100:
            return "business"
        elif plan_tier in ["professional", "pro"] or monthly_value >= 30:
            return "professional"
        else:
            return "basic"

    def _estimate_tenure(self, signal: ActionableSignal) -> str:
        """Estimate customer tenure from signal metadata.

        Tiers: new (<3 months), established (3-12 months), loyal (>12 months)
        """
        metadata = signal.metadata
        tenure_months = metadata.get("tenure_months", 0)

        # Determine tenure tier
        if tenure_months >= 12:
            return "loyal"
        elif tenure_months >= 3:
            return "established"
        else:
            return "new"

    def _analyze_complaint_history(self, signal: ActionableSignal) -> Dict[str, Any]:
        """Analyze complaint history from signal metadata."""
        metadata = signal.metadata
        previous_complaints = metadata.get("previous_complaints", 0)
        recent_support_tickets = metadata.get("recent_support_tickets", 0)
        unresolved_issues = metadata.get("unresolved_issues", 0)

        # Determine if this is a recurring issue
        is_recurring = previous_complaints > 0 or unresolved_issues > 0

        # Calculate complaint frequency
        if previous_complaints >= 3:
            frequency = "high"
        elif previous_complaints >= 1:
            frequency = "medium"
        else:
            frequency = "low"

        return {
            "previous_complaints": previous_complaints,
            "recent_support_tickets": recent_support_tickets,
            "unresolved_issues": unresolved_issues,
            "is_recurring": is_recurring,
            "frequency": frequency,
        }

    def _assess_escalation_risk(self, signal: ActionableSignal) -> Dict[str, Any]:
        """Assess risk of public escalation or legal action."""
        text = f"{signal.title} {signal.description}".lower()

        # Public escalation indicators
        public_indicators = [
            "twitter", "reddit", "review", "public", "everyone",
            "warning others", "tell people", "spread the word",
        ]

        # Legal threat indicators
        legal_indicators = [
            "lawyer", "attorney", "legal", "sue", "lawsuit",
            "court", "consumer protection", "ftc", "bbb",
        ]

        # Check for indicators
        has_public_threat = any(indicator in text for indicator in public_indicators)
        has_legal_threat = any(indicator in text for indicator in legal_indicators)

        # Check if already public
        is_public = signal.source_platform in ["twitter", "reddit", "facebook"]

        # Determine escalation level
        if has_legal_threat:
            level = "critical"
        elif has_public_threat or is_public:
            level = "high"
        else:
            level = "low"

        return {
            "level": level,
            "has_public_threat": has_public_threat,
            "has_legal_threat": has_legal_threat,
            "is_already_public": is_public,
            "platform": signal.source_platform,
        }

    def _calculate_severity_score(
        self,
        sentiment: Dict[str, Any],
        account_value: str,
        tenure: str,
        complaint_history: Dict[str, Any],
        escalation_risk: Dict[str, Any],
    ) -> float:
        """Calculate overall severity score (0-1)."""
        score = 0.0

        # Sentiment contribution (0-0.3)
        sentiment_intensity = sentiment.get("intensity", 0)
        has_threat = sentiment.get("has_explicit_threat", False)
        score += sentiment_intensity * 0.25
        if has_threat:
            score += 0.05

        # Account value contribution (0-0.25)
        value_scores = {
            "enterprise": 0.25,
            "business": 0.18,
            "professional": 0.10,
            "basic": 0.05,
        }
        score += value_scores.get(account_value, 0.05)

        # Tenure contribution (0-0.2)
        tenure_scores = {
            "loyal": 0.20,  # Losing loyal customers is critical
            "established": 0.12,
            "new": 0.05,
        }
        score += tenure_scores.get(tenure, 0.05)

        # Complaint history contribution (0-0.15)
        if complaint_history.get("is_recurring"):
            score += 0.10
        if complaint_history.get("unresolved_issues", 0) > 0:
            score += 0.05

        # Escalation risk contribution (0-0.1)
        escalation_scores = {
            "critical": 0.10,
            "high": 0.07,
            "low": 0.02,
        }
        score += escalation_scores.get(escalation_risk.get("level", "low"), 0.02)

        return min(score, 1.0)

    def _determine_severity_level(self, severity_score: float) -> str:
        """Determine severity level from score."""
        if severity_score >= 0.75:
            return "critical"
        elif severity_score >= 0.55:
            return "high"
        elif severity_score >= 0.35:
            return "medium"
        else:
            return "low"

    def _identify_churn_drivers(
        self,
        signal: ActionableSignal,
        sentiment: Dict[str, Any],
    ) -> List[str]:
        """Identify primary drivers of churn risk."""
        text = f"{signal.title} {signal.description}".lower()
        drivers = []

        # Product/feature issues
        if any(word in text for word in ["bug", "broken", "doesn't work", "not working", "error"]):
            drivers.append("product_quality")

        # Pricing issues
        if any(word in text for word in ["expensive", "price", "cost", "too much", "overpriced"]):
            drivers.append("pricing")

        # Support issues
        if any(word in text for word in ["support", "help", "response", "ticket", "ignored"]):
            drivers.append("support_quality")

        # Performance issues
        if any(word in text for word in ["slow", "lag", "performance", "speed", "timeout"]):
            drivers.append("performance")

        # Missing features
        if any(word in text for word in ["missing", "need", "lack", "doesn't have", "no way to"]):
            drivers.append("missing_features")

        # Competitor comparison
        if any(word in text for word in ["competitor", "alternative", "other tool", "switch to"]):
            drivers.append("competitor_offering")

        # If no specific drivers found, use sentiment
        if not drivers:
            if sentiment.get("has_explicit_threat"):
                drivers.append("general_dissatisfaction")
            else:
                drivers.append("unclear")

        return drivers

    def _assess_urgency(
        self,
        signal: ActionableSignal,
        escalation_risk: Dict[str, Any],
    ) -> str:
        """Assess urgency of response needed."""
        text = f"{signal.title} {signal.description}".lower()

        # Critical urgency indicators
        if escalation_risk.get("has_legal_threat"):
            return "critical"

        # High urgency indicators
        high_urgency_words = [
            "immediately", "asap", "urgent", "right now", "today",
            "canceling today", "canceling now", "already canceled",
        ]

        if any(word in text for word in high_urgency_words):
            return "high"

        # Medium urgency (explicit threat but no timeline)
        if escalation_risk.get("has_public_threat"):
            return "medium"

        # Use signal urgency score
        if signal.urgency_score >= 0.7:
            return "high"
        elif signal.urgency_score >= 0.4:
            return "medium"
        else:
            return "low"

    async def select_retention_strategy(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Select optimal retention strategy based on risk assessment.

        Strategies:
        - immediate_intervention: Critical cases requiring immediate action
        - empathetic_resolution: High-value customers with solvable issues
        - value_reinforcement: Customers who may not see full value
        - win_back_offer: Price-sensitive customers
        - feedback_collection: Low-risk cases to gather insights

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Selected retention strategy with tactics
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)
        assessment = execution.context.get("churn_assessment", {})

        logger.info(f"Selecting retention strategy for signal {signal.id}")

        severity = assessment.get("severity_level", "medium")
        account_value = assessment.get("account_value", "basic")
        churn_drivers = assessment.get("churn_drivers", [])
        urgency = assessment.get("urgency", "medium")

        # Select strategy based on assessment
        strategy = self._determine_retention_strategy(
            severity=severity,
            account_value=account_value,
            churn_drivers=churn_drivers,
            urgency=urgency,
        )

        # Select tactics
        tactics = self._select_retention_tactics(
            strategy=strategy,
            churn_drivers=churn_drivers,
            account_value=account_value,
        )

        # Determine response approach
        approach = self._determine_response_approach(strategy, urgency)

        result = {
            "strategy": strategy,
            "tactics": tactics,
            "approach": approach,
            "priority": self._calculate_priority(severity, account_value, urgency),
        }

        # Store in context
        execution.context["retention_strategy"] = result

        logger.info(
            f"Retention strategy selected: {strategy} "
            f"with {len(tactics)} tactics (approach: {approach})"
        )

        return result

    def _determine_retention_strategy(
        self,
        severity: str,
        account_value: str,
        churn_drivers: List[str],
        urgency: str,
    ) -> str:
        """Determine the primary retention strategy."""
        # Critical cases always get immediate intervention
        if severity == "critical" or urgency == "critical":
            return "immediate_intervention"

        # High-value accounts with solvable issues
        if account_value in ["enterprise", "business"] and severity in ["high", "medium"]:
            if any(driver in churn_drivers for driver in ["product_quality", "support_quality", "performance"]):
                return "empathetic_resolution"

        # Pricing-driven churn
        if "pricing" in churn_drivers:
            return "win_back_offer"

        # Missing features or competitor comparison
        if any(driver in churn_drivers for driver in ["missing_features", "competitor_offering"]):
            return "value_reinforcement"

        # Low severity - gather feedback
        if severity == "low":
            return "feedback_collection"

        # Default to empathetic resolution
        return "empathetic_resolution"

    def _select_retention_tactics(
        self,
        strategy: str,
        churn_drivers: List[str],
        account_value: str,
    ) -> List[str]:
        """Select specific tactics for the retention strategy."""
        tactics = []

        # Strategy-specific tactics
        if strategy == "immediate_intervention":
            tactics.extend([
                "executive_outreach",
                "dedicated_support",
                "immediate_resolution",
                "compensation_offer",
            ])
        elif strategy == "empathetic_resolution":
            tactics.extend([
                "acknowledge_issue",
                "provide_solution",
                "timeline_commitment",
                "follow_up_plan",
            ])
        elif strategy == "value_reinforcement":
            tactics.extend([
                "highlight_features",
                "usage_tips",
                "success_stories",
                "roadmap_preview",
            ])
        elif strategy == "win_back_offer":
            tactics.extend([
                "discount_offer",
                "plan_flexibility",
                "value_demonstration",
            ])
        elif strategy == "feedback_collection":
            tactics.extend([
                "ask_feedback",
                "improvement_commitment",
                "stay_connected",
            ])

        # Driver-specific tactics
        if "product_quality" in churn_drivers:
            tactics.append("bug_fix_priority")
        if "support_quality" in churn_drivers:
            tactics.append("support_escalation")
        if "performance" in churn_drivers:
            tactics.append("performance_optimization")

        # Account value-specific tactics
        if account_value in ["enterprise", "business"]:
            tactics.append("account_manager_assignment")

        return list(set(tactics))  # Remove duplicates

    def _determine_response_approach(self, strategy: str, urgency: str) -> str:
        """Determine the communication approach."""
        if strategy == "immediate_intervention":
            return "urgent_personal"
        elif urgency in ["critical", "high"]:
            return "prompt_empathetic"
        elif strategy == "feedback_collection":
            return "casual_helpful"
        else:
            return "professional_supportive"

    def _calculate_priority(
        self,
        severity: str,
        account_value: str,
        urgency: str,
    ) -> str:
        """Calculate overall priority for CS team."""
        # Critical severity or urgency = P0
        if severity == "critical" or urgency == "critical":
            return "P0"

        # High severity + high value = P1
        if severity == "high" and account_value in ["enterprise", "business"]:
            return "P1"

        # High severity or high urgency = P2
        if severity == "high" or urgency == "high":
            return "P2"

        # Medium severity = P3
        if severity == "medium":
            return "P3"

        # Low severity = P4
        return "P4"

    async def generate_retention_content(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Generate personalized retention content.

        Creates empathetic, solution-focused content that:
        - Acknowledges the customer's frustration
        - Provides specific solutions or next steps
        - Reinforces value and commitment
        - Offers escalation path if needed

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Generated retention content
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)
        assessment = execution.context.get("churn_assessment", {})
        strategy_data = execution.context.get("retention_strategy", {})

        logger.info(f"Generating retention content for signal {signal.id}")

        # Enhance signal metadata with retention context
        enhanced_signal = signal.model_copy()
        enhanced_signal.metadata.update({
            "severity_level": assessment.get("severity_level", "medium"),
            "churn_drivers": assessment.get("churn_drivers", []),
            "retention_strategy": strategy_data.get("strategy", "empathetic_resolution"),
            "retention_tactics": strategy_data.get("tactics", []),
            "response_approach": strategy_data.get("approach", "professional_supportive"),
            "account_value": assessment.get("account_value", "basic"),
        })

        # Generate content variants
        num_variants = step.config.get("num_variants", 2)

        variants = await self.response_generator.generate_variants(
            signal=enhanced_signal,
            num_variants=num_variants,
            channel=ResponseChannel.EMAIL,  # Retention typically via email
        )

        generated_content = {}

        if variants:
            generated_content["primary_response"] = {
                "content": variants[0].content,
                "tone": variants[0].tone.value,
                "score": variants[0].overall_score,
            }

            # Store all variants for A/B testing
            generated_content["variants"] = [
                {
                    "content": v.content,
                    "tone": v.tone.value,
                    "score": v.overall_score,
                }
                for v in variants
            ]

        # Store in context
        execution.context["generated_content"] = generated_content

        logger.info(
            f"Retention content generated: {len(variants)} variants "
            f"(best score: {variants[0].overall_score if variants else 0:.2f})"
        )

        return {
            "num_variants": len(variants),
            "best_score": variants[0].overall_score if variants else 0.0,
            "best_content": variants[0].content if variants else "",
            "retention_applied": True,
            "strategy": strategy_data.get("strategy", "empathetic_resolution"),
        }
