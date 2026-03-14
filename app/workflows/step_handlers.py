"""Step handlers for workflow execution.

This module implements the actual logic for each workflow step type.
Each handler is an async function that takes a step and execution context
and returns a result dictionary.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from app.core.signal_models import ActionableSignal, ActionType, SignalStatus, SignalType
from app.intelligence.response_generator import ResponseGenerator
from app.intelligence.response_playbook import ResponseChannel
from app.workflows.workflow_models import WorkflowExecution, WorkflowStep

logger = logging.getLogger(__name__)


class WorkflowStepHandlers:
    """Collection of step handlers for workflow execution."""

    def __init__(
        self,
        response_generator: ResponseGenerator,
    ):
        """Initialize step handlers.

        Args:
            response_generator: Response generator for GENERATE steps
        """
        self.response_generator = response_generator

        # Initialize specialized workflow handlers
        from app.workflows.alternative_seeker_workflow import AlternativeSeekerWorkflow
        from app.workflows.competitor_intelligence_workflow import CompetitorIntelligenceWorkflow
        from app.workflows.churn_prevention_workflow import ChurnPreventionWorkflow

        self.alternative_seeker = AlternativeSeekerWorkflow(response_generator)
        self.competitor_intelligence = CompetitorIntelligenceWorkflow(response_generator)
        self.churn_prevention = ChurnPreventionWorkflow(response_generator)

    async def handle_analyze(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle ANALYZE step - extract information from signal.

        This method routes to specialized workflow handlers based on
        the signal type and step ID for advanced analysis.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Analysis results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        logger.info(f"Analyzing signal {signal.id} (step: {step.id})")

        # Route to specialized handlers based on step ID and signal type
        if step.id == "analyze_intent" and signal.signal_type == SignalType.LEAD_OPPORTUNITY:
            # Use Alternative Seeker workflow
            return await self.alternative_seeker.analyze_lead_intent(step, execution)

        elif step.id == "analyze_complaint" and signal.signal_type == SignalType.COMPETITOR_WEAKNESS:
            # Use Competitor Intelligence workflow
            return await self.competitor_intelligence.analyze_competitor_complaint(step, execution)

        # Default generic analysis
        config = step.config
        extract_fields = config.get("extract_fields", [])

        logger.info(f"Using generic analysis for fields: {extract_fields}")

        # Extract requested fields from signal metadata
        analysis = {}

        for field in extract_fields:
            if field == "pain_points":
                # Extract pain points from description
                analysis["pain_points"] = self._extract_pain_points(signal)
            elif field == "current_tool":
                # Extract current tool from metadata
                analysis["current_tool"] = signal.metadata.get("competitor_name", "unknown")
            elif field == "requirements":
                # Extract requirements from description
                analysis["requirements"] = self._extract_requirements(signal)
            elif field == "competitor_name":
                analysis["competitor_name"] = signal.metadata.get("competitor_name", "unknown")
            elif field == "complaint_type":
                analysis["complaint_type"] = self._classify_complaint(signal)
            elif field == "severity":
                analysis["severity"] = self._assess_severity(signal)
            else:
                # Generic field extraction from metadata
                analysis[field] = signal.metadata.get(field, None)

        # Store analysis in context for next steps
        execution.context["analysis"] = analysis

        return analysis

    async def handle_score(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle SCORE step - score/evaluate something.

        This method routes to specialized workflow handlers for
        advanced scoring logic.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Scoring results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        logger.info(f"Scoring signal {signal.id} (step: {step.id})")

        # Route to specialized handlers
        if step.id == "qualify_lead" and signal.signal_type == SignalType.LEAD_OPPORTUNITY:
            # Use Alternative Seeker workflow
            return await self.alternative_seeker.qualify_lead(step, execution)

        elif step.id == "assess_severity" and signal.signal_type == SignalType.CHURN_RISK:
            # Use Churn Prevention workflow
            return await self.churn_prevention.assess_churn_risk(step, execution)

        # Default generic scoring
        analysis = execution.context.get("analysis", {})
        config = step.config
        scoring_criteria = config.get("scoring_criteria", [])
        min_threshold = config.get("min_threshold", 0.5)

        logger.info(f"Using generic scoring with criteria: {scoring_criteria}")

        scores = {}

        # Calculate scores based on criteria
        if "fit_score" in scoring_criteria:
            scores["fit_score"] = self._calculate_fit_score(signal, analysis)

        if "intent_score" in scoring_criteria:
            scores["intent_score"] = signal.confidence_score

        if "urgency_score" in scoring_criteria:
            scores["urgency_score"] = signal.urgency_score

        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        scores["overall_score"] = overall_score

        # Determine quality level
        if overall_score >= 0.8:
            quality = "high_quality"
        elif overall_score >= min_threshold:
            quality = "medium_quality"
        else:
            quality = "low_quality"

        scores["quality_level"] = quality

        # Store in context
        execution.context["scores"] = scores
        execution.context["lead_quality"] = quality

        return scores

    async def handle_decide(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle DECIDE step - make a decision based on context.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Decision results
        """
        config = step.config
        decision_tree = config.get("decision_tree", {})

        # Get decision input (e.g., lead quality)
        lead_quality = execution.context.get("lead_quality", "low_quality")

        # Make decision
        decision = decision_tree.get(lead_quality, "monitor_only")

        logger.info(f"Decision for {lead_quality}: {decision}")

        # Store decision in context
        execution.context["decision"] = decision

        return {"decision": decision, "input": lead_quality}

    async def handle_generate(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle GENERATE step - generate content using LLM.

        This method routes to specialized workflow handlers for
        advanced content generation with personalization.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Generated content
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        logger.info(f"Generating content for signal {signal.id} (step: {step.id})")

        # Route to specialized handlers
        if step.id == "generate_response" and signal.signal_type == SignalType.LEAD_OPPORTUNITY:
            # Use Alternative Seeker workflow
            return await self.alternative_seeker.generate_personalized_response(step, execution)

        elif step.id == "generate_content" and signal.signal_type == SignalType.COMPETITOR_WEAKNESS:
            # Use Competitor Intelligence workflow
            return await self.competitor_intelligence.generate_positioning_content(step, execution)

        elif step.id == "identify_positioning" and signal.signal_type == SignalType.COMPETITOR_WEAKNESS:
            # This is actually an ANALYZE step but called from GENERATE workflow
            return await self.competitor_intelligence.identify_positioning_angle(step, execution)

        elif step.id == "draft_retention_outreach" and signal.signal_type == SignalType.CHURN_RISK:
            # Use Churn Prevention workflow
            return await self.churn_prevention.generate_retention_content(step, execution)

        # Default generic generation
        config = step.config
        num_variants = config.get("num_variants", 3)
        channel = self._resolve_channel(signal, step.config)

        logger.info(f"Using generic generation: {num_variants} variants on {channel.value}")

        # Generate response variants
        variants = await self.response_generator.generate_variants(
            signal=signal,
            num_variants=num_variants,
            channel=channel,
        )

        # Store best variant in context
        if variants:
            best_variant = variants[0]  # Already sorted by score
            execution.context["generated_response"] = best_variant.content
            execution.context["response_tone"] = best_variant.tone.value
            execution.context["response_score"] = best_variant.overall_score

        return {
            "num_variants": len(variants),
            "best_score": variants[0].overall_score if variants else 0.0,
            "best_content": variants[0].content if variants else "",
        }

    async def handle_notify(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle NOTIFY step - send notifications.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Notification results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        config = step.config
        channels = config.get("notification_channels", ["slack"])
        priority = config.get("priority", "normal")
        now_iso = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Sending {priority} priority notification for signal {signal.id} "
            f"via channels: {channels}"
        )

        # Build a structured notification payload for each channel.
        # In production these would be forwarded to Slack/email/webhook transports.
        notification_payload = {
            "signal_id": str(signal.id),
            "signal_type": signal.signal_type.value,
            "title": signal.title,
            "description": signal.description[:300],
            "action_score": signal.action_score,
            "urgency_score": signal.urgency_score,
            "recommended_action": signal.recommended_action.value,
            "source_url": signal.source_url,
            "priority": priority,
        }

        notifications_sent = []
        for channel_name in channels:
            # Emit a structured log entry that external log-shippers (e.g. Datadog,
            # Splunk) can route to the real notification transport.
            logger.info(
                "NOTIFICATION | channel=%s priority=%s signal_id=%s title=%r",
                channel_name,
                priority,
                signal.id,
                signal.title,
                extra={"notification": notification_payload, "channel": channel_name},
            )
            notifications_sent.append({
                "channel": channel_name,
                "status": "dispatched",
                "timestamp": now_iso,
                "payload_keys": list(notification_payload.keys()),
            })

        # Persist in context so downstream TRACK steps can reference it
        execution.context.setdefault("notifications", []).extend(notifications_sent)

        return {
            "notifications_sent": len(notifications_sent),
            "channels": channels,
            "priority": priority,
            "dispatched_at": now_iso,
        }

    async def handle_execute(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle EXECUTE step - execute an action.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Execution results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        action_type = signal.recommended_action
        channel = self._resolve_channel(signal, step.config)
        now_iso = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Executing action {action_type.value} for signal {signal.id} "
            f"on channel {channel.value}"
        )

        # Pull the generated draft from context (set by a preceding GENERATE step)
        generated_response = execution.context.get("generated_response", "")
        response_score = execution.context.get("response_score", 0.0)

        # Build execution record capturing intent, channel, and content snapshot.
        # In production, this dispatches to platform connector APIs
        # (e.g. RedditConnector.post_comment, TwitterConnector.send_dm).
        execution_record: Dict[str, Any] = {
            "signal_id": str(signal.id),
            "action_type": action_type.value,
            "channel": channel.value,
            "content_preview": generated_response[:200] if generated_response else None,
            "response_score": response_score,
            "executed_at": now_iso,
            "status": "queued",  # 'queued' until connector confirms delivery
        }

        # Dispatch routing:
        # REPLY_PUBLIC / DM_OUTREACH → platform connector (future integration)
        # CREATE_CONTENT → content pipeline (future integration)
        # INTERNAL_ALERT / ESCALATE → already handled by NOTIFY step
        # MONITOR → no action needed
        if action_type in (ActionType.REPLY_PUBLIC, ActionType.DM_OUTREACH):
            logger.info(
                "ACTION_DISPATCH | type=%s channel=%s signal=%s content_len=%d",
                action_type.value, channel.value, signal.id,
                len(generated_response),
            )
            execution_record["status"] = "dispatched"
        elif action_type == ActionType.CREATE_CONTENT:
            logger.info(
                "CONTENT_DISPATCH | channel=%s signal=%s", channel.value, signal.id
            )
            execution_record["status"] = "dispatched"
        else:
            execution_record["status"] = "acknowledged"

        # Mark signal as acted in context
        execution.context["signal_status"] = SignalStatus.ACTED.value
        execution.context["last_action"] = execution_record
        execution.context.setdefault("execution_history", []).append(execution_record)

        return {
            "action_executed": True,
            "action_type": action_type.value,
            "channel": channel.value,
            "status": execution_record["status"],
            "signal_id": str(signal.id),
            "executed_at": now_iso,
        }

    async def handle_track(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle TRACK step - track metrics and outcomes.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Tracking results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        config = step.config
        metrics = config.get("metrics", [])
        now_iso = datetime.now(timezone.utc).isoformat()

        logger.info(f"Tracking metrics for signal {signal.id}: {metrics}")

        # Collect values for all requested metric names from signal + context
        metric_values: Dict[str, Any] = {
            "signal_id": str(signal.id),
            "signal_type": signal.signal_type.value,
            "tracked_at": now_iso,
        }

        # Built-in signal scores always captured
        score_map = {
            "urgency_score": signal.urgency_score,
            "impact_score": signal.impact_score,
            "confidence_score": signal.confidence_score,
            "action_score": signal.action_score,
        }
        for metric in metrics:
            if metric in score_map:
                metric_values[metric] = score_map[metric]
            elif metric in execution.context:
                metric_values[metric] = execution.context[metric]

        # Always capture response quality score if available (set by GENERATE step)
        if "response_score" in execution.context:
            metric_values["response_score"] = execution.context["response_score"]

        # Persist tracking record in context for the learning loop
        execution.context.setdefault("tracking_records", []).append(metric_values)

        logger.info(
            "METRICS_TRACKED | signal=%s metrics=%s",
            signal.id,
            list(metric_values.keys()),
        )

        return {
            "signal_id": str(signal.id),
            "metrics_tracked": metrics,
            "metric_values": metric_values,
            "tracking_started": True,
            "tracked_at": now_iso,
        }

    async def handle_wait(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle WAIT step - wait for external input or time.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Wait results
        """
        wait_seconds = step.config.get("wait_seconds", 0)
        wait_reason = step.config.get("wait_reason", "configured delay")
        # Hard cap: prevent runaway sleeps in automated pipelines (max 30 s).
        # For longer waits (approval gates, SLA windows), the orchestrator
        # should persist state and resume via an external trigger.
        max_sleep = 30
        actual_sleep = min(float(wait_seconds), max_sleep)

        if actual_sleep > 0:
            logger.info(
                f"Waiting {actual_sleep:.1f}s for signal {execution.execution_id} "
                f"(reason: {wait_reason})"
            )
            await asyncio.sleep(actual_sleep)
        else:
            logger.debug(f"Wait step: no sleep required (wait_seconds={wait_seconds})")

        return {
            "waited": True,
            "wait_seconds": wait_seconds,
            "actual_sleep_seconds": actual_sleep,
            "wait_reason": wait_reason,
            "capped": wait_seconds > max_sleep,
        }

    # Helper methods for analysis

    def _extract_pain_points(self, signal: ActionableSignal) -> list:
        """Extract pain points from signal description."""
        # Simple extraction - in production, use NLP
        description = signal.description.lower()
        pain_points = []

        pain_keywords = ["slow", "expensive", "complicated", "broken", "missing", "frustrated"]
        for keyword in pain_keywords:
            if keyword in description:
                pain_points.append(keyword)

        return pain_points or ["general_dissatisfaction"]

    def _extract_requirements(self, signal: ActionableSignal) -> list:
        """Extract requirements from signal description."""
        # Simple extraction - in production, use NLP
        description = signal.description.lower()
        requirements = []

        requirement_keywords = ["need", "want", "require", "looking for", "must have"]
        for keyword in requirement_keywords:
            if keyword in description:
                requirements.append(keyword)

        return requirements or ["general_requirements"]

    def _classify_complaint(self, signal: ActionableSignal) -> str:
        """Classify type of complaint."""
        description = signal.description.lower()

        if any(word in description for word in ["price", "expensive", "cost"]):
            return "pricing"
        elif any(word in description for word in ["slow", "performance", "lag"]):
            return "performance"
        elif any(word in description for word in ["support", "help", "response"]):
            return "support"
        elif any(word in description for word in ["bug", "broken", "error"]):
            return "reliability"
        else:
            return "general"

    def _assess_severity(self, signal: ActionableSignal) -> str:
        """Assess severity of issue."""
        # Use signal scores to determine severity
        if signal.urgency_score >= 0.8 and signal.impact_score >= 0.7:
            return "critical"
        elif signal.urgency_score >= 0.6 or signal.impact_score >= 0.6:
            return "high"
        elif signal.urgency_score >= 0.4 or signal.impact_score >= 0.4:
            return "medium"
        else:
            return "low"

    def _calculate_fit_score(self, signal: ActionableSignal, analysis: Dict[str, Any]) -> float:
        """Calculate product fit score for lead."""
        # Simple scoring - in production, use ML model
        score = 0.5  # Base score

        # Boost score if pain points match our strengths
        pain_points = analysis.get("pain_points", [])
        if "expensive" in pain_points:
            score += 0.2  # We're more affordable
        if "complicated" in pain_points:
            score += 0.2  # We're easier to use

        # Cap at 1.0
        return min(score, 1.0)

    def _resolve_channel(
        self,
        signal: ActionableSignal,
        config: Dict[str, Any],
    ) -> ResponseChannel:
        """Resolve the ResponseChannel for a signal.

        Priority order:
        1. Explicit override in step config ("channel" key)
        2. Signal's suggested_channel field
        3. Fallback to REDDIT

        Args:
            signal: The actionable signal
            config: Step configuration dict

        Returns:
            Resolved ResponseChannel enum value
        """
        # 1. Explicit step-level override
        config_channel = config.get("channel")
        if config_channel:
            try:
                return ResponseChannel(config_channel)
            except ValueError:
                logger.warning(f"Unknown channel in step config: {config_channel!r}; using signal default")

        # 2. Signal's suggested channel
        if signal.suggested_channel:
            try:
                return ResponseChannel(signal.suggested_channel.lower())
            except ValueError:
                logger.warning(
                    f"Cannot map suggested_channel {signal.suggested_channel!r} to ResponseChannel; "
                    "falling back to REDDIT"
                )

        # 3. Safe fallback
        return ResponseChannel.REDDIT
