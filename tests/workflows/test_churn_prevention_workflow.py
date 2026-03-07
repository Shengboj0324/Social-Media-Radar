"""Tests for Churn Prevention workflow."""

import pytest
from uuid import uuid4

from app.core.signal_models import (
    ActionableSignal,
    SignalType,
    ActionType,
    ResponseTone,
)
from app.intelligence.response_generator import ResponseGenerator
from app.workflows.churn_prevention_workflow import ChurnPreventionWorkflow
from app.workflows.workflow_models import WorkflowExecution, WorkflowStep, StepType, WorkflowType


@pytest.fixture
def response_generator():
    """Create response generator instance."""
    return ResponseGenerator()


@pytest.fixture
def workflow(response_generator):
    """Create Churn Prevention workflow instance."""
    return ChurnPreventionWorkflow(response_generator)


@pytest.fixture
def churn_signal_critical():
    """Create a critical churn risk signal."""
    return ActionableSignal(
        user_id=uuid4(),
        signal_type=SignalType.CHURN_RISK,
        source_platform="email",
        source_url="https://example.com/support/ticket/456",
        source_author="enterprise_customer",
        title="Canceling our enterprise account - legal action pending",
        description=(
            "I'm the VP of Engineering at a Fortune 500 company and we're canceling our "
            "enterprise account ($5000/month) effective immediately. We've had 3 major outages "
            "in the past month with zero response from your support team. Our legal team is "
            "reviewing our contract for breach of SLA. This is unacceptable for a mission-critical "
            "system. We're switching to CompetitorX next week."
        ),
        context="Critical churn risk from high-value enterprise customer with legal threats",
        urgency_score=0.95,
        impact_score=0.9,
        confidence_score=0.95,
        action_score=0.9,
        recommended_action=ActionType.ESCALATE,
        suggested_channel="email",
        suggested_tone=ResponseTone.SUPPORTIVE,
        metadata={
            "account_type": "enterprise",
            "monthly_value": 5000,
            "tenure_months": 24,
            "previous_complaints": 3,
            "unresolved_issues": 2,
        },
    )


@pytest.fixture
def churn_signal_medium():
    """Create a medium churn risk signal."""
    return ActionableSignal(
        user_id=uuid4(),
        signal_type=SignalType.CHURN_RISK,
        source_platform="twitter",
        source_url="https://twitter.com/user/status/789",
        source_author="frustrated_user",
        title="Disappointed with recent changes",
        description=(
            "I've been a loyal customer for over a year but the recent price increase is "
            "really disappointing. The product is good but I'm not sure it's worth the new price. "
            "Looking at alternatives that offer similar features for less."
        ),
        context="Medium churn risk from price-sensitive customer",
        urgency_score=0.6,
        impact_score=0.5,
        confidence_score=0.7,
        action_score=0.6,
        recommended_action=ActionType.REPLY_PUBLIC,
        suggested_channel="twitter",
        suggested_tone=ResponseTone.HELPFUL,
        metadata={
            "account_type": "professional",
            "monthly_value": 49,
            "tenure_months": 14,
            "previous_complaints": 0,
        },
    )


@pytest.fixture
def execution_critical(churn_signal_critical):
    """Create workflow execution for critical churn."""
    execution = WorkflowExecution(
        workflow_id=uuid4(),
        workflow_type=WorkflowType.CHURN_PREVENTION,
        signal_id=churn_signal_critical.id,
        user_id=churn_signal_critical.user_id,
    )
    execution.context["signal"] = churn_signal_critical.model_dump()
    return execution


@pytest.fixture
def execution_medium(churn_signal_medium):
    """Create workflow execution for medium churn."""
    execution = WorkflowExecution(
        workflow_id=uuid4(),
        workflow_type=WorkflowType.CHURN_PREVENTION,
        signal_id=churn_signal_medium.id,
        user_id=churn_signal_medium.user_id,
    )
    execution.context["signal"] = churn_signal_medium.model_dump()
    return execution


class TestChurnPreventionWorkflow:
    """Test Churn Prevention workflow functionality."""

    @pytest.mark.asyncio
    async def test_assess_churn_risk_critical(self, workflow, execution_critical):
        """Test churn risk assessment for critical case."""
        step = WorkflowStep(
            id="assess_severity",
            type=StepType.SCORE,
            name="Assess Churn Risk",
            description="Test step",
            config={
                "risk_factors": ["sentiment", "account_value", "tenure", "complaint_history"],
            },
        )

        result = await workflow.assess_churn_risk(step, execution_critical)

        # Verify assessment results
        assert "severity_score" in result
        assert "severity_level" in result
        assert "sentiment" in result
        assert "account_value" in result
        assert "tenure" in result
        assert "complaint_history" in result
        assert "escalation_risk" in result
        assert "churn_drivers" in result
        assert "urgency" in result
        assert "requires_escalation" in result

        # Verify critical severity
        assert result["severity_level"] == "critical"
        assert result["severity_score"] >= 0.75
        assert result["requires_escalation"] is True

        # Verify account value
        assert result["account_value"] == "enterprise"

        # Verify tenure
        assert result["tenure"] == "loyal"

        # Verify escalation risk
        escalation = result["escalation_risk"]
        assert escalation["has_legal_threat"] is True
        assert escalation["level"] == "critical"

        # Verify urgency
        assert result["urgency"] in ["critical", "high"]

    @pytest.mark.asyncio
    async def test_assess_churn_risk_medium(self, workflow, execution_medium):
        """Test churn risk assessment for medium case."""
        step = WorkflowStep(
            id="assess_severity",
            type=StepType.SCORE,
            name="Assess Churn Risk",
            description="Test step",
        )

        result = await workflow.assess_churn_risk(step, execution_medium)

        # Verify medium severity
        assert result["severity_level"] in ["medium", "high"]
        assert 0.3 <= result["severity_score"] < 0.75

        # Verify account value
        assert result["account_value"] == "professional"

        # Verify tenure
        assert result["tenure"] == "loyal"

        # Verify churn drivers include pricing
        assert "pricing" in result["churn_drivers"]

    @pytest.mark.asyncio
    async def test_select_retention_strategy_critical(self, workflow, execution_critical, churn_signal_critical):
        """Test retention strategy selection for critical case."""
        # First assess risk
        assess_step = WorkflowStep(
            id="assess_severity",
            type=StepType.SCORE,
            name="Assess",
            description="Test",
        )
        assessment = await workflow.assess_churn_risk(assess_step, execution_critical)
        execution_critical.context["churn_assessment"] = assessment

        # Now select strategy
        strategy_step = WorkflowStep(
            id="select_strategy",
            type=StepType.DECIDE,
            name="Select Strategy",
            description="Test step",
        )

        result = await workflow.select_retention_strategy(strategy_step, execution_critical)

        # Verify strategy results
        assert "strategy" in result
        assert "tactics" in result
        assert "approach" in result
        assert "priority" in result

        # Critical cases should get immediate intervention
        assert result["strategy"] == "immediate_intervention"
        assert result["priority"] == "P0"
        assert result["approach"] == "urgent_personal"

        # Verify tactics
        tactics = result["tactics"]
        assert len(tactics) > 0
        assert "executive_outreach" in tactics or "immediate_resolution" in tactics

    @pytest.mark.asyncio
    async def test_select_retention_strategy_pricing(self, workflow, execution_medium, churn_signal_medium):
        """Test retention strategy for pricing-driven churn."""
        # Setup assessment
        assessment = {
            "severity_level": "medium",
            "account_value": "professional",
            "churn_drivers": ["pricing"],
            "urgency": "medium",
        }
        execution_medium.context["churn_assessment"] = assessment

        strategy_step = WorkflowStep(
            id="select_strategy",
            type=StepType.DECIDE,
            name="Select Strategy",
            description="Test step",
        )

        result = await workflow.select_retention_strategy(strategy_step, execution_medium)

        # Pricing-driven churn should get win-back offer
        assert result["strategy"] == "win_back_offer"
        assert "discount_offer" in result["tactics"] or "plan_flexibility" in result["tactics"]

    @pytest.mark.asyncio
    async def test_generate_retention_content(self, workflow, execution_critical):
        """Test retention content generation."""
        # Setup context
        execution_critical.context["churn_assessment"] = {
            "severity_level": "critical",
            "churn_drivers": ["support_quality", "product_quality"],
            "account_value": "enterprise",
            "urgency": "critical",
        }
        execution_critical.context["retention_strategy"] = {
            "strategy": "immediate_intervention",
            "tactics": ["executive_outreach", "immediate_resolution"],
            "approach": "urgent_personal",
            "priority": "P0",
        }

        step = WorkflowStep(
            id="draft_retention_outreach",
            type=StepType.GENERATE,
            name="Generate Content",
            description="Test step",
            config={"num_variants": 2},
        )

        result = await workflow.generate_retention_content(step, execution_critical)

        # Verify generation results
        assert "num_variants" in result
        assert "best_score" in result
        assert "best_content" in result
        assert result["retention_applied"] is True
        assert result["strategy"] == "immediate_intervention"

        # Verify content was generated
        assert result["num_variants"] >= 1
        assert len(result["best_content"]) > 0

    def test_sentiment_analysis(self, workflow):
        """Test churn sentiment analysis."""
        # Test with explicit threat
        signal = ActionableSignal(
            user_id=uuid4(),
            signal_type=SignalType.CHURN_RISK,
            source_platform="email",
            source_url="https://example.com/test",
            source_author="user",
            title="Canceling my subscription",
            description="I'm canceling my subscription today. This is a waste of money.",
            context="Test",
            urgency_score=0.8,
            impact_score=0.6,
            confidence_score=0.7,
            action_score=0.7,
            recommended_action=ActionType.MONITOR,
            suggested_channel="email",
            suggested_tone=ResponseTone.SUPPORTIVE,
        )

        sentiment = workflow._analyze_churn_sentiment(signal)

        assert "intensity" in sentiment
        assert "dominant_emotion" in sentiment
        assert "has_explicit_threat" in sentiment
        assert sentiment["has_explicit_threat"] is True
        assert sentiment["dominant_emotion"] == "threat"
        assert sentiment["threat_count"] > 0

    def test_account_value_estimation(self, workflow):
        """Test account value estimation."""
        # Test enterprise
        signal = ActionableSignal(
            user_id=uuid4(),
            signal_type=SignalType.CHURN_RISK,
            source_platform="email",
            source_url="https://example.com/test",
            source_author="user",
            title="Test",
            description="Test",
            context="Test",
            urgency_score=0.5,
            impact_score=0.5,
            confidence_score=0.5,
            action_score=0.5,
            recommended_action=ActionType.MONITOR,
            suggested_channel="email",
            suggested_tone=ResponseTone.PROFESSIONAL,
            metadata={"account_type": "enterprise", "monthly_value": 2000},
        )

        value = workflow._estimate_account_value(signal)
        assert value == "enterprise"

    def test_severity_calculation(self, workflow):
        """Test severity score calculation."""
        # High severity case
        score = workflow._calculate_severity_score(
            sentiment={"intensity": 0.8, "has_explicit_threat": True},
            account_value="enterprise",
            tenure="loyal",
            complaint_history={"is_recurring": True, "unresolved_issues": 2},
            escalation_risk={"level": "critical"},
        )

        assert score >= 0.75  # Should be critical
        assert score <= 1.0

    def test_churn_driver_identification(self, workflow):
        """Test churn driver identification."""
        signal = ActionableSignal(
            user_id=uuid4(),
            signal_type=SignalType.CHURN_RISK,
            source_platform="email",
            source_url="https://example.com/test",
            source_author="user",
            title="Product is too expensive and buggy",
            description="The price is way too high and there are constant bugs. Support never responds.",
            context="Test",
            urgency_score=0.7,
            impact_score=0.6,
            confidence_score=0.7,
            action_score=0.7,
            recommended_action=ActionType.MONITOR,
            suggested_channel="email",
            suggested_tone=ResponseTone.SUPPORTIVE,
        )

        sentiment = {"intensity": 0.7, "has_explicit_threat": False}
        drivers = workflow._identify_churn_drivers(signal, sentiment)

        assert "pricing" in drivers
        assert "product_quality" in drivers
        assert "support_quality" in drivers

