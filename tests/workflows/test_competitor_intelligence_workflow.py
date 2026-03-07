"""Tests for Competitor Intelligence workflow."""

import pytest
from uuid import uuid4

from app.core.signal_models import (
    ActionableSignal,
    SignalType,
    ActionType,
    ResponseTone,
)
from app.intelligence.response_generator import ResponseGenerator
from app.workflows.competitor_intelligence_workflow import CompetitorIntelligenceWorkflow
from app.workflows.workflow_models import WorkflowExecution, WorkflowStep, StepType, WorkflowType


@pytest.fixture
def response_generator():
    """Create response generator instance."""
    return ResponseGenerator()


@pytest.fixture
def workflow(response_generator):
    """Create Competitor Intelligence workflow instance."""
    return CompetitorIntelligenceWorkflow(response_generator)


@pytest.fixture
def competitor_signal():
    """Create a competitor weakness signal."""
    return ActionableSignal(
        user_id=uuid4(),
        signal_type=SignalType.COMPETITOR_WEAKNESS,
        source_platform="twitter",
        source_url="https://twitter.com/user/status/123",
        source_author="angry_customer",
        title="CompetitorX support is absolutely terrible",
        description=(
            "I've been waiting 5 days for a response from CompetitorX support. "
            "This is unacceptable for a paid product. Their customer service is the worst "
            "I've ever experienced. No response to emails, no live chat, nothing. "
            "Looking for alternatives with better support."
        ),
        context="Public complaint about competitor support quality",
        urgency_score=0.7,
        impact_score=0.65,
        confidence_score=0.9,
        action_score=0.7,
        recommended_action=ActionType.CREATE_CONTENT,
        suggested_channel="twitter",
        suggested_tone=ResponseTone.HELPFUL,
        metadata={"competitor_name": "CompetitorX"},
    )


@pytest.fixture
def execution(competitor_signal):
    """Create workflow execution instance."""
    execution = WorkflowExecution(
        workflow_id=uuid4(),
        workflow_type=WorkflowType.COMPETITOR_INTELLIGENCE,
        signal_id=competitor_signal.id,
        user_id=competitor_signal.user_id,
    )
    execution.context["signal"] = competitor_signal.model_dump()
    return execution


class TestCompetitorIntelligenceWorkflow:
    """Test Competitor Intelligence workflow functionality."""

    @pytest.mark.asyncio
    async def test_analyze_competitor_complaint(self, workflow, execution):
        """Test competitor complaint analysis."""
        step = WorkflowStep(
            id="analyze_complaint",
            type=StepType.ANALYZE,
            name="Analyze Complaint",
            description="Test step",
            config={"extract_fields": ["competitor_name", "complaint_type", "severity"]},
        )

        result = await workflow.analyze_competitor_complaint(step, execution)

        # Verify analysis results
        assert "competitor_name" in result
        assert "complaint_type" in result
        assert "pain_points" in result
        assert "severity" in result
        assert "sentiment" in result

        # Verify competitor name
        assert result["competitor_name"] == "CompetitorX"

        # Verify complaint type
        assert result["complaint_type"] == "support"

        # Verify severity
        assert result["severity"] in ["low", "medium", "high", "critical"]

        # Verify sentiment
        sentiment = result["sentiment"]
        assert "polarity" in sentiment
        assert "intensity" in sentiment
        assert sentiment["polarity"] < 0  # Should be negative

    @pytest.mark.asyncio
    async def test_identify_positioning_angle(self, workflow, execution):
        """Test positioning angle identification."""
        # Setup context with analysis
        execution.context["competitor_analysis"] = {
            "competitor_name": "CompetitorX",
            "complaint_type": "support",
            "pain_points": ["no_response", "poor_documentation"],
            "severity": "high",
            "sentiment": {"polarity": -0.7, "intensity": 0.8},
        }

        step = WorkflowStep(
            id="identify_positioning",
            type=StepType.ANALYZE,
            name="Identify Positioning",
            description="Test step",
        )

        result = await workflow.identify_positioning_angle(step, execution)

        # Verify positioning results
        assert "positioning" in result
        assert "differentiators" in result
        assert "angle" in result
        assert "content_opportunities" in result

        # Verify positioning
        positioning = result["positioning"]
        assert "our_advantage" in positioning
        assert "message" in positioning
        assert "features" in positioning

        # Verify differentiators
        differentiators = result["differentiators"]
        assert len(differentiators) > 0
        assert any("support" in d for d in differentiators)

        # Verify angle
        assert result["angle"] in ["empathetic_helper", "helpful_alternative", "thought_leader"]

        # Verify content opportunities
        content_ops = result["content_opportunities"]
        assert "public_response" in content_ops

    @pytest.mark.asyncio
    async def test_generate_positioning_content(self, workflow, execution):
        """Test positioning content generation."""
        # Setup context
        execution.context["competitor_analysis"] = {
            "competitor_name": "CompetitorX",
            "complaint_type": "support",
            "pain_points": ["no_response"],
            "severity": "high",
        }
        execution.context["positioning"] = {
            "angle": "empathetic_helper",
            "differentiators": ["fast_response", "dedicated_support"],
            "content_opportunities": ["public_response"],
        }

        step = WorkflowStep(
            id="generate_content",
            type=StepType.GENERATE,
            name="Generate Content",
            description="Test step",
        )

        result = await workflow.generate_positioning_content(step, execution)

        # Verify generation results
        assert "content_types_generated" in result
        assert "primary_content" in result
        assert result["positioning_applied"] is True

        # Verify content was generated
        assert len(result["content_types_generated"]) > 0
        assert len(result["primary_content"]) > 0

    def test_categorize_complaint_types(self, workflow):
        """Test complaint categorization."""
        # Test pricing complaint
        signal = ActionableSignal(
            user_id=uuid4(),
            signal_type=SignalType.COMPETITOR_WEAKNESS,
            source_platform="reddit",
            source_url="https://reddit.com/test",
            source_author="user",
            title="Too expensive",
            description="The pricing is way too high for what you get",
            context="Test",
            urgency_score=0.5,
            impact_score=0.5,
            confidence_score=0.5,
            action_score=0.5,
            recommended_action=ActionType.MONITOR,
            suggested_channel="reddit",
            suggested_tone=ResponseTone.PROFESSIONAL,
        )

        complaint_type = workflow._categorize_complaint(signal)
        assert complaint_type == "pricing"

    def test_sentiment_analysis(self, workflow):
        """Test sentiment analysis."""
        # Test negative sentiment
        signal = ActionableSignal(
            user_id=uuid4(),
            signal_type=SignalType.COMPETITOR_WEAKNESS,
            source_platform="twitter",
            source_url="https://twitter.com/test",
            source_author="user",
            title="Terrible product",
            description="This is the worst software I've ever used. Absolutely horrible.",
            context="Test",
            urgency_score=0.5,
            impact_score=0.5,
            confidence_score=0.5,
            action_score=0.5,
            recommended_action=ActionType.MONITOR,
            suggested_channel="twitter",
            suggested_tone=ResponseTone.PROFESSIONAL,
        )

        sentiment = workflow._analyze_sentiment(signal)

        assert "polarity" in sentiment
        assert "intensity" in sentiment
        assert sentiment["polarity"] < 0  # Should be negative
        assert sentiment["intensity"] > 0  # Should have some intensity

