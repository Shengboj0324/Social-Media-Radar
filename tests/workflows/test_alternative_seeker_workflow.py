"""Tests for Alternative Seeker workflow."""

import pytest
from uuid import uuid4

from app.core.signal_models import (
    ActionableSignal,
    SignalType,
    ActionType,
    ResponseTone,
)
from app.intelligence.response_generator import ResponseGenerator
from app.workflows.alternative_seeker_workflow import AlternativeSeekerWorkflow
from app.workflows.workflow_models import WorkflowExecution, WorkflowStep, StepType, WorkflowType


@pytest.fixture
def response_generator():
    """Create response generator instance."""
    return ResponseGenerator()


@pytest.fixture
def workflow(response_generator):
    """Create Alternative Seeker workflow instance."""
    return AlternativeSeekerWorkflow(response_generator)


@pytest.fixture
def lead_signal():
    """Create a high-quality lead signal."""
    return ActionableSignal(
        user_id=uuid4(),
        signal_type=SignalType.LEAD_OPPORTUNITY,
        source_platform="reddit",
        source_url="https://reddit.com/r/saas/comments/123",
        source_author="frustrated_user",
        title="Looking for Salesforce alternative - too expensive and complicated",
        description=(
            "We're a small startup and Salesforce is killing us with pricing ($150/user/month). "
            "It's also way too complicated for our needs. We just need simple CRM with good "
            "integrations. Looking for something affordable (under $30/user) and easy to use. "
            "Need to switch ASAP as our contract ends next month."
        ),
        context="High-intent lead with clear pain points and timeline",
        urgency_score=0.85,
        impact_score=0.75,
        confidence_score=0.9,
        action_score=0.8,
        recommended_action=ActionType.REPLY_PUBLIC,
        suggested_channel="reddit",
        suggested_tone=ResponseTone.HELPFUL,
        metadata={"competitor_name": "Salesforce"},
    )


@pytest.fixture
def execution(lead_signal):
    """Create workflow execution instance."""
    execution = WorkflowExecution(
        workflow_id=uuid4(),
        workflow_type=WorkflowType.ALTERNATIVE_SEEKER,
        signal_id=lead_signal.id,
        user_id=lead_signal.user_id,
    )
    execution.context["signal"] = lead_signal.model_dump()
    return execution


class TestAlternativeSeekerWorkflow:
    """Test Alternative Seeker workflow functionality."""

    @pytest.mark.asyncio
    async def test_analyze_lead_intent(self, workflow, execution):
        """Test lead intent analysis."""
        step = WorkflowStep(
            id="analyze_intent",
            type=StepType.ANALYZE,
            name="Analyze Lead Intent",
            description="Test step",
            config={"extract_fields": ["pain_points", "current_tool", "requirements"]},
        )

        result = await workflow.analyze_lead_intent(step, execution)

        # Verify analysis results
        assert "pain_points" in result
        assert "current_tool" in result
        assert "requirements" in result
        assert "budget_signals" in result
        assert "urgency_indicators" in result
        assert "intent_clarity" in result

        # Verify pain points extracted
        pain_points = result["pain_points"]
        assert "pricing" in pain_points or "expensive" in pain_points
        assert "complexity" in pain_points

        # Verify current tool
        assert result["current_tool"] == "Salesforce"

        # Verify requirements
        requirements = result["requirements"]
        assert "affordable" in requirements
        assert "ease_of_use" in requirements

        # Verify budget signals
        budget = result["budget_signals"]
        assert budget["price_sensitive"] is True

        # Verify urgency
        urgency = result["urgency_indicators"]
        assert urgency["urgency_level"] in ["medium", "high"]

        # Verify intent clarity is high
        assert result["intent_clarity"] >= 0.5

    @pytest.mark.asyncio
    async def test_qualify_lead(self, workflow, execution, lead_signal):
        """Test lead qualification scoring."""
        # First run analysis
        analysis_step = WorkflowStep(
            id="analyze_intent",
            type=StepType.ANALYZE,
            name="Analyze",
            description="Test",
        )
        analysis = await workflow.analyze_lead_intent(analysis_step, execution)
        execution.context["lead_analysis"] = analysis

        # Now qualify
        qualify_step = WorkflowStep(
            id="qualify_lead",
            type=StepType.SCORE,
            name="Qualify Lead",
            description="Test step",
            config={"scoring_criteria": ["fit_score", "intent_score", "urgency_score"]},
        )

        result = await workflow.qualify_lead(qualify_step, execution)

        # Verify scores
        assert "fit_score" in result
        assert "intent_score" in result
        assert "urgency_score" in result
        assert "overall_score" in result
        assert "quality_level" in result

        # Verify scores are in valid range
        assert 0.0 <= result["fit_score"] <= 1.0
        assert 0.0 <= result["intent_score"] <= 1.0
        assert 0.0 <= result["overall_score"] <= 1.0

        # This should be a high-quality lead
        assert result["quality_level"] in ["medium_quality", "high_quality"]

    @pytest.mark.asyncio
    async def test_generate_personalized_response(self, workflow, execution, lead_signal):
        """Test personalized response generation."""
        # Setup context
        execution.context["lead_analysis"] = {
            "pain_points": ["pricing", "complexity"],
            "requirements": ["affordable", "ease_of_use"],
            "current_tool": "Salesforce",
        }
        execution.context["lead_scores"] = {
            "overall_score": 0.8,
            "quality_level": "high_quality",
        }

        step = WorkflowStep(
            id="generate_response",
            type=StepType.GENERATE,
            name="Generate Response",
            description="Test step",
            config={"num_variants": 2},
        )

        result = await workflow.generate_personalized_response(step, execution)

        # Verify generation results
        assert "num_variants" in result
        assert "best_score" in result
        assert "best_content" in result
        assert result["personalization_applied"] is True

        # Verify content was generated
        assert result["num_variants"] >= 1
        assert len(result["best_content"]) > 0

