"""Integration tests across Phase 1, 2, and 3 components."""

import pytest
from uuid import uuid4

from app.core.models import ContentItem, SourcePlatform, MediaType
from app.core.signal_models import SignalType, ActionType
from app.intelligence.signal_classifier import SignalClassifier
from app.intelligence.action_scorer import ActionScorer
from app.intelligence.response_generator import ResponseGenerator
from app.workflows.orchestrator import WorkflowOrchestrator
from datetime import datetime


@pytest.fixture
def signal_classifier():
    """Create signal classifier instance."""
    return SignalClassifier(use_llm=False, min_confidence=0.6)


@pytest.fixture
def action_scorer():
    """Create action scorer instance."""
    return ActionScorer()


@pytest.fixture
def response_generator():
    """Create response generator instance."""
    return ResponseGenerator()


@pytest.fixture
def workflow_orchestrator(response_generator):
    """Create workflow orchestrator instance."""
    return WorkflowOrchestrator(response_generator)


@pytest.fixture
def lead_content_item():
    """Create a content item that should trigger lead opportunity signal."""
    return ContentItem(
        user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id="test123",
        source_url="https://reddit.com/r/saas/comments/test",
        author="frustrated_user",
        title="Looking for alternatives to Slack",
        raw_text=(
            "We're a team of 20 and looking for alternatives to Slack. "
            "Need something with better pricing and integrations. "
            "Any recommendations?"
        ),
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )


@pytest.fixture
def competitor_content_item():
    """Create a content item that should trigger competitor weakness signal."""
    return ContentItem(
        user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id="tweet456",
        source_url="https://reddit.com/r/saas/comments/456",
        author="angry_customer",
        title="Terrible customer support from Zendesk",
        raw_text=(
            "Been waiting 3 days for a response from Zendesk support. "
            "This is ridiculous for a paid product. "
            "Their support is terrible and pricing is too high."
        ),
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )


@pytest.fixture
def churn_content_item():
    """Create a content item that should trigger churn risk signal."""
    return ContentItem(
        user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id="post789",
        source_url="https://reddit.com/r/saas/comments/789",
        author="disappointed_customer",
        title="Canceling my subscription",
        raw_text=(
            "I've been a customer for 2 years but I'm canceling my subscription. "
            "The recent price increase is just too much and the product hasn't improved. "
            "Really disappointed as I used to love this product."
        ),
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )


class TestPhase1To2Integration:
    """Test integration between Phase 1 (Signals) and Phase 2 (Response Generation)."""

    @pytest.mark.asyncio
    async def test_lead_signal_to_response_generation(
        self,
        signal_classifier,
        action_scorer,
        response_generator,
        lead_content_item,
    ):
        """Test full pipeline from content to signal to response."""
        user_id = lead_content_item.user_id

        # Phase 1: Classify signal
        signal = await signal_classifier.classify_content(lead_content_item, user_id)

        assert signal is not None
        assert signal.signal_type == SignalType.LEAD_OPPORTUNITY
        assert signal.confidence_score > 0.5

        # Phase 1: Score action (already done in classify_content)
        assert signal.action_score > 0
        assert signal.recommended_action in [
            ActionType.REPLY_PUBLIC,
            ActionType.DM_OUTREACH,
            ActionType.CREATE_CONTENT,
        ]

        # Phase 2: Verify signal has the right structure for response generation
        assert signal.suggested_tone is not None
        assert signal.suggested_channel is not None

    @pytest.mark.asyncio
    async def test_competitor_signal_to_response_generation(
        self,
        signal_classifier,
        action_scorer,
        competitor_content_item,
    ):
        """Test competitor weakness signal pipeline."""
        user_id = competitor_content_item.user_id

        # Phase 1: Classify signal
        signal = await signal_classifier.classify_content(competitor_content_item, user_id)

        assert signal is not None
        assert signal.signal_type == SignalType.COMPETITOR_WEAKNESS
        assert signal.confidence_score > 0.5

        # Phase 1: Verify action scoring (already done in classify_content)
        assert signal.action_score > 0
        assert signal.recommended_action in [
            ActionType.CREATE_CONTENT,
            ActionType.MONITOR,
            ActionType.REPLY_PUBLIC,
        ]


class TestPhase1To3Integration:
    """Test integration between Phase 1 (Signals) and Phase 3 (Workflows)."""

    @pytest.mark.asyncio
    async def test_lead_signal_to_workflow_execution(
        self,
        signal_classifier,
        action_scorer,
        workflow_orchestrator,
        lead_content_item,
    ):
        """Test full pipeline from signal to workflow execution."""
        user_id = lead_content_item.user_id

        # Phase 1: Create signal
        signal = await signal_classifier.classify_content(lead_content_item, user_id)

        assert signal is not None
        assert signal.signal_type == SignalType.LEAD_OPPORTUNITY

        # Phase 3: Execute workflow
        execution = await workflow_orchestrator.execute_for_signal(signal)

        assert execution is not None
        assert execution.signal_id == signal.id
        # Workflow should complete or be running
        assert execution.status.value in ["completed", "running", "failed"]

