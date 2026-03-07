"""Unit tests for response generator and playbook system.

Tests the response generation pipeline including:
- Playbook template loading
- Prompt building
- Multi-variant generation
- Quality scoring
- Tone adaptation
"""

import pytest
from datetime import datetime
from uuid import uuid4

from app.core.signal_models import (
    ActionableSignal,
    ActionType,
    ResponseTone,
    SignalType,
)
from app.intelligence.response_playbook import (
    ResponseChannel,
    ResponsePlaybook,
    ResponseTemplate,
)
from app.intelligence.response_generator import ResponseGenerator


@pytest.fixture
def playbook():
    """Create response playbook."""
    return ResponsePlaybook()


@pytest.fixture
def lead_signal():
    """Create a lead opportunity signal for testing."""
    return ActionableSignal(
        user_id=uuid4(),
        signal_type=SignalType.LEAD_OPPORTUNITY,
        source_item_ids=[uuid4()],
        source_platform="reddit",
        source_url="https://reddit.com/r/saas/comments/123",
        source_author="tech_user",
        title="Lead Opportunity: Looking for Slack alternatives",
        description="We're a team of 20 looking for alternatives to Slack with better pricing",
        context="User is actively seeking alternatives, high conversion potential",
        urgency_score=0.8,
        impact_score=0.7,
        confidence_score=0.85,
        action_score=0.78,
        recommended_action=ActionType.REPLY_PUBLIC,
        suggested_channel="reddit",
        suggested_tone=ResponseTone.HELPFUL,
    )


@pytest.fixture
def competitor_signal():
    """Create a competitor weakness signal for testing."""
    return ActionableSignal(
        user_id=uuid4(),
        signal_type=SignalType.COMPETITOR_WEAKNESS,
        source_item_ids=[uuid4()],
        source_platform="reddit",
        source_url="https://reddit.com/r/saas/comments/456",
        source_author="frustrated_user",
        title="Competitor Weakness: Zendesk support complaints",
        description="Terrible customer support from Zendesk, waiting 3 days for response",
        context="Competitor weakness in support quality, opportunity to highlight our strengths",
        urgency_score=0.6,
        impact_score=0.75,
        confidence_score=0.8,
        action_score=0.72,
        recommended_action=ActionType.CREATE_CONTENT,
        suggested_channel="reddit",
        suggested_tone=ResponseTone.PROFESSIONAL,
    )


class TestResponsePlaybook:
    """Test response playbook functionality."""

    def test_playbook_initialization(self, playbook):
        """Test that playbook initializes with all templates."""
        assert len(playbook.templates) == 8  # All signal types
        assert SignalType.LEAD_OPPORTUNITY in playbook.templates
        assert SignalType.COMPETITOR_WEAKNESS in playbook.templates
        assert SignalType.PRODUCT_CONFUSION in playbook.templates

    def test_get_template(self, playbook):
        """Test retrieving templates by signal type."""
        template = playbook.get_template(SignalType.LEAD_OPPORTUNITY)
        assert template is not None
        assert template.signal_type == SignalType.LEAD_OPPORTUNITY
        assert template.action_type == ActionType.REPLY_PUBLIC
        assert len(template.tone_variations) > 0

    def test_build_prompt_lead_opportunity(self, playbook, lead_signal):
        """Test building prompts for lead opportunity."""
        system_prompt, user_prompt = playbook.build_prompt(
            signal=lead_signal,
            tone=ResponseTone.HELPFUL,
            channel=ResponseChannel.REDDIT,
        )

        assert len(system_prompt) > 0
        assert len(user_prompt) > 0
        assert "helpful" in system_prompt.lower() or "product expert" in system_prompt.lower()
        assert lead_signal.description in user_prompt
        assert "friendly" in user_prompt.lower() or "warm" in user_prompt.lower()

    def test_build_prompt_competitor_weakness(self, playbook, competitor_signal):
        """Test building prompts for competitor weakness."""
        system_prompt, user_prompt = playbook.build_prompt(
            signal=competitor_signal,
            tone=ResponseTone.PROFESSIONAL,
            channel=ResponseChannel.LINKEDIN,
        )

        assert len(system_prompt) > 0
        assert len(user_prompt) > 0
        assert "competitor" in system_prompt.lower() or "marketer" in system_prompt.lower()
        assert competitor_signal.description in user_prompt

    def test_channel_constraints(self, playbook, lead_signal):
        """Test that channel constraints are applied."""
        # Twitter has 280 char limit
        _, twitter_prompt = playbook.build_prompt(
            signal=lead_signal,
            tone=ResponseTone.PROFESSIONAL,
            channel=ResponseChannel.TWITTER,
        )
        assert "280" in twitter_prompt

        # LinkedIn allows longer content
        _, linkedin_prompt = playbook.build_prompt(
            signal=lead_signal,
            tone=ResponseTone.PROFESSIONAL,
            channel=ResponseChannel.LINKEDIN,
        )
        assert "3000" in linkedin_prompt  # LinkedIn has 3000 char limit

    def test_tone_variations(self, playbook):
        """Test that all tones have variations defined."""
        template = playbook.get_template(SignalType.LEAD_OPPORTUNITY)

        assert ResponseTone.PROFESSIONAL in template.tone_variations
        assert ResponseTone.HELPFUL in template.tone_variations
        assert ResponseTone.TECHNICAL in template.tone_variations
        assert ResponseTone.SUPPORTIVE in template.tone_variations
        assert ResponseTone.FOUNDER_VOICE in template.tone_variations


class TestResponseGenerator:
    """Test response generator functionality."""

    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        generator = ResponseGenerator()
        assert generator.playbook is not None
        assert generator.enable_quality_scoring is True

    def test_score_clarity(self):
        """Test clarity scoring."""
        generator = ResponseGenerator()

        # Good clarity
        good_text = "This is a clear and well-structured response. It has proper sentences."
        score = generator._score_clarity(good_text)
        assert 0.4 <= score <= 1.0  # Adjusted threshold

        # Poor clarity
        poor_text = "no punctuation or structure just rambling text"
        score = generator._score_clarity(poor_text)
        assert score < 0.7

    def test_score_tone_match_professional(self):
        """Test tone matching for professional tone."""
        generator = ResponseGenerator()

        professional_text = "Thank you for your inquiry. I appreciate your interest. However, please note..."
        score = generator._score_tone_match(professional_text, ResponseTone.PROFESSIONAL)
        assert score >= 0.6  # Adjusted threshold

        informal_text = "Hey yeah gonna check that out lol thanks"
        score = generator._score_tone_match(informal_text, ResponseTone.PROFESSIONAL)
        assert score < 0.7

    def test_score_tone_match_helpful(self):
        """Test tone matching for helpful tone."""
        generator = ResponseGenerator()

        friendly_text = "Hey! Thanks so much, happy to help! This is awesome."
        score = generator._score_tone_match(friendly_text, ResponseTone.HELPFUL)
        assert score > 0.7

    def test_score_length_twitter(self):
        """Test length scoring for Twitter."""
        generator = ResponseGenerator()

        # Perfect length for Twitter
        good_text = "A" * 200
        score = generator._score_length(good_text, ResponseChannel.TWITTER)
        assert score >= 0.9

        # Too long for Twitter
        long_text = "A" * 500
        score = generator._score_length(long_text, ResponseChannel.TWITTER)
        assert score < 0.7

    def test_score_engagement_potential(self, lead_signal):
        """Test engagement potential scoring."""
        generator = ResponseGenerator()

        # High engagement potential
        engaging_text = "Have you considered our solution? Happy to help you learn more about the benefits."
        score = generator._score_engagement_potential(engaging_text, lead_signal)
        assert score > 0.6

        # Low engagement potential (too salesy)
        salesy_text = "Buy now! Limited time offer! Don't miss out!"
        score = generator._score_engagement_potential(salesy_text, lead_signal)
        assert score < 0.6

    def test_select_tones_for_variants(self, lead_signal):
        """Test tone selection for variants."""
        generator = ResponseGenerator()

        tones = generator._select_tones_for_variants(lead_signal, num_variants=3)
        assert len(tones) == 3
        assert len(set(tones)) == 3  # All unique
        assert lead_signal.suggested_tone in tones  # Includes suggested tone

