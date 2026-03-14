"""Integration tests for output generation pipeline."""

import pytest
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.models import Cluster, ContentItem, MediaType, SourcePlatform
from app.output.models import (
    OutputFormat,
    OutputPreferences,
    OutputRequest,
    TextStyle,
    TonePreference,
    LengthPreference,
)
from app.output.manager import OutputManager
from app.output.generators.text_generator import MarkdownGenerator
from app.output.generators.visual_generator import InfographicGenerator


@pytest.fixture
def sample_clusters():
    """Create sample clusters for testing."""
    user_id = uuid4()

    items1 = [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="post1",
            source_url="https://reddit.com/r/test/post1",
            title="AI Breakthrough in Language Models",
            raw_text="Researchers announce new breakthrough in AI...",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.YOUTUBE,
            source_id="video1",
            source_url="https://youtube.com/watch?v=video1",
            title="Explaining the Latest AI Research",
            media_type=MediaType.VIDEO,
            published_at=datetime.utcnow(),
        ),
    ]

    cluster1 = Cluster(
        user_id=user_id,
        topic="AI Research Advances",
        summary="Multiple sources report significant advances in AI research",
        items=items1,
        item_ids=[item.id for item in items1],
        relevance_score=0.92,
        platforms_represented=[SourcePlatform.REDDIT, SourcePlatform.YOUTUBE],
    )

    return [cluster1]


@pytest.fixture
def sample_preferences():
    """Create sample output preferences."""
    return OutputPreferences(
        user_id=uuid4(),
        name="Test Preferences",
        primary_format=OutputFormat.MARKDOWN,
        text_style=TextStyle.PROFESSIONAL,
        tone=TonePreference.NEUTRAL,
        length=LengthPreference.MEDIUM,
        include_sources=True,
        include_visualizations=False,
    )


@pytest.mark.asyncio
async def test_markdown_generation(sample_clusters, sample_preferences):
    """Test Markdown output generation."""
    mock_router = MagicMock()
    mock_router.generate_simple = AsyncMock(
        return_value=(
            "# AI Research Advances\n\n"
            "This is a test summary covering the latest breakthroughs in AI research. "
            "Multiple sources report significant advances in language models and neural networks. "
            "The field continues to evolve rapidly with new architectures and training techniques."
        )
    )
    mock_router.get_stats = MagicMock(return_value={"primary_model": "gpt-4o"})

    manager = OutputManager(llm_router=mock_router)

    request = OutputRequest(
        user_id=sample_preferences.user_id,
        preferences_id=sample_preferences.id,
    )

    output = await manager.generate_output(
        request=request,
        preferences=sample_preferences,
        clusters=sample_clusters,
        items=sample_clusters[0].items,
    )

    assert output.format == OutputFormat.MARKDOWN
    assert output.success is True
    assert len(output.content) > 0
    assert output.metadata.word_count > 0
    assert "AI Research Advances" in output.content


@pytest.mark.asyncio
async def test_infographic_generation(sample_clusters, sample_preferences):
    """Test infographic generation (stub - visual generators don't require LLM)."""
    # Update preferences for image output
    sample_preferences.primary_format = OutputFormat.IMAGE

    manager = OutputManager()

    request = OutputRequest(
        user_id=sample_preferences.user_id,
        preferences_id=sample_preferences.id,
    )

    output = await manager.generate_output(
        request=request,
        preferences=sample_preferences,
        clusters=sample_clusters,
        items=sample_clusters[0].items,
    )

    assert output.format == OutputFormat.IMAGE
    assert output.success is True


@pytest.mark.asyncio
async def test_multi_format_generation(sample_clusters, sample_preferences):
    """Test generating multiple formats concurrently."""
    mock_router = MagicMock()
    mock_router.generate_simple = AsyncMock(
        return_value=(
            "# Test Summary\n\n"
            "This is a comprehensive test summary with enough words to pass the quality "
            "validation threshold for the output generator. The content covers multiple "
            "topics including artificial intelligence, machine learning, and neural networks. "
            "Researchers have made significant advances in these areas recently, with new "
            "architectures and training techniques showing promising results on benchmarks."
        )
    )
    mock_router.get_stats = MagicMock(return_value={"primary_model": "gpt-4o"})

    manager = OutputManager(llm_router=mock_router)

    request = OutputRequest(
        user_id=sample_preferences.user_id,
        preferences_id=sample_preferences.id,
    )

    formats = [OutputFormat.MARKDOWN, OutputFormat.IMAGE]

    outputs = await manager.generate_multi_format(
        request=request,
        preferences=sample_preferences,
        clusters=sample_clusters,
        items=sample_clusters[0].items,
        formats=formats,
    )

    assert len(outputs) == 2
    assert OutputFormat.MARKDOWN in outputs
    assert OutputFormat.IMAGE in outputs


@pytest.mark.asyncio
async def test_quality_check_and_retry(sample_clusters, sample_preferences):
    """Test quality checking with retry logic."""
    # Disable fallback formats so generate_output returns low-quality output
    # instead of raising, allowing generate_with_quality_check to retry
    sample_preferences.fallback_formats = []

    mock_router = MagicMock()
    # First attempt returns low quality, second returns high quality
    mock_router.generate_simple = AsyncMock(
        side_effect=[
            "Short",  # Low quality - will fail validation (word_count < 50)
            (
                "# High Quality Summary\n\nThis is a much better summary with more detail "
                "and proper structure covering all the important topics in depth. "
                "The content includes comprehensive analysis of recent developments in "
                "artificial intelligence and machine learning research."
            ),
        ]
    )
    mock_router.get_stats = MagicMock(return_value={"primary_model": "gpt-4o"})

    manager = OutputManager(llm_router=mock_router)

    request = OutputRequest(
        user_id=sample_preferences.user_id,
        preferences_id=sample_preferences.id,
    )

    output = await manager.generate_with_quality_check(
        request=request,
        preferences=sample_preferences,
        clusters=sample_clusters,
        items=sample_clusters[0].items,
        min_quality_score=0.7,
        max_retries=3,
    )

    # Should have retried and gotten better output
    assert output.metadata.word_count > 10


@pytest.mark.asyncio
async def test_fallback_format_on_failure(sample_clusters, sample_preferences):
    """Test fallback to alternative format on failure."""
    sample_preferences.fallback_formats = [OutputFormat.PLAIN_TEXT, OutputFormat.JSON]

    with patch("app.output.generators.text_generator.MarkdownGenerator.generate") as mock_gen:
        # Make primary format fail
        mock_gen.side_effect = Exception("Generation failed")

        manager = OutputManager()

        request = OutputRequest(
            user_id=sample_preferences.user_id,
            preferences_id=sample_preferences.id,
        )

        # Should fall back to alternative format
        # In this test, we'll just verify the fallback logic is triggered
        with pytest.raises(ValueError, match="All output generation attempts failed"):
            await manager.generate_output(
                request=request,
                preferences=sample_preferences,
                clusters=sample_clusters,
                items=sample_clusters[0].items,
            )


@pytest.mark.asyncio
async def test_custom_prompt_integration(sample_clusters, sample_preferences):
    """Test custom prompt integration."""
    captured_prompts: list = []

    async def capture_and_return(prompt: str, **kwargs) -> str:
        captured_prompts.append(prompt)
        return (
            "# Custom Technical Analysis\n\n"
            "Custom formatted output covering technical details and machine learning "
            "with neural network explanations and code examples for practitioners. "
            "This comprehensive guide explores the latest advances in deep learning, "
            "transformer architectures, and their practical applications in industry. "
            "Researchers have demonstrated significant improvements across multiple benchmarks."
        )

    mock_router = MagicMock()
    mock_router.generate_simple = capture_and_return
    mock_router.get_stats = MagicMock(return_value={"primary_model": "gpt-4o"})

    manager = OutputManager(llm_router=mock_router)

    request = OutputRequest(
        user_id=sample_preferences.user_id,
        preferences_id=sample_preferences.id,
        custom_prompt="Focus on technical details and include code examples",
        focus_topics=["machine learning", "neural networks"],
    )

    output = await manager.generate_output(
        request=request,
        preferences=sample_preferences,
        clusters=sample_clusters,
        items=sample_clusters[0].items,
    )

    # Verify custom prompt was passed to the LLM
    assert len(captured_prompts) > 0
    assert "Focus on technical details" in captured_prompts[0]
    assert "machine learning" in captured_prompts[0]

