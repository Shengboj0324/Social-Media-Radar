"""End-to-end tests for complete pipeline."""

import pytest
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.models import (
    ContentItem,
    DigestRequest,
    MediaType,
    SourcePlatform,
    UserInterestProfile,
)
from app.core.ranking import ContentClusterer, RelevanceScorer
from app.output.manager import OutputManager
from app.output.models import OutputFormat, OutputPreferences, OutputRequest


@pytest.mark.asyncio
async def test_complete_digest_pipeline():
    """Test complete pipeline from content ingestion to output generation."""
    user_id = uuid4()

    # Step 1: Create user profile
    user_profile = UserInterestProfile(
        user_id=user_id,
        topics=["artificial intelligence", "machine learning", "technology"],
        keywords=["AI", "ML", "neural networks", "deep learning"],
        preferred_sources=[SourcePlatform.REDDIT, SourcePlatform.YOUTUBE],
    )

    # Step 2: Simulate content items
    content_items = [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="post1",
            source_url="https://reddit.com/r/MachineLearning/post1",
            title="New Breakthrough in Neural Network Architecture",
            raw_text="Researchers have developed a new neural network architecture...",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.YOUTUBE,
            source_id="video1",
            source_url="https://youtube.com/watch?v=video1",
            title="Explaining Transformers in Deep Learning",
            raw_text="This video explains how transformer models work...",
            media_type=MediaType.VIDEO,
            published_at=datetime.utcnow(),
        ),
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.RSS,
            source_id="article1",
            source_url="https://techcrunch.com/article1",
            title="AI Startup Raises $100M",
            raw_text="A new AI startup focused on enterprise solutions...",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
    ]

    # Step 3: Score relevance (RelevanceScorer is now synchronous, no LLM dependency)
    scorer = RelevanceScorer(interest_profile=user_profile)

    scores = [scorer.score_item(item) for item in content_items]

    # All items should have relevance scores in [0, 1]
    assert all(score is not None for score in scores)
    assert all(0 <= score <= 1 for score in scores)

    # Step 4: Cluster content (ContentClusterer is now synchronous, no LLM dependency)
    clusterer = ContentClusterer()
    clusters = clusterer.cluster_items(content_items, user_id)

    # Should have created at least one cluster (single-cluster fallback when no embeddings)
    assert len(clusters) > 0
    assert all(cluster.topic for cluster in clusters)

    # Step 5: Generate output
    mock_router = MagicMock()
    mock_router.generate_simple = AsyncMock(
        return_value=(
            "# Daily Intelligence Digest\n\n"
            "## AI and Machine Learning\n\n"
            "Today's top stories cover major breakthroughs in neural network architecture, "
            "transformer models, and enterprise AI funding. Researchers have developed new "
            "approaches that significantly improve performance on benchmark tasks. "
            "The field continues to advance rapidly with both academic and industry contributions."
        )
    )
    mock_router.get_stats = MagicMock(return_value={"primary_model": "gpt-4o"})

    output_manager = OutputManager(llm_router=mock_router)

    preferences = OutputPreferences(
        user_id=user_id,
        name="Default",
        primary_format=OutputFormat.MARKDOWN,
    )

    request = OutputRequest(
        user_id=user_id,
        preferences_id=preferences.id,
    )

    output = await output_manager.generate_output(
        request=request,
        preferences=preferences,
        clusters=clusters,
        items=content_items,
    )

    # Verify output
    assert output.success is True
    assert output.format == OutputFormat.MARKDOWN
    assert len(output.content) > 0
    assert "Daily Intelligence Digest" in output.content


@pytest.mark.asyncio
async def test_error_recovery_in_pipeline():
    """Test error recovery and graceful degradation."""
    user_id = uuid4()

    content_items = [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="post1",
            source_url="https://reddit.com/r/test/post1",
            title="Test Post",
            raw_text="Test content",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
    ]

    # Test LLM failure: when no fallback formats are configured, the manager raises
    mock_router = MagicMock()
    mock_router.generate_simple = AsyncMock(side_effect=Exception("API timeout"))
    mock_router.get_stats = MagicMock(return_value={"primary_model": "gpt-4o"})

    output_manager = OutputManager(llm_router=mock_router)

    preferences = OutputPreferences(
        user_id=user_id,
        name="Test",
        primary_format=OutputFormat.MARKDOWN,
        fallback_formats=[],  # No fallback - should raise
    )

    request = OutputRequest(
        user_id=user_id,
        preferences_id=preferences.id,
    )

    # Cluster without LLM (synchronous)
    clusterer = ContentClusterer()
    clusters = clusterer.cluster_items(content_items, user_id)

    # Without fallback formats, the manager should propagate the error
    import pytest as _pytest
    with _pytest.raises(Exception):
        await output_manager.generate_output(
            request=request,
            preferences=preferences,
            clusters=clusters,
            items=content_items,
        )


@pytest.mark.asyncio
async def test_multi_platform_aggregation():
    """Test aggregation from multiple platforms."""
    user_id = uuid4()

    # Content from different platforms
    platforms = [
        SourcePlatform.REDDIT,
        SourcePlatform.YOUTUBE,
        SourcePlatform.RSS,
        SourcePlatform.TIKTOK,
    ]

    content_items = []
    for i, platform in enumerate(platforms):
        item = ContentItem(
            user_id=user_id,
            source_platform=platform,
            source_id=f"item{i}",
            source_url=f"https://example.com/item{i}",
            title=f"Content from {platform.value}",
            raw_text=f"This is content from {platform.value}",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        )
        content_items.append(item)

    # Cluster should identify multi-platform coverage (synchronous, no LLM)
    clusterer = ContentClusterer()
    clusters = clusterer.cluster_items(content_items, user_id)

    # Should have identified platforms
    for cluster in clusters:
        assert len(cluster.platforms_represented) > 0

