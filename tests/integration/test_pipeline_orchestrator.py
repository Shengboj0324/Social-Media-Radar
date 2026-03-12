"""Integration tests for PipelineOrchestrator - end-to-end pipeline testing."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

from app.core.models import ContentItem, SourcePlatform, MediaType
from app.core.signal_models import SignalType, ActionableSignal
from app.ingestion.pipeline_orchestrator import PipelineOrchestrator


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.add = MagicMock()
    session.get = AsyncMock()
    return session


@pytest.fixture
def sample_content_items():
    """Create sample content items."""
    user_id = uuid4()
    return [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="test1",
            source_url="https://reddit.com/r/saas/test1",
            author="user1",
            title="Looking for alternatives to Slack",
            raw_text="We need a better team communication tool. Any recommendations?",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id="test2",
            source_url="https://reddit.com/r/saas/test2",
            author="user2",
            title="Zendesk support is terrible",
            raw_text="Been waiting 3 days for a response. This is unacceptable.",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        ),
    ]


class TestPipelineOrchestrator:
    """Test PipelineOrchestrator integration."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, mock_db_session):
        """Test pipeline orchestrator initialization."""
        orchestrator = PipelineOrchestrator(mock_db_session)

        assert orchestrator.db == mock_db_session
        assert orchestrator.content_ingestor is not None
        assert orchestrator.normalization_engine is not None
        assert orchestrator.enrichment_service is not None
        assert orchestrator.signal_classifier is not None
        assert orchestrator.workflow_orchestrator is not None

    @pytest.mark.asyncio
    async def test_normalize_items(self, mock_db_session, sample_content_items):
        """Test content normalization stage."""
        orchestrator = PipelineOrchestrator(mock_db_session)

        normalized = await orchestrator._normalize_items(sample_content_items)

        assert len(normalized) == len(sample_content_items)
        # Check that normalization was applied
        for item in normalized:
            assert item.title is not None
            assert item.raw_text is not None

    @pytest.mark.asyncio
    async def test_enrich_items(self, mock_db_session, sample_content_items):
        """Test content enrichment stage."""
        orchestrator = PipelineOrchestrator(mock_db_session)

        # Mock enrichment service to avoid actual LLM calls
        orchestrator.enrichment_service.enable_embedding_generation = False
        orchestrator.enrichment_service.enable_entity_extraction = False

        enriched = await orchestrator._enrich_items(sample_content_items)

        assert len(enriched) == len(sample_content_items)

    @pytest.mark.asyncio
    async def test_classify_signals(self, mock_db_session, sample_content_items):
        """Test signal classification stage."""
        orchestrator = PipelineOrchestrator(mock_db_session)

        # Use pattern-only classification to avoid LLM calls
        orchestrator.signal_classifier.use_llm = False
        orchestrator.signal_classifier.min_confidence = 0.5

        user_id = sample_content_items[0].user_id
        signals = await orchestrator._classify_signals(sample_content_items, user_id)

        # Should detect at least one signal from the sample content
        assert len(signals) >= 1
        assert all(isinstance(s, ActionableSignal) for s in signals)

    @pytest.mark.asyncio
    async def test_full_pipeline_without_workflows(self, mock_db_session):
        """Test full pipeline without workflow execution."""
        orchestrator = PipelineOrchestrator(mock_db_session)

        # Mock content ingestor to return sample items
        user_id = uuid4()
        sample_items = [
            ContentItem(
                user_id=user_id,
                source_platform=SourcePlatform.REDDIT,
                source_id="test1",
                source_url="https://reddit.com/r/saas/test1",
                author="user1",
                title="Looking for alternatives to Slack",
                raw_text="We need better team communication. Any recommendations?",
                media_type=MediaType.TEXT,
                published_at=datetime.utcnow(),
            ),
        ]

        orchestrator.content_ingestor.fetch_from_sources = AsyncMock(return_value=sample_items)
        orchestrator.signal_classifier.use_llm = False
        orchestrator.enrichment_service.enable_embedding_generation = False
        orchestrator.enrichment_service.enable_entity_extraction = False

        # Run pipeline without workflows
        result = await orchestrator.run_full_pipeline(
            user_id=user_id,
            execute_workflows=False,
            store_content=False,
            store_signals=False,
        )

        assert result["status"] == "success"
        assert result["items_fetched"] >= 0
        assert result["items_normalized"] >= 0
        assert result["items_enriched"] >= 0
        assert "duration_seconds" in result

    @pytest.mark.asyncio
    async def test_pipeline_metrics(self, mock_db_session):
        """Test pipeline metrics tracking."""
        orchestrator = PipelineOrchestrator(mock_db_session)

        metrics = orchestrator.get_metrics()

        assert "total_items_fetched" in metrics
        assert "total_signals_detected" in metrics
        assert "pipeline_failures" in metrics

