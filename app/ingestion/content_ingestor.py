"""Production-grade content ingestor for multi-source data collection.

This module provides the ContentIngestor class that orchestrates content fetching
from multiple platforms with:
- Priority queue management for intelligent source scheduling
- Rate limiting and backoff strategies
- Deduplication using Bloom filters
- Error recovery and retry logic
- Metrics and monitoring
- Batch processing optimization
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID
import asyncio
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.base import ConnectorConfig
from app.connectors.registry import ConnectorRegistry
from app.core.models import ContentItem, SourcePlatform
from app.core.db_models import PlatformConfigDB
from app.core.errors import ConnectorError, RateLimitError
from app.scraping.probabilistic_structures import BloomFilter

logger = logging.getLogger(__name__)


class IngestionMetrics:
    """Metrics for ingestion operations."""

    def __init__(self):
        """Initialize metrics."""
        self.total_fetched = 0
        self.total_duplicates = 0
        self.total_errors = 0
        self.fetch_times: Dict[SourcePlatform, List[float]] = defaultdict(list)
        self.error_counts: Dict[SourcePlatform, int] = defaultdict(int)

    def record_fetch(
        self,
        platform: SourcePlatform,
        items_count: int,
        duration: float,
        errors: int = 0,
    ):
        """Record fetch metrics."""
        self.total_fetched += items_count
        self.total_errors += errors
        self.fetch_times[platform].append(duration)
        if errors > 0:
            self.error_counts[platform] += errors

    def record_duplicate(self):
        """Record duplicate detection."""
        self.total_duplicates += 1

    def get_summary(self) -> Dict:
        """Get metrics summary."""
        return {
            "total_fetched": self.total_fetched,
            "total_duplicates": self.total_duplicates,
            "total_errors": self.total_errors,
            "avg_fetch_times": {
                platform.value: sum(times) / len(times) if times else 0
                for platform, times in self.fetch_times.items()
            },
            "error_counts": {
                platform.value: count
                for platform, count in self.error_counts.items()
            },
        }


class SourcePriority:
    """Priority information for a content source."""

    def __init__(
        self,
        platform: SourcePlatform,
        config_id: UUID,
        priority: float,
        last_fetch: Optional[datetime] = None,
        consecutive_errors: int = 0,
    ):
        """Initialize source priority."""
        self.platform = platform
        self.config_id = config_id
        self.priority = priority
        self.last_fetch = last_fetch
        self.consecutive_errors = consecutive_errors

    def calculate_priority(self) -> float:
        """Calculate dynamic priority based on recency and errors.

        Returns:
            Priority score (higher = more urgent)
        """
        base_priority = self.priority

        # Increase priority if not fetched recently
        if self.last_fetch:
            hours_since_fetch = (datetime.utcnow() - self.last_fetch).total_seconds() / 3600
            recency_boost = min(hours_since_fetch / 24.0, 2.0)  # Max 2x boost
            base_priority *= (1.0 + recency_boost)

        # Decrease priority if consecutive errors
        if self.consecutive_errors > 0:
            error_penalty = 0.5 ** self.consecutive_errors  # Exponential decay
            base_priority *= error_penalty

        return base_priority

    def __lt__(self, other):
        """Compare priorities for heap operations."""
        return self.calculate_priority() > other.calculate_priority()  # Max heap


class ContentIngestor:
    """Production-grade content ingestor with intelligent scheduling.

    Features:
    - Priority queue for source scheduling
    - Bloom filter for deduplication
    - Rate limiting per platform
    - Error recovery with exponential backoff
    - Batch processing optimization
    - Comprehensive metrics
    """

    def __init__(
        self,
        db_session: AsyncSession,
        bloom_filter_size: int = 1000000,
        bloom_filter_fp_rate: float = 0.01,
        max_concurrent_fetches: int = 5,
        enable_deduplication: bool = True,
    ):
        """Initialize content ingestor.

        Args:
            db_session: Database session
            bloom_filter_size: Expected number of items for Bloom filter
            bloom_filter_fp_rate: False positive rate for Bloom filter
            max_concurrent_fetches: Maximum concurrent fetch operations
            enable_deduplication: Enable Bloom filter deduplication
        """
        self.db = db_session
        self.max_concurrent_fetches = max_concurrent_fetches
        self.enable_deduplication = enable_deduplication

        # Initialize Bloom filter for deduplication
        if enable_deduplication:
            self.bloom_filter = BloomFilter(
                expected_elements=bloom_filter_size,
                false_positive_rate=bloom_filter_fp_rate,
            )
        else:
            self.bloom_filter = None

        # Metrics
        self.metrics = IngestionMetrics()

        # Rate limiting state
        self.rate_limits: Dict[SourcePlatform, datetime] = {}

        logger.info(
            f"ContentIngestor initialized: max_concurrent={max_concurrent_fetches}, "
            f"dedup={enable_deduplication}"
        )


    async def fetch_from_sources(
        self,
        user_id: UUID,
        platform_configs: Optional[List[PlatformConfigDB]] = None,
        since: Optional[datetime] = None,
    ) -> List[ContentItem]:
        """Fetch content from multiple sources with intelligent scheduling.

        Args:
            user_id: User ID
            platform_configs: List of platform configurations (if None, fetch all)
            since: Fetch content published after this timestamp

        Returns:
            List of fetched content items
        """
        start_time = datetime.utcnow()

        # Get platform configs if not provided
        if platform_configs is None:
            result = await self.db.execute(
                select(PlatformConfigDB)
                .where(PlatformConfigDB.user_id == user_id)
                .where(PlatformConfigDB.enabled.is_(True))
            )
            platform_configs = result.scalars().all()

        if not platform_configs:
            logger.warning(f"No enabled platform configs for user {user_id}")
            return []

        # Build priority queue
        source_priorities = []
        for config in platform_configs:
            priority = SourcePriority(
                platform=config.platform,
                config_id=config.id,
                priority=config.settings.get("priority", 1.0) if config.settings else 1.0,
                last_fetch=config.last_fetch_time,
                consecutive_errors=0,  # TODO: Track in database
            )
            source_priorities.append(priority)

        # Sort by priority (highest first)
        source_priorities.sort()

        # Fetch from sources with concurrency control
        all_items = []
        semaphore = asyncio.Semaphore(self.max_concurrent_fetches)

        async def fetch_with_semaphore(priority: SourcePriority):
            async with semaphore:
                return await self._fetch_from_source(
                    user_id=user_id,
                    config_id=priority.config_id,
                    since=since,
                )

        # Execute fetches concurrently
        fetch_tasks = [fetch_with_semaphore(p) for p in source_priorities]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Process results
        for priority, result in zip(source_priorities, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error fetching from {priority.platform}: {result}",
                    exc_info=result,
                )
                self.metrics.record_fetch(priority.platform, 0, 0, errors=1)
            elif isinstance(result, list):
                all_items.extend(result)

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"Fetched {len(all_items)} items from {len(platform_configs)} sources "
            f"in {duration:.2f}s"
        )

        return all_items

    async def _fetch_from_source(
        self,
        user_id: UUID,
        config_id: UUID,
        since: Optional[datetime] = None,
    ) -> List[ContentItem]:
        """Fetch content from a single source.

        Args:
            user_id: User ID
            config_id: Platform configuration ID
            since: Fetch content published after this timestamp

        Returns:
            List of fetched content items
        """
        start_time = datetime.utcnow()

        # Get platform config
        config = await self.db.get(PlatformConfigDB, config_id)
        if not config:
            logger.error(f"Platform config {config_id} not found")
            return []

        # Check rate limits
        if self._is_rate_limited(config.platform):
            logger.warning(f"Rate limited for {config.platform}, skipping")
            return []

        # Decrypt credentials
        credentials = {}
        if config.encrypted_credentials:
            try:
                from app.core.security import CredentialEncryption

                encryption = CredentialEncryption()
                credentials = encryption.decrypt(config.encrypted_credentials)
            except Exception as e:
                logger.error(f"Failed to decrypt credentials: {e}")
                self.metrics.record_fetch(config.platform, 0, 0, errors=1)
                return []

        # Create connector
        try:
            connector = ConnectorRegistry.get_connector(
                platform=config.platform,
                config=ConnectorConfig(
                    platform=config.platform,
                    credentials=credentials,
                    settings=config.settings or {},
                ),
                user_id=user_id,
            )
        except ConnectorError as e:
            logger.error(f"Failed to create connector for {config.platform}: {e}")
            self.metrics.record_fetch(config.platform, 0, 0, errors=1)
            return []

        # Fetch content
        try:
            # Use last fetch time or default to 24 hours ago
            if since is None:
                since = config.last_fetch_time or (datetime.utcnow() - timedelta(hours=24))

            result = await connector.fetch_content(since=since, max_items=100)

            # Filter duplicates
            unique_items = self._filter_duplicates(result.items)

            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.record_fetch(
                config.platform,
                len(unique_items),
                duration,
                errors=len(result.errors),
            )

            # Update last fetch time
            config.last_fetch_time = datetime.utcnow()
            await self.db.commit()

            logger.info(
                f"Fetched {len(unique_items)} items from {config.platform} "
                f"({len(result.items) - len(unique_items)} duplicates filtered)"
            )

            return unique_items

        except RateLimitError as e:
            logger.warning(f"Rate limit hit for {config.platform}: {e}")
            self._set_rate_limit(config.platform, e.reset_at)
            self.metrics.record_fetch(config.platform, 0, 0, errors=1)
            return []

        except Exception as e:
            logger.error(f"Error fetching from {config.platform}: {e}", exc_info=True)
            self.metrics.record_fetch(config.platform, 0, 0, errors=1)
            return []

    def _filter_duplicates(self, items: List[ContentItem]) -> List[ContentItem]:
        """Filter duplicate items using Bloom filter.

        Args:
            items: List of content items

        Returns:
            List of unique items
        """
        if not self.enable_deduplication or not self.bloom_filter:
            return items

        unique_items = []
        for item in items:
            # Create unique key from source_platform + source_id
            item_key = f"{item.source_platform.value}:{item.source_id}"

            # Check if item exists in Bloom filter
            if not self.bloom_filter.contains(item_key):
                # Add to Bloom filter
                self.bloom_filter.add(item_key)
                unique_items.append(item)
            else:
                # Potential duplicate (may be false positive)
                self.metrics.record_duplicate()
                logger.debug(f"Duplicate detected: {item_key}")

        return unique_items

    def _is_rate_limited(self, platform: SourcePlatform) -> bool:
        """Check if platform is currently rate limited.

        Args:
            platform: Source platform

        Returns:
            True if rate limited, False otherwise
        """
        if platform not in self.rate_limits:
            return False

        reset_time = self.rate_limits[platform]
        if datetime.utcnow() >= reset_time:
            # Rate limit expired
            del self.rate_limits[platform]
            return False

        return True

    def _set_rate_limit(self, platform: SourcePlatform, reset_at: datetime):
        """Set rate limit for platform.

        Args:
            platform: Source platform
            reset_at: Time when rate limit resets
        """
        self.rate_limits[platform] = reset_at
        logger.info(f"Rate limit set for {platform} until {reset_at}")

    def get_metrics(self) -> Dict:
        """Get ingestion metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.get_summary()

    def reset_bloom_filter(self):
        """Reset Bloom filter (useful for periodic cleanup)."""
        if self.bloom_filter:
            self.bloom_filter = BloomFilter(
                expected_elements=1000000,
                false_positive_rate=0.01,
            )
            logger.info("Bloom filter reset")
