"""Comprehensive unit tests for Priority Queue with Min-Heap.

Tests priority-based scheduling, heap operations, and deduplication.
Verifies algorithm correctness with peak skepticism.
"""

from datetime import datetime, timedelta
import pytest

from app.scraping.priority_queue import (
    CrawlItem,
    PriorityLevel,
    PriorityQueue,
    PriorityScorer,
)


class TestCrawlItem:
    """Test CrawlItem dataclass."""

    def test_crawl_item_creation(self):
        """Test creating a CrawlItem."""
        item = CrawlItem(
            priority_score=0.5,
            url="https://example.com",
            priority_level=PriorityLevel.HIGH,
        )

        assert item.priority_score == 0.5
        assert item.url == "https://example.com"
        assert item.priority_level == PriorityLevel.HIGH
        assert item.item_id is not None
        assert item.retry_count == 0

    def test_crawl_item_comparison(self):
        """Test that CrawlItems are compared by priority_score."""
        item1 = CrawlItem(priority_score=0.3, url="url1")
        item2 = CrawlItem(priority_score=0.7, url="url2")

        # Lower priority_score = higher priority
        assert item1 < item2
        assert item2 > item1

    def test_crawl_item_defaults(self):
        """Test CrawlItem default values."""
        item = CrawlItem(priority_score=0.5, url="https://example.com")

        assert item.priority_level == PriorityLevel.MEDIUM
        assert item.retry_count == 0
        assert item.metadata == {}
        assert item.estimated_freshness == 0.5
        assert item.estimated_relevance == 0.5
        assert item.engagement_score == 0.0


class TestPriorityScorer:
    """Test PriorityScorer implementation."""

    def test_priority_scorer_initialization(self):
        """Test PriorityScorer initialization."""
        scorer = PriorityScorer(
            freshness_weight=0.4,
            relevance_weight=0.3,
            engagement_weight=0.2,
            urgency_weight=0.1,
        )

        # Weights should be normalized to sum to 1.0
        total = (
            scorer.freshness_weight
            + scorer.relevance_weight
            + scorer.engagement_weight
            + scorer.urgency_weight
        )
        assert abs(total - 1.0) < 0.0001

    def test_priority_scorer_weight_normalization(self):
        """Test that weights are normalized."""
        scorer = PriorityScorer(
            freshness_weight=2.0,
            relevance_weight=2.0,
            engagement_weight=2.0,
            urgency_weight=2.0,
        )

        # Should normalize to 0.25 each
        assert abs(scorer.freshness_weight - 0.25) < 0.0001
        assert abs(scorer.relevance_weight - 0.25) < 0.0001
        assert abs(scorer.engagement_weight - 0.25) < 0.0001
        assert abs(scorer.urgency_weight - 0.25) < 0.0001

    def test_priority_calculation_critical(self):
        """Test priority calculation for CRITICAL items."""
        scorer = PriorityScorer()

        item = CrawlItem(
            priority_score=0.0,  # Will be recalculated
            url="https://example.com",
            priority_level=PriorityLevel.CRITICAL,
            estimated_freshness=1.0,
            estimated_relevance=1.0,
            engagement_score=1.0,
        )

        priority = scorer.calculate_priority(item)

        # CRITICAL with max freshness/relevance/engagement should have very low score
        assert priority < 0.1, f"CRITICAL priority too high: {priority}"

    def test_priority_calculation_deferred(self):
        """Test priority calculation for DEFERRED items."""
        scorer = PriorityScorer()

        item = CrawlItem(
            priority_score=0.0,
            url="https://example.com",
            priority_level=PriorityLevel.DEFERRED,
            estimated_freshness=0.0,
            estimated_relevance=0.0,
            engagement_score=0.0,
        )

        priority = scorer.calculate_priority(item)

        # DEFERRED with min freshness/relevance/engagement should have high score
        assert priority > 0.9, f"DEFERRED priority too low: {priority}"

    def test_priority_score_inversion(self):
        """Test that higher freshness/relevance/engagement = lower priority score."""
        scorer = PriorityScorer()

        # High quality item
        high_quality = CrawlItem(
            priority_score=0.0,
            url="url1",
            priority_level=PriorityLevel.MEDIUM,
            estimated_freshness=1.0,
            estimated_relevance=1.0,
            engagement_score=1.0,
        )

        # Low quality item
        low_quality = CrawlItem(
            priority_score=0.0,
            url="url2",
            priority_level=PriorityLevel.MEDIUM,
            estimated_freshness=0.0,
            estimated_relevance=0.0,
            engagement_score=0.0,
        )


    def test_priority_queue_push_and_pop(self):
        """Test pushing and popping items."""
        pq = PriorityQueue()

        item1 = CrawlItem(priority_score=0.5, url="url1")
        item2 = CrawlItem(priority_score=0.3, url="url2")
        item3 = CrawlItem(priority_score=0.7, url="url3")

        pq.push(item1, recalculate_priority=False)
        pq.push(item2, recalculate_priority=False)
        pq.push(item3, recalculate_priority=False)

        assert pq.size() == 3

        # Pop should return items in priority order (lowest score first)
        popped1 = pq.pop()
        assert popped1.url == "url2"  # priority_score=0.3

        popped2 = pq.pop()
        assert popped2.url == "url1"  # priority_score=0.5

        popped3 = pq.pop()
        assert popped3.url == "url3"  # priority_score=0.7

        assert pq.is_empty() is True

    def test_priority_queue_min_heap_property(self):
        """Test that queue maintains min-heap property."""
        pq = PriorityQueue()

        # Add items in random order
        items = [
            CrawlItem(priority_score=0.8, url="url1"),
            CrawlItem(priority_score=0.2, url="url2"),
            CrawlItem(priority_score=0.5, url="url3"),
            CrawlItem(priority_score=0.1, url="url4"),
            CrawlItem(priority_score=0.9, url="url5"),
        ]

        for item in items:
            pq.push(item, recalculate_priority=False)

        # Pop all items - should come out in sorted order
        popped_scores = []
        while not pq.is_empty():
            item = pq.pop()
            popped_scores.append(item.priority_score)

        # CRITICAL: Must be in ascending order (min-heap)
        assert popped_scores == sorted(popped_scores)

    def test_priority_queue_peek(self):
        """Test peeking at highest priority item."""
        pq = PriorityQueue()

        item1 = CrawlItem(priority_score=0.5, url="url1")
        item2 = CrawlItem(priority_score=0.3, url="url2")

        pq.push(item1, recalculate_priority=False)
        pq.push(item2, recalculate_priority=False)

        # Peek should return highest priority without removing
        peeked = pq.peek()
        assert peeked.url == "url2"
        assert pq.size() == 2  # Size unchanged

        # Pop should return same item
        popped = pq.pop()
        assert popped.url == "url2"
        assert pq.size() == 1

    def test_priority_queue_deduplication(self):
        """Test URL deduplication."""
        pq = PriorityQueue(enable_deduplication=True)

        item1 = CrawlItem(priority_score=0.5, url="https://example.com")
        item2 = CrawlItem(priority_score=0.3, url="https://example.com")  # Duplicate

        assert pq.push(item1, recalculate_priority=False) is True
        assert pq.push(item2, recalculate_priority=False) is False  # Should be rejected

        assert pq.size() == 1
        assert pq.total_duplicates == 1

    def test_priority_queue_no_deduplication(self):
        """Test queue without deduplication."""
        pq = PriorityQueue(enable_deduplication=False)

        item1 = CrawlItem(priority_score=0.5, url="https://example.com")
        item2 = CrawlItem(priority_score=0.3, url="https://example.com")  # Duplicate

        assert pq.push(item1, recalculate_priority=False) is True
        assert pq.push(item2, recalculate_priority=False) is True  # Should be accepted

        assert pq.size() == 2

    def test_priority_queue_max_size(self):
        """Test queue max size enforcement."""
        pq = PriorityQueue(max_size=3)

        item1 = CrawlItem(priority_score=0.1, url="url1")
        item2 = CrawlItem(priority_score=0.2, url="url2")
        item3 = CrawlItem(priority_score=0.3, url="url3")
        item4 = CrawlItem(priority_score=0.4, url="url4")

        assert pq.push(item1, recalculate_priority=False) is True
        assert pq.push(item2, recalculate_priority=False) is True
        assert pq.push(item3, recalculate_priority=False) is True
        assert pq.push(item4, recalculate_priority=False) is False  # Should be rejected

        assert pq.size() == 3

    def test_priority_queue_clear(self):
        """Test clearing the queue."""
        pq = PriorityQueue()

        for i in range(10):
            item = CrawlItem(priority_score=float(i), url=f"url{i}")
            pq.push(item, recalculate_priority=False)

        assert pq.size() == 10

        pq.clear()

        assert pq.size() == 0
        assert pq.is_empty() is True
        assert len(pq.seen_urls) == 0

    def test_priority_queue_update_priority(self):
        """Test updating item priority."""
        pq = PriorityQueue()

        item = CrawlItem(
            priority_score=0.5,
            url="url1",
            priority_level=PriorityLevel.MEDIUM,
        )

        pq.push(item, recalculate_priority=False)

        # Update priority
        success = pq.update_priority(item.item_id, PriorityLevel.CRITICAL)
        assert success is True

        # Item should have new priority level
        assert item.priority_level == PriorityLevel.CRITICAL

    def test_priority_queue_update_nonexistent(self):
        """Test updating nonexistent item."""
        pq = PriorityQueue()

        success = pq.update_priority("nonexistent_id", PriorityLevel.HIGH)
        assert success is False

    def test_priority_queue_get_top_n(self):
        """Test getting top N items."""
        pq = PriorityQueue()

        items = [
            CrawlItem(priority_score=0.5, url="url1"),
            CrawlItem(priority_score=0.2, url="url2"),
            CrawlItem(priority_score=0.8, url="url3"),
            CrawlItem(priority_score=0.1, url="url4"),
            CrawlItem(priority_score=0.9, url="url5"),
        ]

        for item in items:
            pq.push(item, recalculate_priority=False)

        # Get top 3
        top_3 = pq.get_top_n(3)

        assert len(top_3) == 3
        assert top_3[0].priority_score == 0.1
        assert top_3[1].priority_score == 0.2
        assert top_3[2].priority_score == 0.5

        # Queue should be unchanged
        assert pq.size() == 5

    def test_priority_queue_statistics(self):
        """Test queue statistics."""
        pq = PriorityQueue(max_size=100)

        # Add items with different priorities
        pq.push(CrawlItem(priority_score=0.1, url="url1", priority_level=PriorityLevel.CRITICAL), recalculate_priority=False)
        pq.push(CrawlItem(priority_score=0.2, url="url2", priority_level=PriorityLevel.HIGH), recalculate_priority=False)
        pq.push(CrawlItem(priority_score=0.3, url="url3", priority_level=PriorityLevel.MEDIUM), recalculate_priority=False)

        stats = pq.get_statistics()

        assert stats["current_size"] == 3
        assert stats["max_size"] == 100
        assert stats["total_added"] == 3
        assert stats["total_popped"] == 0
        assert stats["utilization"] == 0.03

    def test_priority_queue_empty_pop(self):
        """Test popping from empty queue."""
        pq = PriorityQueue()

        item = pq.pop()
        assert item is None

    def test_priority_queue_empty_peek(self):
        """Test peeking at empty queue."""
        pq = PriorityQueue()

        item = pq.peek()
        assert item is None


class TestPriorityQueueIntegration:
    """Integration tests for priority queue."""

    def test_priority_queue_realistic_crawl(self):
        """Test realistic crawl scenario."""
        pq = PriorityQueue()

        # Add breaking news (CRITICAL)
        breaking_news = CrawlItem(
            priority_score=0.0,
            url="https://news.com/breaking",
            priority_level=PriorityLevel.CRITICAL,
            estimated_freshness=1.0,
            estimated_relevance=1.0,
            engagement_score=1.0,
        )

        # Add recent post (HIGH)
        recent_post = CrawlItem(
            priority_score=0.0,
            url="https://social.com/recent",
            priority_level=PriorityLevel.HIGH,
            estimated_freshness=0.8,
            estimated_relevance=0.7,
            engagement_score=0.6,
        )

        # Add old content (LOW)
        old_content = CrawlItem(
            priority_score=0.0,
            url="https://archive.com/old",
            priority_level=PriorityLevel.LOW,
            estimated_freshness=0.1,
            estimated_relevance=0.2,
            engagement_score=0.1,
        )

        pq.push(old_content)
        pq.push(recent_post)
        pq.push(breaking_news)

        # Should pop in priority order
        first = pq.pop()
        assert first.url == "https://news.com/breaking"

        second = pq.pop()
        assert second.url == "https://social.com/recent"

        third = pq.pop()
        assert third.url == "https://archive.com/old"

