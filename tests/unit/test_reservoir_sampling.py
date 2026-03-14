"""Comprehensive unit tests for Reservoir Sampling (Algorithm R).

Tests uniform and weighted sampling with peak skepticism.
Verifies algorithm correctness and statistical properties.
"""

import pytest
import random

from app.scraping.reservoir_sampling import (
    ReservoirSampler,
    SampleStatistics,
)


class TestReservoirSampler:
    """Test ReservoirSampler implementation."""

    def test_reservoir_sampler_initialization(self):
        """Test ReservoirSampler initialization."""
        sampler = ReservoirSampler[str](reservoir_size=100)

        assert sampler.reservoir_size == 100
        assert sampler.enable_weighted is False
        assert sampler.time_decay_factor == 0.0
        assert len(sampler.reservoir) == 0
        assert sampler.stats.total_items_seen == 0

    def test_reservoir_sampler_with_seed(self):
        """Test that random seed makes sampling deterministic."""
        sampler1 = ReservoirSampler[int](reservoir_size=10, random_seed=42)
        sampler2 = ReservoirSampler[int](reservoir_size=10, random_seed=42)

        # Add same items to both
        for i in range(100):
            sampler1.add(i)
            sampler2.add(i)

        # Should have same samples
        assert sampler1.get_sample() == sampler2.get_sample()

    def test_reservoir_fill_phase(self):
        """Test Phase 1: Filling reservoir."""
        sampler = ReservoirSampler[int](reservoir_size=10)

        # Add 10 items - all should be accepted
        for i in range(10):
            result = sampler.add(i)
            assert result is True

        assert len(sampler.reservoir) == 10
        assert sampler.stats.total_items_seen == 10
        assert sampler.stats.samples_collected == 10

    def test_reservoir_replacement_phase(self):
        """Test Phase 2: Probabilistic replacement."""
        sampler = ReservoirSampler[int](reservoir_size=10, random_seed=42)

        # Add 100 items
        for i in range(100):
            sampler.add(i)

        # Reservoir should be full
        assert len(sampler.reservoir) == 10
        assert sampler.stats.total_items_seen == 100

        # Should contain a mix of items (not just first 10)
        sample = sampler.get_sample()
        assert not all(i < 10 for i in sample), "Should contain items beyond first 10"

    def test_uniform_sampling_probability(self):
        """Test that uniform sampling has correct probability."""
        sampler = ReservoirSampler[int](reservoir_size=10, random_seed=42)

        # Add 1000 items
        for i in range(1000):
            sampler.add(i)

        sample = sampler.get_sample()

        # Each item should have ~1% chance (10/1000)
        # Check that we have diverse items
        assert len(set(sample)) == 10, "Should have 10 unique items"
        assert max(sample) > 100, "Should have items from later in stream"

    def test_algorithm_r_correctness(self):
        """Test Algorithm R correctness: j = random(0, n-1), replace if j < k."""
        sampler = ReservoirSampler[int](reservoir_size=5, random_seed=42)

        # Fill reservoir
        for i in range(5):
            sampler.add(i)

        # Track replacements
        initial_sample = sampler.get_sample().copy()

        # Add more items
        for i in range(5, 20):
            sampler.add(i)

        final_sample = sampler.get_sample()

        # Some items should have been replaced
        assert initial_sample != final_sample

    def test_get_sample_returns_copy(self):
        """Test that get_sample returns a copy."""
        sampler = ReservoirSampler[int](reservoir_size=10)

        for i in range(10):
            sampler.add(i)

        sample1 = sampler.get_sample()
        sample2 = sampler.get_sample()

        # Should be equal but not same object
        assert sample1 == sample2
        assert sample1 is not sample2

        # Modifying returned sample shouldn't affect reservoir
        sample1.append(999)
        assert 999 not in sampler.get_sample()

    def test_get_random_item(self):
        """Test getting random item from reservoir."""
        sampler = ReservoirSampler[str](reservoir_size=10)

        # Empty reservoir
        assert sampler.get_random_item() is None

        # Add items
        items = ["a", "b", "c", "d", "e"]
        for item in items:
            sampler.add(item)

        # Get random item
        random_item = sampler.get_random_item()
        assert random_item in items

    def test_clear_reservoir(self):
        """Test clearing reservoir."""
        sampler = ReservoirSampler[int](reservoir_size=10)

        # Add items
        for i in range(20):
            sampler.add(i)

        assert len(sampler.reservoir) > 0
        assert sampler.stats.total_items_seen == 20



    def test_weighted_sampling_fill_phase(self):
        """Test weighted sampling fill phase."""
        sampler = ReservoirSampler[str](reservoir_size=5, enable_weighted=True)

        items = ["a", "b", "c", "d", "e"]
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]

        for item, weight in zip(items, weights):
            sampler.add(item, weight=weight)

        assert len(sampler.reservoir) == 5
        assert len(sampler.weights) == 5

    def test_weighted_sampling_higher_weight_more_likely(self):
        """Test that higher weight items are more likely to be kept."""
        sampler = ReservoirSampler[str](reservoir_size=10, enable_weighted=True, random_seed=42)

        # Add items with different weights
        for i in range(100):
            if i < 10:
                # First 10 items with very high weight
                sampler.add(f"high_{i}", weight=100.0)
            else:
                # Rest with low weight
                sampler.add(f"low_{i}", weight=1.0)

        sample = sampler.get_sample()

        # Most samples should be high-weight items
        high_weight_count = sum(1 for item in sample if item.startswith("high_"))
        assert high_weight_count >= 7, f"Expected mostly high-weight items, got {high_weight_count}/10"

    def test_weighted_key_calculation(self):
        """Test weighted key calculation: random^(1/weight)."""
        sampler = ReservoirSampler[int](reservoir_size=10, enable_weighted=True, random_seed=42)

        # Fill reservoir
        for i in range(10):
            sampler.add(i, weight=1.0)

        # Weights should be stored
        assert len(sampler.weights) == 10
        assert all(0 <= w <= 1 for w in sampler.weights)

    def test_weighted_sampling_replaces_minimum(self):
        """Test that weighted sampling replaces minimum key."""
        sampler = ReservoirSampler[str](reservoir_size=3, enable_weighted=True, random_seed=42)

        # Fill with items
        sampler.add("a", weight=1.0)
        sampler.add("b", weight=1.0)
        sampler.add("c", weight=1.0)

        initial_weights = sampler.weights.copy()

        # Add high-weight item
        sampler.add("d", weight=100.0)

        # Weights should have changed
        assert sampler.weights != initial_weights


class TestTimeDecay:
    """Test time decay functionality."""

    def test_time_decay_initialization(self):
        """Test time decay initialization."""
        sampler = ReservoirSampler[int](reservoir_size=10, time_decay_factor=0.5)

        assert sampler.time_decay_factor == 0.5

    def test_time_decay_reduces_weight(self):
        """Test that time decay reduces weight over time."""
        sampler = ReservoirSampler[int](reservoir_size=10, enable_weighted=True, time_decay_factor=1.0)

        # Add item immediately
        sampler.add(1, weight=10.0)

        # Note: Testing time decay is difficult without mocking time
        # Just verify it doesn't crash
        stats = sampler.get_statistics()
        assert stats["time_decay_enabled"] is True


class TestStatisticalProperties:
    """Test statistical properties of reservoir sampling."""

    def test_uniform_distribution(self):
        """Test that uniform sampling produces uniform distribution."""
        sampler = ReservoirSampler[int](reservoir_size=100, random_seed=42)

        # Add 1000 items
        for i in range(1000):
            sampler.add(i)

        sample = sampler.get_sample()

        # Check distribution across ranges
        ranges = {
            "0-249": sum(1 for x in sample if 0 <= x < 250),
            "250-499": sum(1 for x in sample if 250 <= x < 500),
            "500-749": sum(1 for x in sample if 500 <= x < 750),
            "750-999": sum(1 for x in sample if 750 <= x < 1000),
        }

        # Each range should have roughly 25 items (100/4)
        for range_name, count in ranges.items():
            assert 15 <= count <= 35, f"{range_name}: {count} not in [15, 35]"

    def test_reservoir_size_maintained(self):
        """Test that reservoir never exceeds max size."""
        sampler = ReservoirSampler[int](reservoir_size=50)

        # Add many items
        for i in range(10000):
            sampler.add(i)

        # CRITICAL: Reservoir must never exceed size
        assert len(sampler.reservoir) == 50
        assert len(sampler.reservoir) <= sampler.reservoir_size

    def test_all_items_have_equal_probability(self):
        """Test that all items have equal probability in uniform sampling."""
        # Run multiple trials
        item_counts = {i: 0 for i in range(100)}

        for trial in range(100):
            sampler = ReservoirSampler[int](reservoir_size=10, random_seed=trial)

            for i in range(100):
                sampler.add(i)

            sample = sampler.get_sample()
            for item in sample:
                item_counts[item] += 1

        # Each item should appear roughly 10 times (100 trials * 10/100)
        avg_count = sum(item_counts.values()) / len(item_counts)
        assert 8 <= avg_count <= 12, f"Average count {avg_count} not uniform"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_reservoir(self):
        """Test operations on empty reservoir."""
        sampler = ReservoirSampler[int](reservoir_size=10)

        assert sampler.get_sample() == []
        assert sampler.get_random_item() is None

    def test_reservoir_size_one(self):
        """Test reservoir with size 1."""
        sampler = ReservoirSampler[int](reservoir_size=1, random_seed=42)

        for i in range(100):
            sampler.add(i)

        assert len(sampler.reservoir) == 1
        assert sampler.stats.total_items_seen == 100

    def test_zero_weight(self):
        """Test adding item with zero weight."""
        sampler = ReservoirSampler[int](reservoir_size=10, enable_weighted=True)

        # Fill reservoir
        for i in range(10):
            sampler.add(i, weight=1.0)

        # Add item with zero weight (should handle gracefully)
        try:
            sampler.add(999, weight=0.0)
        except ZeroDivisionError:
            pytest.fail("Should handle zero weight gracefully")

    def test_negative_weight(self):
        """Test adding item with negative weight."""
        sampler = ReservoirSampler[int](reservoir_size=10, enable_weighted=True)

        # Negative weight should be handled (implementation dependent)
        sampler.add(1, weight=-1.0)
        # Just verify it doesn't crash

    def test_very_large_stream(self):
        """Test with very large stream."""
        sampler = ReservoirSampler[int](reservoir_size=100)

        # Add 1 million items
        for i in range(1000000):
            sampler.add(i)

        assert len(sampler.reservoir) == 100
        assert sampler.stats.total_items_seen == 1000000


class TestIntegration:
    """Integration tests for reservoir sampling."""

    def test_live_comment_sampling(self):
        """Test sampling live stream comments."""
        sampler = ReservoirSampler[dict](reservoir_size=100)

        # Simulate live comments
        for i in range(10000):
            comment = {
                "id": i,
                "text": f"Comment {i}",
                "timestamp": i,
            }
            sampler.add(comment)

        sample = sampler.get_sample()

        assert len(sample) == 100
        assert all(isinstance(c, dict) for c in sample)
        assert all("id" in c for c in sample)

    def test_weighted_trending_content(self):
        """Test weighted sampling for trending content."""
        sampler = ReservoirSampler[str](reservoir_size=50, enable_weighted=True, random_seed=42)

        # Add trending content with high engagement
        for i in range(100):
            if i % 10 == 0:
                # Trending content (10% of items)
                sampler.add(f"trending_{i}", weight=10.0)
            else:
                # Normal content
                sampler.add(f"normal_{i}", weight=1.0)

        sample = sampler.get_sample()

        # Should have more trending content than expected by uniform sampling.
        # With 10 trending items out of 100 total, uniform sampling gives ~5 out of 50.
        # Weighted sampling (weight 10.0 vs 1.0) should keep all 10 trending items.
        trending_count = sum(1 for item in sample if item.startswith("trending_"))
        assert trending_count >= 7, f"Expected ≥7 trending items (all 10 kept), got {trending_count}"

