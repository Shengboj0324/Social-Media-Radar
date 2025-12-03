"""Comprehensive unit tests for Probabilistic Data Structures.

Tests Bloom Filter, Count-Min Sketch, and HyperLogLog with peak skepticism.
Verifies algorithm correctness, statistical properties, and edge cases.
"""

import math
import pytest

from app.scraping.probabilistic_structures import (
    BloomFilter,
    CountMinSketch,
    HyperLogLog,
)


class TestBloomFilter:
    """Test Bloom Filter implementation."""

    def test_bloom_filter_initialization(self):
        """Test Bloom filter initialization."""
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)

        assert bf.expected_elements == 1000
        assert bf.false_positive_rate == 0.01
        assert bf.size > 0
        assert bf.num_hashes > 0
        assert len(bf.bit_array) == bf.size
        assert bf.elements_added == 0

    def test_bloom_filter_optimal_size(self):
        """Test optimal size calculation."""
        # Formula: m = -(n * ln(p)) / (ln(2)^2)
        n = 1000
        p = 0.01

        expected_size = -(n * math.log(p)) / (math.log(2) ** 2)
        actual_size = BloomFilter._optimal_size(n, p)

        assert actual_size == int(math.ceil(expected_size))
        assert actual_size > 0

    def test_bloom_filter_optimal_hash_count(self):
        """Test optimal hash count calculation."""
        # Formula: k = (m/n) * ln(2)
        m = 10000
        n = 1000

        expected_hashes = (m / n) * math.log(2)
        actual_hashes = BloomFilter._optimal_hash_count(m, n)

        assert actual_hashes == max(1, int(math.ceil(expected_hashes)))
        assert actual_hashes >= 1

    def test_bloom_filter_add_and_contains(self):
        """Test adding items and checking membership."""
        bf = BloomFilter(expected_elements=100, false_positive_rate=0.01)

        # Add items
        items = ["url1", "url2", "url3", "url4", "url5"]
        for item in items:
            bf.add(item)

        # All added items should be found (no false negatives)
        for item in items:
            assert bf.contains(item) is True

        assert bf.elements_added == 5

    def test_bloom_filter_no_false_negatives(self):
        """Test that Bloom filter has NO false negatives."""
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)

        # Add 100 items
        items = [f"item_{i}" for i in range(100)]
        for item in items:
            bf.add(item)

        # CRITICAL: All added items MUST be found (no false negatives)
        for item in items:
            assert bf.contains(item) is True, f"False negative for {item}"

    def test_bloom_filter_false_positives(self):
        """Test that false positive rate is reasonable."""
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)

        # Add 1000 items
        added_items = [f"added_{i}" for i in range(1000)]
        for item in added_items:
            bf.add(item)

        # Check 1000 items that were NOT added
        not_added_items = [f"not_added_{i}" for i in range(1000)]
        false_positives = sum(1 for item in not_added_items if bf.contains(item))

        # False positive rate should be close to expected (within 5x tolerance)
        actual_fpr = false_positives / len(not_added_items)
        assert actual_fpr < 0.05, f"FPR too high: {actual_fpr}"

    def test_bloom_filter_empty(self):
        """Test Bloom filter with no items."""
        bf = BloomFilter(expected_elements=100, false_positive_rate=0.01)

        # Should not contain any items
        assert bf.contains("nonexistent") is False
        assert bf.elements_added == 0

    def test_bloom_filter_hash_deterministic(self):
        """Test that hash function is deterministic."""
        bf = BloomFilter(expected_elements=100, false_positive_rate=0.01)

        item = "test_item"
        seed = 5

        hash1 = bf._hash(item, seed)
        hash2 = bf._hash(item, seed)
        hash3 = bf._hash(item, seed)

        # CRITICAL: Hash must be deterministic
        assert hash1 == hash2 == hash3

    def test_bloom_filter_hash_different_seeds(self):
        """Test that different seeds produce different hashes."""
        bf = BloomFilter(expected_elements=100, false_positive_rate=0.01)

        item = "test_item"

        hashes = [bf._hash(item, seed) for seed in range(10)]

        # Most hashes should be different
        unique_hashes = len(set(hashes))
        assert unique_hashes >= 8, "Hash function not diverse enough"

    def test_bloom_filter_statistics(self):
        """Test Bloom filter statistics."""
        bf = BloomFilter(expected_elements=100, false_positive_rate=0.01)

        # Add some items
        for i in range(50):
            bf.add(f"item_{i}")

        stats = bf.get_statistics()

        assert stats["size_bits"] == bf.size
        assert stats["num_hashes"] == bf.num_hashes
        assert stats["elements_added"] == 50
        assert stats["bits_set"] > 0
        assert 0 <= stats["utilization"] <= 1


    def test_count_min_sketch_minimum_estimate(self):
        """Test that estimate returns MINIMUM across all hash tables."""
        cms = CountMinSketch(width=100, depth=5)

        # Add item
        cms.update("test_item", 10)

        # Manually check that estimate is minimum
        item = "test_item"
        estimates = []
        for i in range(cms.depth):
            index = cms._hash(item, i) % cms.width
            estimates.append(cms.table[i][index])

        actual_estimate = cms.estimate(item)

        # CRITICAL: Must return minimum
        assert actual_estimate == min(estimates)

    def test_count_min_sketch_hash_deterministic(self):
        """Test that hash function is deterministic."""
        cms = CountMinSketch(width=1000, depth=5)

        item = "test_item"
        seed = 3

        hash1 = cms._hash(item, seed)
        hash2 = cms._hash(item, seed)
        hash3 = cms._hash(item, seed)

        # CRITICAL: Hash must be deterministic
        assert hash1 == hash2 == hash3

    def test_count_min_sketch_statistics(self):
        """Test Count-Min Sketch statistics."""
        cms = CountMinSketch(width=1000, depth=5)

        # Add items
        for i in range(100):
            cms.update(f"item_{i}", i)

        stats = cms.get_statistics()

        assert stats["width"] == 1000
        assert stats["depth"] == 5
        assert stats["total_count"] == sum(range(100))
        assert stats["memory_kb"] > 0


class TestHyperLogLog:
    """Test HyperLogLog implementation."""

    def test_hyperloglog_initialization(self):
        """Test HyperLogLog initialization."""
        hll = HyperLogLog(precision=14)

        assert hll.precision == 14
        assert hll.m == 1 << 14  # 2^14 = 16384
        assert len(hll.registers) == hll.m
        assert all(r == 0 for r in hll.registers)

    def test_hyperloglog_precision_validation(self):
        """Test precision parameter validation."""
        # Valid precision
        hll = HyperLogLog(precision=10)
        assert hll.precision == 10

        # Invalid precision (too low)
        with pytest.raises(ValueError):
            HyperLogLog(precision=3)

        # Invalid precision (too high)
        with pytest.raises(ValueError):
            HyperLogLog(precision=17)

    def test_hyperloglog_alpha_constants(self):
        """Test alpha constant selection."""
        # m >= 128
        hll1 = HyperLogLog(precision=8)  # m = 256
        assert hll1.alpha == 0.7213 / (1 + 1.079 / hll1.m)

        # m >= 64
        hll2 = HyperLogLog(precision=6)  # m = 64
        assert hll2.alpha == 0.709

        # m >= 32
        hll3 = HyperLogLog(precision=5)  # m = 32
        assert hll3.alpha == 0.697

        # m < 32
        hll4 = HyperLogLog(precision=4)  # m = 16
        assert hll4.alpha == 0.673

    def test_hyperloglog_add_and_cardinality(self):
        """Test adding items and estimating cardinality."""
        hll = HyperLogLog(precision=14)

        # Add 1000 unique items
        for i in range(1000):
            hll.add(f"item_{i}")

        cardinality = hll.cardinality()

        # Should be close to 1000 (within 10% error)
        assert 900 <= cardinality <= 1100, f"Cardinality {cardinality} not close to 1000"

    def test_hyperloglog_cardinality_accuracy(self):
        """Test cardinality estimation accuracy."""
        hll = HyperLogLog(precision=14)

        # Test different cardinalities
        test_cases = [100, 500, 1000, 5000, 10000]

        for expected_count in test_cases:
            hll = HyperLogLog(precision=14)
            for i in range(expected_count):
                hll.add(f"item_{i}")

            estimated = hll.cardinality()

            # Should be within 10% error
            error = abs(estimated - expected_count) / expected_count
            assert error < 0.10, f"Error {error:.2%} too high for count {expected_count}"

    def test_hyperloglog_duplicates(self):
        """Test that duplicates don't increase cardinality."""
        hll = HyperLogLog(precision=14)

        # Add same item 100 times
        for _ in range(100):
            hll.add("duplicate_item")

        cardinality = hll.cardinality()

        # Should estimate ~1 unique item
        assert cardinality <= 10, f"Cardinality {cardinality} too high for 1 unique item"

    def test_hyperloglog_register_updates(self):
        """Test that registers are updated with maximum."""
        hll = HyperLogLog(precision=14)

        # Add item
        hll.add("test_item")

        # At least one register should be non-zero
        assert any(r > 0 for r in hll.registers)

        # Save register state
        initial_registers = hll.registers.copy()

        # Add same item again
        hll.add("test_item")

        # Registers should not change (already at maximum)
        assert hll.registers == initial_registers

    def test_hyperloglog_leading_zeros_count(self):
        """Test leading zeros counting."""
        hll = HyperLogLog(precision=14)

        # Test specific values
        assert hll._count_leading_zeros(0) == 64 - 14
        assert hll._count_leading_zeros(1) == 64 - 14 - 1

        # Test power of 2
        value = 1 << 30  # Bit at position 30
        expected_zeros = 64 - 14 - 31  # 64 - precision - (position + 1)
        assert hll._count_leading_zeros(value) == expected_zeros

    def test_hyperloglog_merge(self):
        """Test merging two HyperLogLogs."""
        hll1 = HyperLogLog(precision=14)
        hll2 = HyperLogLog(precision=14)

        # Add different items to each
        for i in range(500):
            hll1.add(f"item_{i}")

        for i in range(500, 1000):
            hll2.add(f"item_{i}")

        # Merge
        hll1.merge(hll2)

        cardinality = hll1.cardinality()

        # Should estimate ~1000 unique items
        assert 900 <= cardinality <= 1100, f"Merged cardinality {cardinality} not close to 1000"

    def test_hyperloglog_merge_different_precision(self):
        """Test that merging different precisions raises error."""
        hll1 = HyperLogLog(precision=14)
        hll2 = HyperLogLog(precision=12)

        with pytest.raises(ValueError):
            hll1.merge(hll2)

    def test_hyperloglog_merge_takes_maximum(self):
        """Test that merge takes maximum of registers."""
        hll1 = HyperLogLog(precision=10)
        hll2 = HyperLogLog(precision=10)

        # Add items
        hll1.add("item1")
        hll2.add("item2")

        # Save initial registers
        initial_hll1 = hll1.registers.copy()
        initial_hll2 = hll2.registers.copy()

        # Merge
        hll1.merge(hll2)

        # Check that each register is maximum
        for i in range(hll1.m):
            assert hll1.registers[i] == max(initial_hll1[i], initial_hll2[i])

    def test_hyperloglog_hash_deterministic(self):
        """Test that hash function is deterministic."""
        hll = HyperLogLog(precision=14)

        item = "test_item"

        hash1 = hll._hash(item)
        hash2 = hll._hash(item)
        hash3 = hll._hash(item)

        # CRITICAL: Hash must be deterministic
        assert hash1 == hash2 == hash3

    def test_hyperloglog_statistics(self):
        """Test HyperLogLog statistics."""
        hll = HyperLogLog(precision=14)

        # Add items
        for i in range(1000):
            hll.add(f"item_{i}")

        stats = hll.get_statistics()

        assert stats["precision"] == 14
        assert stats["num_registers"] == 1 << 14
        assert stats["memory_kb"] > 0
        assert stats["estimated_cardinality"] > 0
        assert stats["standard_error"] > 0


class TestProbabilisticStructuresIntegration:
    """Integration tests for all probabilistic structures."""

    def test_bloom_filter_url_deduplication(self):
        """Test Bloom filter for URL deduplication use case."""
        bf = BloomFilter(expected_elements=10000, false_positive_rate=0.001)

        # Simulate URL crawling
        urls = [f"https://example.com/page_{i}" for i in range(1000)]

        # Add URLs
        for url in urls:
            bf.add(url)

        # Check all URLs are found
        for url in urls:
            assert bf.contains(url) is True

        # Check new URLs are not found
        new_urls = [f"https://example.com/new_page_{i}" for i in range(100)]
        false_positives = sum(1 for url in new_urls if bf.contains(url))

        # Should have very few false positives
        assert false_positives < 5

    def test_count_min_sketch_hashtag_frequency(self):
        """Test Count-Min Sketch for hashtag frequency tracking."""
        cms = CountMinSketch(width=1000, depth=5)

        # Simulate hashtag stream
        hashtags = ["#python"] * 100 + ["#ai"] * 50 + ["#ml"] * 75

        for tag in hashtags:
            cms.update(tag)

        # Estimate frequencies
        python_freq = cms.estimate("#python")
        ai_freq = cms.estimate("#ai")
        ml_freq = cms.estimate("#ml")

        # Should not underestimate
        assert python_freq >= 100
        assert ai_freq >= 50
        assert ml_freq >= 75

    def test_hyperloglog_unique_users(self):
        """Test HyperLogLog for unique user counting."""
        hll = HyperLogLog(precision=14)

        # Simulate user visits (with duplicates)
        user_ids = [f"user_{i % 1000}" for i in range(10000)]  # 1000 unique users

        for user_id in user_ids:
            hll.add(user_id)

        cardinality = hll.cardinality()

        # Should estimate ~1000 unique users
        assert 900 <= cardinality <= 1100

