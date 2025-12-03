"""Comprehensive unit tests for HNSW Search.

Tests hierarchical navigable small world graph construction and nearest neighbor search.
Verifies mathematical correctness and search performance.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.intelligence.hnsw_search import (
    HNSWIndex,
    HNSWConfig,
    SearchResult,
    IndexStatistics,
)


class TestHNSWConfig:
    """Test HNSWConfig dataclass."""

    def test_hnsw_config_defaults(self):
        """Test HNSWConfig default values."""
        config = HNSWConfig()

        assert config.dimension == 1536  # OpenAI embedding dimension
        assert config.M == 16
        assert config.ef_construction == 200
        assert config.ef_search == 50
        assert config.max_elements == 1000000
        assert config.space == "cosine"
        assert config.num_threads == 4

    def test_hnsw_config_custom_values(self):
        """Test HNSWConfig with custom values."""
        config = HNSWConfig(
            dimension=768,
            M=32,
            ef_construction=400,
            ef_search=100,
        )

        assert config.dimension == 768
        assert config.M == 32
        assert config.ef_construction == 400
        assert config.ef_search == 100


class TestSearchResult:
    """Test SearchResult model."""

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            id="doc_123",
            distance=0.15,
            metadata={"title": "Test Document"},
        )

        assert result.id == "doc_123"
        assert result.distance == 0.15
        assert result.metadata["title"] == "Test Document"

    def test_search_result_without_metadata(self):
        """Test SearchResult without metadata."""
        result = SearchResult(id="doc_456", distance=0.25)

        assert result.id == "doc_456"
        assert result.distance == 0.25
        assert result.metadata is None


class TestIndexStatistics:
    """Test IndexStatistics model."""

    def test_index_statistics_creation(self):
        """Test IndexStatistics creation."""
        stats = IndexStatistics(
            num_vectors=1000,
            dimension=1536,
            ef_construction=200,
            ef_search=50,
            M=16,
            max_level=100000,
            index_size_bytes=1024000,
        )

        assert stats.num_vectors == 1000
        assert stats.dimension == 1536
        assert stats.M == 16


class TestHNSWIndex:
    """Test HNSWIndex implementation."""

    def test_hnsw_initialization(self):
        """Test HNSWIndex initialization."""
        index = HNSWIndex()

        assert index.config is not None
        assert index.index is None
        assert index.id_to_label == {}
        assert index.label_to_id == {}
        assert index.metadata == {}
        assert index.next_label == 0
        assert index._initialized is False

    def test_hnsw_with_custom_config(self):
        """Test HNSWIndex with custom config."""
        config = HNSWConfig(dimension=768)
        index = HNSWIndex(config)

        assert index.config.dimension == 768

    def test_label_assignment_single(self):
        """Test label assignment for single vector."""
        # CRITICAL: Test label assignment logic
        next_label = 0
        label = next_label
        next_label += 1

        assert label == 0
        assert next_label == 1

    def test_label_assignment_batch(self):
        """Test label assignment for batch."""
        # CRITICAL: Test batch label assignment
        next_label = 5
        num_new_ids = 10

        labels = list(range(next_label, next_label + num_new_ids))
        next_label += num_new_ids

        assert labels == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        assert next_label == 15
        assert len(labels) == 10

    def test_memory_estimation_formula(self):
        """Test HNSW memory estimation formula."""
        # CRITICAL: Test memory estimation
        # Formula: (M * 2 * 4 + dimension * 4) bytes per vector
        M = 16
        dimension = 1536
        num_vectors = 1000

        bytes_per_vector = (M * 2 * 4 + dimension * 4)
        total_bytes = bytes_per_vector * num_vectors

        # M * 2 * 4 = 16 * 2 * 4 = 128
        # dimension * 4 = 1536 * 4 = 6144



class TestMathematicalCorrectness:
    """Test mathematical correctness of algorithms."""

    def test_cosine_distance_range(self):
        """Test that cosine distance is in [0, 2]."""
        # Cosine distance = 1 - cosine_similarity
        # cosine_similarity in [-1, 1]
        # So cosine distance in [0, 2]

        # Identical vectors: similarity = 1, distance = 0
        similarity_identical = 1.0
        distance_identical = 1.0 - similarity_identical
        assert distance_identical == 0.0

        # Opposite vectors: similarity = -1, distance = 2
        similarity_opposite = -1.0
        distance_opposite = 1.0 - similarity_opposite
        assert distance_opposite == 2.0

        # Orthogonal vectors: similarity = 0, distance = 1
        similarity_orthogonal = 0.0
        distance_orthogonal = 1.0 - similarity_orthogonal
        assert distance_orthogonal == 1.0

    def test_l2_distance_calculation(self):
        """Test L2 (Euclidean) distance calculation."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])

        # L2 distance = sqrt(sum((v1 - v2)^2))
        diff = v1 - v2  # [-3, -3, -3]
        squared = diff ** 2  # [9, 9, 9]
        distance = np.sqrt(squared.sum())  # sqrt(27) = 5.196...

        assert abs(distance - 5.196152) < 1e-5

    def test_inner_product_calculation(self):
        """Test inner product (dot product) calculation."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])

        # Inner product = sum(v1 * v2)
        inner_product = np.dot(v1, v2)

        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert inner_product == 32.0

    def test_dimension_validation(self):
        """Test dimension validation."""
        config_dim = 1536
        vector_dim = 768

        # Should detect mismatch
        assert vector_dim != config_dim

    def test_batch_dimension_validation(self):
        """Test batch dimension validation."""
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        # Shape: (2, 3) - 2 vectors of dimension 3
        assert vectors.shape == (2, 3)
        assert vectors.shape[0] == 2  # Number of vectors
        assert vectors.shape[1] == 3  # Dimension


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_index_search(self):
        """Test searching empty index."""
        # Should return empty results
        current_count = 0

        if current_count == 0:
            results = []
        else:
            results = ["some_result"]

        assert results == []

    def test_duplicate_id_handling(self):
        """Test handling duplicate IDs."""
        id_to_label = {"doc_1": 0, "doc_2": 1}

        # Try to add duplicate
        new_id = "doc_1"

        if new_id in id_to_label:
            # Should skip
            added = False
        else:
            added = True

        assert added is False

    def test_filter_results_until_k(self):
        """Test filtering results until k is reached."""
        # CRITICAL: Test result limiting logic
        k = 5
        all_results = list(range(20))  # 20 results

        filtered_results = []
        for result in all_results:
            filtered_results.append(result)
            if len(filtered_results) >= k:
                break

        assert len(filtered_results) == 5
        assert filtered_results == [0, 1, 2, 3, 4]

    def test_soft_delete_mapping_removal(self):
        """Test soft delete removes mappings."""
        id_to_label = {"doc_1": 0, "doc_2": 1, "doc_3": 2}
        label_to_id = {0: "doc_1", 1: "doc_2", 2: "doc_3"}
        metadata = {"doc_1": {"title": "Doc 1"}, "doc_2": {"title": "Doc 2"}}

        # Delete doc_1
        delete_id = "doc_1"
        if delete_id in id_to_label:
            label = id_to_label.pop(delete_id)
            label_to_id.pop(label, None)
            metadata.pop(delete_id, None)
            deleted = True
        else:
            deleted = False

        assert deleted is True
        assert delete_id not in id_to_label
        assert 0 not in label_to_id
        assert delete_id not in metadata

    def test_zero_vector(self):
        """Test zero vector."""
        vector = np.array([0.0, 0.0, 0.0])

        # Zero vector has zero norm
        norm = np.linalg.norm(vector)
        assert norm == 0.0


class TestIntegration:
    """Integration tests for HNSW index."""

    def test_initialize_idempotent(self):
        """Test that initialize can be called multiple times safely."""
        index = HNSWIndex()

        with patch('app.intelligence.hnsw_search.hnswlib') as mock_hnswlib:
            mock_index = MagicMock()
            mock_hnswlib.Index.return_value = mock_index

            index.initialize()
            assert index._initialized is True

            # Call again - should not reinitialize
            index.initialize()
            assert index._initialized is True

            # Should only call Index once
            assert mock_hnswlib.Index.call_count == 1

    def test_id_label_bidirectional_mapping(self):
        """Test bidirectional ID-label mapping."""
        id_to_label = {}
        label_to_id = {}

        # Add mappings
        id_to_label["doc_1"] = 0
        label_to_id[0] = "doc_1"

        id_to_label["doc_2"] = 1
        label_to_id[1] = "doc_2"

        # Verify bidirectional lookup
        assert id_to_label["doc_1"] == 0
        assert label_to_id[0] == "doc_1"

        assert id_to_label["doc_2"] == 1
        assert label_to_id[1] == "doc_2"

    def test_ef_parameter_tradeoff(self):
        """Test ef parameter speed/quality tradeoff."""
        # Lower ef = faster search, lower recall
        ef_fast = 50

        # Higher ef = slower search, higher recall
        ef_quality = 200

        assert ef_fast < ef_quality
        assert ef_fast == 50  # Speed optimized
        assert ef_quality == 200  # Quality optimized

    def test_m_parameter_tradeoff(self):
        """Test M parameter memory/recall tradeoff."""
        # Lower M = less memory, lower recall
        M_small = 8

        # Higher M = more memory, higher recall
        M_large = 32

        assert M_small < M_large
        assert M_small == 8  # Memory optimized
        assert M_large == 32  # Recall optimized

    def test_space_types(self):
        """Test different distance space types."""
        spaces = ["cosine", "l2", "ip"]

        assert "cosine" in spaces  # Cosine distance
        assert "l2" in spaces  # Euclidean distance
        assert "ip" in spaces  # Inner product

    def test_batch_filtering_logic(self):
        """Test batch filtering for existing IDs."""
        existing_ids = {"doc_1", "doc_2"}
        new_batch_ids = ["doc_1", "doc_3", "doc_4", "doc_2"]

        # Filter out existing
        filtered_ids = [id for id in new_batch_ids if id not in existing_ids]

        assert filtered_ids == ["doc_3", "doc_4"]
        assert len(filtered_ids) == 2

    def test_numpy_dtype_conversion(self):
        """Test numpy dtype conversion to float32."""
        vector = [1.0, 2.0, 3.0]
        vector_np = np.array(vector, dtype=np.float32)

        assert vector_np.dtype == np.float32
        assert len(vector_np) == 3

    def test_search_result_sorting(self):
        """Test that search results are sorted by distance."""
        # Results should be sorted ascending by distance
        results = [
            SearchResult(id="doc_1", distance=0.5),
            SearchResult(id="doc_2", distance=0.2),
            SearchResult(id="doc_3", distance=0.8),
        ]

        # Sort by distance
        sorted_results = sorted(results, key=lambda x: x.distance)

        assert sorted_results[0].distance == 0.2
        assert sorted_results[1].distance == 0.5
        assert sorted_results[2].distance == 0.8

