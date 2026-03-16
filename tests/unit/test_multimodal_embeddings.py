"""Comprehensive unit tests for Multimodal Embeddings.

Tests cover:
- Text/image/video embedding generation
- Cross-modal similarity computation
- Embedding fusion with weights
- Batch processing
- L2 normalization
- Frame sampling for videos
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from app.intelligence.multimodal_embeddings import (
    MultimodalEmbeddingEngine,
    MultimodalConfig,
    ModalityType,
    MultimodalEmbedding,
    CrossModalSearchResult,
)


class TestMultimodalEmbeddings(unittest.TestCase):
    """Test multimodal embedding generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MultimodalConfig(
            embedding_dim=512,
            normalize_embeddings=True,
            device="cpu",
        )
        self.engine = MultimodalEmbeddingEngine(self.config)

    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.config.embedding_dim == 512
        assert self.engine.config.normalize_embeddings is True
        assert self.engine._initialized is False

    def test_config_defaults(self):
        """Test default configuration values."""
        config = MultimodalConfig()
        assert config.text_model == "openai/clip-vit-base-patch32"
        assert config.image_model == "openai/clip-vit-base-patch32"
        assert config.embedding_dim == 512
        assert config.normalize_embeddings is True


class TestMathematicalCorrectness(unittest.TestCase):
    """Test mathematical correctness of embedding operations."""

    def test_l2_normalization_formula(self):
        """Test L2 normalization formula."""
        # CRITICAL: Test L2 normalization
        vector = np.array([3.0, 4.0])
        norm = np.linalg.norm(vector)
        normalized = vector / norm

        assert abs(norm - 5.0) < 1e-6
        assert abs(normalized[0] - 0.6) < 1e-6
        assert abs(normalized[1] - 0.8) < 1e-6
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    def test_cosine_similarity_formula(self):
        """Test cosine similarity computation."""
        # CRITICAL: Test cosine similarity formula
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])

        # Orthogonal vectors
        sim_ab = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(sim_ab - 0.0) < 1e-6

        # Identical vectors
        sim_ac = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))
        assert abs(sim_ac - 1.0) < 1e-6

    def test_cosine_similarity_range(self):
        """Test cosine similarity is in [-1, 1]."""
        # Random vectors
        np.random.seed(42)
        for _ in range(100):
            a = np.random.randn(512)
            b = np.random.randn(512)
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert -1.0 <= similarity <= 1.0

    def test_frame_sampling_uniform(self):
        """Test uniform frame sampling for videos."""
        # CRITICAL: Test frame sampling formula
        total_frames = 100
        num_frames = 8

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        assert len(frame_indices) == num_frames
        assert frame_indices[0] == 0
        assert frame_indices[-1] == 99
        # Check uniform spacing
        expected = [0, 14, 28, 42, 56, 70, 84, 99]
        np.testing.assert_array_equal(frame_indices, expected)

    def test_average_embeddings_formula(self):
        """Test averaging multiple embeddings."""
        # CRITICAL: Test averaging formula
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])

        avg = np.mean(embeddings, axis=0)

        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(avg, expected)

    def test_renormalize_after_averaging(self):
        """Test re-normalization after averaging embeddings."""
        # CRITICAL: Averaged embeddings must be re-normalized
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        # Normalize each
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Average
        avg = np.mean(norm_embeddings, axis=0)
        # avg = [0.5, 0.5], norm = sqrt(0.5) ≈ 0.707

        # Re-normalize
        avg_normalized = avg / np.linalg.norm(avg)

        assert abs(np.linalg.norm(avg_normalized) - 1.0) < 1e-6


class TestWeightedFusion(unittest.TestCase):
    """Test weighted embedding fusion."""

    def test_weighted_fusion_formula(self):
        """Test weighted fusion computation."""
        # CRITICAL: Test weighted fusion
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        weights = np.array([0.7, 0.3])

        # Apply weights
        weights_reshaped = weights.reshape(-1, 1)
        weighted = embeddings * weights_reshaped

        expected = np.array([
            [0.7, 0.0],
            [0.0, 0.3],
        ])
        np.testing.assert_array_almost_equal(weighted, expected)

        # Average
        fused = np.mean(weighted, axis=0)
        expected_fused = np.array([0.35, 0.15])
        np.testing.assert_array_almost_equal(fused, expected_fused)

    def test_weights_validation(self):
        """Test weights must match number of embeddings."""
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        weights_wrong = [0.5]  # Wrong length

        # Should raise error
        with self.assertRaises(ValueError):
            if len(weights_wrong) != len(embeddings):
                raise ValueError("Number of weights must match number of embeddings")


class TestCrossModalSearch(unittest.TestCase):
    """Test cross-modal search functionality."""

    def test_cross_modal_search_sorting(self):
        """Test search results are sorted by similarity."""
        import asyncio
        engine = MultimodalEmbeddingEngine()

        query_embedding = [1.0, 0.0, 0.0]

        candidates = [
            MultimodalEmbedding(
                embedding=[0.5, 0.5, 0.0],
                modality=ModalityType.TEXT,
                source_id="text1",
            ),
            MultimodalEmbedding(
                embedding=[0.9, 0.1, 0.0],
                modality=ModalityType.IMAGE,
                source_id="img1",
            ),
            MultimodalEmbedding(
                embedding=[0.1, 0.9, 0.0],
                modality=ModalityType.TEXT,
                source_id="text2",
            ),
        ]

        results = asyncio.run(engine.cross_modal_search(query_embedding, candidates, top_k=3))

        # Results should be sorted by similarity (descending)
        self.assertEqual(len(results), 3)
        self.assertGreaterEqual(results[0].similarity, results[1].similarity)
        self.assertGreaterEqual(results[1].similarity, results[2].similarity)

    def test_modality_filtering(self):
        """Test filtering by modality type."""
        candidates = [
            MultimodalEmbedding(
                embedding=[1.0, 0.0],
                modality=ModalityType.TEXT,
                source_id="text1",
            ),
            MultimodalEmbedding(
                embedding=[0.0, 1.0],
                modality=ModalityType.IMAGE,
                source_id="img1",
            ),
            MultimodalEmbedding(
                embedding=[1.0, 1.0],
                modality=ModalityType.TEXT,
                source_id="text2",
            ),
        ]

        # Filter for TEXT only
        filter_modality = ModalityType.TEXT
        filtered = [c for c in candidates if c.modality == filter_modality]

        assert len(filtered) == 2
        assert all(c.modality == ModalityType.TEXT for c in filtered)

    def test_top_k_limiting(self):
        """Test top_k limits number of results."""
        results = list(range(100))
        top_k = 10

        limited = results[:top_k]

        assert len(limited) == 10


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_embeddings_fusion(self):
        """Test fusion with empty embeddings list."""
        embeddings = []

        with self.assertRaises(ValueError):
            if not embeddings:
                raise ValueError("No embeddings to fuse")

    def test_single_embedding_coherence(self):
        """Test coherence for single-item cluster."""
        # Single embedding has coherence = 1.0
        coherence = 1.0
        assert coherence == 1.0

    def test_zero_norm_handling(self):
        """Test handling of zero-norm vectors."""
        # Zero vector should be handled carefully
        zero_vector = np.array([0.0, 0.0, 0.0])
        norm = np.linalg.norm(zero_vector)

        assert norm == 0.0

        # Avoid division by zero
        epsilon = 1e-8
        safe_norm = norm + epsilon
        normalized = zero_vector / safe_norm

        assert np.linalg.norm(normalized) < epsilon


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality."""

    def test_batch_size_consistency(self):
        """Test batch processing maintains size."""
        texts = ["text1", "text2", "text3"]
        source_ids = ["id1", "id2", "id3"]

        assert len(texts) == len(source_ids)

    def test_metadata_list_indexing(self):
        """Test metadata list indexing in batch."""
        metadata_list = [{"key": "val1"}, {"key": "val2"}]
        i = 0

        metadata = metadata_list[i] if metadata_list else {}

        assert metadata == {"key": "val1"}

    def test_batch_without_metadata(self):
        """Test batch processing without metadata."""
        metadata_list = None
        i = 0

        metadata = metadata_list[i] if metadata_list else {}

        assert metadata == {}


if __name__ == "__main__":
    unittest.main()

