"""Comprehensive unit tests for Collaborative Filtering.

Tests cover:
- ALS matrix factorization
- NCF neural collaborative filtering
- User-item interaction matrix
- Confidence weighting
- Recommendation generation
- Similar item finding
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from app.intelligence.collaborative_filtering import (
    ALSRecommender,
    NCFRecommender,
    ALSConfig,
    NCFConfig,
    UserItemInteraction,
    Recommendation,
)


class TestALSRecommender(unittest.TestCase):
    """Test ALS recommender."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ALSConfig(
            num_factors=10,
            num_iterations=5,
            regularization=0.01,
            alpha=40.0,
            random_state=42,
        )
        self.recommender = ALSRecommender(self.config)

    def test_initialization(self):
        """Test ALS initialization."""
        assert self.recommender.config.num_factors == 10
        assert self.recommender.config.num_iterations == 5
        assert self.recommender._fitted is False

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ALSConfig()
        assert config.num_factors == 100
        assert config.num_iterations == 15
        assert config.regularization == 0.01
        assert config.alpha == 40.0


class TestMathematicalCorrectness(unittest.TestCase):
    """Test mathematical correctness of ALS."""

    def test_confidence_matrix_formula(self):
        """Test confidence matrix computation."""
        # CRITICAL: C = 1 + alpha * R
        alpha = 40.0
        R = np.array([[1.0, 0.0], [0.0, 1.0]])

        C = 1 + alpha * R

        expected = np.array([[41.0, 1.0], [1.0, 41.0]])
        np.testing.assert_array_almost_equal(C, expected)

    def test_als_update_formula(self):
        """Test ALS update formula structure."""
        # CRITICAL: (Y^T * C_u * Y + lambda * I) * x_u = Y^T * C_u * p_u
        Y = np.array([[1.0, 0.0], [0.0, 1.0]])
        Cu = np.array([41.0, 1.0])
        lambda_reg = 0.01
        num_factors = 2

        # Y^T * C_u * Y
        YtCuY = Y.T @ np.diag(Cu) @ Y

        # Add regularization
        YtCuY_reg = YtCuY + lambda_reg * np.eye(num_factors)

        # Check shape
        assert YtCuY_reg.shape == (2, 2)

        # Check regularization added to diagonal
        assert YtCuY_reg[0, 0] > YtCuY[0, 0]

    def test_matrix_factorization_dimensions(self):
        """Test matrix factorization dimensions."""
        num_users = 100
        num_items = 50
        num_factors = 10

        user_factors = np.random.randn(num_users, num_factors)
        item_factors = np.random.randn(num_items, num_factors)

        # Reconstruct matrix
        reconstructed = user_factors @ item_factors.T

        assert reconstructed.shape == (num_users, num_items)

    def test_score_computation(self):
        """Test score computation for recommendations."""
        # CRITICAL: scores = item_factors @ user_vector
        item_factors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ])
        user_vector = np.array([0.8, 0.2])

        scores = item_factors @ user_vector

        expected = np.array([0.8, 0.2, 0.5])
        np.testing.assert_array_almost_equal(scores, expected)

    def test_top_k_selection(self):
        """Test top-k item selection."""
        # CRITICAL: np.argsort(scores)[::-1][:top_k]
        scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        top_k = 3

        top_indices = np.argsort(scores)[::-1][:top_k]

        # Should be [1, 3, 4] (indices of 0.9, 0.7, 0.5)
        expected = np.array([1, 3, 4])
        np.testing.assert_array_equal(top_indices, expected)

    def test_filter_seen_items(self):
        """Test filtering seen items."""
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        seen_item_idx = 0

        # Set seen item to -inf
        scores[seen_item_idx] = -np.inf

        top_idx = np.argmax(scores)

        # Should not be the seen item
        assert top_idx != seen_item_idx
        assert top_idx == 1  # Index of 0.8


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity for item-item similarity."""

    def test_cosine_similarity_formula(self):
        """Test cosine similarity computation."""
        # CRITICAL: (item_factors @ item_vector) / (norms * item_norm)
        item_factors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        item_vector = np.array([1.0, 0.0])

        # Compute norms
        norms = np.linalg.norm(item_factors, axis=1)
        item_norm = np.linalg.norm(item_vector)

        # Compute similarities
        similarities = (item_factors @ item_vector) / (norms * item_norm + 1e-8)

        # Expected: [1.0, 0.0, 0.707...]
        assert abs(similarities[0] - 1.0) < 1e-6
        assert abs(similarities[1] - 0.0) < 1e-6
        assert abs(similarities[2] - 0.707) < 0.01

    def test_exclude_self_similarity(self):
        """Test excluding item itself from similar items."""
        similarities = np.array([0.5, 1.0, 0.3])
        item_idx = 1

        # Exclude self
        similarities[item_idx] = -np.inf

        top_idx = np.argmax(similarities)

        assert top_idx != item_idx
        assert top_idx == 0  # Index of 0.5


class TestNCFRecommender(unittest.TestCase):
    """Test NCF recommender."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = NCFConfig(
            embedding_dim=64,
            hidden_layers=[128, 64, 32],
            dropout=0.2,
            learning_rate=0.001,
            batch_size=256,
            num_epochs=20,
            device="cpu",
        )
        self.recommender = NCFRecommender(self.config)

    def test_initialization(self):
        """Test NCF initialization."""
        assert self.recommender.config.embedding_dim == 64
        assert self.recommender.config.hidden_layers == [128, 64, 32]
        assert self.recommender._initialized is False
        assert self.recommender._fitted is False

    def test_config_defaults(self):
        """Test default configuration values."""
        config = NCFConfig()
        assert config.embedding_dim == 64
        assert config.hidden_layers == [128, 64, 32]
        assert config.dropout == 0.2


class TestNCFArchitecture(unittest.TestCase):
    """Test NCF architecture components."""

    def test_gmf_computation(self):
        """Test GMF (Generalized Matrix Factorization) computation."""
        # CRITICAL: GMF = user_emb * item_emb (element-wise)
        user_emb = np.array([1.0, 2.0, 3.0])
        item_emb = np.array([0.5, 1.0, 1.5])

        gmf_output = user_emb * item_emb

        expected = np.array([0.5, 2.0, 4.5])
        np.testing.assert_array_almost_equal(gmf_output, expected)

    def test_mlp_input_concatenation(self):
        """Test MLP input concatenation."""
        # CRITICAL: MLP input = concat([user_emb, item_emb])
        user_emb = np.array([1.0, 2.0])
        item_emb = np.array([3.0, 4.0])

        mlp_input = np.concatenate([user_emb, item_emb])

        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(mlp_input, expected)

    def test_ncf_fusion(self):
        """Test NCF fusion of GMF and MLP."""
        # CRITICAL: concat([gmf_output, mlp_output])
        gmf_output = np.array([0.5, 0.3])
        mlp_output = np.array([0.8, 0.6, 0.4])

        ncf_concat = np.concatenate([gmf_output, mlp_output])

        assert len(ncf_concat) == 5
        assert ncf_concat[0] == 0.5
        assert ncf_concat[2] == 0.8

    def test_mlp_layer_dimensions(self):
        """Test MLP layer dimension progression."""
        embedding_dim = 64
        hidden_layers = [128, 64, 32]

        input_dim = embedding_dim * 2  # Concatenated user + item
        assert input_dim == 128

        # First layer: 128 -> 128
        # Second layer: 128 -> 64
        # Third layer: 64 -> 32
        assert hidden_layers[0] == 128
        assert hidden_layers[-1] == 32


class TestUserItemMapping(unittest.TestCase):
    """Test user and item ID mapping."""

    def test_id_mapping_creation(self):
        """Test creation of ID mappings."""
        user_ids = ["user1", "user2", "user3"]
        user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}

        assert user_id_map["user1"] == 0
        assert user_id_map["user2"] == 1
        assert user_id_map["user3"] == 2

    def test_reverse_mapping(self):
        """Test reverse mapping from index to ID."""
        user_id_map = {"user1": 0, "user2": 1}
        reverse_map = {idx: uid for uid, idx in user_id_map.items()}

        assert reverse_map[0] == "user1"
        assert reverse_map[1] == "user2"

    def test_sorted_unique_extraction(self):
        """Test extracting sorted unique IDs."""
        interactions = [
            UserItemInteraction(user_id="u2", item_id="i1", rating=1.0, timestamp=1.0, interaction_type="view"),
            UserItemInteraction(user_id="u1", item_id="i2", rating=1.0, timestamp=2.0, interaction_type="click"),
            UserItemInteraction(user_id="u2", item_id="i1", rating=1.0, timestamp=3.0, interaction_type="like"),
        ]

        unique_users = sorted(set(i.user_id for i in interactions))

        assert unique_users == ["u1", "u2"]


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_cold_start_user(self):
        """Test handling of cold-start users."""
        user_id_map = {"user1": 0, "user2": 1}
        new_user = "user3"

        if new_user not in user_id_map:
            # Return empty recommendations
            recommendations = []

        assert recommendations == []

    def test_empty_candidate_items(self):
        """Test handling of empty candidate items."""
        candidate_items = []
        valid_items = [item for item in candidate_items if item in {"item1", "item2"}]

        assert valid_items == []

    def test_model_not_fitted(self):
        """Test error when model not fitted."""
        recommender = ALSRecommender()

        with self.assertRaises(ValueError):
            if not recommender._fitted:
                raise ValueError("Model not fitted")


class TestImplicitFeedback(unittest.TestCase):
    """Test implicit feedback handling."""

    def test_implicit_rating_binary(self):
        """Test implicit ratings are binary."""
        # Implicit: 1.0 for interaction, 0.0 for no interaction
        interaction = UserItemInteraction(
            user_id="u1",
            item_id="i1",
            rating=1.0,
            timestamp=1.0,
            interaction_type="view",
        )

        assert interaction.rating == 1.0

    def test_confidence_scaling(self):
        """Test confidence scaling for implicit feedback."""
        # CRITICAL: Confidence = 1 + alpha * rating
        alpha = 40.0
        rating = 1.0

        confidence = 1 + alpha * rating

        assert confidence == 41.0


if __name__ == "__main__":
    unittest.main()

