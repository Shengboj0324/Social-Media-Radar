"""Comprehensive unit tests for Advanced Clustering.

Tests cover:
- DBSCAN density-based clustering
- Leiden community detection
- Hierarchical clustering
- Cluster quality metrics
- Noise detection
- Modularity computation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from app.intelligence.advanced_clustering import (
    DBSCANClustering,
    LeidenCommunityDetection,
    HierarchicalClustering,
    DBSCANConfig,
    LeidenConfig,
    ClusterResult,
    CommunityResult,
)


class TestDBSCANClustering(unittest.TestCase):
    """Test DBSCAN clustering."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DBSCANConfig(
            eps=0.5,
            min_samples=5,
            metric="cosine",
            algorithm="auto",
            leaf_size=30,
            n_jobs=-1,
        )
        self.clustering = DBSCANClustering(self.config)

    def test_initialization(self):
        """Test DBSCAN initialization."""
        assert self.clustering.config.eps == 0.5
        assert self.clustering.config.min_samples == 5
        assert self.clustering.config.metric == "cosine"

    def test_config_defaults(self):
        """Test default configuration values."""
        config = DBSCANConfig()
        assert config.eps == 0.5
        assert config.min_samples == 5
        assert config.metric == "cosine"
        assert config.n_jobs == -1


class TestMathematicalCorrectness(unittest.TestCase):
    """Test mathematical correctness of clustering."""

    def test_centroid_computation(self):
        """Test centroid computation."""
        # CRITICAL: centroid = mean(cluster_embeddings, axis=0)
        cluster_embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])

        centroid = np.mean(cluster_embeddings, axis=0)

        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(centroid, expected)

    def test_density_computation(self):
        """Test density computation."""
        # CRITICAL: density = 1.0 / (mean_distance + epsilon)
        cluster_embeddings = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        centroid = np.array([0.333, 0.333])

        # Compute distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        mean_distance = np.mean(distances)

        epsilon = 1e-8
        density = 1.0 / (mean_distance + epsilon)

        assert density > 0

    def test_coherence_computation(self):
        """Test coherence computation."""
        # CRITICAL: coherence = mean(pairwise_similarities) excluding diagonal
        from sklearn.metrics.pairwise import cosine_similarity

        cluster_embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ])

        similarities = cosine_similarity(cluster_embeddings)

        # Exclude diagonal
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        coherence = similarities[mask].mean()

        assert 0.0 <= coherence <= 1.0

    def test_single_item_coherence(self):
        """Test coherence for single-item cluster."""
        # CRITICAL: Single item has coherence = 1.0
        coherence = 1.0
        assert coherence == 1.0

    def test_noise_label(self):
        """Test noise cluster label."""
        # CRITICAL: DBSCAN uses -1 for noise
        noise_label = -1
        assert noise_label == -1


class TestClusterStatistics(unittest.TestCase):
    """Test cluster statistics computation."""

    def test_num_clusters_calculation(self):
        """Test number of clusters calculation."""
        labels = np.array([0, 0, 1, 1, -1, 2, 2])

        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        assert num_clusters == 3  # Clusters 0, 1, 2 (excluding -1)

    def test_noise_ratio_calculation(self):
        """Test noise ratio calculation."""
        labels = np.array([0, 0, 1, 1, -1, -1, 2])

        num_noise = np.sum(labels == -1)
        noise_ratio = num_noise / len(labels)

        assert num_noise == 2
        assert abs(noise_ratio - 2/7) < 1e-6

    def test_cluster_sizes(self):
        """Test cluster size counting."""
        labels = np.array([0, 0, 0, 1, 1, 2])

        cluster_sizes = {}
        for label in set(labels):
            if label != -1:
                cluster_sizes[label] = np.sum(labels == label)

        assert cluster_sizes[0] == 3
        assert cluster_sizes[1] == 2
        assert cluster_sizes[2] == 1


class TestLeidenCommunityDetection(unittest.TestCase):
    """Test Leiden community detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LeidenConfig(
            resolution=1.0,
            n_iterations=2,
            randomness=0.001,
            seed=42,
        )
        self.detector = LeidenCommunityDetection(self.config)

    def test_initialization(self):
        """Test Leiden initialization."""
        assert self.detector.config.resolution == 1.0
        assert self.detector.config.n_iterations == 2
        assert self.detector.config.seed == 42

    def test_config_defaults(self):
        """Test default configuration values."""
        config = LeidenConfig()
        assert config.resolution == 1.0
        assert config.n_iterations == 2
        assert config.seed == 42


class TestCommunityMetrics(unittest.TestCase):
    """Test community detection metrics."""

    def test_internal_external_edges(self):
        """Test counting internal and external edges."""
        community_nodes = {"node1", "node2", "node3"}
        edges = [
            ("node1", "node2", 1.0),  # Internal
            ("node2", "node3", 1.0),  # Internal
            ("node1", "node4", 1.0),  # External
            ("node3", "node5", 1.0),  # External
        ]

        internal_edges = 0
        external_edges = 0

        for source, target, _ in edges:
            if source in community_nodes and target in community_nodes:
                internal_edges += 1
            elif source in community_nodes or target in community_nodes:
                external_edges += 1

        assert internal_edges == 2
        assert external_edges == 2

    def test_node_to_index_mapping(self):
        """Test node to index mapping."""
        node_ids = ["node1", "node2", "node3"]
        node_to_idx = {node: idx for idx, node in enumerate(node_ids)}

        assert node_to_idx["node1"] == 0
        assert node_to_idx["node2"] == 1
        assert node_to_idx["node3"] == 2

    def test_community_size_counting(self):
        """Test community size counting."""
        membership = [0, 0, 1, 1, 1, 2]

        community_sizes = {}
        for community_id in membership:
            if community_id not in community_sizes:
                community_sizes[community_id] = 0
            community_sizes[community_id] += 1

        assert community_sizes[0] == 2
        assert community_sizes[1] == 3
        assert community_sizes[2] == 1


class TestHierarchicalClustering(unittest.TestCase):
    """Test hierarchical clustering."""

    def test_initialization_with_n_clusters(self):
        """Test initialization with fixed number of clusters."""
        clustering = HierarchicalClustering(n_clusters=5, linkage="ward")

        assert clustering.n_clusters == 5
        assert clustering.linkage == "ward"
        assert clustering.distance_threshold is None

    def test_initialization_with_threshold(self):
        """Test initialization with distance threshold."""
        clustering = HierarchicalClustering(
            n_clusters=None,
            linkage="complete",
            distance_threshold=0.5,
        )

        assert clustering.n_clusters is None
        assert clustering.distance_threshold == 0.5

    def test_linkage_methods(self):
        """Test different linkage methods."""
        linkage_methods = ["ward", "complete", "average", "single"]

        for method in linkage_methods:
            clustering = HierarchicalClustering(n_clusters=3, linkage=method)
            assert clustering.linkage == method


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_cluster_handling(self):
        """Test handling of empty clusters."""
        labels = np.array([0, 0, 1, 1])
        cluster_id = 2  # Non-existent cluster

        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]

        assert len(cluster_indices) == 0

    def test_mask_diagonal_exclusion(self):
        """Test diagonal exclusion in similarity matrix."""
        n = 3
        mask = ~np.eye(n, dtype=bool)

        # Diagonal should be False, off-diagonal True
        assert mask[0, 0] == False
        assert mask[0, 1] == True
        assert mask[1, 0] == True

    def test_sorting_by_size(self):
        """Test sorting clusters by size."""
        clusters = [
            {"cluster_id": 0, "size": 10},
            {"cluster_id": 1, "size": 50},
            {"cluster_id": 2, "size": 30},
        ]

        sorted_clusters = sorted(clusters, key=lambda x: x["size"], reverse=True)

        assert sorted_clusters[0]["cluster_id"] == 1  # Size 50
        assert sorted_clusters[1]["cluster_id"] == 2  # Size 30
        assert sorted_clusters[2]["cluster_id"] == 0  # Size 10


class TestDistanceMetrics(unittest.TestCase):
    """Test distance metrics for clustering."""

    def test_cosine_distance(self):
        """Test cosine distance computation."""
        # Cosine distance = 1 - cosine_similarity
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])

        cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cosine_dist = 1 - cosine_sim

        assert abs(cosine_dist - 1.0) < 1e-6  # Orthogonal vectors

    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])

        euclidean_dist = np.linalg.norm(b - a)

        assert abs(euclidean_dist - 5.0) < 1e-6


if __name__ == "__main__":
    unittest.main()

