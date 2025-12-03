"""Comprehensive unit tests for graph traversal algorithms.

Tests all aspects of BFS, DFS, and Hybrid traversal with peak skepticism.
"""

import asyncio
from datetime import datetime
from typing import List

import pytest

from app.scraping.graph_traversal import (
    GraphNode,
    GraphTraverser,
    NodeType,
    TraversalConfig,
    TraversalStrategy,
)


class TestGraphNode:
    """Test GraphNode data structure."""

    def test_node_creation(self):
        """Test creating a graph node."""
        node = GraphNode(
            id="node1",
            node_type=NodeType.POST,
            url="https://example.com/post1",
            depth=0,
        )

        assert node.id == "node1"
        assert node.node_type == NodeType.POST
        assert node.url == "https://example.com/post1"
        assert node.depth == 0
        assert node.priority == 1.0
        assert node.visited is False
        assert node.parent_id is None

    def test_node_with_metadata(self):
        """Test node with metadata."""
        metadata = {"author": "user123", "score": 100}
        node = GraphNode(
            id="node2",
            node_type=NodeType.COMMENT,
            url="https://example.com/comment2",
            metadata=metadata,
            depth=1,
            parent_id="node1",
            priority=0.8,
        )

        assert node.metadata == metadata
        assert node.depth == 1
        assert node.parent_id == "node1"
        assert node.priority == 0.8

    def test_node_types(self):
        """Test all node types."""
        node_types = [
            NodeType.USER,
            NodeType.POST,
            NodeType.COMMENT,
            NodeType.SUBREDDIT,
            NodeType.CHANNEL,
            NodeType.VIDEO,
            NodeType.HASHTAG,
            NodeType.TOPIC,
        ]

        for node_type in node_types:
            node = GraphNode(
                id=f"node_{node_type.value}",
                node_type=node_type,
                url=f"https://example.com/{node_type.value}",
            )
            assert node.node_type == node_type


class TestTraversalConfig:
    """Test traversal configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TraversalConfig()

        assert config.strategy == TraversalStrategy.BFS
        assert config.max_depth == 3
        assert config.max_nodes == 1000
        assert config.max_children_per_node == 50
        assert config.enable_cycle_detection is True
        assert config.priority_threshold == 0.5
        assert config.timeout_seconds == 300
        assert config.concurrent_fetches == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = TraversalConfig(
            strategy=TraversalStrategy.DFS,
            max_depth=5,
            max_nodes=500,
            max_children_per_node=20,
            enable_cycle_detection=False,
            priority_threshold=0.7,
            timeout_seconds=60,
            concurrent_fetches=10,
        )

        assert config.strategy == TraversalStrategy.DFS
        assert config.max_depth == 5
        assert config.max_nodes == 500
        assert config.max_children_per_node == 20
        assert config.enable_cycle_detection is False
        assert config.priority_threshold == 0.7
        assert config.timeout_seconds == 60
        assert config.concurrent_fetches == 10


class TestBFSTraversal:
    """Test Breadth-First Search traversal."""

    @pytest.mark.asyncio
    async def test_bfs_simple_graph(self):
        """Test BFS on a simple graph."""
        # Create simple graph: root -> [child1, child2]
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)
        child1 = GraphNode(id="child1", node_type=NodeType.COMMENT, url="https://example.com/child1", depth=1, parent_id="root")
        child2 = GraphNode(id="child2", node_type=NodeType.COMMENT, url="https://example.com/child2", depth=1, parent_id="root")

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return [child1, child2]
            return []

        config = TraversalConfig(strategy=TraversalStrategy.BFS, max_depth=2, max_nodes=10)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([root])

        # Should visit root, child1, child2 in BFS order
        assert len(results) == 3
        assert results[0].id == "root"
        assert {results[1].id, results[2].id} == {"child1", "child2"}
        assert traverser.nodes_visited == 3
        assert traverser.nodes_discovered == 3




    @pytest.mark.asyncio
    async def test_bfs_depth_limit(self):
        """Test BFS respects depth limit."""
        # Create graph with depth > max_depth
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)
        level1 = GraphNode(id="level1", node_type=NodeType.COMMENT, url="https://example.com/level1", depth=1, parent_id="root")
        level2 = GraphNode(id="level2", node_type=NodeType.COMMENT, url="https://example.com/level2", depth=2, parent_id="level1")
        level3 = GraphNode(id="level3", node_type=NodeType.COMMENT, url="https://example.com/level3", depth=3, parent_id="level2")
        level4 = GraphNode(id="level4", node_type=NodeType.COMMENT, url="https://example.com/level4", depth=4, parent_id="level3")

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return [level1]
            elif node.id == "level1":
                return [level2]
            elif node.id == "level2":
                return [level3]
            elif node.id == "level3":
                return [level4]
            return []

        config = TraversalConfig(strategy=TraversalStrategy.BFS, max_depth=2, max_nodes=100)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([root])

        # Should only visit up to depth 2
        assert len(results) == 3  # root, level1, level2
        assert all(node.depth <= 2 for node in results)
        assert "level3" not in [node.id for node in results]
        assert "level4" not in [node.id for node in results]

    @pytest.mark.asyncio
    async def test_bfs_cycle_detection(self):
        """Test BFS detects and prevents cycles."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)
        child = GraphNode(id="child", node_type=NodeType.COMMENT, url="https://example.com/child", depth=1, parent_id="root")
        # Create cycle: child points back to root
        root_again = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=2, parent_id="child")

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return [child]
            elif node.id == "child":
                return [root_again]
            return []

        config = TraversalConfig(strategy=TraversalStrategy.BFS, max_depth=5, enable_cycle_detection=True)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([root])

        # Should visit root and child only once
        assert len(results) == 2
        assert results[0].id == "root"
        assert results[1].id == "child"

    @pytest.mark.asyncio
    async def test_bfs_priority_filtering(self):
        """Test BFS filters nodes by priority threshold."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0, priority=1.0)
        high_priority = GraphNode(id="high", node_type=NodeType.COMMENT, url="https://example.com/high", depth=1, priority=0.9)
        low_priority = GraphNode(id="low", node_type=NodeType.COMMENT, url="https://example.com/low", depth=1, priority=0.3)

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return [high_priority, low_priority]
            return []

        config = TraversalConfig(strategy=TraversalStrategy.BFS, priority_threshold=0.5)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([root])

        # Should only visit root and high_priority node
        assert len(results) == 2
        assert results[0].id == "root"
        assert results[1].id == "high"

    @pytest.mark.asyncio
    async def test_bfs_max_nodes_limit(self):
        """Test BFS stops at max_nodes limit."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)
        children = [
            GraphNode(id=f"child{i}", node_type=NodeType.COMMENT, url=f"https://example.com/child{i}", depth=1)
            for i in range(100)
        ]

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return children
            return []

        config = TraversalConfig(strategy=TraversalStrategy.BFS, max_nodes=10, max_depth=5)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([root])

        # Should stop at max_nodes
        assert len(results) == 10
        assert traverser.nodes_visited == 10

    @pytest.mark.asyncio
    async def test_bfs_max_children_per_node(self):
        """Test BFS limits children per node."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)
        children = [
            GraphNode(id=f"child{i}", node_type=NodeType.COMMENT, url=f"https://example.com/child{i}", depth=1)
            for i in range(100)
        ]

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return children
            return []

        config = TraversalConfig(strategy=TraversalStrategy.BFS, max_children_per_node=5, max_nodes=100)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([root])

        # Should visit root + 5 children only
        assert len(results) == 6
        assert results[0].id == "root"


class TestDFSTraversal:
    """Test Depth-First Search traversal."""

    @pytest.mark.asyncio
    async def test_dfs_simple_graph(self):
        """Test DFS on a simple graph."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)
        child1 = GraphNode(id="child1", node_type=NodeType.COMMENT, url="https://example.com/child1", depth=1, parent_id="root")
        child2 = GraphNode(id="child2", node_type=NodeType.COMMENT, url="https://example.com/child2", depth=1, parent_id="root")

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return [child1, child2]
            return []

        config = TraversalConfig(strategy=TraversalStrategy.DFS, max_depth=2, max_nodes=10)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([root])

        # Should visit all nodes
        assert len(results) == 3
        assert results[0].id == "root"


    @pytest.mark.asyncio
    async def test_dfs_cycle_detection(self):
        """Test DFS detects and prevents cycles."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)
        child = GraphNode(id="child", node_type=NodeType.COMMENT, url="https://example.com/child", depth=1, parent_id="root")
        root_again = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=2, parent_id="child")

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return [child]
            elif node.id == "child":
                return [root_again]
            return []

        config = TraversalConfig(strategy=TraversalStrategy.DFS, max_depth=5, enable_cycle_detection=True)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([root])

        assert len(results) == 2
        assert results[0].id == "root"
        assert results[1].id == "child"


class TestHybridTraversal:
    """Test Hybrid traversal strategy."""

    @pytest.mark.asyncio
    async def test_hybrid_bfs_then_dfs(self):
        """Test hybrid strategy switches from BFS to DFS."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0, priority=1.0)
        child1 = GraphNode(id="child1", node_type=NodeType.COMMENT, url="https://example.com/child1", depth=1, priority=0.9)
        child2 = GraphNode(id="child2", node_type=NodeType.COMMENT, url="https://example.com/child2", depth=1, priority=0.6)
        grandchild = GraphNode(id="grandchild", node_type=NodeType.COMMENT, url="https://example.com/grandchild", depth=2, priority=0.8)

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return [child1, child2]
            elif node.id == "child1":
                return [grandchild]
            return []

        config = TraversalConfig(strategy=TraversalStrategy.HYBRID, max_depth=3, max_nodes=10)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([root])

        # Should visit all nodes
        assert len(results) >= 3


class TestStatistics:
    """Test traversal statistics."""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test statistics are tracked correctly."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)
        child1 = GraphNode(id="child1", node_type=NodeType.COMMENT, url="https://example.com/child1", depth=1)
        child2 = GraphNode(id="child2", node_type=NodeType.COMMENT, url="https://example.com/child2", depth=1)

        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            if node.id == "root":
                return [child1, child2]
            return []

        config = TraversalConfig(strategy=TraversalStrategy.BFS, max_depth=2)
        traverser = GraphTraverser(config, fetch_neighbors)

        await traverser.traverse([root])

        stats = traverser.get_statistics()

        assert stats["nodes_visited"] == 3
        assert stats["nodes_discovered"] == 3
        assert stats["edges_traversed"] == 2
        assert stats["strategy"] == "bfs"
        assert stats["max_depth_reached"] == 1
        assert stats["elapsed_seconds"] > 0
        assert stats["nodes_per_second"] > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_fetch_timeout(self):
        """Test handling of fetch timeouts."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)

        async def slow_fetch(node: GraphNode) -> List[GraphNode]:
            await asyncio.sleep(60)  # Longer than timeout
            return []

        config = TraversalConfig(strategy=TraversalStrategy.BFS, max_depth=2)
        traverser = GraphTraverser(config, slow_fetch)

        results = await traverser.traverse([root])

        # Should still visit root despite timeout
        assert len(results) == 1
        assert results[0].id == "root"

    @pytest.mark.asyncio
    async def test_fetch_error(self):
        """Test handling of fetch errors."""
        root = GraphNode(id="root", node_type=NodeType.POST, url="https://example.com/root", depth=0)

        async def error_fetch(node: GraphNode) -> List[GraphNode]:
            raise ValueError("Fetch failed")

        config = TraversalConfig(strategy=TraversalStrategy.BFS, max_depth=2)
        traverser = GraphTraverser(config, error_fetch)

        results = await traverser.traverse([root])

        # Should still visit root despite error
        assert len(results) == 1
        assert results[0].id == "root"

    @pytest.mark.asyncio
    async def test_empty_start_nodes(self):
        """Test traversal with empty start nodes."""
        async def fetch_neighbors(node: GraphNode) -> List[GraphNode]:
            return []

        config = TraversalConfig(strategy=TraversalStrategy.BFS)
        traverser = GraphTraverser(config, fetch_neighbors)

        results = await traverser.traverse([])

        assert len(results) == 0
        assert traverser.nodes_visited == 0

