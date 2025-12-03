"""Comprehensive unit tests for Contextual Bandits (UCB1).

Tests UCB1 algorithm for proxy selection with exploration-exploitation balance.
Verifies mathematical correctness and learning behavior.
"""

import pytest
import math
from datetime import datetime, timedelta

from app.scraping.contextual_bandits import (
    ProxyArm,
    BanditContext,
    UCB1ProxySelector,
)


class TestProxyArm:
    """Test ProxyArm dataclass."""

    def test_proxy_arm_creation(self):
        """Test ProxyArm creation."""
        arm = ProxyArm(proxy_id="proxy1", host="1.2.3.4", port=8080)

        assert arm.proxy_id == "proxy1"
        assert arm.host == "1.2.3.4"
        assert arm.port == 8080
        assert arm.total_pulls == 0
        assert arm.ucb_score == float("inf")

    def test_proxy_arm_with_country(self):
        """Test ProxyArm with country."""
        arm = ProxyArm(proxy_id="proxy1", host="1.2.3.4", port=8080, country="US")

        assert arm.country == "US"

    def test_proxy_arm_initial_ucb_score(self):
        """Test that initial UCB score is infinity."""
        arm = ProxyArm(proxy_id="proxy1", host="1.2.3.4", port=8080)

        # CRITICAL: Initial UCB score must be infinity for exploration
        assert arm.ucb_score == float("inf")


class TestBanditContext:
    """Test BanditContext dataclass."""

    def test_bandit_context_creation(self):
        """Test BanditContext creation."""
        context = BanditContext(
            platform="reddit",
            content_type="post",
            time_of_day=14,
        )

        assert context.platform == "reddit"
        assert context.content_type == "post"
        assert context.time_of_day == 14
        assert context.is_trending is False
        assert context.requires_auth is False


class TestUCB1ProxySelector:
    """Test UCB1ProxySelector implementation."""

    def test_ucb1_initialization(self):
        """Test UCB1ProxySelector initialization."""
        selector = UCB1ProxySelector(
            exploration_factor=2.0,
            min_pulls_before_exploitation=10,
            block_threshold=3,
        )

        assert selector.exploration_factor == 2.0
        assert selector.min_pulls_before_exploitation == 10
        assert selector.block_threshold == 3
        assert len(selector.arms) == 0

    def test_add_proxy(self):
        """Test adding proxy."""
        selector = UCB1ProxySelector()

        selector.add_proxy("proxy1", "1.2.3.4", 8080, country="US")

        assert len(selector.arms) == 1
        assert "proxy1" in selector.arms
        assert selector.arms["proxy1"].host == "1.2.3.4"

    def test_add_duplicate_proxy(self):
        """Test adding duplicate proxy."""
        selector = UCB1ProxySelector()

        selector.add_proxy("proxy1", "1.2.3.4", 8080)
        selector.add_proxy("proxy1", "1.2.3.4", 8080)  # Duplicate

        # Should only have one proxy
        assert len(selector.arms) == 1

    def test_select_proxy_no_proxies(self):
        """Test selecting proxy when none available."""
        selector = UCB1ProxySelector()

        selected = selector.select_proxy()

        assert selected is None

    def test_select_proxy_single_proxy(self):
        """Test selecting proxy with single option."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        selected = selector.select_proxy()

        assert selected is not None
        assert selected.proxy_id == "proxy1"

    def test_select_proxy_multiple_proxies(self):
        """Test selecting proxy with multiple options."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080)
        selector.add_proxy("proxy2", "5.6.7.8", 8080)
        selector.add_proxy("proxy3", "9.10.11.12", 8080)

        selected = selector.select_proxy()

        # Should select one of the proxies
        assert selected is not None
        assert selected.proxy_id in ["proxy1", "proxy2", "proxy3"]

    def test_update_reward_success(self):
        """Test updating reward for successful request."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        selector.update_reward("proxy1", success=True, response_time=0.5)

        arm = selector.arms["proxy1"]
        assert arm.total_pulls == 1
        assert arm.total_successes == 1
        assert arm.total_failures == 0
        assert arm.success_rate == 1.0
        assert arm.consecutive_failures == 0

    def test_update_reward_failure(self):
        """Test updating reward for failed request."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        selector.update_reward("proxy1", success=False)



    def test_ucb1_formula(self):
        """Test UCB1 formula: mean + sqrt(c * ln(N) / n)."""
        selector = UCB1ProxySelector(exploration_factor=2.0, min_pulls_before_exploitation=1)
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        # Pull proxy 10 times with 80% success
        for i in range(10):
            selector.update_reward("proxy1", success=(i < 8))

        # Update UCB scores
        arm = selector.arms["proxy1"]
        selector._update_ucb_scores([arm], None)

        # Calculate expected UCB score
        mean_reward = 0.8
        exploration_bonus = math.sqrt((2.0 * math.log(10)) / 10)
        expected_ucb = mean_reward + exploration_bonus

        assert abs(arm.ucb_score - expected_ucb) < 0.01

    def test_exploration_phase(self):
        """Test that proxies with few pulls get infinite UCB score."""
        selector = UCB1ProxySelector(min_pulls_before_exploitation=10)
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        # Pull only 5 times (less than min_pulls_before_exploitation)
        for i in range(5):
            selector.update_reward("proxy1", success=True)

        arm = selector.arms["proxy1"]
        selector._update_ucb_scores([arm], None)

        # Should have infinite UCB score
        assert arm.ucb_score == float("inf")

    def test_exploitation_phase(self):
        """Test that proxies with enough pulls get finite UCB score."""
        selector = UCB1ProxySelector(min_pulls_before_exploitation=10)
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        # Pull 15 times (more than min_pulls_before_exploitation)
        for i in range(15):
            selector.update_reward("proxy1", success=True)

        arm = selector.arms["proxy1"]
        selector._update_ucb_scores([arm], None)

        # Should have finite UCB score
        assert arm.ucb_score != float("inf")
        assert arm.ucb_score > 0

    def test_blocking_detection(self):
        """Test automatic blocking detection."""
        selector = UCB1ProxySelector(block_threshold=3)
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        # 3 consecutive failures
        selector.update_reward("proxy1", success=False)
        selector.update_reward("proxy1", success=False)
        selector.update_reward("proxy1", success=False)

        arm = selector.arms["proxy1"]
        assert arm.is_blocked is True
        assert arm.consecutive_failures == 3

    def test_blocking_reset_on_success(self):
        """Test that blocking is reset on success."""
        selector = UCB1ProxySelector(block_threshold=3)
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        # 2 failures, then success
        selector.update_reward("proxy1", success=False)
        selector.update_reward("proxy1", success=False)
        selector.update_reward("proxy1", success=True)

        arm = selector.arms["proxy1"]
        assert arm.is_blocked is False
        assert arm.consecutive_failures == 0

    def test_blocked_proxy_not_selected(self):
        """Test that blocked proxies are not selected."""
        selector = UCB1ProxySelector(block_threshold=2)
        selector.add_proxy("proxy1", "1.2.3.4", 8080)
        selector.add_proxy("proxy2", "5.6.7.8", 8080)

        # Block proxy1
        selector.update_reward("proxy1", success=False)
        selector.update_reward("proxy1", success=False)

        # Select proxy
        selected = selector.select_proxy()

        # Should select proxy2 (proxy1 is blocked)
        assert selected.proxy_id == "proxy2"

    def test_exponential_moving_average_response_time(self):
        """Test exponential moving average for response time."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        # First update
        selector.update_reward("proxy1", success=True, response_time=1.0)
        arm = selector.arms["proxy1"]
        assert abs(arm.avg_response_time - 1.0) < 0.01

        # Second update (EMA with alpha=0.3)
        selector.update_reward("proxy1", success=True, response_time=2.0)
        expected = 0.3 * 2.0 + 0.7 * 1.0  # 1.3
        assert abs(arm.avg_response_time - expected) < 0.01

    def test_platform_specific_statistics(self):
        """Test platform-specific statistics tracking."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        context_reddit = BanditContext(platform="reddit", content_type="post", time_of_day=14)
        context_tiktok = BanditContext(platform="tiktok", content_type="video", time_of_day=14)

        # Update with different platforms
        selector.update_reward("proxy1", success=True, context=context_reddit)
        selector.update_reward("proxy1", success=True, context=context_reddit)
        selector.update_reward("proxy1", success=False, context=context_tiktok)

        arm = selector.arms["proxy1"]
        assert "reddit" in arm.platform_stats
        assert "tiktok" in arm.platform_stats
        assert arm.platform_stats["reddit"]["successes"] == 2
        assert arm.platform_stats["tiktok"]["failures"] == 1

    def test_contextual_selection(self):
        """Test contextual proxy selection."""
        selector = UCB1ProxySelector(min_pulls_before_exploitation=1)
        selector.add_proxy("proxy1", "1.2.3.4", 8080)
        selector.add_proxy("proxy2", "5.6.7.8", 8080)

        context = BanditContext(platform="reddit", content_type="post", time_of_day=14)

        # Train proxy1 to be good for reddit
        for i in range(10):
            selector.update_reward("proxy1", success=True, context=context)

        # Train proxy2 to be bad for reddit
        for i in range(10):
            selector.update_reward("proxy2", success=False, context=context)

        # Select with reddit context
        selected = selector.select_proxy(context)

        # Should prefer proxy1 for reddit
        assert selected.proxy_id == "proxy1"

    def test_get_statistics(self):
        """Test getting bandit statistics."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080)
        selector.add_proxy("proxy2", "5.6.7.8", 8080)

        # Update some rewards
        selector.update_reward("proxy1", success=True)
        selector.update_reward("proxy2", success=False)

        stats = selector.get_statistics()

        assert stats["total_proxies"] == 2
        assert stats["total_pulls"] == 2
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 1

    def test_get_proxy_statistics(self):
        """Test getting proxy-specific statistics."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080, country="US")

        selector.update_reward("proxy1", success=True, response_time=0.5)

        stats = selector.get_proxy_statistics("proxy1")

        assert stats is not None
        assert stats["proxy_id"] == "proxy1"
        assert stats["host"] == "1.2.3.4"
        assert stats["country"] == "US"
        assert stats["total_pulls"] == 1
        assert stats["success_rate"] == 1.0

    def test_reset_proxy(self):
        """Test resetting proxy statistics."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        # Update some statistics
        selector.update_reward("proxy1", success=True)
        selector.update_reward("proxy1", success=False)

        # Reset
        result = selector.reset_proxy("proxy1")

        assert result is True
        arm = selector.arms["proxy1"]
        assert arm.total_pulls == 0
        assert arm.total_successes == 0
        assert arm.total_failures == 0
        assert arm.ucb_score == float("inf")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_update_reward_unknown_proxy(self):
        """Test updating reward for unknown proxy."""
        selector = UCB1ProxySelector()

        # Should not crash
        selector.update_reward("unknown", success=True)

    def test_all_proxies_blocked(self):
        """Test when all proxies are blocked."""
        selector = UCB1ProxySelector(block_threshold=1)
        selector.add_proxy("proxy1", "1.2.3.4", 8080)
        selector.add_proxy("proxy2", "5.6.7.8", 8080)

        # Block all proxies
        selector.update_reward("proxy1", success=False)
        selector.update_reward("proxy2", success=False)

        # Should return None
        selected = selector.select_proxy()
        assert selected is None

    def test_reset_unknown_proxy(self):
        """Test resetting unknown proxy."""
        selector = UCB1ProxySelector()

        result = selector.reset_proxy("unknown")
        assert result is False


class TestIntegration:
    """Integration tests for contextual bandits."""

    def test_learning_behavior(self):
        """Test that UCB1 learns to prefer better proxies."""
        selector = UCB1ProxySelector(min_pulls_before_exploitation=5)
        selector.add_proxy("good_proxy", "1.2.3.4", 8080)
        selector.add_proxy("bad_proxy", "5.6.7.8", 8080)

        # Train: good_proxy has 90% success, bad_proxy has 10% success
        for i in range(100):
            # Select proxy
            selected = selector.select_proxy()

            # Simulate success based on proxy quality
            if selected.proxy_id == "good_proxy":
                success = (i % 10) != 0  # 90% success
            else:
                success = (i % 10) == 0  # 10% success

            selector.update_reward(selected.proxy_id, success=success)

        # After learning, should prefer good_proxy
        good_arm = selector.arms["good_proxy"]
        bad_arm = selector.arms["bad_proxy"]

        assert good_arm.success_rate > bad_arm.success_rate
        assert good_arm.total_pulls > bad_arm.total_pulls

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        selector = UCB1ProxySelector()
        selector.add_proxy("proxy1", "1.2.3.4", 8080)

        # 7 successes, 3 failures = 70% success rate
        for i in range(10):
            selector.update_reward("proxy1", success=(i < 7))

        arm = selector.arms["proxy1"]
        assert abs(arm.success_rate - 0.7) < 0.01

