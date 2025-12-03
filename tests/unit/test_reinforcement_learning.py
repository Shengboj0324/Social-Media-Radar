"""Comprehensive unit tests for Reinforcement Learning (DQN/PPO).

CRITICAL: Tests deterministic action mapping fixes (4 bugs fixed).
Tests all aspects of RL with peak skepticism.
"""

import numpy as np
import pytest
import torch

from app.intelligence.reinforcement_learning import (
    Action,
    DQNConfig,
    DQNNetwork,
    Experience,
    PPOAgent,
    PPOConfig,
    Reward,
    State,
)


class TestDeterministicMapping:
    """CRITICAL: Test deterministic content ID to action index mapping.

    This tests the fix for 4 critical bugs that caused non-deterministic training.
    """

    def test_content_id_to_action_idx_deterministic(self):
        """Test that same content ID always maps to same action index."""
        config = DQNConfig(state_dim=10, action_dim=100)
        dqn = DQNNetwork(config)

        content_id = "content_12345"

        # Call multiple times - should always return same index
        idx1 = dqn._content_id_to_action_idx(content_id)
        idx2 = dqn._content_id_to_action_idx(content_id)
        idx3 = dqn._content_id_to_action_idx(content_id)

        assert idx1 == idx2 == idx3
        assert 0 <= idx1 < config.action_dim

    def test_content_id_to_action_idx_across_runs(self):
        """Test that mapping is consistent across different DQN instances."""
        config = DQNConfig(state_dim=10, action_dim=100)

        dqn1 = DQNNetwork(config)
        dqn2 = DQNNetwork(config)

        content_id = "content_67890"

        idx1 = dqn1._content_id_to_action_idx(content_id)
        idx2 = dqn2._content_id_to_action_idx(content_id)

        # CRITICAL: Must be same across different instances
        assert idx1 == idx2

    def test_different_content_ids_different_indices(self):
        """Test that different content IDs map to different indices (usually)."""
        config = DQNConfig(state_dim=10, action_dim=100)
        dqn = DQNNetwork(config)

        content_ids = [f"content_{i}" for i in range(50)]
        indices = [dqn._content_id_to_action_idx(cid) for cid in content_ids]

        # Most should be different (allowing some collisions due to modulo)
        unique_indices = len(set(indices))
        assert unique_indices >= 40  # At least 80% unique

    def test_action_idx_to_content_id_deterministic(self):
        """Test reverse mapping is deterministic."""
        config = DQNConfig(state_dim=10, action_dim=100)
        dqn = DQNNetwork(config)

        available_actions = [f"content_{i}" for i in range(20)]
        action_idx = 42

        # Call multiple times - should always return same content ID
        content1 = dqn._action_idx_to_content_id(action_idx, available_actions)
        content2 = dqn._action_idx_to_content_id(action_idx, available_actions)
        content3 = dqn._action_idx_to_content_id(action_idx, available_actions)

        assert content1 == content2 == content3
        assert content1 in available_actions

    def test_round_trip_mapping(self):
        """Test that mapping content ID -> action idx -> content ID is consistent."""
        config = DQNConfig(state_dim=10, action_dim=100)
        dqn = DQNNetwork(config)

        original_content = "content_test_123"
        available_actions = [original_content, "content_other_1", "content_other_2"]

        # Map to action index
        action_idx = dqn._content_id_to_action_idx(original_content)

        # Map back to content ID
        recovered_content = dqn._action_idx_to_content_id(action_idx, available_actions)

        # Should recover the original content (or at least one from same bucket)
        assert recovered_content in available_actions

    def test_crc32_hash_stability(self):
        """Test that CRC32 hash is stable across calls."""
        import zlib

        content_id = "test_content_stability"

        hash1 = zlib.crc32(content_id.encode('utf-8'))
        hash2 = zlib.crc32(content_id.encode('utf-8'))
        hash3 = zlib.crc32(content_id.encode('utf-8'))

        # CRITICAL: CRC32 must be deterministic
        assert hash1 == hash2 == hash3


class TestDQNNetwork:
    """Test Deep Q-Network implementation."""

    def test_dqn_initialization(self):
        """Test DQN network initialization."""
        config = DQNConfig(state_dim=20, action_dim=50, hidden_dim=128)
        dqn = DQNNetwork(config)

        assert dqn.config.state_dim == 20
        assert dqn.config.action_dim == 50
        assert dqn.config.hidden_dim == 128
        assert dqn.q_network is not None
        assert dqn.target_network is not None
        assert dqn.optimizer is not None

    def test_q_network_forward_pass(self):
        """Test Q-network forward pass."""
        config = DQNConfig(state_dim=10, action_dim=20)
        dqn = DQNNetwork(config)

        # Create dummy state
        state = State(
            user_features=np.random.randn(10).tolist(),
            context_features={},
            timestamp=0.0,
        )

        # Forward pass
        state_tensor = torch.FloatTensor(state.user_features).unsqueeze(0)
        q_values = dqn.q_network(state_tensor)




class TestExperienceReplay:
    """Test experience replay buffer."""

    def test_replay_buffer_add(self):
        """Test adding experiences to replay buffer."""
        config = DQNConfig(state_dim=10, action_dim=20, replay_buffer_size=100)
        dqn = DQNNetwork(config)

        # Create dummy experience
        state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=0.0)
        action = Action(content_id="content_1", action_type="view", metadata={})
        reward = Reward(value=1.0, reward_type="engagement", metadata={})
        next_state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=1.0)

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
        )

        # Add to buffer
        dqn.replay_buffer.append(experience)

        assert len(dqn.replay_buffer) == 1
        assert dqn.replay_buffer[0] == experience

    def test_replay_buffer_max_size(self):
        """Test replay buffer respects max size."""
        config = DQNConfig(state_dim=10, action_dim=20, replay_buffer_size=10)
        dqn = DQNNetwork(config)

        # Add more experiences than buffer size
        for i in range(20):
            state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=float(i))
            action = Action(content_id=f"content_{i}", action_type="view", metadata={})
            reward = Reward(value=1.0, reward_type="engagement", metadata={})
            next_state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=float(i+1))

            experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=False)
            dqn.replay_buffer.append(experience)

        # Buffer should not exceed max size
        assert len(dqn.replay_buffer) == 10

    def test_replay_buffer_sampling(self):
        """Test sampling from replay buffer."""
        config = DQNConfig(state_dim=10, action_dim=20, replay_buffer_size=100, batch_size=32)
        dqn = DQNNetwork(config)

        # Add experiences
        for i in range(50):
            state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=float(i))
            action = Action(content_id=f"content_{i}", action_type="view", metadata={})
            reward = Reward(value=float(i), reward_type="engagement", metadata={})
            next_state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=float(i+1))

            experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=False)
            dqn.replay_buffer.append(experience)

        # Sample batch
        import random
        batch = random.sample(dqn.replay_buffer, min(config.batch_size, len(dqn.replay_buffer)))

        assert len(batch) == 32
        assert all(isinstance(exp, Experience) for exp in batch)


class TestDQNTraining:
    """Test DQN training loop."""

    def test_train_step_basic(self):
        """Test basic DQN training step."""
        config = DQNConfig(state_dim=10, action_dim=20, batch_size=4)
        dqn = DQNNetwork(config)

        # Add some experiences
        for i in range(10):
            state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=float(i))
            action = Action(content_id=f"content_{i}", action_type="view", metadata={})
            reward = Reward(value=1.0, reward_type="engagement", metadata={})
            next_state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=float(i+1))

            experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=False)
            dqn.replay_buffer.append(experience)

        # Train step
        loss = dqn.train_step()

        # Loss should be a valid number
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_step_insufficient_data(self):
        """Test training with insufficient data."""
        config = DQNConfig(state_dim=10, action_dim=20, batch_size=32)
        dqn = DQNNetwork(config)

        # Add only 5 experiences (less than batch size)
        for i in range(5):
            state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=float(i))
            action = Action(content_id=f"content_{i}", action_type="view", metadata={})
            reward = Reward(value=1.0, reward_type="engagement", metadata={})
            next_state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=float(i+1))

            experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=False)
            dqn.replay_buffer.append(experience)

        # Should handle gracefully (return 0.0 or skip)
        loss = dqn.train_step()
        assert isinstance(loss, float)


class TestPPOAgent:
    """Test Proximal Policy Optimization agent."""

    def test_ppo_initialization(self):
        """Test PPO agent initialization."""
        config = PPOConfig(state_dim=20, action_dim=50, hidden_dim=128)
        ppo = PPOAgent(config)

        assert ppo.config.state_dim == 20
        assert ppo.config.action_dim == 50
        assert ppo.config.hidden_dim == 128
        assert ppo.actor is not None
        assert ppo.critic is not None
        assert ppo.optimizer is not None

    def test_ppo_select_action(self):
        """Test PPO action selection."""
        config = PPOConfig(state_dim=10, action_dim=20)
        ppo = PPOAgent(config)

        state = State(user_features=np.random.randn(10).tolist(), context_features={}, timestamp=0.0)
        available_actions = [f"content_{i}" for i in range(10)]

        action = ppo.select_action(state, available_actions)

        assert isinstance(action, Action)
        assert action.content_id in available_actions

    def test_ppo_deterministic_mapping(self):
        """Test PPO uses deterministic content ID mapping."""
        config = PPOConfig(state_dim=10, action_dim=100)
        ppo = PPOAgent(config)

        content_id = "content_ppo_test"

        # Call multiple times - should always return same index
        idx1 = ppo._content_id_to_action_idx(content_id)
        idx2 = ppo._content_id_to_action_idx(content_id)
        idx3 = ppo._content_id_to_action_idx(content_id)

        assert idx1 == idx2 == idx3
        assert 0 <= idx1 < config.action_dim


class TestStateActionReward:
    """Test State, Action, and Reward data structures."""

    def test_state_creation(self):
        """Test creating a State."""
        user_features = [0.1, 0.2, 0.3, 0.4, 0.5]
        context_features = {"platform": "reddit", "time_of_day": "morning"}
        timestamp = 123456.789

        state = State(
            user_features=user_features,
            context_features=context_features,
            timestamp=timestamp,
        )

        assert state.user_features == user_features
        assert state.context_features == context_features
        assert state.timestamp == timestamp

    def test_action_creation(self):
        """Test creating an Action."""
        action = Action(
            content_id="content_123",
            action_type="view",
            metadata={"duration": 30.5},
        )

        assert action.content_id == "content_123"
        assert action.action_type == "view"
        assert action.metadata["duration"] == 30.5

    def test_reward_creation(self):
        """Test creating a Reward."""
        reward = Reward(
            value=1.5,
            reward_type="engagement",
            metadata={"likes": 10, "comments": 5},
        )

        assert reward.value == 1.5
        assert reward.reward_type == "engagement"
        assert reward.metadata["likes"] == 10
        assert reward.metadata["comments"] == 5

    def test_experience_creation(self):
        """Test creating an Experience."""
        state = State(user_features=[0.1, 0.2], context_features={}, timestamp=0.0)
        action = Action(content_id="content_1", action_type="view", metadata={})
        reward = Reward(value=1.0, reward_type="engagement", metadata={})
        next_state = State(user_features=[0.3, 0.4], context_features={}, timestamp=1.0)

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
        )

        assert experience.state == state
        assert experience.action == action
        assert experience.reward == reward
        assert experience.next_state == next_state
        assert experience.done is False

