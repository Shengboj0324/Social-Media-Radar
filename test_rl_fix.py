#!/usr/bin/env python3
"""Test script to verify reinforcement learning bug fixes."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('=' * 80)
print('CRITICAL BUG FIX VERIFICATION - REINFORCEMENT LEARNING')
print('=' * 80)
print()

# Test deterministic mapping
from app.intelligence.reinforcement_learning import DQNNetwork, PPOAgent, DQNConfig, PPOConfig

print('1. Testing DQN deterministic action mapping...')
dqn = DQNNetwork(DQNConfig(state_dim=10, action_dim=100))

# Test content_id_to_action_idx is deterministic
content_ids = ['content_1', 'content_2', 'content_3', 'content_123', 'content_xyz']
for cid in content_ids:
    idx1 = dqn._content_id_to_action_idx(cid)
    idx2 = dqn._content_id_to_action_idx(cid)
    idx3 = dqn._content_id_to_action_idx(cid)
    assert idx1 == idx2 == idx3, f'Non-deterministic mapping for {cid}'
    assert 0 <= idx1 < 100, f'Index out of range: {idx1}'
    print(f'  ✅ {cid} -> {idx1} (deterministic)')

print()
print('2. Testing DQN action_idx_to_content_id mapping...')
available_actions = ['content_a', 'content_b', 'content_c', 'content_d', 'content_e']
for action_idx in [0, 10, 50, 99]:
    content_id = dqn._action_idx_to_content_id(action_idx, available_actions)
    assert content_id in available_actions, f'Invalid content_id: {content_id}'
    print(f'  ✅ Action {action_idx} -> {content_id}')

print()
print('3. Testing PPO deterministic action mapping...')
ppo = PPOAgent(PPOConfig(state_dim=10, action_dim=100))

for cid in content_ids:
    idx1 = ppo._content_id_to_action_idx(cid)
    idx2 = ppo._content_id_to_action_idx(cid)
    idx3 = ppo._content_id_to_action_idx(cid)
    assert idx1 == idx2 == idx3, f'Non-deterministic mapping for {cid}'
    assert 0 <= idx1 < 100, f'Index out of range: {idx1}'
    print(f'  ✅ {cid} -> {idx1} (deterministic)')

print()
print('4. Testing PPO action_idx_to_content_id mapping...')
for action_idx in [0, 10, 50, 99]:
    content_id = ppo._action_idx_to_content_id(action_idx, available_actions)
    assert content_id in available_actions, f'Invalid content_id: {content_id}'
    print(f'  ✅ Action {action_idx} -> {content_id}')

print()
print('5. Testing consistency across multiple runs...')
# Run 100 times to ensure determinism
for i in range(100):
    idx = dqn._content_id_to_action_idx('test_content_123')
    assert idx == dqn._content_id_to_action_idx('test_content_123'), 'Inconsistent mapping!'

print('  ✅ 100 runs - all consistent')

print()
print('=' * 80)
print('✅ ALL CRITICAL BUGS FIXED - DETERMINISTIC MAPPING VERIFIED')
print('=' * 80)

