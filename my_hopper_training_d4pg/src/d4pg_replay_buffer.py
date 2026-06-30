#!/usr/bin/env python3
"""
d4pg_replay_buffer.py
Standard uniform replay buffer for D4PG.

D4PG in the original DeepMind paper used a prioritised replay buffer (PER),
but PER requires careful tuning and is sensitive to reward scale.
Given that the hop reward already has high variance (alive=5, liftoff_bonus=20,
done_reward=-100), uniform replay is safer for the first training run.
PER can be added later once the curriculum stages are working reliably.
"""

import numpy as np
import torch
from typing import Dict


class ReplayBuffer:
    """
    Circular buffer storing (obs, act, rew, next_obs, done) transitions.

    Capacity of 500_000 transitions matches the SAC BUFFER_SIZE so that
    exploration coverage is equivalent.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 capacity: int = 500_000, device: str = "cuda"):
        self.capacity = capacity
        self.device   = torch.device(device)
        self.ptr      = 0
        self.size     = 0

        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act      = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew      = np.zeros((capacity, 1),       dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done     = np.zeros((capacity, 1),       dtype=np.float32)

    def push(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr]      = obs
        self.act[self.ptr]      = act
        self.rew[self.ptr]      = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr]     = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs':      torch.FloatTensor(self.obs[idx]).to(self.device),
            'act':      torch.FloatTensor(self.act[idx]).to(self.device),
            'rew':      torch.FloatTensor(self.rew[idx]).to(self.device),
            'next_obs': torch.FloatTensor(self.next_obs[idx]).to(self.device),
            'done':     torch.FloatTensor(self.done[idx]).to(self.device),
        }

    def __len__(self):
        return self.size