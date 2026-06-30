#!/usr/bin/env python3
"""
d4pg_agent.py
Distributed Distributional Deep Deterministic Policy Gradient (D4PG) agent.

Architecture choices tied to the monoped hopping task:
- Actor:  3-layer MLP [400, 300] matching the SAC net_arch, tanh output
          scaled to the action range [-0.10, +0.10] rad/step.
- Critic: Distributional (C51-style) head predicts N_ATOMS return buckets
          instead of a scalar Q. Handles the high-variance hop reward better
          than a scalar estimate.
- Noise:  Ornstein-Uhlenbeck for temporally-correlated exploration, which
          works better than Gaussian for the slow joint dynamics here.
- No entropy term: exploration is purely noise-driven (unlike SAC).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


# ===========================================================================
#  DISTRIBUTIONAL CRITIC CONSTANTS
# ===========================================================================

N_ATOMS   = 51       # number of return distribution buckets (C51 default)
V_MIN     = -200.0   # minimum expected return — covers done_reward=-100 + decay
V_MAX     =  500.0   # maximum expected return — covers long successful episodes


# ===========================================================================
#  NETWORKS
# ===========================================================================

class Actor(nn.Module):
    """
    Deterministic actor: obs → action in [-1, 1]^3, then scaled to [-0.10, 0.10].

    Uses the same hidden layer sizes [400, 300] as the SAC net_arch so that
    the functional capacity is comparable and curriculum stages transfer.
    """

    def __init__(self, obs_dim: int, act_dim: int, act_limit: float = 0.10):
        super().__init__()
        self.act_limit = act_limit

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 400), nn.LayerNorm(400), nn.ReLU(),
            nn.Linear(400, 300),     nn.LayerNorm(300), nn.ReLU(),
            nn.Linear(300, act_dim), nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act_limit * self.net(obs)


class DistributionalCritic(nn.Module):
    """
    C51-style distributional critic: (obs, act) → distribution over N_ATOMS buckets.

    Returns logits; apply softmax externally when computing the projected distribution.
    The support atoms are fixed at construction and should be registered as a buffer
    so they move to GPU automatically if needed.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 n_atoms: int = N_ATOMS, v_min: float = V_MIN, v_max: float = V_MAX):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min   = v_min
        self.v_max   = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Support vector — register as buffer so it travels with the model
        support = torch.linspace(v_min, v_max, n_atoms)
        self.register_buffer('support', support)

        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 400), nn.LayerNorm(400), nn.ReLU(),
            nn.Linear(400, 300),              nn.LayerNorm(300), nn.ReLU(),
            nn.Linear(300, n_atoms),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Returns logits [batch, n_atoms]."""
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

    def get_q_value(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Scalar expected Q: dot product of softmax probs × support atoms."""
        logits = self.forward(obs, act)
        probs  = F.softmax(logits, dim=-1)
        return (probs * self.support).sum(dim=-1, keepdim=True)


# ===========================================================================
#  ORNSTEIN-UHLENBECK NOISE
# ===========================================================================

class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration.

    Better than Gaussian for the slow PID-controlled joints: a single large
    deviation one step is likely to persist for a few steps, which gives the
    agent meaningful exploratory trajectories rather than jittery noise.

    theta=0.15, sigma=0.20 are DeepMind's D4PG paper defaults; they work
    well for joint-space control tasks.
    """

    def __init__(self, size: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.20, dt: float = 1e-2):
        self.size  = size
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt    = dt
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = (self.theta * (self.mu - self.state) * self.dt
              + self.sigma * np.sqrt(self.dt) * np.random.randn(self.size))
        self.state = self.state + dx
        return self.state.copy()

    def decay(self, factor: float = 0.9995):
        """Anneal sigma each episode so exploitation improves over time."""
        self.sigma = max(0.02, self.sigma * factor)


# ===========================================================================
#  D4PG AGENT
# ===========================================================================

class D4PGAgent:
    """
    Full D4PG agent wrapping actor, critic, target networks, and update logic.

    Design decisions:
    - Two critics (like TD3) to reduce overestimation bias. The distributional
      target is computed from the critic whose expected Q is lower.
    - Target networks updated via Polyak averaging (tau=0.005, same as SAC).
    - Gradient clipping at 1.0 prevents the large hop-event bonus (+20) from
      causing gradient explosions in the early training phase.
    """

    def __init__(self,
                 obs_dim:      int,
                 act_dim:      int,
                 act_limit:    float = 0.10,
                 lr_actor:     float = 1e-4,
                 lr_critic:    float = 3e-4,
                 gamma:        float = 0.99,
                 tau:          float = 0.005,
                 n_atoms:      int   = N_ATOMS,
                 v_min:        float = V_MIN,
                 v_max:        float = V_MAX,
                 device:       str   = "cuda"):

        self.gamma    = gamma
        self.tau      = tau
        self.n_atoms  = n_atoms
        self.v_min    = v_min
        self.v_max    = v_max
        self.delta_z  = (v_max - v_min) / (n_atoms - 1)
        self.device   = torch.device(device)

        # ── Networks ────────────────────────────────────────────────────────
        self.actor  = Actor(obs_dim, act_dim, act_limit).to(self.device)
        self.critic1 = DistributionalCritic(obs_dim, act_dim, n_atoms, v_min, v_max).to(self.device)
        self.critic2 = DistributionalCritic(obs_dim, act_dim, n_atoms, v_min, v_max).to(self.device)

        # Target networks (frozen copy, updated via Polyak)
        self.actor_target   = Actor(obs_dim, act_dim, act_limit).to(self.device)
        self.critic1_target = DistributionalCritic(obs_dim, act_dim, n_atoms, v_min, v_max).to(self.device)
        self.critic2_target = DistributionalCritic(obs_dim, act_dim, n_atoms, v_min, v_max).to(self.device)

        self._hard_update(self.actor_target,   self.actor)
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)

        # ── Optimisers ──────────────────────────────────────────────────────
        self.actor_optim   = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # ── Noise ───────────────────────────────────────────────────────────
        self.noise = OUNoise(size=act_dim, sigma=0.20)

        # ── Stats ───────────────────────────────────────────────────────────
        self.total_steps   = 0
        self.critic_losses = []
        self.actor_losses  = []

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray,
                      add_noise: bool = True) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().squeeze()
        if add_noise:
            action = action + self.noise.sample()
        return np.clip(action, -0.10, 0.10).astype(np.float32)

    # ── Distributional projection ─────────────────────────────────────────

    def _project_distribution(self,
                               rewards:    torch.Tensor,
                               dones:      torch.Tensor,
                               next_probs: torch.Tensor) -> torch.Tensor:
        """
        Categorical projection (Algorithm 1 from Bellemare et al., 2017).
        Projects the Bellman-updated distribution onto the fixed support.

        rewards:    [batch, 1]
        dones:      [batch, 1]  (1.0 if terminal)
        next_probs: [batch, n_atoms]  (softmax of target critic logits)

        Returns:    [batch, n_atoms]  target distribution
        """
        batch = rewards.shape[0]
        support = self.critic1.support.unsqueeze(0)          # [1, n_atoms]
        rewards = rewards.expand_as(support.expand(batch, -1))
        dones   = dones.expand_as(support.expand(batch, -1))

        # Bellman operator applied to each atom
        Tz = rewards + self.gamma * (1.0 - dones) * support  # [batch, n_atoms]
        Tz = Tz.clamp(self.v_min, self.v_max)

        # Map onto the fixed support grid
        b  = (Tz - self.v_min) / self.delta_z                # [batch, n_atoms]
        l  = b.floor().long().clamp(0, self.n_atoms - 1)
        u  = b.ceil().long().clamp(0, self.n_atoms - 1)

        # Distribute probability mass
        target_dist = torch.zeros(batch, self.n_atoms, device=self.device)
        offset = torch.linspace(0, (batch - 1) * self.n_atoms, batch,
                                device=self.device).long().unsqueeze(1).expand(batch, self.n_atoms)

        target_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_probs * (u.float() - b)).view(-1))
        target_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_probs * (b - l.float())).view(-1))

        return target_dist.detach()

    # ── Training update ───────────────────────────────────────────────────

    def update(self, batch: dict) -> dict:
        """
        Single gradient update step.

        batch keys: 'obs', 'act', 'rew', 'next_obs', 'done'
        Each is a torch.Tensor on self.device.

        Returns dict with scalar loss values for logging.
        """
        obs      = batch['obs']
        act      = batch['act']
        rew      = batch['rew']
        next_obs = batch['next_obs']
        done     = batch['done']

        # ── Compute target distribution ──────────────────────────────────
        with torch.no_grad():
            next_act = self.actor_target(next_obs)

            logits1 = self.critic1_target(next_obs, next_act)
            logits2 = self.critic2_target(next_obs, next_act)

            # Pick the critic with the lower expected Q (conservative target)
            q1 = (F.softmax(logits1, dim=-1) * self.critic1_target.support).sum(-1)
            q2 = (F.softmax(logits2, dim=-1) * self.critic2_target.support).sum(-1)

            # Use the lower-Q critic's distribution as the target
            next_probs = torch.where(
                (q1 < q2).unsqueeze(-1).expand_as(logits1),
                F.softmax(logits1, dim=-1),
                F.softmax(logits2, dim=-1)
            )
            target_dist = self._project_distribution(rew, done, next_probs)

        # ── Critic losses (cross-entropy between projected target and predictions) ──
        logits1_pred = self.critic1(obs, act)
        logits2_pred = self.critic2(obs, act)

        critic1_loss = -(target_dist * F.log_softmax(logits1_pred, dim=-1)).sum(-1).mean()
        critic2_loss = -(target_dist * F.log_softmax(logits2_pred, dim=-1)).sum(-1).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optim.step()

        # ── Actor loss (maximize expected Q from critic1) ────────────────
        actor_loss = -self.critic1.get_q_value(obs, self.actor(obs)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ── Polyak-average target networks ───────────────────────────────
        self._soft_update(self.actor_target,   self.actor)
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        self.total_steps += 1

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss':   actor_loss.item(),
        }

    # ── Network utilities ─────────────────────────────────────────────────

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    # ── Checkpoint I/O ────────────────────────────────────────────────────

    def save(self, path: str):
        """Save all networks and optimiser states to `path` (no .pt extension needed)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'actor':          self.actor.state_dict(),
            'critic1':        self.critic1.state_dict(),
            'critic2':        self.critic2.state_dict(),
            'actor_target':   self.actor_target.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optim':    self.actor_optim.state_dict(),
            'critic1_optim':  self.critic1_optim.state_dict(),
            'critic2_optim':  self.critic2_optim.state_dict(),
            'total_steps':    self.total_steps,
            'noise_sigma':    self.noise.sigma,
        }, path + '.pt')

    def load(self, path: str):
        """Load from a checkpoint saved by save()."""
        if not path.endswith('.pt'):
            path = path + '.pt'
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic1_target.load_state_dict(ckpt['critic1_target'])
        self.critic2_target.load_state_dict(ckpt['critic2_target'])
        self.actor_optim.load_state_dict(ckpt['actor_optim'])
        self.critic1_optim.load_state_dict(ckpt['critic1_optim'])
        self.critic2_optim.load_state_dict(ckpt['critic2_optim'])
        self.total_steps  = ckpt.get('total_steps', 0)
        self.noise.sigma  = ckpt.get('noise_sigma', 0.20)