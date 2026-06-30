#!/usr/bin/env python3
"""
start_training_d4pg.py
Training script for the HOP-FORWARD task using D4PG.

Mirrors start_training_hop.py in structure so the same curriculum and
reward logic (monoped_state.py) are preserved identically.
The only changes are:
  - SAC (SB3) replaced with D4PGAgent + ReplayBuffer
  - Explicit training loop replaces model.learn()
  - OUNoise replaces SAC entropy-based exploration
  - Checkpoint format: .pt files instead of .zip
"""

import gym
import rospy
import numpy as np
import os
import yaml
from datetime import datetime
from tqdm import tqdm

import monoped_env                         # registers Monoped-v0
from d4pg_agent import D4PGAgent
from d4pg_replay_buffer import ReplayBuffer


# ===========================================================================
#  CONFIGURATION
# ===========================================================================

class D4PGConfig:

    # ── Steps ──────────────────────────────────────────────────────────────
    TOTAL_TIMESTEPS  = 50000
    LEARNING_STARTS  =   2000   # same as SAC: collect random experience first
    CHECKPOINT_FREQ  =  10000
    EVAL_FREQ        =  10000
    EVAL_EPISODES    =      10

    # ── D4PG hyperparameters ────────────────────────────────────────────────
    LR_ACTOR         = 1e-4     # actor LR slightly lower than critic (standard for DDPG family)
    LR_CRITIC        = 3e-4     # same as SAC
    BUFFER_SIZE      = 500_000
    BATCH_SIZE       = 256
    GAMMA            = 0.99
    TAU              = 0.005
    GRADIENT_STEPS   = 1        # updates per env step (1 = online, like SAC default)
    TRAIN_FREQ       = 1

    # ── Distributional parameters ────────────────────────────────────────────
    N_ATOMS          = 51
    V_MIN            = -200.0
    V_MAX            =  500.0

    NOISE_DECAY_PER_STEP = 0.9999723   # gives sigma=0.05 at 50k steps
    NOISE_MIN            = 0.02

    # ── Paths ────────────────────────────────────────────────────────────────
    BASE_SAVE_DIR    = "/root/monoped_ws/src/model"
    CONFIG_PATH      = "/root/monoped_ws/src/my_hopper_training/config/learn_params_d4pg.yaml"

# ===========================================================================
#  HELPERS
# ===========================================================================

def load_hop_parameters(config_path: str):
    if os.path.exists(config_path):
        rospy.loginfo("Loading hop parameters from: %s", config_path)
        os.system(f"rosparam load {config_path}")
        return True
    rospy.logwarn("Hop config not found: %s", config_path)
    return False


def get_run_directory() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir  = os.path.expanduser(D4PGConfig.BASE_SAVE_DIR)
    run_dir   = os.path.join(base_dir, f"d4pg_hop_run_{timestamp}")
    for sub in ("checkpoints", "logs", "models"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def save_config(run_dir: str):
    cfg = {k: v for k, v in D4PGConfig.__dict__.items() if not k.startswith('_')}
    with open(os.path.join(run_dir, "d4pg_training_config.yaml"), 'w') as f:
        yaml.dump(cfg, f)


def evaluate(env, agent: D4PGAgent, n_episodes: int = 10) -> float:
    """Run deterministic rollouts and return mean episode reward."""
    total = 0.0
    for _ in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            act = agent.select_action(np.array(obs, dtype=np.float32),
                                      add_noise=False)
            obs, rew, done, _ = env.step(act)
            ep_r += rew
        total += ep_r
    return total / n_episodes


# ===========================================================================
#  MAIN TRAINING LOOP
# ===========================================================================

def main():
    rospy.init_node('monoped_d4pg_hop', anonymous=True, log_level=rospy.INFO)
    load_hop_parameters(D4PGConfig.CONFIG_PATH)

    run_dir = get_run_directory()
    save_config(run_dir)

    print(f"\n{'='*65}")
    print(f"  D4PG HOP-FORWARD TRAINING")
    print(f"  Run dir:     {run_dir}")
    print(f"  Steps:       {D4PGConfig.TOTAL_TIMESTEPS:,}")
    print(f"  V_MIN/V_MAX: {D4PGConfig.V_MIN} / {D4PGConfig.V_MAX}")
    print(f"  N_ATOMS:     {D4PGConfig.N_ATOMS}")
    print(f"{'='*65}\n")

    # ── Environments ────────────────────────────────────────────────────────
    env      = gym.make('Monoped-v0')
    eval_env = gym.make('Monoped-v0')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"  obs_dim={obs_dim}  act_dim={act_dim}")

    # ── Agent and buffer ─────────────────────────────────────────────────────
    agent = D4PGAgent(
        obs_dim   = obs_dim,
        act_dim   = act_dim,
        act_limit = 0.10,
        lr_actor  = D4PGConfig.LR_ACTOR,
        lr_critic = D4PGConfig.LR_CRITIC,
        gamma     = D4PGConfig.GAMMA,
        tau       = D4PGConfig.TAU,
        n_atoms   = D4PGConfig.N_ATOMS,
        v_min     = D4PGConfig.V_MIN,
        v_max     = D4PGConfig.V_MAX,
        device    = "cuda",   # change to "cuda" if RTX 3050 is accessible from Docker
    )

    buffer = ReplayBuffer(
        obs_dim  = obs_dim,
        act_dim  = act_dim,
        capacity = D4PGConfig.BUFFER_SIZE,
        device   = "cuda",
    )

    # ── Training state ───────────────────────────────────────────────────────
    total_steps    = 0
    episode        = 0
    best_eval_rew  = -np.inf

    log_path = os.path.join(run_dir, "logs", "training_log.csv")
    with open(log_path, 'w') as f:
        f.write("step,episode,ep_reward,ep_len,critic1_loss,critic2_loss,actor_loss,noise_sigma\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    pbar = tqdm(total=D4PGConfig.TOTAL_TIMESTEPS, desc="D4PG Hop Training", unit="steps", ncols=80)

    try:
        while total_steps < D4PGConfig.TOTAL_TIMESTEPS:
            obs  = env.reset()
            obs  = np.array(obs, dtype=np.float32)
            done = False
            ep_reward = 0.0
            ep_len    = 0
            ep_critic1_loss = []
            ep_critic2_loss = []
            ep_actor_loss   = []

            agent.noise.reset()    # reset OU noise at episode start

            while not done and total_steps < D4PGConfig.TOTAL_TIMESTEPS:
                # ── Action selection ──────────────────────────────────────
                if total_steps < D4PGConfig.LEARNING_STARTS:
                    # Pure random exploration before learning starts
                    act = env.action_space.sample()
                else:
                    act = agent.select_action(obs, add_noise=True)

                # ── Environment step ──────────────────────────────────────
                next_obs, rew, done, _ = env.step(act)
                next_obs = np.array(next_obs, dtype=np.float32)

                # Store transition (do not store terminal=True for time-limit done)
                buffer.push(obs, act, rew, next_obs, float(done))

                obs        = next_obs
                ep_reward += rew
                ep_len    += 1
                total_steps += 1
                pbar.update(1)

                # ── Gradient updates ──────────────────────────────────────
                if (total_steps >= D4PGConfig.LEARNING_STARTS and
                        total_steps % D4PGConfig.TRAIN_FREQ == 0):
                    for _ in range(D4PGConfig.GRADIENT_STEPS):
                        batch  = buffer.sample(D4PGConfig.BATCH_SIZE)
                        losses = agent.update(batch)
                        ep_critic1_loss.append(losses['critic1_loss'])
                        ep_critic2_loss.append(losses['critic2_loss'])
                        ep_actor_loss.append(losses['actor_loss'])
                
                # ── Per-step noise decay ──────────────────────────────────
                agent.noise.sigma = max(D4PGConfig.NOISE_MIN,
                                        agent.noise.sigma * D4PGConfig.NOISE_DECAY_PER_STEP)

                # ── Checkpoint ────────────────────────────────────────────
                if total_steps % D4PGConfig.CHECKPOINT_FREQ == 0:
                    ckpt_path = os.path.join(
                        run_dir, "checkpoints",
                        f"d4pg_checkpoint_{total_steps}_steps")
                    agent.save(ckpt_path)
                    print(f"\n  [ckpt] saved → {ckpt_path}.pt")

                # ── Evaluation ────────────────────────────────────────────
                if total_steps % D4PGConfig.EVAL_FREQ == 0:
                    mean_eval = evaluate(eval_env, agent, D4PGConfig.EVAL_EPISODES)
                    print(f"\n  [eval] step={total_steps:,}  mean_ep_rew={mean_eval:.2f}")
                    if mean_eval > best_eval_rew:
                        best_eval_rew = mean_eval
                        best_path = os.path.join(run_dir, "models", "best_model")
                        agent.save(best_path)
                        print(f"  [best] new best → {best_path}.pt")

            # ── Episode end ───────────────────────────────────────────────
            episode += 1

            c1 = np.mean(ep_critic1_loss) if ep_critic1_loss else 0.0
            c2 = np.mean(ep_critic2_loss) if ep_critic2_loss else 0.0
            al = np.mean(ep_actor_loss)   if ep_actor_loss   else 0.0

            with open(log_path, 'a') as f:
                f.write(f"{total_steps},{episode},{ep_reward:.3f},{ep_len},"
                        f"{c1:.6f},{c2:.6f},{al:.6f},{agent.noise.sigma:.4f}\n")

            if episode % 20 == 0:
                print(f"\n{'='*65}")
                print(f"  Ep {episode:>5} | Steps {total_steps:>8,} | "
                      f"Reward {ep_reward:>8.1f} | Len {ep_len:>4} | σ={agent.noise.sigma:.3f}")
                print(f"{'='*65}")

    except KeyboardInterrupt:
        print("\n  Training interrupted — saving model...")

    finally:
        pbar.close()

        final_path = os.path.join(run_dir, "models", "final_model")
        agent.save(final_path)
        print(f"\n  Final model: {final_path}.pt")
        print(f"  Best model:  {os.path.join(run_dir, 'models', 'best_model.pt')}")
        print(f"  Log:         {log_path}")

        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()