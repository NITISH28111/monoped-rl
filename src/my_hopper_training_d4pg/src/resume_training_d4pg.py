#!/usr/bin/env python3
"""
resume_training_d4pg.py
Resume D4PG hop training from a checkpoint .pt file.
"""

import gym
import rospy
import numpy as np
import os
import re
import yaml
from datetime import datetime
from tqdm import tqdm

import monoped_env
from d4pg_agent import D4PGAgent
from d4pg_replay_buffer import ReplayBuffer
from start_training_d4pg import (D4PGConfig, load_hop_parameters,
                                  evaluate, save_config)


def find_latest_checkpoint(directory: str) -> str:
    """Return the checkpoint .pt file with the highest step count."""
    pattern = re.compile(r'd4pg_checkpoint_(\d+)_steps\.pt')
    best_path, best_steps = None, -1
    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps = steps
                best_path  = os.path.join(directory, fname)
    return best_path


def extract_steps(checkpoint_file: str) -> int:
    m = re.search(r'd4pg_checkpoint_(\d+)_steps\.pt', os.path.basename(checkpoint_file))
    return int(m.group(1)) if m else 0


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python resume_training_d4pg.py <checkpoint_path_or_dir> [additional_steps]")
        print("Example: python resume_training_d4pg.py "
              "/root/monoped_ws/src/model/d4pg_hop_run_20250624_120000/checkpoints/ 50000")
        sys.exit(1)

    path             = sys.argv[1]
    additional_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000

    # Resolve checkpoint file
    if os.path.isdir(path):
        ckpt_file = find_latest_checkpoint(path)
        if ckpt_file is None:
            print(f"No d4pg_checkpoint_*_steps.pt found in {path}")
            sys.exit(1)
    else:
        ckpt_file = path if path.endswith('.pt') else path + '.pt'

    current_steps = extract_steps(ckpt_file)
    run_dir = os.path.dirname(os.path.dirname(ckpt_file))   # up from checkpoints/

    print(f"\n{'='*65}")
    print(f"  RESUME D4PG HOP TRAINING")
    print(f"  Checkpoint:    {os.path.basename(ckpt_file)}")
    print(f"  Current steps: {current_steps:,}")
    print(f"  Additional:    {additional_steps:,}")
    print(f"  Total after:   {current_steps + additional_steps:,}")
    print(f"{'='*65}\n")

    rospy.init_node('monoped_d4pg_hop_resume', anonymous=True, log_level=rospy.INFO)
    load_hop_parameters(D4PGConfig.CONFIG_PATH)

    env      = gym.make('Monoped-v0')
    eval_env = gym.make('Monoped-v0')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = D4PGAgent(
        obs_dim=obs_dim, act_dim=act_dim,
        lr_actor=D4PGConfig.LR_ACTOR, lr_critic=D4PGConfig.LR_CRITIC,
        gamma=D4PGConfig.GAMMA, tau=D4PGConfig.TAU,
        n_atoms=D4PGConfig.N_ATOMS, v_min=D4PGConfig.V_MIN, v_max=D4PGConfig.V_MAX,
        device="cuda",
    )
    agent.load(ckpt_file)
    agent.total_steps = current_steps
    print(f"  Loaded checkpoint — noise sigma={agent.noise.sigma:.3f}\n")

    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                          capacity=D4PGConfig.BUFFER_SIZE, device="cuda")

    total_steps   = current_steps
    episode       = 0
    best_eval_rew = -np.inf

    log_path = os.path.join(run_dir, "logs", f"resume_log_{datetime.now().strftime('%H%M%S')}.csv")
    with open(log_path, 'w') as f:
        f.write("step,episode,ep_reward,ep_len,critic1_loss,actor_loss,noise_sigma\n")

    target_steps = current_steps + additional_steps
    pbar = tqdm(total=additional_steps, desc="Resuming D4PG", unit="steps", ncols=80)

    try:
        while total_steps < target_steps:
            obs  = np.array(env.reset(), dtype=np.float32)
            done = False
            ep_reward, ep_len = 0.0, 0
            ep_c1_losses, ep_al_losses = [], []
            agent.noise.reset()

            while not done and total_steps < target_steps:
                act = agent.select_action(obs, add_noise=True)
                next_obs, rew, done, _ = env.step(act)
                next_obs = np.array(next_obs, dtype=np.float32)

                buffer.push(obs, act, rew, next_obs, float(done))
                obs        = next_obs
                ep_reward += rew
                ep_len    += 1
                total_steps += 1
                pbar.update(1)

                agent.noise.sigma = max(D4PGConfig.NOISE_MIN,
                        agent.noise.sigma * D4PGConfig.NOISE_DECAY_PER_STEP)

                if len(buffer) >= D4PGConfig.BATCH_SIZE:
                    losses = agent.update(buffer.sample(D4PGConfig.BATCH_SIZE))
                    ep_c1_losses.append(losses['critic1_loss'])
                    ep_al_losses.append(losses['actor_loss'])

                if total_steps % D4PGConfig.CHECKPOINT_FREQ == 0:
                    ckpt = os.path.join(run_dir, "checkpoints",
                                        f"d4pg_checkpoint_{total_steps}_steps")
                    agent.save(ckpt)

                if total_steps % D4PGConfig.EVAL_FREQ == 0:
                    mean_eval = evaluate(eval_env, agent, D4PGConfig.EVAL_EPISODES)
                    print(f"\n  [eval] step={total_steps:,}  mean_ep_rew={mean_eval:.2f}")
                    if mean_eval > best_eval_rew:
                        best_eval_rew = mean_eval
                        agent.save(os.path.join(run_dir, "models", "best_model"))

            episode += 1
            agent.noise.decay()

            with open(log_path, 'a') as f:
                c1 = np.mean(ep_c1_losses) if ep_c1_losses else 0.0
                al = np.mean(ep_al_losses) if ep_al_losses else 0.0
                f.write(f"{total_steps},{episode},{ep_reward:.3f},{ep_len},"
                        f"{c1:.6f},{al:.6f},{agent.noise.sigma:.4f}\n")

    except KeyboardInterrupt:
        print("\n  Interrupted — saving...")

    finally:
        pbar.close()
        agent.save(os.path.join(run_dir, "models", "final_model"))
        env.close()
        eval_env.close()
        print(f"\n  Run dir: {run_dir}")


if __name__ == '__main__':
    main()