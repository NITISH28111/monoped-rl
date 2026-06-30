#!/usr/bin/env python3
"""
test_model_d4pg.py — visualise a trained D4PG checkpoint in Gazebo.
Drop-in replacement for test_model_visual.py.
"""

import gym
import rospy
import numpy as np
import os, sys, time
import monoped_env
from d4pg_agent import D4PGAgent


def find_latest_model(base_dir="/root/monoped_ws/src/model"):
    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
            if d.startswith("d4pg_hop_run_")]
    if not runs:
        return None
    latest = max(runs, key=os.path.getmtime)
    for name in ("best_model.pt", "final_model.pt"):
        p = os.path.join(latest, "models", name)
        if os.path.exists(p):
            return p
    return None


def main():
    model_path   = sys.argv[1] if len(sys.argv) > 1 else find_latest_model()
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not model_path:
        print("No model found. Pass the .pt path as argument.")
        sys.exit(1)

    rospy.init_node('monoped_d4pg_test', anonymous=True)
    env = gym.make('Monoped-v0')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = D4PGAgent(obs_dim=obs_dim, act_dim=act_dim,  device="cuda")
    agent.load(model_path)
    print(f"Loaded: {model_path}\n")

    rewards = []
    for ep in range(num_episodes):
        obs  = np.array(env.reset(), dtype=np.float32)
        done = False
        ep_r = 0.0
        steps = 0
        while not done:
            act = agent.select_action(obs, add_noise=False)
            obs, rew, done, _ = env.step(act)
            obs   = np.array(obs, dtype=np.float32)
            ep_r += rew
            steps += 1
            time.sleep(0.05)
        rewards.append(ep_r)
        print(f"Episode {ep+1}: reward={ep_r:.2f}  steps={steps}")

    print(f"\nMean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    env.close()


if __name__ == '__main__':
    main()