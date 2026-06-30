#!/usr/bin/env python3
"""
start_training_hop.py
Training script for HOP-FORWARD task.

Key differences from start_training_v3.py:
  - Loads learn_params_hop.yaml instead of learn_params.yaml
  - Larger TOTAL_TIMESTEPS (hopping needs more exploration)
  - Smaller LEARNING_STARTS (shorter episodes mean less data per timestep)
  - Larger BUFFER_SIZE (more diverse experiences needed for dynamic task)
  - Larger NET_ARCH (more complex behaviour to approximate)
"""

import gym
import rospy
import numpy as np
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback)
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import monoped_env
import yaml


# ===========================================================================
#  TRAINING CONFIGURATION
# ===========================================================================

class HopTrainingConfig:

    # ── Steps ──────────────────────────────────────────────────────────────
    TOTAL_TIMESTEPS  = 50000   # hopping needs ~5-10× more than standing
    CHECKPOINT_FREQ  =  10000
    EVAL_FREQ        =  10000
    EVAL_EPISODES    =  10

    # ── SAC hyperparameters ─────────────────────────────────────────────────
    LEARNING_RATE    = 3e-4
    BUFFER_SIZE      = 500_000
    LEARNING_STARTS  = 2_000    # shorter: episodes are ~25 steps, need data fast
    BATCH_SIZE       = 256
    TAU              = 0.005
    GAMMA            = 0.99
    TRAIN_FREQ       = 1
    GRADIENT_STEPS   = 1

    # ── Network ─────────────────────────────────────────────────────────────
    NET_ARCH         = [400, 300]   # deeper than standing for complex behaviour

    # ── Paths ───────────────────────────────────────────────────────────────
    BASE_SAVE_DIR    = "/root/monoped_ws/src/model"
    CONFIG_PATH      = "/root/monoped_ws/src/my_hopper_training/config/learn_params_hop.yaml"

# ===========================================================================
#  CALLBACKS
# ===========================================================================

class HopProgressLogger(BaseCallback):
    """Prints episode summary every N episodes with hop-relevant metrics."""

    def __init__(self, log_every=20, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.log_every     = log_every

    def _on_step(self):
        if self.locals.get('done', False):
            self.episode_count += 1
            if self.episode_count % self.log_every == 0:
                ep_rew = self.locals.get('episode_reward', 0)
                ep_len = self.locals.get('episode_length', 0)
                print(f"\n{'='*65}")
                print(f"  Episode {self.episode_count:>5}  |  "
                      f"Reward: {ep_rew:>8.1f}  |  Steps: {ep_len:>4}")
                print(f"{'='*65}")
        return True


class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar, verbose=0):
        super().__init__(verbose)
        self.pbar = pbar

    def _on_step(self):
        self.pbar.update(1)
        return True


# ===========================================================================
#  HELPERS
# ===========================================================================

def load_hop_parameters(config_path):
    if os.path.exists(config_path):
        rospy.loginfo("Loading hop parameters from: %s", config_path)
        os.system(f"rosparam load {config_path}")
        return True
    rospy.logwarn("Hop config not found: %s", config_path)
    return False


def get_run_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir  = os.path.expanduser(HopTrainingConfig.BASE_SAVE_DIR)
    run_dir   = os.path.join(base_dir, f"hop_run_{timestamp}")
    for sub in ("checkpoints", "logs", "models"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def save_config(run_dir):
    config_path = os.path.join(run_dir, "hop_training_config.yaml")
    config_dict = {k: v for k, v in HopTrainingConfig.__dict__.items()
                   if not k.startswith('_')}
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    rospy.init_node('monoped_hop_gym', anonymous=True, log_level=rospy.INFO)
    load_hop_parameters(HopTrainingConfig.CONFIG_PATH)

    run_dir = get_run_directory()
    print(f"\n{'='*65}")
    print(f"  HOP-FORWARD TRAINING STARTED")
    print(f"  Run dir:       {run_dir}")
    print(f"  Total steps:   {HopTrainingConfig.TOTAL_TIMESTEPS:,}")
    print(f"  Net arch:      {HopTrainingConfig.NET_ARCH}")
    print(f"{'='*65}\n")

    # Create environments
    env      = Monitor(gym.make('Monoped-v0'))
    eval_env = gym.make('Monoped-v0')

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq   = HopTrainingConfig.CHECKPOINT_FREQ,
        save_path   = os.path.join(run_dir, "checkpoints"),
        name_prefix = "hop_checkpoint",
        verbose     = 1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(run_dir, "models"),
        log_path             = os.path.join(run_dir, "logs"),
        eval_freq            = HopTrainingConfig.EVAL_FREQ,
        n_eval_episodes      = HopTrainingConfig.EVAL_EPISODES,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )
    hop_logger = HopProgressLogger(log_every=20)

    # SAC model
    model = SAC(
        policy        = 'MlpPolicy',
        env           = env,
        learning_rate = HopTrainingConfig.LEARNING_RATE,
        buffer_size   = HopTrainingConfig.BUFFER_SIZE,
        learning_starts = HopTrainingConfig.LEARNING_STARTS,
        batch_size    = HopTrainingConfig.BATCH_SIZE,
        tau           = HopTrainingConfig.TAU,
        gamma         = HopTrainingConfig.GAMMA,
        train_freq    = HopTrainingConfig.TRAIN_FREQ,
        gradient_steps= HopTrainingConfig.GRADIENT_STEPS,
        policy_kwargs = {'net_arch': HopTrainingConfig.NET_ARCH},
        verbose       = 1,
        tensorboard_log = run_dir,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  SAC policy parameters: {total_params:,}")
    save_config(run_dir)
    print(f"  TensorBoard: tensorboard --logdir={run_dir}\n")

    try:
        pbar = tqdm(
            total    = HopTrainingConfig.TOTAL_TIMESTEPS,
            desc     = "Hop Training",
            unit     = "steps",
            ncols    = 80,
        )
        pbar_cb = ProgressBarCallback(pbar)

        model.learn(
            total_timesteps = HopTrainingConfig.TOTAL_TIMESTEPS,
            callback        = [checkpoint_cb, eval_cb, hop_logger, pbar_cb],
            log_interval    = 100,
            tb_log_name     = f"SAC_hop_{datetime.now().strftime('%H%M%S')}",
        )
        pbar.close()
        model.save(os.path.join(run_dir, "models", "final_model"))

        print(f"\n{'='*65}")
        print(f"  TRAINING COMPLETE")
        print(f"  Best model:  {os.path.join(run_dir, 'models', 'best_model.zip')}")
        print(f"  Final model: {os.path.join(run_dir, 'models', 'final_model.zip')}")
        print(f"{'='*65}\n")

    except KeyboardInterrupt:
        print("\n  Training interrupted — saving checkpoint...")
        model.save(os.path.join(run_dir, "models", "interrupted_model"))
        if 'pbar' in locals():
            pbar.close()

    finally:
        env.close()


if __name__ == '__main__':
    main()