#!/usr/bin/env python3
"""
Monoped RL Training Script - Main entry point for training
Supports: Standing, Hopping, and Walking tasks
"""

import gym
import rospy
import numpy as np
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import monoped_env
import yaml
import sys

# ===== CUSTOM LOGGING CALLBACK =====
class PrettyLogger(BaseCallback):
    """Custom callback for cleaner logging with spacing"""
    def __init__(self, verbose=0):
        super(PrettyLogger, self).__init__(verbose)
        self.episode_count = 0
        self.last_print = 0
        
    def _on_step(self):
        # Print episode info when episode ends
        if self.locals.get('done', False):
            self.episode_count += 1
            if self.episode_count % 10 == 0:  # Print every 10 episodes
                ep_rew = self.locals.get('episode_reward', 0)
                ep_len = self.locals.get('episode_length', 0)
                print(f"\n{'='*60}")
                print(f"📊 Episode {self.episode_count}")
                print(f"   Reward: {ep_rew:.2f}")
                print(f"   Length: {ep_len} steps")
                print(f"{'='*60}\n")
        return True

class TrainingConfig:
    """Training configuration - modify these as needed"""
    
    # ===== TRAINING PARAMETERS =====
    TOTAL_TIMESTEPS = 50000      # Total training steps
    CHECKPOINT_FREQ = 10000       # Save checkpoint every N steps
    EVAL_FREQ = 10000             # Evaluate every N steps
    EVAL_EPISODES = 10            # Episodes for evaluation
    
    # ===== SAC HYPERPARAMETERS =====
    LEARNING_RATE = 0.0003
    BUFFER_SIZE = 1000000
    LEARNING_STARTS = 5000
    BATCH_SIZE = 256
    TAU = 0.005
    GAMMA = 0.99
    TRAIN_FREQ = 1
    GRADIENT_STEPS = 1
    
    # ===== NEURAL NETWORK ARCHITECTURE =====
    NET_ARCH = [256, 256]
    
    # ===== LOGGING =====
    TENSORBOARD_LOG = True
    VERBOSE = 1
    
    # ===== MODEL SAVE PATH =====
    BASE_SAVE_DIR = "/root/monoped_ws/src/model"

def create_env():
    """Create and wrap environment with monitoring"""
    env = gym.make('Monoped-v0')
    env = Monitor(env)
    return env

def load_ros_parameters():
    """Load ROS parameters from yaml file"""
    config_path = "/root/monoped_ws/src/my_hopper_training_sac/config/learn_params.yaml"
    if os.path.exists(config_path):
        rospy.loginfo(f"Loading parameters from: {config_path}")
        os.system(f"rosparam load {config_path}")
        return True
    else:
        rospy.logwarn(f"Parameter file not found: {config_path}")
        return False

def get_run_directory():
    """Create a unique run directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.expanduser(TrainingConfig.BASE_SAVE_DIR)
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    return run_dir

def setup_callbacks(run_dir):
    """Setup training callbacks"""
    
    checkpoint_callback = CheckpointCallback(
        save_freq=TrainingConfig.CHECKPOINT_FREQ,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="monoped_checkpoint",
        verbose=1
    )
    
    eval_env = gym.make('Monoped-v0')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "models"),
        log_path=os.path.join(run_dir, "logs"),
        eval_freq=TrainingConfig.EVAL_FREQ,
        n_eval_episodes=TrainingConfig.EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Add pretty logger
    pretty_logger = PrettyLogger()
    
    return [checkpoint_callback, eval_callback, pretty_logger]

def save_config(run_dir):
    """Save training configuration"""
    config_path = os.path.join(run_dir, "training_config.yaml")
    config_dict = {k: v for k, v in TrainingConfig.__dict__.items() if not k.startswith('_')}
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)

class ProgressBarCallback(BaseCallback):
    """Custom callback for progress bar"""
    def __init__(self, pbar, total_steps, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.pbar = pbar
        self.total_steps = total_steps
        self.current_steps = 0
        
    def _on_step(self):
        self.current_steps += 1
        self.pbar.update(1)
        return True

def main():
    # Initialize ROS node
    rospy.init_node('monoped_gym', anonymous=True, log_level=rospy.INFO)

    # Load ROS parameters
    load_ros_parameters()
    
    # Create run directory
    run_dir = get_run_directory()
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING STARTED")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")
    print(f"Total steps: {TrainingConfig.TOTAL_TIMESTEPS:,}")
    print(f"Checkpoint freq: {TrainingConfig.CHECKPOINT_FREQ:,}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = create_env()
    
    # Setup callbacks
    callbacks = setup_callbacks(run_dir)
    
    # Model configuration
    model_kwargs = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': TrainingConfig.LEARNING_RATE,
        'buffer_size': TrainingConfig.BUFFER_SIZE,
        'learning_starts': TrainingConfig.LEARNING_STARTS,
        'batch_size': TrainingConfig.BATCH_SIZE,
        'tau': TrainingConfig.TAU,
        'gamma': TrainingConfig.GAMMA,
        'train_freq': TrainingConfig.TRAIN_FREQ,
        'gradient_steps': TrainingConfig.GRADIENT_STEPS,
        'policy_kwargs': {
            'net_arch': TrainingConfig.NET_ARCH,
        },
        'verbose': TrainingConfig.VERBOSE,
        'tensorboard_log': run_dir if TrainingConfig.TENSORBOARD_LOG else None,
    }
    
    # Create model
    model = SAC(**model_kwargs)
    
    # Calculate and display model parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model Architecture: {TrainingConfig.NET_ARCH}")
    print(f"Total parameters: {total_params:,}\n")
    
    # Save configuration
    save_config(run_dir)
    
    print(f"TensorBoard logs: {run_dir}")
    print(f"Run: tensorboard --logdir={run_dir}\n")
    
    try:
        # Create progress bar
        progress_bar = tqdm(
            total=TrainingConfig.TOTAL_TIMESTEPS,
            desc="Training Progress",
            unit="steps",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Progress bar callback
        progress_callback = ProgressBarCallback(progress_bar, TrainingConfig.TOTAL_TIMESTEPS)
        
        # Combine all callbacks
        all_callbacks = callbacks + [progress_callback]
        
        # Train
        model.learn(
            total_timesteps=TrainingConfig.TOTAL_TIMESTEPS,
            callback=all_callbacks,
            log_interval=100,
            tb_log_name=f"SAC_run_{datetime.now().strftime('%H%M%S')}"
        )

        progress_bar.close()
        
        # Save final model
        final_model_path = os.path.join(run_dir, "models", "final_model")
        model.save(final_model_path)
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Final model: {final_model_path}")
        print(f"Best model: {os.path.join(run_dir, 'models', 'best_model.zip')}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user. Saving current model...")
        model.save(os.path.join(run_dir, "models", "interrupted_model"))
        print("Model saved.\n")
    
    finally:
        env.close()
        print(f"\n📁 All files saved in: {run_dir}")
        print(f"  📂 checkpoints/: Checkpoint models")
        print(f"  📂 models/: Best and final models")
        print(f"  📂 logs/: Evaluation logs")
        print(f"  📄 training_config.yaml: Training configuration")
        print(f"\n{'='*60}\n")

if __name__ == '__main__':
    main()