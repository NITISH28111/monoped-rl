#!/usr/bin/env python3
"""
Resume Training from Checkpoint
Use this to continue training from a saved checkpoint
"""

import gym
import rospy
from tqdm import tqdm
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys
from datetime import datetime
import monoped_env
import yaml

from start_training_v3 import load_ros_parameters

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

class ResumeConfig:
    """Configuration for resuming training"""
    
    # ===== RESUME PARAMETERS =====
    ADDITIONAL_STEPS = 100000      # Additional steps to train
    CHECKPOINT_FREQ = 10000        # Save checkpoint every N steps
    EVAL_FREQ = 10000              # Evaluate every N steps
    EVAL_EPISODES = 10              # Episodes for evaluation
    
    # ===== SAVE PATH =====
    BASE_SAVE_DIR = "/root/monoped_ws/src/model"

def find_checkpoint(checkpoint_path):
    """Find a checkpoint file"""
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # If path is a directory, find latest checkpoint
    if os.path.isdir(checkpoint_path):
        checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.zip')]
        if checkpoints:
            # Sort by modification time
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))
            return os.path.join(checkpoint_path, checkpoints[-1])
    
    return None

def get_run_directory(checkpoint_path):
    """Use the same directory as the checkpoint"""
    # Get the parent run directory (go up two levels from checkpoint file)
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    
    return run_dir

def setup_callbacks(run_dir, save_freq=50000, initial_step=0):
    """Setup training callbacks for resume"""
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="monoped_checkpoint",
        initial_step=initial_step,
        verbose=1
    )
    
    eval_env = gym.make('Monoped-v0')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "models"),
        log_path=os.path.join(run_dir, "logs"),
        eval_freq=ResumeConfig.EVAL_FREQ,
        n_eval_episodes=ResumeConfig.EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Add pretty logger
    pretty_logger = PrettyLogger()
    
    return [checkpoint_callback, eval_callback, pretty_logger]

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
    if len(sys.argv) < 2:
        print("Usage: python resume_training.py <checkpoint_path> [additional_steps]")
        print("Example: python resume_training.py /root/monoped_ws/training_runs/run_20240101_120000/checkpoints/monoped_checkpoint_100000_steps.zip 500000")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    additional_steps = int(sys.argv[2]) if len(sys.argv) > 2 else ResumeConfig.ADDITIONAL_STEPS
    
    # Find checkpoint
    checkpoint_file = find_checkpoint(checkpoint_path)
    if not checkpoint_file:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"🔄 RESUMING TRAINING")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Additional steps: {additional_steps:,}")
    print(f"{'='*60}\n")
    
    # Initialize ROS node
    rospy.init_node('monoped_resume', anonymous=True, log_level=rospy.INFO)

    load_ros_parameters()
    
    # Create environment
    env = gym.make('Monoped-v0')
    env = Monitor(env)
    
    # Load model from checkpoint
    model = SAC.load(checkpoint_file)
    model.set_env(env)
    
    # Get total trained steps from checkpoint name
    try:
        # Extract step count from filename
        filename = os.path.basename(checkpoint_file)
        if '_' in filename and 'steps' in filename:
            step_str = filename.split('_')[-1].replace('steps.zip', '')
            current_steps = int(step_str)
        else:
            current_steps = 0
    except:
        current_steps = 0
    
    resume_dir = get_run_directory(checkpoint_file) 
    
    # Setup callbacks - PASS current_steps to continue numbering
    callbacks = setup_callbacks(resume_dir, ResumeConfig.CHECKPOINT_FREQ, current_steps)
    
    print(f"Resume directory: {resume_dir}")
    print(f"Current model has ~{current_steps:,} steps trained")
    print(f"Will train for {additional_steps:,} additional steps")
    print(f"Total after training: {current_steps + additional_steps:,} steps\n")
    
    print(f"TensorBoard logs: {resume_dir}")
    print(f"Run: tensorboard --logdir={resume_dir}\n")
    
    try:
        # Create progress bar
        progress_bar = tqdm(
            total=additional_steps,
            desc="Resuming Training",
            unit="steps",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Add progress bar callback
        progress_callback = ProgressBarCallback(progress_bar, additional_steps)
        
        # Combine all callbacks
        all_callbacks = callbacks + [progress_callback]
        
        # Continue training - use fixed tb_log_name to append to same log
        model.learn(
            total_timesteps=additional_steps,
            callback=all_callbacks,
            log_interval=100,
            tb_log_name="monoped_training"
        )
        
        progress_bar.close()
        
        # Save final models
        model.save(os.path.join(resume_dir, "models", "final_model"))
        
        print(f"\n{'='*60}")
        print(f"✅ RESUME COMPLETE!")
        print(f"{'='*60}")
        print(f"Resumed model: {os.path.join(resume_dir, 'models', 'resumed_model')}")
        print(f"Final model: {os.path.join(resume_dir, 'models', 'final_model')}")
        print(f"Best model: {os.path.join(resume_dir, 'models', 'best_model.zip')}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Resume interrupted by user. Saving current model...")
        model.save(os.path.join(resume_dir, "models", "interrupted_resume"))
        print("Model saved.\n")
        if 'progress_bar' in locals():
            progress_bar.close()
    
    finally:
        env.close()
        print(f"\n📁 All files saved in: {resume_dir}")
        print(f"  📂 checkpoints/: Checkpoint models (continuing numbering)")
        print(f"  📂 models/: Best, final, and resumed models")
        print(f"  📂 logs/: Evaluation logs")
        print(f"\n{'='*60}\n")

if __name__ == '__main__':
    main()