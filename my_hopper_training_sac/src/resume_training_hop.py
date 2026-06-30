#!/usr/bin/env python3
"""
resume_training_hop.py
Resume training for HOP-FORWARD task with CONTINUOUS checkpoint numbering.
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
import re

# ===== IMPORT CONFIG FROM TRAINING SCRIPT =====
from start_training_hop import HopTrainingConfig, load_hop_parameters

# ===== CUSTOM CHECKPOINT CALLBACK WITH CONTINUOUS NUMBERING =====
class ContinuousCheckpointCallback(BaseCallback):
    """
    Custom checkpoint callback that continues numbering from a specified step.
    This fixes the issue where CheckpointCallback resets numbering to 1.
    """
    
    def __init__(self, save_freq, save_path, name_prefix, initial_step=0, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.initial_step = initial_step
        self.step_count = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
    def _on_step(self):
        self.step_count += 1
        
        # Check if it's time to save
        if self.step_count % self.save_freq == 0:
            # Calculate total steps (initial + current)
            total_steps = self.initial_step + self.step_count
            
            # Save checkpoint with proper numbering
            model_path = os.path.join(
                self.save_path, 
                f"{self.name_prefix}_{total_steps}_steps"
            )
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"\n💾 Checkpoint saved: {model_path}.zip (total steps: {total_steps:,})")
                
        return True

# ===== OTHER CALLBACKS =====
class HopProgressLogger(BaseCallback):
    """Prints episode summary every N episodes with hop-relevant metrics."""
    
    def __init__(self, log_every=20, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.log_every = log_every

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
    """Progress bar for training"""
    def __init__(self, pbar, verbose=0):
        super().__init__(verbose)
        self.pbar = pbar

    def _on_step(self):
        self.pbar.update(1)
        return True

class ResumeHopConfig:
    """Configuration for resuming hop training"""
    
    ADDITIONAL_STEPS = 100000      # Additional steps to train (default)
    CHECKPOINT_FREQ = 10000       # Save checkpoint every N steps
    EVAL_FREQ = 10000             # Evaluate every N steps
    EVAL_EPISODES = 10            # Episodes for evaluation
    CHECKPOINT_PREFIX = "hop_checkpoint"  # Must match start_training_hop.py

def find_checkpoint(checkpoint_path):
    """Find a checkpoint file with hop naming convention."""
    if os.path.exists(checkpoint_path) and checkpoint_path.endswith('.zip'):
        return checkpoint_path
    
    if os.path.isdir(checkpoint_path):
        checkpoints = []
        for f in os.listdir(checkpoint_path):
            if f.endswith('.zip') and ResumeHopConfig.CHECKPOINT_PREFIX in f:
                checkpoints.append(f)
        
        if checkpoints:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)), reverse=True)
            return os.path.join(checkpoint_path, checkpoints[0])
    
    return None

def extract_step_count(checkpoint_file):
    """Extract step count from checkpoint filename."""
    try:
        filename = os.path.basename(checkpoint_file)
        pattern = r'hop_checkpoint_(\d+)_steps\.zip'
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        return 0
    except:
        return 0

def get_run_directory(checkpoint_file):
    """Get the parent run directory from checkpoint path."""
    run_dir = os.path.dirname(os.path.dirname(checkpoint_file))
    if not os.path.basename(run_dir).startswith('hop_run_'):
        run_dir = os.path.dirname(os.path.dirname(checkpoint_file))
    return run_dir

def setup_callbacks(run_dir, current_steps):
    """
    Setup training callbacks for resume with CONTINUOUS numbering.
    Uses custom ContinuousCheckpointCallback instead of built-in CheckpointCallback.
    """
    
    # ✅ FIXED: Use custom callback with continuous numbering
    checkpoint_cb = ContinuousCheckpointCallback(
        save_freq=ResumeHopConfig.CHECKPOINT_FREQ,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix=ResumeHopConfig.CHECKPOINT_PREFIX,
        initial_step=current_steps,  # ← This continues numbering from 90000!
        verbose=1
    )
    
    # Eval callback
    eval_env = gym.make('Monoped-v0')
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "models"),
        log_path=os.path.join(run_dir, "logs"),
        eval_freq=ResumeHopConfig.EVAL_FREQ,
        n_eval_episodes=ResumeHopConfig.EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Progress logger
    hop_logger = HopProgressLogger(log_every=20)
    
    return [checkpoint_cb, eval_cb, hop_logger]

def save_resume_config(run_dir, checkpoint_file, additional_steps, current_steps):
    """Save resume configuration for reference"""
    config_path = os.path.join(run_dir, "resume_config.yaml")
    config_dict = {
        'resume_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'checkpoint_file': checkpoint_file,
        'additional_steps': additional_steps,
        'current_steps': current_steps,
        'total_steps_after_resume': current_steps + additional_steps,
        'checkpoint_prefix': ResumeHopConfig.CHECKPOINT_PREFIX,
        'checkpoint_freq': ResumeHopConfig.CHECKPOINT_FREQ,
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print(f"\n{'='*65}")
        print("  RESUME HOP TRAINING - USAGE")
        print(f"{'='*65}")
        print("  From checkpoint file:")
        print("    python resume_training_hop.py /path/to/hop_checkpoint_90000_steps.zip [additional_steps]")
        print("\n  From checkpoint directory (uses latest):")
        print("    python resume_training_hop.py /path/to/hop_run_20240101_120000/checkpoints/ [additional_steps]")
        print("\n  Examples:")
        print("    python resume_training_hop.py /root/monoped_ws/src/model/hop_run_20250623_120000/checkpoints/hop_checkpoint_90000_steps.zip 50000")
        print("    python resume_training_hop.py /root/monoped_ws/src/model/hop_run_20250623_120000/checkpoints/")
        print(f"{'='*65}\n")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    additional_steps = int(sys.argv[2]) if len(sys.argv) > 2 else ResumeHopConfig.ADDITIONAL_STEPS
    
    # Find checkpoint
    checkpoint_file = find_checkpoint(checkpoint_path)
    if not checkpoint_file:
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("   Make sure the path exists and contains hop_checkpoint_*_steps.zip files\n")
        sys.exit(1)
    
    # Extract current step count
    current_steps = extract_step_count(checkpoint_file)
    
    print(f"\n{'='*65}")
    print(f"  RESUME HOP-FORWARD TRAINING")
    print(f"{'='*65}")
    print(f"  Checkpoint:     {os.path.basename(checkpoint_file)}")
    print(f"  Current steps:  {current_steps:,}")
    print(f"  Additional:     {additional_steps:,}")
    print(f"  Total after:    {current_steps + additional_steps:,}")
    print(f"{'='*65}\n")
    
    # Initialize ROS node
    rospy.init_node('monoped_hop_resume', anonymous=True, log_level=rospy.INFO)
    
    # Load hop parameters
    load_hop_parameters(HopTrainingConfig.CONFIG_PATH)
    
    # Get run directory
    run_dir = get_run_directory(checkpoint_file)
    
    # Verify run directory structure
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    # Create environment
    env = gym.make('Monoped-v0')
    env = Monitor(env)
    
    # Load model from checkpoint
    print(f"  Loading model from: {checkpoint_file}")
    model = SAC.load(checkpoint_file)
    model.set_env(env)
    print(f"  ✅ Model loaded successfully\n")
    
    # Setup callbacks with CONTINUOUS numbering
    callbacks = setup_callbacks(run_dir, current_steps)
    
    # Save resume configuration
    save_resume_config(run_dir, checkpoint_file, additional_steps, current_steps)
    
    print(f"  Resume directory:   {run_dir}")
    print(f"  Checkpoint prefix:  {ResumeHopConfig.CHECKPOINT_PREFIX}")
    print(f"  ✅ Checkpoints will continue numbering from {current_steps:,}")
    print(f"  TensorBoard:        tensorboard --logdir={run_dir}")
    print(f"  Logs will be saved in: {os.path.join(run_dir, 'logs')}\n")
    
    print(f"{'='*65}")
    print("  STARTING RESUME TRAINING...")
    print(f"{'='*65}\n")
    
    try:
        # Create progress bar
        progress_bar = tqdm(
            total=additional_steps,
            desc="Resuming Hop Training",
            unit="steps",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Add progress bar callback
        progress_cb = ProgressBarCallback(progress_bar)
        
        # Combine all callbacks
        all_callbacks = callbacks + [progress_cb]
        
        # Continue training
        tb_log_name = f"SAC_hop_resume_{datetime.now().strftime('%H%M%S')}"
        
        model.learn(
            total_timesteps=additional_steps,
            callback=all_callbacks,
            log_interval=100,
            tb_log_name=tb_log_name
        )
        
        progress_bar.close()
        
        # Save final models
        final_model_path = os.path.join(run_dir, "models", "final_model")
        model.save(final_model_path)
        
        # Save final checkpoint with correct numbering
        final_checkpoint_path = os.path.join(
            run_dir, "checkpoints", 
            f"hop_checkpoint_{current_steps + additional_steps}_steps"
        )
        model.save(final_checkpoint_path)
        
        print(f"\n{'='*65}")
        print(f"  ✅ RESUME TRAINING COMPLETE!")
        print(f"{'='*65}")
        print(f"  Final model:        {final_model_path}.zip")
        print(f"  Final checkpoint:   {final_checkpoint_path}.zip")
        print(f"  Best model:         {os.path.join(run_dir, 'models', 'best_model.zip')}")
        print(f"  Total steps:        {current_steps + additional_steps:,}")
        print(f"{'='*65}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Resume interrupted by user. Saving current model...")
        interrupt_path = os.path.join(run_dir, "models", f"interrupted_{datetime.now().strftime('%H%M%S')}")
        model.save(interrupt_path)
        print(f"  ✅ Model saved: {interrupt_path}.zip")
        if 'progress_bar' in locals():
            progress_bar.close()
    
    finally:
        env.close()
        print(f"\n📁  All files saved in: {run_dir}")
        print(f"   📂 checkpoints/: {ResumeHopConfig.CHECKPOINT_PREFIX}_XXXXX_steps.zip")
        print(f"      → Numbering continues from {current_steps:,}")
        print(f"   📂 models/: final_model.zip, best_model.zip")
        print(f"   📂 logs/: Evaluation logs")
        print(f"\n{'='*65}\n")

if __name__ == '__main__':
    main()