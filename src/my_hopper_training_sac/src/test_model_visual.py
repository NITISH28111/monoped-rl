#!/usr/bin/env python3
"""
Test a trained model with visualization in Gazebo
"""

import gym
import rospy
import numpy as np
from stable_baselines3 import SAC
import monoped_env
import time
import os
import sys

class ModelTester:
    def __init__(self, model_path):
        """Initialize tester with model"""
        rospy.init_node('monoped_test', anonymous=True, log_level=rospy.INFO)
        
        # Create environment
        self.env = gym.make('Monoped-v0')
        
        # Load model
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Loading model from: {model_path}")
        self.model = SAC.load(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"{'='*60}\n")
        
    def test(self, num_episodes=5, delay=0.05):
        """Test the model with visualization"""
        
        print(f"Starting test for {num_episodes} episodes...")
        print("Watch the robot in Gazebo window!\n")
        
        all_rewards = []
        all_steps = []
        successes = 0
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            steps = 0
            episode_done = False
            
            print(f"🏁 Episode {episode + 1} starting...")
            
            while not episode_done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                steps += 1
                episode_done = done
                
                # Small delay to see movement
                if delay > 0:
                    time.sleep(delay)
                
                # Print progress every 100 steps
                if steps % 100 == 0:
                    print(f"  Steps: {steps}, Current Reward: {episode_reward:.2f}")
            
            # Episode complete
            all_rewards.append(episode_reward)
            all_steps.append(steps)
            
            if episode_reward > 0:
                successes += 1
            
            print(f"  ✅ Episode {episode + 1} complete!")
            print(f"     Total Reward: {episode_reward:.2f}")
            print(f"     Steps: {steps}")
            print(f"     Success: {'Yes' if episode_reward > 0 else 'No'}\n")
        
        # Summary
        self.print_summary(all_rewards, all_steps, successes, num_episodes)
        
    def print_summary(self, rewards, steps, successes, total):
        """Print test summary"""
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Episodes: {total}")
        print(f"Success Rate: {successes}/{total} ({successes/total*100:.1f}%)")
        print(f"\nReward Statistics:")
        print(f"  Average: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Min: {np.min(rewards):.2f}")
        print(f"  Max: {np.max(rewards):.2f}")
        print(f"\nEpisode Length:")
        print(f"  Average Steps: {np.mean(steps):.1f}")
        print(f"  Min Steps: {np.min(steps)}")
        print(f"  Max Steps: {np.max(steps)}")
        print(f"\n💡 Performance Interpretation:")
        if np.mean(rewards) > 5:
            print("  🟢 Excellent! Robot is performing very well.")
        elif np.mean(rewards) > 0:
            print("  🟡 Good. Robot can maintain balance and move.")
        elif np.mean(rewards) > -5:
            print("  🟠 Moderate. Robot is learning but still struggles.")
        else:
            print("  🔴 Poor. Robot needs more training.")
        print(f"{'='*60}\n")
        
    def close(self):
        """Close the environment"""
        self.env.close()

def find_latest_model():
    """Find the latest trained model"""
    base_dir = os.path.expanduser("/root/monoped_ws/training_runs")
    if not os.path.exists(base_dir):
        return None
    
    # Find all run directories
    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(("run_", "resume_"))]
    
    if not runs:
        return None
    
    # Sort by modification time (latest first)
    latest_run = max(runs, key=os.path.getmtime)
    
    # Try best_model first, then final_model
    model_path = os.path.join(latest_run, "models", "best_model.zip")
    if os.path.exists(model_path):
        return model_path
    else:
        model_path = os.path.join(latest_run, "models", "final_model.zip")
        if os.path.exists(model_path):
            return model_path
    
    return None

def main():
    model_path = None
    num_episodes = 5
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if len(sys.argv) > 2:
            num_episodes = int(sys.argv[2])
    else:
        # Auto-find latest model
        model_path = find_latest_model()
        if model_path:
            print(f"🔍 Found latest model: {model_path}")
        else:
            print("❌ No model found. Please specify model path:")
            print("Usage: python test_model_visual.py [model_path] [num_episodes]")
            print("Example: python test_model_visual.py /root/monoped_ws/training_runs/run_*/models/best_model.zip 10")
            sys.exit(1)
    
    # Run test
    tester = ModelTester(model_path)
    
    try:
        tester.test(num_episodes=num_episodes, delay=0.05)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    finally:
        tester.close()

if __name__ == '__main__':
    main()