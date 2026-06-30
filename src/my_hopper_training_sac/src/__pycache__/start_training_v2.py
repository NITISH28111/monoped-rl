#!/usr/bin/env python3

'''
    Original Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Moded by Miguel Angel Rodriguez <duckfrost@theconstructsim.com>
    Visit our website at www.theconstructsim.com
'''
import gym
import time
import numpy
import random
from gym import wrappers
from std_msgs.msg import Float64
# ROS packages required
import rospy
import rospkg


# import our training environment
import monoped_env
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import A2C
from stable_baselines3 import SAC
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':
    
    rospy.init_node('monoped_gym', anonymous=True, log_level=rospy.INFO)

    # Create the Gym environment
    env = gym.make('Monoped-v0')
    # check_env(env)

    # Create the RL Agent
    # n_actions = 3
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

    model = SAC("MlpPolicy", env, verbose=1)

    # model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000, log_interval=10)
    model.save("rl_monoped")

    del model # remove to demonstrate saving and loading
    model = SAC.load("rl_monoped")
    # model = A2C.load("rl_monoped")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    
    env.close()
