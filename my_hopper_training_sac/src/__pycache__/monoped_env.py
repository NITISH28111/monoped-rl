#!/usr/bin/env python

import gym
import rospy
import numpy as np
import time
from gym import utils, spaces
from geometry_msgs.msg import Pose
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection
from joint_publisher import JointPub
from monoped_state import MonopedState
from controllers_connection import ControllersConnection

# Register the training environment
reg = register(
    id='Monoped-v0',
    entry_point='monoped_env:MonopedEnv',
    max_episode_steps=500,
)


class MonopedEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        # ── Load parameters from ROS param server ──────────────────────────
        self.desired_pose = Pose()
        self.desired_pose.position.x = rospy.get_param("/desired_pose/x")
        self.desired_pose.position.y = rospy.get_param("/desired_pose/y")
        self.desired_pose.position.z = rospy.get_param("/desired_pose/z")

        self.running_step         = rospy.get_param("/running_step")
        self.max_incl             = rospy.get_param("/max_incl")
        self.max_height           = rospy.get_param("/max_height")
        self.min_height           = rospy.get_param("/min_height")
        self.joint_increment_value= rospy.get_param("/joint_increment_value")
        self.done_reward          = rospy.get_param("/done_reward")
        self.alive_reward         = rospy.get_param("/alive_reward")
        self.desired_force        = rospy.get_param("/desired_force")
        self.desired_yaw          = rospy.get_param("/desired_yaw")

        self.weight_r1 = rospy.get_param("/weight_r1")
        self.weight_r2 = rospy.get_param("/weight_r2")
        self.weight_r3 = rospy.get_param("/weight_r3")
        self.weight_r4 = rospy.get_param("/weight_r4")
        self.weight_r5 = rospy.get_param("/weight_r5")

        # ── MODE FLAGS ──────────────────────────────────────────────────────
        # Both must be True to activate hop-forward reward.
        self.hopping_mode = rospy.get_param("/hopping_mode", False)
        self.walking_mode = rospy.get_param("/walking_mode", False)

        rospy.loginfo("[MonopedEnv] hopping_mode=%s  walking_mode=%s",
                      self.hopping_mode, self.walking_mode)

        # ── Simulator connections ───────────────────────────────────────────
        self.gazebo             = GazeboConnection()
        self.controllers_object = ControllersConnection(namespace="monoped")

        # ── State object (contains reward logic) ───────────────────────────
        self.monoped_state_object = MonopedState(
            max_height            = self.max_height,
            min_height            = self.min_height,
            abs_max_roll          = self.max_incl,
            abs_max_pitch         = self.max_incl,
            joint_increment_value = self.joint_increment_value,
            done_reward           = self.done_reward,
            alive_reward          = self.alive_reward,
            desired_force         = self.desired_force,
            desired_yaw           = self.desired_yaw,
            weight_r1             = self.weight_r1,
            weight_r2             = self.weight_r2,
            weight_r3             = self.weight_r3,
            weight_r4             = self.weight_r4,
            weight_r5             = self.weight_r5,
            hopping_mode          = self.hopping_mode,
            walking_mode          = self.walking_mode,
        )

        self.monoped_state_object.set_desired_world_point(
            self.desired_pose.position.x,
            self.desired_pose.position.y,
            self.desired_pose.position.z)

        self.monoped_joint_pubisher_object = JointPub()

        # ── Action space: delta joint angles ±0.25 rad ─────────────────────
        self.action_space = spaces.Box(
            low=-0.25, high=0.25, shape=(3,), dtype=np.float32)

        # ── Observation space: dimension from MonopedState ──────────────────
        # Standing mode  → 11 observations
        # Hop-forward    → 13 observations (adds vx and is_airborne)
        # Querying MonopedState prevents silent shape mismatches.
        obs_dim = self.monoped_state_object.get_obs_dim()
        rospy.loginfo("[MonopedEnv] observation_space shape=(%d,)", obs_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reward_range = (-np.inf, np.inf)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ========================================================================
    #  RESET
    # ========================================================================

    def reset(self):
        # 0. Pause simulator
        rospy.logdebug("Pausing SIM...")
        self.gazebo.pauseSim()

        # 1. Full simulation reset
        rospy.logdebug("Reset SIM...")
        self.gazebo.resetSim()

        # 2. Zero gravity so joints can be set without falling
        rospy.logdebug("Remove Gravity...")
        self.gazebo.change_gravity(0.0, 0.0, 0.0)

        # 3. Reset joint controllers (fixes TF time-travel after sim reset)
        rospy.logdebug("reset_monoped_joint_controllers...")
        self.controllers_object.reset_monoped_joint_controllers()

        # 4. Move joints to initial pose
        rospy.logdebug("set_init_pose...")
        self.monoped_joint_pubisher_object.set_init_pose()

        # 5. Wait for all sensor data to arrive; resets prev_x baseline
        rospy.logdebug("check_all_systems_ready...")
        self.monoped_state_object.check_all_systems_ready()

        # 6. *** Reset hop phase state machine ***
        #    Without this the _was_airborne flag and airborne_steps counter
        #    carry over from the previous episode, corrupting hop event bonuses.
        self.monoped_state_object.reset_jump_state()

        # 7. Get initial observation
        rospy.logdebug("get_observations...")
        observation = self.monoped_state_object.get_observations()

        # 8. Restore gravity
        rospy.logdebug("Restore Gravity...")
        self.gazebo.change_gravity(0.0, 0.0, -9.81)

        # 9. Pause again until step() is called
        rospy.logdebug("Pause SIM...")
        self.gazebo.pauseSim()

        return self.get_state(observation)

    # ========================================================================
    #  STEP
    # ========================================================================

    def step(self, action):
        # Unpause, apply action, let physics run, pause again
        self.gazebo.unpauseSim()
        next_action_position = self.monoped_state_object.get_action_to_position(action)
        self.monoped_joint_pubisher_object.move_joints(next_action_position)
        time.sleep(self.running_step)
        self.gazebo.pauseSim()

        # Read sensor state and compute reward
        observation       = self.monoped_state_object.get_observations()
        reward, done      = self.monoped_state_object.process_data()
        state             = self.get_state(observation)

        return state, reward, done, {}

    def get_state(self, observation):
        return np.array(observation)