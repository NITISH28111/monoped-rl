#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
import tf
import numpy
import math

"""
Contact state and message format reference preserved from original.
"""

# ===========================================================================
#  REWARD WEIGHT TABLES
#  Edit weights here — nowhere else — to tune behaviour.
# ===========================================================================

# ── STANDING mode  (hopping_mode=False OR walking_mode=False) ──────────────
# Weights are unchanged from the working standing policy so that stand-only
# training still produces the same behaviour you already have.
STAND_W = dict(
    alive        = 10.0,   # per-step survival
    height       =  3.0,   # raw height reward  (weight_r6)
    orientation  =  1.0,   # roll+pitch penalty  (weight_r4)
    joint_pos    =  1.0,   # HAA deviation penalty  (weight_r1)
    joint_effort =  0.02,  # torque penalty  (weight_r2)
    joint_smooth =  0.10,  # velocity penalty
    knee_bend    =  3.0,   # HFE×(−KFE) coordination
)

# ── HOP-FORWARD mode  (hopping_mode=True AND walking_mode=True) ────────────
# Priority is enforced by weight magnitude:
#   stability (biggest) > hop quality > forward progress > effort
HOP_W = dict(
    # 1. STABILITY — must not fall
    alive            =  5.0,   # per-step survival (reduced; episode length matters less)
    orientation_roll =  8.0,   # squared roll penalty — kills the -Y head tilt
    orientation_pitch=  2.0,   # REDUCED: was 5.0 — large pitch penalty was fighting forward lean
    lateral_vel      =  3.0,   # penalty for Y-axis velocity (no sideways drift)
    height_floor     =  5.0,   # penalty when body drops below standing baseline
    haa_penalty      =  2.0,   # penalise HAA deviation (keeps leg centred under body)

    # 2. HOP QUALITY
    air_per_step     =  4.0,   # reward every step the foot is truly airborne
    air_height_bonus =  8.0,   # extra reward scaled by height-above-baseline while airborne
    liftoff_bonus    = 30.0,   # one-shot bonus on landing after a valid hop
    push_vel_z       =  6.0,   # reward upward body velocity while on the ground (push-off)

    # 3. FORWARD PROGRESS
    forward_vel      =  5.0,   # reward positive X velocity every step
    forward_delta    =  4.0,   # reward per-step X displacement (positive only)
    backward_penalty =  3.0,   # penalty for negative X velocity

    # 4. BODY ALIGNMENT — keeps torso coordinated with leg motion
    # Problem: leg pushes forward but torso lags behind (pitches backward).
    # Fix: penalise backward pitch (negative pitch_y) when robot is moving forward,
    # and reward forward pitch coordination during push-off.
    pitch_vel_misalign =  6.0,  # penalty when pitch < 0 AND vx > 0 (torso lagging leg)
    forward_pitch_bonus=  3.0,  # reward small forward lean (0 < pitch < 0.3 rad) during fwd motion

    # 5. EFFORT / SMOOTHNESS (tiny, just keeps motion clean)
    joint_effort     =  0.01,
    joint_smooth     =  0.03,
)

# Physical constants
_STAND_HEIGHT_BASELINE = 0.50   # m — approximate standing CoM height
_AIRBORNE_FORCE_THRESH = 0.5    # N — contact force below this = airborne
_VALID_HOP_CLEARANCE   = 0.04   # m — liftoff must exceed baseline by this for a "real" hop
_MIN_AIRBORNE_STEPS    = 3      # consecutive airborne steps to count as a hop


class MonopedState(object):

    def __init__(self,
                 max_height, min_height,
                 abs_max_roll, abs_max_pitch,
                 joint_increment_value=0.05,
                 done_reward=-1000.0,
                 alive_reward=10.0,
                 desired_force=7.08,
                 desired_yaw=0.0,
                 weight_r1=1.0, weight_r2=1.0, weight_r3=1.0,
                 weight_r4=1.0, weight_r5=1.0,
                 discrete_division=10,
                 hopping_mode=False,
                 walking_mode=False):

        rospy.logdebug("Starting MonopedState Class object...")

        # ── geometry / termination limits ──────────────────────────────────
        self.desired_world_point      = Vector3(0.0, 0.0, 0.0)
        self._min_height              = min_height
        self._max_height              = max_height
        self._abs_max_roll            = abs_max_roll
        self._abs_max_pitch           = abs_max_pitch
        self._joint_increment_value   = joint_increment_value
        self._done_reward             = done_reward
        self._alive_reward            = alive_reward
        self._desired_force           = desired_force
        self._desired_yaw             = desired_yaw
        self._discrete_division       = discrete_division

        # Keep original weight params (used by standing reward path)
        self._weight_r1 = weight_r1
        self._weight_r2 = weight_r2
        self._weight_r3 = weight_r3
        self._weight_r4 = weight_r4
        self._weight_r5 = weight_r5
        self._weight_r6 = rospy.get_param("/weight_r6", 3.0)

        # ── MODE FLAGS ──────────────────────────────────────────────────────
        # Both flags must be True to activate hop-forward reward.
        # Any other combination uses the standing reward (backward compatible).
        self.hopping_mode       = hopping_mode
        self.walking_mode       = walking_mode
        self._hop_forward_active = hopping_mode and walking_mode

        rospy.loginfo("[MonopedState] hop_forward_active=%s  (hop=%s, walk=%s)",
                      self._hop_forward_active, hopping_mode, walking_mode)

        # ── HOP PHASE STATE MACHINE ─────────────────────────────────────────
        self._was_airborne      = False
        self._airborne_steps    = 0       # consecutive steps in air this hop
        self._liftoff_height    = 0.0     # CoM height at moment foot left ground
        self._liftoff_x         = 0.0     # X position at liftoff

        # ── FORWARD PROGRESS TRACKING ───────────────────────────────────────
        self.prev_x             = 0.0

        # ── SENSOR STATE ────────────────────────────────────────────────────
        self.base_position            = Point()
        self.base_orientation         = Quaternion()
        self.base_linear_acceleration = Vector3()
        self.base_linear_velocity     = Vector3()
        self.contact_force            = Vector3()
        self.joints_state             = JointState()

        # ── OBSERVATION LIST ────────────────────────────────────────────────
        # Standing mode: 11 observations  (shape must match monoped_env.py)
        # Hop-forward mode: 13 observations — adds vx and foot-contact flag
        self._list_of_observations = [
            "base_roll",
            "base_pitch",
            "contact_force",
            "joint_states_haa",
            "joint_states_hfe",
            "joint_states_kfe",
            "joint_vel_haa",
            "joint_vel_hfe",
            "joint_vel_kfe",
            "base_z",
            "base_z_velocity",
        ]
        if self._hop_forward_active:
            self._list_of_observations += [
                "base_x_velocity",   # forward velocity — critical for hop-forward shaping
                "is_airborne",       # binary 0/1 contact state
            ]

        # ── ROS SUBSCRIBERS ─────────────────────────────────────────────────
        rospy.Subscriber("/odom",                         Odometry,      self.odom_callback)
        rospy.Subscriber("/monoped/imu/data",             Imu,           self.imu_callback)
        rospy.Subscriber("/lowerleg_contactsensor_state", ContactsState, self.contact_callback)
        rospy.Subscriber("/monoped/joint_states",         JointState,    self.joints_state_callback)

    # ========================================================================
    #  SYSTEM READINESS
    # ========================================================================

    def check_all_systems_ready(self):
        data_pose = None
        while data_pose is None and not rospy.is_shutdown():
            try:
                data_pose = rospy.wait_for_message("/odom", Odometry, timeout=0.1)
                self.base_position = data_pose.pose.pose.position
                rospy.logdebug("Current odom READY")
            except:
                rospy.logdebug("Current odom pose not ready yet, retrying")

        imu_data = None
        while imu_data is None and not rospy.is_shutdown():
            try:
                imu_data = rospy.wait_for_message("/monoped/imu/data", Imu, timeout=0.1)
                self.base_orientation         = imu_data.orientation
                self.base_linear_acceleration = imu_data.linear_acceleration
                rospy.logdebug("Current imu_data READY")
            except:
                rospy.logdebug("Current imu_data not ready yet, retrying")

        contacts_data = None
        while contacts_data is None and not rospy.is_shutdown():
            try:
                contacts_data = rospy.wait_for_message(
                    "/lowerleg_contactsensor_state", ContactsState, timeout=0.1)
                for state in contacts_data.states:
                    self.contact_force = state.total_wrench.force
                rospy.logdebug("Current contacts_data READY")
            except:
                rospy.logdebug("Current contacts_data not ready yet, retrying")

        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message(
                    "/monoped/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                rospy.logdebug("Current joint_states not ready yet, retrying==>"+str(e))

        rospy.logdebug("ALL SYSTEMS READY")
        # Reset forward-progress baseline every episode
        self.prev_x = self.base_position.x

    def set_desired_world_point(self, x, y, z):
        self.desired_world_point.x = x
        self.desired_world_point.y = y
        self.desired_world_point.z = z

    # ========================================================================
    #  STATE ACCESSORS
    # ========================================================================

    def get_base_height(self):
        return self.base_position.z

    def get_base_rpy(self):
        euler_rpy = Vector3()
        euler = tf.transformations.euler_from_quaternion(
            [self.base_orientation.x,
             self.base_orientation.y,
             self.base_orientation.z,
             self.base_orientation.w])
        euler_rpy.x = euler[0]
        euler_rpy.y = euler[1]
        euler_rpy.z = euler[2]
        return euler_rpy

    def get_distance_from_point(self, p_end):
        a = numpy.array((self.base_position.x, self.base_position.y, self.base_position.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))
        return numpy.linalg.norm(a - b)

    def get_contact_force_magnitude(self):
        cf = self.contact_force
        return numpy.linalg.norm(numpy.array((cf.x, cf.y, cf.z)))

    def is_airborne(self):
        """True when foot contact force is below the airborne threshold."""
        return self.get_contact_force_magnitude() < _AIRBORNE_FORCE_THRESH

    def get_joint_states(self):
        return self.joints_state

    # ========================================================================
    #  ROS CALLBACKS
    # ========================================================================

    def odom_callback(self, msg):
        self.base_position        = msg.pose.pose.position
        self.base_linear_velocity = msg.twist.twist.linear

    def imu_callback(self, msg):
        self.base_orientation         = msg.orientation
        self.base_linear_acceleration = msg.linear_acceleration

    def contact_callback(self, msg):
        if msg.states:
            self.contact_force = msg.states[0].total_wrench.force
        else:
            self.contact_force = Vector3()

    def joints_state_callback(self, msg):
        self.joints_state = msg

    # ========================================================================
    #  TERMINATION CHECKS
    # ========================================================================

    def monoped_height_ok(self):
        return self._min_height <= self.get_base_height() < self._max_height

    def monoped_orientation_ok(self):
        rpy = self.get_base_rpy()
        return (abs(rpy.x) < self._abs_max_roll and
                abs(rpy.y) < self._abs_max_pitch)

    # ========================================================================
    #  SHARED LOW-LEVEL REWARD PRIMITIVES
    #  (used by both standing and hopping paths)
    # ========================================================================

    def calculate_reward_joint_position(self, weight=1.0):
        """Penalty proportional to HAA (hip abduction) angle — keeps leg centred."""
        haa_pos = abs(self.joints_state.position[0])
        reward  = weight * haa_pos
        rospy.logdebug("calculate_reward_joint_position>>reward=" + str(reward))
        return reward

    def calculate_reward_knee_bend(self, weight=1.0):
        """Reward for HFE × (−KFE) coordination — correct standing posture."""
        hfe          = self.joints_state.position[1]
        kfe          = self.joints_state.position[2]
        coordination = hfe * (-kfe)
        return weight * max(0.0, coordination)

    def calculate_reward_joint_effort(self, weight=1.0):
        """Penalty proportional to summed joint torque."""
        total = sum(abs(e) for e in self.joints_state.effort)
        rospy.logdebug("calculate_reward_joint_effort>>reward=" + str(total))
        return weight * total

    def calculate_reward_contact_force(self, weight=1.0):
        """Penalty for deviation from desired contact force (original, unused in new paths)."""
        force_magnitude    = self.get_contact_force_magnitude()
        force_displacement = force_magnitude - self._desired_force
        reward             = weight * abs(force_displacement)
        rospy.logdebug("calculate_reward_contact_force>>reward=" + str(reward))
        return reward

    def calculate_reward_orientation(self, weight=1.0):
        """Linear roll+pitch penalty (used in standing path)."""
        rpy        = self.get_base_rpy()
        rp_penalty = abs(rpy.x) + abs(rpy.y)
        rospy.logdebug("calculate_reward_orientation>>reward=" + str(rp_penalty))
        return weight * rp_penalty

    def calculate_reward_height(self, weight=1.0):
        """Raw height reward (used in standing path)."""
        height = self.get_base_height()
        rospy.logdebug("calculate_reward_height>>reward=" + str(weight * height))
        return weight * height

    def calculate_reward_joint_smoothness(self, weight=1.0):
        """Penalty for high joint velocities."""
        vel     = list(self.joints_state.velocity[:3])
        penalty = sum(max(0.0, abs(v) - 0.3)**2 for v in vel)
        return weight * penalty

    def calculate_reward_distance_from_des_point(self, weight=1.0):
        """Distance penalty from desired XY point (original, kept for compatibility)."""
        dx       = self.base_position.x - self.desired_world_point.x
        dy       = self.base_position.y - self.desired_world_point.y
        distance = numpy.sqrt(dx**2 + dy**2)
        rospy.logdebug("calculate_reward_distance_from_des_point>>reward=" + str(distance))
        return weight * distance

    # ========================================================================
    #  HOP-FORWARD REWARD COMPONENTS
    #  Each method is self-contained and documented with its purpose.
    # ========================================================================

    def _hop_r_stability_orientation(self):
        """
        SQUARED roll+pitch penalty with extra weight on roll.
        Using squared (not linear) means small angles are almost free but any
        significant tilt is very expensive.  This directly fixes the -Y head
        tilt observed in current training: roll deviation is penalised at 2×
        the pitch coefficient.
        Returns a positive number — caller subtracts it.
        """
        rpy      = self.get_base_rpy()
        roll_sq  = rpy.x ** 2
        pitch_sq = rpy.y ** 2
        return (HOP_W['orientation_roll']  * roll_sq +
                HOP_W['orientation_pitch'] * pitch_sq)

    def _hop_r_stability_lateral(self):
        """
        Penalty for Y-axis velocity.  Prevents the robot from drifting
        sideways while trying to hop forward.
        Returns a positive number — caller subtracts it.
        """
        return HOP_W['lateral_vel'] * abs(self.base_linear_velocity.y)

    def _hop_r_stability_height_floor(self):
        """
        Soft height-floor reward.
        +  when above baseline  → body is standing or airborne (good)
        −  when below baseline  → body is crouching too low (bad)
        Not intended to replace the hard termination check; it shapes
        the approach to standing before a hop.
        """
        h     = self.get_base_height()
        delta = h - _STAND_HEIGHT_BASELINE
        if delta >= 0:
            return  HOP_W['height_floor'] * delta          # reward for standing tall / being airborne
        else:
            return  HOP_W['height_floor'] * delta * 2.0   # double penalty for crouching below baseline

    def _hop_r_haa_penalty(self):
        """
        Penalty for HAA (hip abduction) deviation.
        In hopping mode the foot must stay directly under the body CoM for
        stable push-off; any sideways lean is penalised.
        Returns a positive number — caller subtracts it.
        """
        return HOP_W['haa_penalty'] * abs(self.joints_state.position[0])

    def _hop_r_air_time(self):
        """
        Per-step reward while the foot is truly airborne.
        Scaled by height above baseline so the agent is incentivised to jump
        higher, not just to barely lift off.
        Returns 0 when on the ground.
        """
        if not self.is_airborne():
            return 0.0
        h_above = max(0.0, self.get_base_height() - _STAND_HEIGHT_BASELINE)
        return HOP_W['air_per_step'] + HOP_W['air_height_bonus'] * h_above

    def _hop_r_hop_event(self):
        """
        State-machine hop event tracker.
        Gives a one-shot liftoff_bonus on the step when the robot lands after
        a VALID hop (airborne for at least _MIN_AIRBORNE_STEPS consecutive steps
        AND liftoff height was above baseline + clearance).

        Also updates:  self._was_airborne, self._airborne_steps,
                       self._liftoff_height, self._liftoff_x
        """
        currently_airborne = self.is_airborne()
        reward = 0.0
        h      = self.get_base_height()

        if currently_airborne:
            if not self._was_airborne:
                # Liftoff event
                self._liftoff_height = h
                self._liftoff_x      = self.base_position.x
                self._was_airborne   = True
                self._airborne_steps = 1
            else:
                self._airborne_steps += 1
        else:
            if self._was_airborne:
                # Landing event — check whether this was a real hop
                valid_clearance = self._liftoff_height > (_STAND_HEIGHT_BASELINE + _VALID_HOP_CLEARANCE)
                valid_duration  = self._airborne_steps >= _MIN_AIRBORNE_STEPS
                if valid_clearance and valid_duration:
                    # Scale bonus by airtime: longer hop = larger reward (capped at 3×)
                    time_scale = min(3.0, float(self._airborne_steps) / _MIN_AIRBORNE_STEPS)
                    reward     = HOP_W['liftoff_bonus'] * time_scale
                    rospy.logdebug(
                        "[HOP EVENT] bonus=%.1f  airborne_steps=%d  liftoff_h=%.3f",
                        reward, self._airborne_steps, self._liftoff_height)
                # Reset phase state
                self._was_airborne   = False
                self._airborne_steps = 0
                self._liftoff_height = 0.0
                self._liftoff_x      = 0.0

        return reward

    def _hop_r_push_velocity(self):
        """
        Reward upward body velocity (vz > 0) WHILE on the ground.
        This teaches the push-off motion: the only way to get vz > 0 while
        still in contact is to actively push against the ground.
        Returns 0 when airborne (air-time reward handles that phase).
        """
        if self.is_airborne():
            return 0.0
        vz = self.base_linear_velocity.z
        if vz > 0:
            return HOP_W['push_vel_z'] * vz
        return 0.0

    def _hop_r_forward_velocity(self):
        """
        Reward positive X velocity; penalise negative X velocity.
        Using velocity (not displacement) gives a smoother signal and helps
        SAC learn the policy faster than sparse displacement rewards alone.
        """
        vx = self.base_linear_velocity.x
        if vx > 0:
            return  HOP_W['forward_vel'] * vx
        else:
            return -HOP_W['backward_penalty'] * abs(vx)

    def _hop_r_forward_delta(self):
        """
        Reward actual per-step X displacement.
        Complements velocity reward: even at low velocity a consistent positive
        delta accumulates.  Uses a properly reset self.prev_x so the delta is
        valid on the very first step of each episode.
        """
        current_x   = self.base_position.x
        x_delta     = current_x - self.prev_x
        self.prev_x = current_x
        return HOP_W['forward_delta'] * x_delta  # negative if moving backward

    def _hop_r_effort(self):
        """
        Tiny effort + smoothness penalty.
        Weaker than in standing mode because hopping requires more joint work.
        Returns a positive number — caller subtracts it.
        """
        total_effort = sum(abs(e) for e in self.joints_state.effort)
        vel_penalty  = sum(max(0.0, abs(v) - 0.5) ** 2
                          for v in list(self.joints_state.velocity[:3]))
        return (HOP_W['joint_effort'] * total_effort +
                HOP_W['joint_smooth'] * vel_penalty)

    def _hop_r_body_alignment(self):
        """
        Body alignment reward — the core fix for 'head lags behind legs'.

        The problem: during push-off the leg extends and contacts the ground
        ahead of the CoM projection, rotating the torso BACKWARD (negative
        pitch in this URDF convention).  The original orientation_pitch penalty
        penalises ALL pitch equally, so the policy learned to keep the torso
        perfectly vertical even while the leg shoots forward — decoupling them.

        Fix has two parts:
        (a) PENALTY: when the torso is pitching backward (pitch < 0) AND the
            robot is moving forward (vx > 0), apply an extra penalty
            proportional to |pitch| × vx.  This directly targets the observed
            mismatch: the larger the forward velocity and the more the head
            lags, the bigger the penalty.

        (b) BONUS: when the torso has a small forward lean (0 < pitch < 0.3 rad)
            AND vx > 0, give a small bonus.  This teaches the robot to lean
            INTO the hop direction, transferring momentum from the torso to the
            direction of travel rather than wasting it in backward rotation.
            The 0.3 rad cap prevents the agent from simply diving face-first.

        Returns a signed value: positive = good alignment, negative = penalty.
        Caller ADDS this value (it can be negative).
        """
        rpy   = self.get_base_rpy()
        pitch = rpy.y          # negative = torso leaning backward
        vx    = self.base_linear_velocity.x

        result = 0.0

        # (a) Penalty for backward-lean while moving forward
        if pitch < 0 and vx > 0:
            # Both factors amplify each other: badly lagging torso at high speed
            # is much worse than a slight lag at low speed
            result -= HOP_W['pitch_vel_misalign'] * abs(pitch) * vx

        # (b) Bonus for small forward lean while moving forward
        if 0 < pitch < 0.3 and vx > 0:
            result += HOP_W['forward_pitch_bonus'] * pitch * vx

        return result

    # ========================================================================
    #  HOP PHASE RESET  (call at episode start AND on termination)
    # ========================================================================

    def reset_jump_state(self):
        """Reset the hop phase state machine.  Must be called every episode."""
        self._was_airborne   = False
        self._airborne_steps = 0
        self._liftoff_height = 0.0
        self._liftoff_x      = 0.0

    # ========================================================================
    #  MASTER REWARD DISPATCHER
    # ========================================================================

    def calculate_total_reward(self):
        """
        Routes to the correct reward function based on mode flags.
        hopping_mode=True AND walking_mode=True  → hop-forward reward
        anything else                            → original standing reward
        """
        if self._hop_forward_active:
            return self._calculate_reward_hop_forward()
        else:
            return self._calculate_reward_standing()

    def _calculate_reward_standing(self):
        """
        Original standing reward — identical logic to the old calculate_total_reward().
        Kept separate so changes here cannot accidentally break hop-forward training.
        """
        w = STAND_W
        r_alive   = w['alive']
        r_height  = self.calculate_reward_height(self._weight_r6)   # uses yaml weight
        r_knee    = self.calculate_reward_knee_bend(w['knee_bend'])
        r_orient  = self.calculate_reward_orientation(w['orientation'])   # subtracted
        r_jpos    = self.calculate_reward_joint_position(w['joint_pos'])  # subtracted
        r_effort  = self.calculate_reward_joint_effort(w['joint_effort']) # subtracted
        r_smooth  = self.calculate_reward_joint_smoothness(w['joint_smooth'])  # subtracted

        total = r_alive + r_height + r_knee - r_orient - r_jpos - r_effort - r_smooth

        rospy.logdebug("###############")
        rospy.logdebug("alive=%s  height=%s  knee=%s", r_alive, r_height, r_knee)
        rospy.logdebug("orient=%s  jpos=%s  effort=%s  smooth=%s",
                       r_orient, r_jpos, r_effort, r_smooth)
        rospy.logdebug("total=%s", total)
        rospy.logdebug("###############")
        return total

    def _calculate_reward_hop_forward(self):
        """
        Hop-forward reward.

        TERM                  SIGN    PRIORITY
        ──────────────────────────────────────────────────────
        alive                  +      1  stability
        orientation penalty    -      1  stability (squared, fixes -Y tilt)
        lateral vel penalty    -      1  stability
        height floor           ±      1  stability
        HAA penalty            -      1  stability
        air-time per step      +      2  hop quality
        hop event bonus        +      2  hop quality (one-shot on landing)
        push velocity          +      2  hop quality (teaches push-off)
        forward velocity       ±      3  forward progress
        forward delta          ±      3  forward progress
        body alignment         ±      3  torso-leg coordination (NEW)
        effort penalty         -      4  efficiency
        """
        # 1. Stability
        r_alive    = HOP_W['alive']
        pen_orient = self._hop_r_stability_orientation()  # subtracted
        pen_lat    = self._hop_r_stability_lateral()      # subtracted
        r_height   = self._hop_r_stability_height_floor() # can be negative
        pen_haa    = self._hop_r_haa_penalty()            # subtracted

        # 2. Hop quality
        r_air      = self._hop_r_air_time()
        r_hop_evt  = self._hop_r_hop_event()
        r_push     = self._hop_r_push_velocity()

        # 3. Forward progress
        r_fwd_vel  = self._hop_r_forward_velocity()
        r_fwd_dlt  = self._hop_r_forward_delta()

        # 4. Body alignment (signed — adds reward for forward lean, penalty for backward lag)
        r_align    = self._hop_r_body_alignment()

        # 5. Effort
        pen_effort = self._hop_r_effort()                 # subtracted

        total = (r_alive
                 + r_height
                 + r_air
                 + r_hop_evt
                 + r_push
                 + r_fwd_vel
                 + r_fwd_dlt
                 + r_align
                 - pen_orient
                 - pen_lat
                 - pen_haa
                 - pen_effort)

        rospy.logdebug(
            "[HOP-FWD] alive=%.1f h=%.2f air=%.2f evt=%.1f push=%.2f "
            "fvl=%.2f fdt=%.2f aln=%.2f | ori=%.2f lat=%.2f haa=%.2f eff=%.2f | TOT=%.2f",
            r_alive, r_height, r_air, r_hop_evt, r_push,
            r_fwd_vel, r_fwd_dlt, r_align,
            pen_orient, pen_lat, pen_haa, pen_effort, total)

        return total

    # ========================================================================
    #  OBSERVATIONS
    # ========================================================================

    def get_observations(self):
        distance_from_desired_point = self.get_distance_from_point(self.desired_world_point)
        base_orientation            = self.get_base_rpy()
        base_roll                   = base_orientation.x
        base_pitch                  = base_orientation.y
        base_yaw                    = base_orientation.z
        contact_force               = self.get_contact_force_magnitude()
        joint_states                = self.get_joint_states()
        joint_states_haa            = joint_states.position[0]
        joint_states_hfe            = joint_states.position[1]
        joint_states_kfe            = joint_states.position[2]

        observation = []
        for obs_name in self._list_of_observations:
            if obs_name == "distance_from_desired_point":
                observation.append(distance_from_desired_point)
            elif obs_name == "base_roll":
                observation.append(base_roll)
            elif obs_name == "base_pitch":
                observation.append(base_pitch)
            elif obs_name == "base_yaw":
                observation.append(base_yaw)
            elif obs_name == "contact_force":
                observation.append(contact_force)
            elif obs_name == "joint_states_haa":
                observation.append(joint_states_haa)
            elif obs_name == "joint_states_hfe":
                observation.append(joint_states_hfe)
            elif obs_name == "joint_states_kfe":
                observation.append(joint_states_kfe)
            elif obs_name == "joint_vel_haa":
                observation.append(self.joints_state.velocity[0])
            elif obs_name == "joint_vel_hfe":
                observation.append(self.joints_state.velocity[1])
            elif obs_name == "joint_vel_kfe":
                observation.append(self.joints_state.velocity[2])
            elif obs_name == "base_z_velocity":
                observation.append(self.base_linear_velocity.z)
            elif obs_name == "base_z":
                observation.append(self.base_position.z)
            elif obs_name == "base_x_velocity":
                observation.append(self.base_linear_velocity.x)
            elif obs_name == "is_airborne":
                observation.append(1.0 if self.is_airborne() else 0.0)
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))

        return observation

    def get_obs_dim(self):
        """Return observation dimension so monoped_env.py can query it dynamically."""
        return len(self._list_of_observations)

    # ========================================================================
    #  ACTION → JOINT POSITIONS
    # ========================================================================

    def get_action_to_position(self, action):
        """
        Converts a delta action into absolute joint position commands.

        Standing mode:  HFE in [0.0, 1.0],  KFE in [-1.0, 0.0]
        Hop-forward:    HFE in [-0.1, 1.2],  KFE in [-1.2, 0.1]
          The wider bounds allow the leg to extend nearly straight for push-off.
          The old [0,1]/[-1,0] bounds prevented full leg extension and blocked liftoff.
        """
        joint_states          = self.get_joint_states()
        joint_states_position = joint_states.position
        action_position       = [0.0, 0.0, 0.0]

        action_position[0] = joint_states_position[0] + action[0]
        action_position[1] = joint_states_position[1] + action[1]
        action_position[2] = joint_states_position[2] + action[2]

        action_position[0] = numpy.clip(action_position[0], -0.5,  0.5)

        if self._hop_forward_active:
            # Wider HFE/KFE range enables full leg extension for push-off
            action_position[1] = numpy.clip(action_position[1], -0.1,  1.2)
            action_position[2] = numpy.clip(action_position[2], -1.2,  0.1)
        else:
            action_position[1] = numpy.clip(action_position[1],  0.0,  1.0)
            action_position[2] = numpy.clip(action_position[2], -1.0,  0.0)

        return action_position

    # ========================================================================
    #  STEP PROCESSING
    # ========================================================================

    def process_data(self):
        """
        Called every step.  Returns (reward, done).
        On termination: applies done_reward and resets hop state machine.
        """
        height_ok      = self.monoped_height_ok()
        orientation_ok = self.monoped_orientation_ok()
        done           = not (height_ok and orientation_ok)

        if done:
            rospy.logdebug("Robot fell — applying done_reward and resetting hop state")
            self.reset_jump_state()
            return self._done_reward, True

        return self.calculate_total_reward(), False

    # ========================================================================
    #  TESTING HELPER
    # ========================================================================

    def testing_loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.calculate_total_reward()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node('monoped_state_node', anonymous=True)
    monoped_state = MonopedState(max_height=3.0,
                                 min_height=0.6,
                                 abs_max_roll=0.7,
                                 abs_max_pitch=0.7)
    monoped_state.testing_loop()