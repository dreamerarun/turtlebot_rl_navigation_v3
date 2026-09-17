"""
Gymnasium environment that drives TurtleBot3 in Gazebo via live ROS 2 topics.

CHANGES vs. original turtlebot_gym_env.py (see report.pdf Section 5/7.4 for
why these were needed):

  1. `allow_reverse` (default False): when True, the action space's linear
     velocity lower bound is -0.5*MAX_LIN_VEL instead of 0.0. The report's
     own Section 7.3 notes the dead-end scenario's policy CANNOT genuinely
     recover because the action space is forward-only -- this makes
     recovery physically possible. A small `reverse_penalty` reward term
     discourages using it except when actually needed, so it doesn't
     degrade normal navigation efficiency.

  2. `dynamic_obstacle` (default False): when True, the env itself drives
     `moving_obstacle` back and forth via /set_entity_state during
     training (not just during evaluate.py's held-out evaluation). Without
     this, the policy has literally zero training exposure to motion before
     being scored on the dynamic-obstacle scenario -- that's why the
     original run collapsed to 50%/50% there. Use this ONLY for an
     additional short curriculum stage after Stage 2 (static) converges;
     do not train Stage 2 itself with this on, or you conflate two
     difficulty axes at once.

  Both flags default to False/off, so plain `TurtleBot3NavEnv()` behaves
  exactly like the original for Stage 1/Stage 2 training.
"""
import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from gazebo_msgs.msg import EntityState

# ---- Physical / task constants -------------------------------------------
MAX_LIN_VEL = 0.22      # m/s, TurtleBot3 Burger physical limit
MAX_ANG_VEL = 2.84      # rad/s, TurtleBot3 Burger physical limit
LIDAR_MAX_RANGE = 3.5   # m, clip range
N_BEAMS = 20            # downsampled beam count
CONTROL_HZ = 5.0        # matched to /scan publish rate — see module docstring
CONTROL_DT = 1.0 / CONTROL_HZ
MAX_EPISODE_STEPS = 300         # ~60s wall-clock per episode at 5 Hz
GOAL_TOLERANCE = 0.15           # m
COLLISION_RANGE = 0.15          # m, min LiDAR range considered a collision
WORLD_BOUNDS = 4.0               # m, half-width of the arena (out-of-bounds check)

# Reward term weights -- table form, see report.pdf section 2 for tuning notes
REWARD_WEIGHTS = {
    "goal_reached": 100.0,
    "collision": -100.0,
    "progress": 40.0,        # coefficient on (prev_dist - curr_dist)
    "step_penalty": -0.05,   # per-step cost, encourages efficient paths
    "smoothness": -0.02,     # coefficient on |delta angular velocity|
    "reverse_penalty": -0.03,  # NEW: per-step cost while v < 0, only relevant if allow_reverse=True
}

DYNAMIC_OBSTACLE_NAME = "moving_obstacle"
DYNAMIC_OBSTACLE_CENTER = (0.5, 0.0)
DYNAMIC_OBSTACLE_AMPLITUDE = 0.8
DYNAMIC_OBSTACLE_PERIOD_S = 4.0


def yaw_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class _RosBridge(Node):
    """Thin ROS2 node: subscribes sensors, publishes cmd_vel, offers a
    synchronous spin_until_fresh_scan() so the Gym env can block until the
    next real LiDAR frame arrives (never reusing a stale one)."""

    def __init__(self):
        super().__init__("rl_nav_gym_bridge")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)
        self.scan = None
        self.odom = None
        self._scan_seq = 0
        self._last_consumed_seq = -1

        self.create_subscription(LaserScan, "/scan", self._scan_cb, qos)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # NOTE: this ROS2 Gazebo bridge (libgazebo_ros_state) advertises the
        # service as plain "/set_entity_state" -- the ROS1-style prefixed
        # name "/gazebo/set_entity_state" does NOT exist here. Confirmed via
        # `ros2 service list` on the VM; using the wrong name meant every
        # reset_robot_pose() call silently timed out (2s wait_for_service),
        # so the robot was never actually repositioned between episodes.
        self.set_state_cli = self.create_client(SetEntityState, "/set_entity_state")

    def _scan_cb(self, msg):
        self.scan = msg
        self._scan_seq += 1

    def _odom_cb(self, msg):
        self.odom = msg

    def publish_cmd(self, v, w):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)

    def spin_until_fresh_scan(self, timeout_s=2.0):
        """Block (spinning the node) until a LiDAR scan newer than the last
        one we consumed arrives. Guarantees no stale-observation reuse."""
        start = time.time()
        target_seq = self._scan_seq
        while self._scan_seq <= target_seq:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - start > timeout_s:
                break
        self._last_consumed_seq = self._scan_seq

    def reset_robot_pose(self, x, y, yaw, entity_name="burger"):
        if not self.set_state_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("set_entity_state service unavailable")
            return
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = entity_name
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.01
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        future = self.set_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def set_entity_xy(self, name, x, y, z=0.25):
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = name
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = z
        self.set_state_cli.call_async(req)  # fire-and-forget, matches evaluate.py's approach


class TurtleBot3NavEnv(gym.Env):
    """Gymnasium env: TurtleBot3 nav via live Gazebo/ROS2 topics."""

    metadata = {"render_modes": []}

    def __init__(self, randomize_obstacles=False, curriculum_stage=0,
                 allow_reverse=False, dynamic_obstacle=False):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()
        self.node = _RosBridge()

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * (N_BEAMS + 4),), dtype=np.float32
        )
        self.allow_reverse = allow_reverse
        min_v = -0.5 * MAX_LIN_VEL if allow_reverse else 0.0
        self.action_space = spaces.Box(
            low=np.array([min_v, -MAX_ANG_VEL], dtype=np.float32),
            high=np.array([MAX_LIN_VEL, MAX_ANG_VEL], dtype=np.float32),
        )

        self.randomize_obstacles = randomize_obstacles
        self.curriculum_stage = curriculum_stage
        self.dynamic_obstacle = dynamic_obstacle
        self._frame_buffer = None
        self._prev_w = 0.0
        self._step_count = 0
        self._episode_t0 = None
        self.goal = np.array([1.5, 0.0])

    # -- helpers -------------------------------------------------------
    def _get_lidar_vec(self):
        scan = self.node.scan
        if scan is None:
            return np.ones(N_BEAMS, dtype=np.float32)
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=LIDAR_MAX_RANGE, posinf=LIDAR_MAX_RANGE,
                                neginf=0.0)
        ranges = np.clip(ranges, 0.0, LIDAR_MAX_RANGE)
        n = len(ranges)
        edges = np.linspace(0, n, N_BEAMS + 1, dtype=int)
        pooled = np.array([ranges[edges[i]:edges[i + 1]].min() if edges[i + 1] > edges[i]
                            else LIDAR_MAX_RANGE for i in range(N_BEAMS)])
        return (pooled / LIDAR_MAX_RANGE).astype(np.float32)

    def _get_pose(self):
        odom = self.node.odom
        if odom is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        p = odom.pose.pose.position
        yaw = yaw_from_quaternion(odom.pose.pose.orientation)
        v = odom.twist.twist.linear.x
        w = odom.twist.twist.angular.z
        return p.x, p.y, yaw, v, w

    def _build_frame(self):
        lidar = self._get_lidar_vec()
        x, y, yaw, v, w = self._get_pose()
        dx, dy = self.goal[0] - x, self.goal[1] - y
        dist = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        heading_err = target_heading - yaw
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))
        extras = np.array([
            min(dist / 5.0, 1.0),
            math.sin(heading_err),
            math.cos(heading_err),
            v / MAX_LIN_VEL,
        ], dtype=np.float32)
        return np.concatenate([lidar, extras]), dist, lidar.min() * LIDAR_MAX_RANGE

    def _drive_dynamic_obstacle_if_enabled(self):
        if not self.dynamic_obstacle or self._episode_t0 is None:
            return
        elapsed = time.time() - self._episode_t0
        y = DYNAMIC_OBSTACLE_AMPLITUDE * math.sin(2 * math.pi * elapsed / DYNAMIC_OBSTACLE_PERIOD_S)
        self.node.set_entity_xy(
            DYNAMIC_OBSTACLE_NAME,
            DYNAMIC_OBSTACLE_CENTER[0],
            DYNAMIC_OBSTACLE_CENTER[1] + y,
        )

    # -- Gym API ---------------------------------------------------------
    # Extra margin above COLLISION_RANGE for validating a spawn as safe --
    # rejects a random start that would already be touching/inside an
    # obstacle, without needing to hardcode each world's obstacle
    # coordinates. See MAX_RESET_ATTEMPTS below for the fallback if 3m x 3m
    # uniform sampling keeps landing in an obstacle footprint.
    SPAWN_SAFETY_MARGIN = 0.10
    MAX_RESET_ATTEMPTS = 20

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.goal = rng.uniform(-2.5, 2.5, size=2)

        # BUGFIX: uniform start sampling over [-1.5, 1.5]^2 is independent of
        # obstacle placement, so roughly 1-in-5 random starts land inside or
        # touching an obstacle's collision margin (confirmed empirically:
        # ~20% of evaluation episodes were terminating on step 1 with
        # path_length 0.0 -- i.e. "collided" before ever moving). Those
        # aren't real navigation failures, they're invalid spawns. Reject
        # and resample until the spawn point reads clear on a fresh scan.
        frame = None
        dist = None
        for attempt in range(self.MAX_RESET_ATTEMPTS):
            start_x, start_y = rng.uniform(-1.5, 1.5, size=2)
            start_yaw = rng.uniform(-math.pi, math.pi)
            self.node.reset_robot_pose(start_x, start_y, start_yaw)
            self.node.publish_cmd(0.0, 0.0)
            self.node.spin_until_fresh_scan()

            frame, dist, min_range = self._build_frame()
            if min_range > COLLISION_RANGE + self.SPAWN_SAFETY_MARGIN:
                break
        else:
            self.node.get_logger().warn(
                f"reset(): no clear spawn found in {self.MAX_RESET_ATTEMPTS} "
                "attempts, proceeding with last (possibly unsafe) pose."
            )

        self._frame_buffer = np.concatenate([frame, frame])  # duplicate on reset
        self._prev_dist = dist
        self._prev_w = 0.0
        self._step_count = 0
        self._episode_t0 = time.time()
        return self._frame_buffer.astype(np.float32), {}

    def step(self, action):
        v, w = float(action[0]), float(action[1])
        min_v = -0.5 * MAX_LIN_VEL if self.allow_reverse else 0.0
        v = float(np.clip(v, min_v, MAX_LIN_VEL))
        w = float(np.clip(w, -MAX_ANG_VEL, MAX_ANG_VEL))
        self.node.publish_cmd(v, w)

        self._drive_dynamic_obstacle_if_enabled()
        self.node.spin_until_fresh_scan()  # blocks until next real 5Hz scan

        frame, dist, min_range = self._build_frame()
        self._frame_buffer = np.concatenate([self._frame_buffer[len(frame):], frame])
        self._step_count += 1

        reward = REWARD_WEIGHTS["step_penalty"]
        reward += REWARD_WEIGHTS["progress"] * (self._prev_dist - dist)
        reward += REWARD_WEIGHTS["smoothness"] * abs(w - self._prev_w)
        if v < 0:
            reward += REWARD_WEIGHTS["reverse_penalty"]
        self._prev_dist = dist
        self._prev_w = w

        terminated = False
        truncated = False
        info = {}

        if dist < GOAL_TOLERANCE:
            reward += REWARD_WEIGHTS["goal_reached"]
            terminated = True
            info["outcome"] = "goal_reached"
        elif min_range < COLLISION_RANGE:
            reward += REWARD_WEIGHTS["collision"]
            terminated = True
            info["outcome"] = "collision"
        elif abs(self._get_pose()[0]) > WORLD_BOUNDS or abs(self._get_pose()[1]) > WORLD_BOUNDS:
            reward += REWARD_WEIGHTS["collision"]
            terminated = True
            info["outcome"] = "out_of_bounds"
        elif self._step_count >= MAX_EPISODE_STEPS:
            truncated = True
            info["outcome"] = "timeout"

        if terminated or truncated:
            self.node.publish_cmd(0.0, 0.0)

        return self._frame_buffer.astype(np.float32), reward, terminated, truncated, info

    def close(self):
        self.node.publish_cmd(0.0, 0.0)
        self.node.destroy_node()
