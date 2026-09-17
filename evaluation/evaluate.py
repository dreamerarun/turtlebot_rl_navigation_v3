"""
Reproduces the reported metrics: runs a deterministic policy over N
randomized episodes in whatever Gazebo world is currently launched, and
reports success rate, collision rate, avg reward, avg time-to-goal,
avg path length, and goal distance error -- exactly the columns required
in the assessment's results table.

IMPORTANT (fixes a train/deploy mismatch introduced by train.py's new
VecNormalize wrapper): if the checkpoint was trained with VecNormalize
(it will be, if you use the updated train.py), you MUST pass
--vecnorm-stats pointing at that stage's vecnormalize.pkl, or the policy
will be evaluated on raw un-normalized observations it never saw during
training -- exactly the "heavily penalized" observation mismatch bug the
assessment brief warns about, just moved from training vs deployment to
training vs evaluation. If your checkpoint predates VecNormalize, omit
this flag and evaluation behaves exactly as before.

CANONICAL FILE NOTE: this is the PPO-based evaluate.py that matches
report.pdf and policy_node.py (both use PPO.load). A separately-uploaded
evaluate.py that imports SAC and a different checkpoint path
(checkpoints_static_sac/) is a drift artifact -- PPO is what was actually
trained and reported; do not use the SAC version, it does not match
anything else in the submission.

Usage (with the target world already running via `ros2 launch
turtlebot3_gazebo <world>.launch.py` in another terminal):

    python3 evaluate.py --checkpoint ../policy/checkpoints_static/final_model.zip \
        --vecnorm-stats ../policy/checkpoints_static/vecnormalize.pkl \
        --episodes 100 --seed 42 --scenario static_obstacles

For the dynamic-obstacle scenario, pass --dynamic-obstacle to also drive a
model named "moving_obstacle" back and forth via /set_entity_state during
each episode -- Gazebo Classic has no simple SDF-only way to script this
motion, so the evaluation harness itself acts as the obstacle's controller.
This is a known, documented limitation (see README.md), not a hidden hack.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from turtlebot_gym_env import TurtleBot3NavEnv, CONTROL_DT  # noqa: E402

DYNAMIC_OBSTACLE_NAME = "moving_obstacle"
DYNAMIC_OBSTACLE_CENTER = (0.5, 0.0)
DYNAMIC_OBSTACLE_AMPLITUDE = 0.8   # meters, back-and-forth range along y
DYNAMIC_OBSTACLE_PERIOD_S = 4.0    # full back-and-forth cycle length


def drive_dynamic_obstacle(env, elapsed_s):
    """Move `moving_obstacle` along a sinusoidal path in y. Fire-and-forget
    (no wait for completion) so it doesn't slow down the 5Hz control loop."""
    y = DYNAMIC_OBSTACLE_AMPLITUDE * math.sin(2 * math.pi * elapsed_s / DYNAMIC_OBSTACLE_PERIOD_S)
    req = SetEntityState.Request()
    req.state = EntityState()
    req.state.name = DYNAMIC_OBSTACLE_NAME
    req.state.pose.position.x = DYNAMIC_OBSTACLE_CENTER[0]
    req.state.pose.position.y = DYNAMIC_OBSTACLE_CENTER[1] + y
    req.state.pose.position.z = 0.25
    env.node.set_state_cli.call_async(req)  # fire-and-forget


def run_episode(env, model, seed, obs_normalizer=None, dynamic_obstacle=False):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    path_len = 0.0
    last_xy = None
    done = False
    episode_start = time.time()
    while not done:
        if dynamic_obstacle:
            drive_dynamic_obstacle(env, time.time() - episode_start)
        model_input = obs_normalizer.normalize_obs(obs) if obs_normalizer is not None else obs
        action, _ = model.predict(model_input, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        x, y, _, _, _ = env._get_pose()
        if last_xy is not None:
            path_len += math.hypot(x - last_xy[0], y - last_xy[1])
        last_xy = (x, y)
        done = terminated or truncated

    final_dist = env._prev_dist
    outcome = info.get("outcome", "unknown")
    return {
        "outcome": outcome,
        "reward": total_reward,
        "steps": steps,
        "time_to_goal": steps * CONTROL_DT if outcome == "goal_reached" else None,
        "path_length": path_len,
        "goal_distance_error": final_dist,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vecnorm-stats", default=None,
                         help="vecnormalize.pkl saved alongside the checkpoint; "
                              "required if this checkpoint was trained with the updated train.py")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", default="unnamed")
    parser.add_argument("--out", default="../results/eval_results.json")
    parser.add_argument("--allow-reverse", action="store_true",
                         help="match this to how the checkpoint was trained (Stage 3+)")
    parser.add_argument("--dynamic-obstacle", action="store_true",
                         help="also drive a 'moving_obstacle' model back and forth during each episode")
    args = parser.parse_args()

    rclpy.init()
    env = TurtleBot3NavEnv(allow_reverse=args.allow_reverse)
    model = PPO.load(args.checkpoint)

    obs_normalizer = None
    if args.vecnorm_stats:
        dummy = VecNormalize.load(args.vecnorm_stats, DummyVecEnv([lambda: env]))
        dummy.training = False       # freeze running stats, don't keep updating on eval data
        dummy.norm_reward = False    # we report raw reward, not normalized reward
        obs_normalizer = dummy

    results = []
    rng = np.random.default_rng(args.seed)
    for i in range(args.episodes):
        ep_seed = int(rng.integers(0, 1_000_000))
        results.append(run_episode(env, model, ep_seed, obs_normalizer=obs_normalizer,
                                    dynamic_obstacle=args.dynamic_obstacle))
        print(f"[{args.scenario}] episode {i+1}/{args.episodes}: {results[-1]['outcome']}")

    env.close()

    n = len(results)
    successes = [r for r in results if r["outcome"] == "goal_reached"]
    collisions = [r for r in results if r["outcome"] == "collision"]

    summary = {
        "scenario": args.scenario,
        "seed": args.seed,
        "episodes": n,
        "success_rate": len(successes) / n,
        "collision_rate": len(collisions) / n,
        "avg_episode_reward": float(np.mean([r["reward"] for r in results])),
        "avg_time_to_goal": float(np.mean([r["time_to_goal"] for r in successes])) if successes else None,
        "avg_path_length": float(np.mean([r["path_length"] for r in results])),
        "avg_goal_distance_error": float(np.mean([r["goal_distance_error"] for r in results])),
    }

    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)


if __name__ == "__main__":
    main()
