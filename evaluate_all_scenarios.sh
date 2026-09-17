#!/usr/bin/env bash
# Run on the VM after Stage 2 (and optionally 3/4) checkpoints exist.
# One scenario at a time: each needs its own Gazebo world running, so this
# launches+kills gzserver per scenario rather than trying to run all 6 at once.
set -euo pipefail
cd "$HOME/turtlebot_rl_navigation_v3"
source /opt/ros/humble/setup.bash
source "$HOME/rl_venv/bin/activate"
export TURTLEBOT3_MODEL=burger

CKPT=policy/checkpoints_static/final_model.zip
VECNORM=policy/checkpoints_static/vecnormalize.pkl

run_scenario () {
  local name=$1 world=$2 extra_flags=${3:-}
  echo "=== $name ==="
  ROS_DOMAIN_ID=99 GAZEBO_MASTER_URI=http://localhost:11399 \
    gzserver --verbose "worlds/${world}" \
    -s libgazebo_ros_init.so -s libgazebo_ros_factory.so &
  GZ_PID=$!

  # Wait for the actual service instead of a fixed sleep -- a fixed sleep is
  # what let training silently hang with no robot spawned earlier.
  echo "waiting for /set_entity_state..."
  for i in $(seq 1 30); do
    ROS_DOMAIN_ID=99 ros2 service list 2>/dev/null | grep -qx "/set_entity_state" && break
    sleep 1
  done

  # THE FIX: spawn the robot. Without this, /scan and /odom never exist and
  # evaluate.py silently blocks forever inside spin_until_fresh_scan().
  ROS_DOMAIN_ID=99 ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py \
    x_pose:=0.0 y_pose:=0.0 &
  SPAWN_PID=$!

  echo "waiting for /scan to publish..."
  for i in $(seq 1 30); do
    ROS_DOMAIN_ID=99 timeout 1 ros2 topic hz /scan >/dev/null 2>&1 && break
    sleep 1
  done

  ROS_DOMAIN_ID=99 GAZEBO_MASTER_URI=http://localhost:11399 \
    python3 evaluation/evaluate.py \
      --checkpoint "$CKPT" --vecnorm-stats "$VECNORM" \
      --episodes 100 --seed 42 --scenario "$name" \
      --out "results/${name}_summary.json" $extra_flags

  kill "$SPAWN_PID" 2>/dev/null || true
  kill "$GZ_PID" 2>/dev/null || true
  sleep 3
}

# Use the REALTIME worlds for evaluation (not _fast) -- 100 eval episodes are
# a one-time cost, and running at real_time_factor=1 lets you watch/record
# them over VNC as they happen, which doubles as demo-video material.
run_scenario "static_obstacles" "static_obstacles.world"
run_scenario "unseen"           "unseen.world"
run_scenario "narrow_passage"   "narrow_passage.world"
run_scenario "dead_end"         "dead_end.world"
run_scenario "dynamic_obstacle" "dynamic_obstacle.world" "--dynamic-obstacle"

echo ""
echo "All scenario summaries written to results/*_summary.json"
python3 - <<'PY'
import json, glob
print(f"{'Scenario':<18}{'Success%':<10}{'Collision%':<12}{'AvgReward':<12}{'AvgT2G(s)':<10}{'PathLen(m)':<12}{'GoalErr(m)'}")
for f in sorted(glob.glob("results/*_summary.json")):
    s = json.load(open(f))["summary"]
    t2g = f"{s['avg_time_to_goal']:.2f}" if s["avg_time_to_goal"] else "n/a"
    print(f"{s['scenario']:<18}{s['success_rate']*100:<10.1f}{s['collision_rate']*100:<12.1f}"
          f"{s['avg_episode_reward']:<12.2f}{t2g:<10}{s['avg_path_length']:<12.2f}{s['avg_goal_distance_error']:.3f}")
PY
