#!/usr/bin/env bash
# Run THIS on the VM (groot-vm-b), inside the new project folder.
# Sets up a clean turtlebot_rl_navigation_v3/ tree, launches headless Gazebo
# against the unthrottled ("fast") world, and runs the 4-stage curriculum in
# tmux so it survives your SSH/VNC session dropping.
#
# One-time folder setup (run once):
#   mkdir -p ~/turtlebot_rl_navigation_v3/{training/config,evaluation,policy,worlds,results,ros2_ws/src}
#   cp -r ~/turtlebot_rl_navigation_v2/ros2_ws/src/rl_navigation ~/turtlebot_rl_navigation_v3/ros2_ws/src/
# (keeps v2 untouched as a fallback / for comparison in your report)
#
# Then scp the patched files from this response into:
#   turtlebot_gym_env.py, train.py          -> ~/turtlebot_rl_navigation_v3/training/
#   ppo_config_stage*.yaml                   -> ~/turtlebot_rl_navigation_v3/training/config/
#   generate_worlds.py, _base_template*.world -> ~/turtlebot_rl_navigation_v3/worlds/
#   evaluate.py                              -> ~/turtlebot_rl_navigation_v3/evaluation/
#   policy_node.py                           -> ~/turtlebot_rl_navigation_v3/ros2_ws/src/rl_navigation/rl_navigation/
#   rl_policy.launch.py                      -> ~/turtlebot_rl_navigation_v3/ros2_ws/src/rl_navigation/launch/

set -euo pipefail
PROJECT_DIR="$HOME/turtlebot_rl_navigation_v3"
cd "$PROJECT_DIR"

source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
source "$HOME/rl_venv/bin/activate"

cd worlds
python3 generate_worlds.py --template fast       # for training (unthrottled)
python3 generate_worlds.py --template realtime   # for the VNC demo/video later
cd ..

# ---- Stage 1: empty world, unthrottled ------------------------------------
tmux new-session -d -s gz_stage1 \
  "source /opt/ros/humble/setup.bash && export TURTLEBOT3_MODEL=burger && \
   ROS_DOMAIN_ID=10 GAZEBO_MASTER_URI=http://localhost:11350 \
   gzserver --verbose worlds/rl_empty_world_fast.world \
   -s libgazebo_ros_init.so -s libgazebo_ros_factory.so"
sleep 8
tmux new-session -d -s train_stage1 \
  "source /opt/ros/humble/setup.bash && source $HOME/rl_venv/bin/activate && \
   export TURTLEBOT3_MODEL=burger ROS_DOMAIN_ID=10 GAZEBO_MASTER_URI=http://localhost:11350 && \
   cd $PROJECT_DIR/training && \
   python3 train.py --config config/ppo_config_stage1_empty.yaml --run-name stage1"
echo "Stage 1 launched in tmux (gz_stage1 / train_stage1)."
echo "Attach with: tmux attach -t train_stage1   (Ctrl-b d to detach)"
echo "Wait for it to finish (watch tmux, or: watch -n30 'ls policy/checkpoints_empty') before Stage 2."

# ---- Stage 2: static obstacles, unthrottled, THE checkpoint ---------------
# Run this block only after Stage 1's final_model.zip exists.
#
# tmux new-session -d -s gz_stage2 \
#   "source /opt/ros/humble/setup.bash && export TURTLEBOT3_MODEL=burger && \
#    ROS_DOMAIN_ID=20 GAZEBO_MASTER_URI=http://localhost:11360 \
#    gzserver --verbose worlds/static_obstacles_fast.world \
#    -s libgazebo_ros_init.so -s libgazebo_ros_factory.so"
# sleep 8
# tmux new-session -d -s train_stage2 \
#   "source /opt/ros/humble/setup.bash && source $HOME/rl_venv/bin/activate && \
#    export TURTLEBOT3_MODEL=burger ROS_DOMAIN_ID=20 GAZEBO_MASTER_URI=http://localhost:11360 && \
#    cd $PROJECT_DIR/training && \
#    python3 train.py --config config/ppo_config_stage2_static.yaml \
#      --resume ../policy/checkpoints_empty/final_model.zip \
#      --vecnorm-stats ../policy/checkpoints_empty/vecnormalize.pkl \
#      --run-name stage2"

# ---- Stage 3: reverse-velocity fine-tune (dead-end recovery) --------------
# tmux new-session -d -s gz_stage3 \
#   "source /opt/ros/humble/setup.bash && export TURTLEBOT3_MODEL=burger && \
#    ROS_DOMAIN_ID=30 GAZEBO_MASTER_URI=http://localhost:11370 \
#    gzserver --verbose worlds/dead_end_fast.world \
#    -s libgazebo_ros_init.so -s libgazebo_ros_factory.so"
# sleep 8
# tmux new-session -d -s train_stage3 \
#   "source /opt/ros/humble/setup.bash && source $HOME/rl_venv/bin/activate && \
#    export TURTLEBOT3_MODEL=burger ROS_DOMAIN_ID=30 GAZEBO_MASTER_URI=http://localhost:11370 && \
#    cd $PROJECT_DIR/training && \
#    python3 train.py --config config/ppo_config_stage3_reverse.yaml \
#      --resume ../policy/checkpoints_static/final_model.zip \
#      --vecnorm-stats ../policy/checkpoints_static/vecnormalize.pkl \
#      --run-name stage3_reverse"

# ---- Stage 4: dynamic-obstacle exposure ------------------------------------
# tmux new-session -d -s gz_stage4 \
#   "source /opt/ros/humble/setup.bash && export TURTLEBOT3_MODEL=burger && \
#    ROS_DOMAIN_ID=40 GAZEBO_MASTER_URI=http://localhost:11380 \
#    gzserver --verbose worlds/dynamic_obstacle_fast.world \
#    -s libgazebo_ros_init.so -s libgazebo_ros_factory.so"
# sleep 8
# tmux new-session -d -s train_stage4 \
#   "source /opt/ros/humble/setup.bash && source $HOME/rl_venv/bin/activate && \
#    export TURTLEBOT3_MODEL=burger ROS_DOMAIN_ID=40 GAZEBO_MASTER_URI=http://localhost:11380 && \
#    cd $PROJECT_DIR/training && \
#    python3 train.py --config config/ppo_config_stage4_dynamic.yaml \
#      --resume ../policy/checkpoints_static/final_model.zip \
#      --vecnorm-stats ../policy/checkpoints_static/vecnormalize.pkl \
#      --run-name stage4_dynamic"

echo ""
echo "Monitor GPU (should show LOW, spiky utilization -- that's expected, sim not GPU is the load):"
echo "  watch -n2 nvidia-smi"
echo "Monitor convergence live:"
echo "  tensorboard --logdir results/logs_static --host 0.0.0.0 --port 6006"
echo "  (then from your laptop: ssh -L 6006:localhost:6006 arunmurugesan0110@35.196.197.45)"
