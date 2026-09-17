# TurtleBot3 Reinforcement Learning Navigation

A ROS 2 + Gazebo simulation project for training and evaluating a TurtleBot3 navigation policy using **Proximal Policy Optimization (PPO)** with Stable-Baselines3.

The objective is to train a learned navigation policy that receives LiDAR and robot-state observations and produces velocity commands for autonomous navigation toward a target while avoiding obstacles.

---

## 1. Project Overview

This project implements an RL-based navigation pipeline for TurtleBot3:

```text
Gazebo / ROS 2
      │
      ├── /scan
      ├── /odom
      │
      ▼
TurtleBot3 Gym Environment
      │
      ▼
Observation Processing
      │
      ▼
PPO Policy
      │
      ▼
Velocity Command
      │
      ▼
/cmd_vel
      │
      ▼
TurtleBot3
```

The trained policy is evaluated across multiple navigation environments, including static obstacles, unseen layouts, narrow passages, dead-end configurations, and dynamic obstacles.

---

## 2. Main Components

* **ROS 2 Humble**
* **Gazebo**
* **TurtleBot3**
* **Stable-Baselines3**
* **PPO (Proximal Policy Optimization)**
* **Python**
* **VecNormalize**
* LiDAR-based obstacle perception
* Odometry-based robot state
* Goal-relative navigation
* Deterministic evaluation with fixed seeds

The training implementation uses `MlpPolicy` with PPO and runs on CPU because the simulation/ROS2 environment is the main computational bottleneck rather than neural-network inference.

---

## 3. Repository Structure

```text
turtlebot_rl_navigation_v3/
│
├── README.md
│
├── training/
│   ├── train.py
│   ├── turtlebot_gym_env.py
│   └── config/
│       ├── ppo_config_stage1.yaml
│       ├── ppo_config_stage2.yaml
│       ├── ppo_config_stage3_reverse.yaml
│       └── ppo_config_stage4_dynamic.yaml
│
├── policy/
│   └── checkpoints_static/
│       ├── final_model.zip
│       └── vecnormalize.pkl
│
├── ros2_ws/
│   └── src/
│       └── rl_navigation/
│           ├── ...
│           └── launch/
│
├── evaluation/
│   ├── evaluate.py
│   └── evaluate_all_scenarios.sh
│
├── results/
│   ├── training_curve.png
│   ├── success_rate.png
│   ├── *_summary.json
│   └── navigation_demo.mp4
│
└── ...
```

> File/folder names above should match the files committed in the repository. If a particular directory has a different name in the final repository, use the actual committed name.

---

# 4. Environment Setup

## Requirements

Recommended environment:

* Ubuntu 22.04
* ROS 2 Humble
* Gazebo
* Python 3
* Stable-Baselines3
* PyYAML
* NumPy
* ROS 2 Python packages

Source ROS 2 before running the project:

```bash
source /opt/ros/humble/setup.bash
```

If the ROS 2 workspace contains the navigation package:

```bash
cd ~/turtlebot_rl_navigation_v3/ros2_ws
source install/setup.bash
```

---

# 5. Training

Training is implemented using Stable-Baselines3 PPO.

The training script supports:

* PPO training
* configuration through YAML files
* checkpoint saving
* TensorBoard logging
* resuming from previous checkpoints
* VecNormalize observation/reward normalization
* curriculum stages

The script creates a `DummyVecEnv`, wraps it with `VecNormalize`, and saves the normalization statistics separately from the PPO checkpoint.

## Stage 1 — Empty Environment

```bash
source /opt/ros/humble/setup.bash

cd ~/turtlebot_rl_navigation_v3/training

python3 train.py \
    --config config/ppo_config_stage1.yaml
```

## Stage 2 — Static Obstacles

```bash
python3 train.py \
    --config config/ppo_config_stage2.yaml
```

## Stage 3 — Reverse Motion

Stage 3 is configured to fine-tune the Stage 2 policy with reverse motion enabled:

```bash
python3 train.py \
    --config config/ppo_config_stage3_reverse.yaml \
    --resume ../policy/checkpoints_static/final_model.zip
```

## Stage 4 — Dynamic Obstacles

Stage 4 is configured for fine-tuning with dynamic obstacles:

```bash
python3 train.py \
    --config config/ppo_config_stage4_dynamic.yaml \
    --resume <stage3_checkpoint>
```

The curriculum configuration is read directly from YAML, including `allow_reverse` and `dynamic_obstacle`.

---

# 6. Resuming Training

A previously trained PPO checkpoint can be loaded using:

```bash
python3 train.py \
    --config config/ppo_config_stage3_reverse.yaml \
    --resume ../policy/checkpoints_static/final_model.zip
```

If VecNormalize statistics from the previous stage are available, they can also be loaded:

```bash
python3 train.py \
    --config config/ppo_config_stage3_reverse.yaml \
    --resume ../policy/checkpoints_static/final_model.zip \
    --vecnorm-stats ../policy/checkpoints_static/vecnormalize.pkl
```

The training script saves:

```text
final_model.zip
vecnormalize.pkl
```

at the end of training.

**Important:** The same observation normalization statistics used during training must be used during inference/evaluation.

---

# 7. PPO Configuration

The PPO configuration is controlled through YAML files.

The main parameters include:

```yaml
learning_rate
n_steps
batch_size
gamma
gae_lambda
ent_coef
total_timesteps
seed
checkpoint_freq
```

The training script passes these parameters to Stable-Baselines3 PPO.

The exact values used for each experiment are stored in the corresponding configuration files.

---

# 8. ROS 2 Policy Execution

The trained checkpoint is intended to be executed through a ROS 2 node rather than performing training during navigation.

The policy node:

1. Receives LiDAR data from `/scan`.
2. Receives odometry from `/odom`.
3. Constructs the same observation representation used during training.
4. Loads the trained PPO checkpoint.
5. Produces velocity commands.
6. Publishes commands to `/cmd_vel`.

The node can therefore be used independently from the training process.

---

# 9. Running the ROS 2 Policy

First build the workspace:

```bash
cd ~/turtlebot_rl_navigation_v3/ros2_ws

source /opt/ros/humble/setup.bash

colcon build --symlink-install

source install/setup.bash
```

Launch the required Gazebo/TurtleBot3 environment according to the launch files in:

```text
ros2_ws/src/rl_navigation/launch/
```

Then launch the RL navigation node using the provided launch file.

Example:

```bash
ros2 launch rl_navigation <launch_file>.launch.py
```

The exact launch command is defined by the launch files committed in the repository.

---

# 10. Observation and Action

The policy uses navigation observations derived from:

* LiDAR measurements
* robot pose/odometry
* relative goal information

The action is a velocity command corresponding to TurtleBot3 motion.

During deployment, the policy generates commands which are published through:

```text
/cmd_vel
```

The policy is not retrained during inference.

---

# 11. Reward Function

The navigation reward combines:

* goal achievement
* collision penalty
* progress toward the goal
* time/step penalty
* angular velocity smoothness
* reverse-motion penalty

The implemented reward formulation is designed to encourage:

* reaching the goal
* reducing distance to the goal
* avoiding collisions
* efficient navigation
* smoother control
* forward navigation where appropriate

The report documents the individual reward terms, their weights, and the observed navigation behavior associated with the reward design.

---

# 12. Evaluation

The evaluation pipeline supports multiple navigation scenarios.

The evaluated scenarios include:

1. Open environment
2. Static obstacles
3. Unseen obstacle configuration
4. Narrow passage
5. Dead-end environment
6. Dynamic obstacle environment

Evaluation uses:

* randomized start positions
* randomized goal positions
* fixed random seed
* deterministic policy inference
* 100 episodes per evaluated scenario

The evaluation script records navigation metrics and failure information.

Run evaluation using:

```bash
cd ~/turtlebot_rl_navigation_v3

python3 evaluation/evaluate.py
```

For all configured scenarios:

```bash
bash evaluation/evaluate_all_scenarios.sh
```

---

# 13. Evaluation Results

The current evaluation results reported for the trained policy are:

| Scenario           | Success Rate | Collision Rate |
| ------------------ | -----------: | -------------: |
| Open               |         100% |             0% |
| Static Obstacles   |          88% |             7% |
| Unseen Environment |          75% |             6% |
| Narrow Passage     |          89% |             1% |
| Dead End           |          81% |             2% |
| Dynamic Obstacles  |          60% |            36% |

The detailed metrics and episode summaries are available under:

```text
results/
```

including the generated summary JSON files and plots.

---

# 14. Failure Analysis

Several characteristic failure modes were identified during evaluation.

### Dead-End Behavior

The policy can enter a dead-end configuration and fail to recover effectively.

This is associated with the limitations of the training distribution and the learned local navigation behavior.

### Narrow Passage

In some narrow configurations, the robot can become overly conservative or freeze around the centerline.

### Dynamic Obstacles

Dynamic obstacles produced substantially more failures than static environments.

The policy was primarily trained around static obstacle interactions, so unseen moving-obstacle behavior remains a limitation.

These observations are documented in the accompanying technical report.

---

# 15. Training Monitoring

TensorBoard can be used to inspect the PPO training process.

For example:

```bash
tensorboard \
    --logdir <log_directory> \
    --host 0.0.0.0 \
    --port 6006
```

The training script also prints a convergence check after training.

Important signals include:

```text
train/explained_variance
train/std
```

The script specifically recommends checking whether explained variance improves toward 1.0 and whether action standard deviation narrows meaningfully.

---

# 16. Results

The `results/` directory contains the main evaluation artifacts:

```text
results/
├── training_curve.png
├── success_rate.png
├── *_summary.json
└── navigation_demo.mp4
```

The demo video provides a visual demonstration of the trained policy navigating the simulated environment.

---

# 17. Reproducibility

To reproduce the experiments:

1. Install ROS 2 Humble and the required Python dependencies.
2. Clone this repository.
3. Build the ROS 2 workspace.
4. Launch the corresponding Gazebo environment.
5. Use the provided YAML configuration.
6. Train or load the provided checkpoint.
7. Use the provided ROS 2 policy node for deployment.
8. Run the evaluation scripts with the configured seed.
9. Compare generated results with the files in `results/`.

Random seeds are explicitly configured in the training configuration.

For resumed training, both the PPO checkpoint and corresponding VecNormalize statistics should be preserved.

---

# 18. Current Scope and Limitations

The main completed working policy in this submission focuses on static/unseen navigation scenarios.

The project also contains configurations for:

* reverse-motion fine-tuning
* dynamic-obstacle fine-tuning

These stages are included in the training pipeline but should not be interpreted as completed experimental results unless corresponding checkpoints and evaluation results are present in the repository.

Known limitations include:

* reduced performance with dynamic obstacles
* dead-end recovery limitations
* narrow-passage freezing in some configurations
* dependence on the training environment/distribution
* limited systematic reward ablation

These limitations are discussed in the technical report.

---

# 19. Nav2 Comparison

A Nav2 comparison was considered as an additional engineering evaluation.

The project primarily investigates a learned RL navigation policy. Nav2 provides a conventional ROS 2 navigation stack and represents an important baseline for practical robotic navigation.

A full quantitative Nav2 comparison is not included in the current results.

---

# 20. Important Files

| File / Directory                       | Purpose                                |
| -------------------------------------- | -------------------------------------- |
| `training/train.py`                    | PPO training implementation            |
| `training/config/`                     | Training and curriculum configurations |
| `policy/`                              | Trained policy checkpoints             |
| `ros2_ws/src/rl_navigation/`           | ROS 2 policy package                   |
| `evaluation/evaluate.py`               | Evaluation script                      |
| `evaluation/evaluate_all_scenarios.sh` | Multi-scenario evaluation              |
| `results/`                             | Evaluation metrics, plots and demo     |
| `README.md`                            | Reproduction and project documentation |

---

# 21. Quick Start

For the shortest deployment workflow:

```bash
# ROS 2
source /opt/ros/humble/setup.bash

# Workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# Launch Gazebo/TurtleBot3
ros2 launch rl_navigation <launch_file>.launch.py
```

Then run the trained policy using the provided ROS 2 launch configuration.

For training:

```bash
cd training

python3 train.py \
    --config config/ppo_config_stage2.yaml
```

For evaluation:

```bash
cd ..

python3 evaluation/evaluate.py
```

---

# 22. Submission Contents

This repository contains the components required for the assessment submission:

* Source code
* Training implementation
* PPO configuration files
* Trained policy/checkpoint
* ROS 2 policy node
* Launch files
* Evaluation scripts
* Evaluation results
* Training/evaluation plots
* Navigation demonstration video
* Technical report

---

## Author

**Arun M.**

B.Tech Robotics & Automation
Karunya Institute of Technology and Sciences

Focus areas:

* Robotics
* Reinforcement Learning
* ROS 2
* Autonomous Navigation
* Robot Simulation
* Physical AI
