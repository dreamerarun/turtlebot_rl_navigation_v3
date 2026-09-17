"""
Train a PPO policy for TurtleBot3 navigation using Stable-Baselines3.

CHANGES vs. original train.py:
  - VecNormalize wraps the env (running mean/std for obs + reward). This is
    a likely second contributor to Stage 2's non-convergence: raw LiDAR/goal
    features and a reward with terms as large as +-100 vs -0.05 are exactly
    the kind of scale mismatch that stalls PPO's value function (matches
    the flat explained_variance ~0.5 reported in report.pdf Section 5).
  - Reads allow_reverse / dynamic_obstacle from the config so the 4-stage
    curriculum below can be run with one script:
        Stage 1: empty world                         (ppo_config_stage1.yaml)
        Stage 2: static obstacles, THE checkpoint     (ppo_config_stage2.yaml)
        Stage 3: fine-tune with reverse allowed       (ppo_config_stage3_reverse.yaml)
        Stage 4: fine-tune with dynamic obstacle      (ppo_config_stage4_dynamic.yaml)
  - Each stage after 1 should --resume from the previous stage's
    final_model.zip so later stages don't relearn basic navigation from
    scratch; only VecNormalize's running stats need to be reloaded
    separately (see load path below).

Usage (on the VM, headless Gazebo already launched against the matching
*_fast.world in a separate terminal/tmux pane -- see run_training.sh):

    source /opt/ros/humble/setup.bash
    python3 train.py --config config/ppo_config_stage2.yaml
    python3 train.py --config config/ppo_config_stage3_reverse.yaml \
        --resume ../policy/checkpoints_static/final_model.zip
"""
import argparse
import os
import yaml
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from turtlebot_gym_env import TurtleBot3NavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ppo_config_stage2.yaml")
    parser.add_argument("--resume", default=None,
                         help="path to a checkpoint .zip to resume/fine-tune from")
    parser.add_argument("--vecnorm-stats", default=None,
                         help="path to a vecnormalize.pkl to warm-start running stats "
                              "from a previous stage (recommended when --resume is set)")
    parser.add_argument("--run-name", default=None, help="label for TensorBoard logs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rclpy.init()

    def make_env():
        env = TurtleBot3NavEnv(
            randomize_obstacles=cfg.get("randomize_obstacles", False),
            curriculum_stage=cfg.get("curriculum_stage", 0),
            allow_reverse=cfg.get("allow_reverse", False),
            dynamic_obstacle=cfg.get("dynamic_obstacle", False),
        )
        return Monitor(env, filename=os.path.join(cfg["log_dir"], "monitor.csv"))

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    venv = DummyVecEnv([make_env])
    if args.vecnorm_stats and os.path.exists(args.vecnorm_stats):
        venv = VecNormalize.load(args.vecnorm_stats, venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # Force CPU: PPO + MlpPolicy on a network this small is faster on CPU than
    # GPU (SB3 warns about exactly this) -- the actual bottleneck here is
    # Gazebo/ROS2 sim throughput, not compute, so the GPU sits idle either
    # way. Forcing CPU removes the per-step host<->device transfer overhead.
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu")
        model.learning_rate = cfg["learning_rate"]  # allow a fresh LR schedule per stage
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            device="cpu",
            verbose=1,
            seed=cfg["seed"],
            learning_rate=cfg["learning_rate"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            ent_coef=cfg["ent_coef"],
            tensorboard_log=cfg["log_dir"],
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["checkpoint_freq"],
        save_path=cfg["checkpoint_dir"],
        name_prefix="ppo_turtlebot3_nav",
    )

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=checkpoint_cb,
        tb_log_name=args.run_name or "PPO",
        reset_num_timesteps=(args.resume is None),
    )
    model.save(os.path.join(cfg["checkpoint_dir"], "final_model"))
    venv.save(os.path.join(cfg["checkpoint_dir"], "vecnormalize.pkl"))
    venv.close()

    print("\nCONVERGENCE CHECK before you trust this checkpoint:")
    print("  tensorboard --logdir", cfg["log_dir"], "--host 0.0.0.0 --port 6006")
    print("  Look at train/explained_variance: it must trend toward 1.0, not stay ~0.5.")
    print("  Look at train/std (action std): it must narrow meaningfully from ~1.0.")
    print("  If neither happened, this run needs MORE total_timesteps, not a different algorithm.")


if __name__ == "__main__":
    main()
