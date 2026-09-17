from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="policy/checkpoints_static/final_model.zip",
        description="Path to the trained PPO checkpoint (.zip)",
    )
    # NEW: pairs with train.py's VecNormalize wrapper. Leave as "" for
    # checkpoints trained without VecNormalize.
    vecnorm_arg = DeclareLaunchArgument(
        "vecnormalize_path",
        default_value="policy/checkpoints_static/vecnormalize.pkl",
        description="Path to vecnormalize.pkl saved alongside the checkpoint ('' to disable)",
    )
    goal_x_arg = DeclareLaunchArgument("goal_x", default_value="1.5")
    goal_y_arg = DeclareLaunchArgument("goal_y", default_value="0.0")

    policy_node = Node(
        package="rl_navigation",
        executable="policy_node",
        name="rl_policy_node",
        output="screen",
        parameters=[{
            "checkpoint_path": LaunchConfiguration("checkpoint_path"),
            "vecnormalize_path": LaunchConfiguration("vecnormalize_path"),
            "goal_x": LaunchConfiguration("goal_x"),
            "goal_y": LaunchConfiguration("goal_y"),
        }],
    )

    return LaunchDescription([
        checkpoint_arg,
        vecnorm_arg,
        goal_x_arg,
        goal_y_arg,
        policy_node,
    ])
