"""Run one pick/place cycle against the already running, verified ROS2 stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    poses_file = os.path.join(
        get_package_share_directory("so101_grasp"), "config", "grasp_poses.yaml"
    )
    # Load robot_description for the current fake-hardware / RViz tests.
    # This launch starts no hardware, controllers, move_group, or RViz processes.
    moveit_config = (
        MoveItConfigsBuilder("so101_follower", package_name="so101_follower_moveit")
        .robot_description(mappings={"use_fake_hardware": "true"})
        .to_moveit_configs()
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "poses_file",
                default_value=poses_file,
                description="ROS parameter YAML with the joint-space targets",
            ),
            Node(
                package="so101_grasp",
                executable="pick_place_node",
                name="so101_pick_place",
                output="screen",
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.joint_limits,
                    LaunchConfiguration("poses_file"),
                    {"use_sim_time": False},
                ],
            ),
        ]
    )
