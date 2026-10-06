"""Run the TCP pick/place client with the existing fake-hardware MoveIt stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("so101_grasp"), "config", "tcp_grasp_poses.yaml"
    )
    moveit_config = (
        MoveItConfigsBuilder("so101_follower", package_name="so101_follower_moveit")
        .robot_description(mappings={"use_fake_hardware": "true"})
        .to_moveit_configs()
    )

    # Start only this client. The separately running so101_follower_moveit
    # demo.launch.py must use the same robot model, SRDF, kinematics and limits.
    # The default YAML disables execution and stops after PRE_GRASP qualification.
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=params_file,
                description="ROS parameter YAML for fake-hardware TCP pick/place",
            ),
            Node(
                package="so101_grasp",
                executable="tcp_pick_place_node",
                name="so101_tcp_pick_place",
                output="screen",
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.joint_limits,
                    LaunchConfiguration("params_file"),
                    {"use_sim_time": False},
                ],
            ),
        ]
    )
