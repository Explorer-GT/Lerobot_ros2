"""Run one XYZ/IK test with the existing fake-hardware MoveIt demo stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("so101_grasp"), "config", "xyz_ik_test.yaml"
    )
    moveit_config = (
        MoveItConfigsBuilder("so101_follower", package_name="so101_follower_moveit")
        .robot_description(mappings={"use_fake_hardware": "true"})
        .to_moveit_configs()
    )

    # The separately launched so101_follower_moveit demo must load the same
    # kinematics.yaml. This launch starts only the test client.
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=params_file,
                description="ROS parameter YAML for the single XYZ/IK test",
            ),
            Node(
                package="so101_grasp",
                executable="xyz_ik_test_node",
                name="so101_xyz_ik_test",
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
