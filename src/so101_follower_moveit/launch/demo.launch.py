import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, NotSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # Declare the launch argument
    declare_use_fake_hardware = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="Bypass hardware interface and use fake controller",
    )

    # Setup MoveIt configs
    moveit_config = (
        MoveItConfigsBuilder("so101_follower", package_name="so101_follower_moveit")
        .robot_description(
            mappings={"use_fake_hardware": LaunchConfiguration("use_fake_hardware")}
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    # RViz
    rviz_config = os.path.join(
        moveit_config.package_path, "config", "moveit.rviz"
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
    )

    # Publish robot TF using the same description as MoveIt
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    # MoveGroup
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # Start the controller manager, but not if you are using fake hardware
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            os.path.join(moveit_config.package_path, "config", "ros2_controllers.yaml"),
        ],
        output="screen",
    )

    # Start the so101_follower_bridge if not using fake hardware
    so101_follower_bridge_node = Node(
        package="so101_follower_driver",
        executable="so101_follower_bridge",
        output="screen",
        condition=IfCondition(NotSubstitution(LaunchConfiguration("use_fake_hardware"))),
    )

    # Spawners
    controller_names = ["joint_state_broadcaster", "arm_controller", "gripper_controller"]
    spawner_nodes = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[controller, "-c", "/controller_manager"],
        ) for controller in controller_names
    ]

    return LaunchDescription(
        [
            declare_use_fake_hardware,
            rviz_node,
            robot_state_publisher_node,
            move_group_node,
            ros2_control_node,
            so101_follower_bridge_node,
        ]
        + spawner_nodes
    )
