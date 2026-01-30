from launch import LaunchDescription
from ament_index_python.packages import get_package_share_path
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.substitutions import Command
import os


def generate_launch_description():
    package_path = get_package_share_path('cam_auto_adjusting')
    urdf_path = os.path.join(package_path, 'urdf', 'robot.urdf.xacro')
    robot_description = ParameterValue(
        Command(['xacro ', urdf_path]),
        value_type=str
    )
    ros2_control_config_path = os.path.join(package_path, 'config', 'robot_controllers.yaml')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_control_config_path],
        output="both",
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
    )

    position_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['forward_position_controller']
    )

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster']
    )

    return LaunchDescription([
        robot_state_publisher_node,
        ros2_control_node,
        position_controller,
        joint_state_broadcaster,
    ])
