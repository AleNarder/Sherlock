from launch_ros.actions import Node
from launch import LaunchDescription
import xacro
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share_dir = get_package_share_directory("er3600_description")

    xacro_file = os.path.join(share_dir, "urdf", "er3600.xacro")
    robot_description_config = xacro.process_file(xacro_file)
    robot_urdf = robot_description_config.toxml()

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": robot_urdf}],
        remappings=[("/robot_description", "/er3600/robot_description")],
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
        ]
    )
