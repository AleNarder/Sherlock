import os
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    FindExecutable,
)
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

# frontend for xml type launch file!
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    ros_bridge = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            [
                os.path.join(get_package_share_directory("rosbridge_server"), "launch"),
                "/rosbridge_websocket_launch.xml",
            ]
        ),
    )

    # Including the Python launch file
    realsense_viewer = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("realsense2_camera"), "launch"
                ),
                "/rs_launch.py",
            ]
        ),
        launch_arguments={"pointcloud.enable": "true"}.items(),
    )

    camera_intrinsic = Node(
        package="camera_intrinsic",
        executable="camera_intrinsic_node",
        name="camera_intrinsic_node",
        output="both",
    )

    return LaunchDescription(
        [
            ros_bridge,
            realsense_viewer,
            camera_intrinsic,
        ]
    )
