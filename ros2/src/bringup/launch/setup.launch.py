import os
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory

# frontend for xml type launch file!
from launch.launch_description_sources import FrontendLaunchDescriptionSource


def generate_launch_description():

    ros_bridge = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            [
                os.path.join(get_package_share_directory("rosbridge_server"), "launch"),
                "/rosbridge_websocket_launch.xml",
            ]
        ),
    )

    camera_intrinsic = Node(
        package="camera_intrinsic",
        executable="camera_intrinsic_node",
        name="camera_intrinsic_node",
        output="both",
    )

    hand_eye = Node(
        package="hand_eye",
        executable="hand_eye_node",
        name="hand_eye_node",
        output="both",
    )

    return LaunchDescription([ros_bridge, camera_intrinsic, hand_eye])
