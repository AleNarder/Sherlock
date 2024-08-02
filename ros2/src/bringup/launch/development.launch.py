import os
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
# frontend for xml type launch file!
from launch.launch_description_sources import FrontendLaunchDescriptionSource
import math

def generate_launch_description():

    ros_bridge = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource([os.path.join(
            get_package_share_directory('rosbridge_server'), 'launch'), '/rosbridge_websocket_launch.xml']),
    )

    intrinsics = Node(
        package='intrinsics',
        executable='intrinsics_node',
        name='intrinsics_node',
        output='both',
    
    ) 
    
    hand_eye = Node(
        package='hand_eye',
        executable='hand_eye_node',
        name='hand_eye_node',
        output='both',
        parameters=[],
    )
    
    scanner = Node(
        package='scanner',
        executable='scanner_node',
        name='scanner_node',
        output='both',
        parameters=[],
    )

    tcp_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', f"{math.pi / 2}", 'link6_1', 'tcp'],
        parameters=[],
    )
    

    return LaunchDescription([
        ros_bridge,
        tcp_tf_node,
        intrinsics,
        scanner,
        hand_eye
    ])
