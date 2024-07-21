from dataclasses import dataclass
from minio import Minio
from rclpy.node import Node
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
import transforms3d
import math
import numpy as np
import logging

#################################
# STATE                         #
#################################
@dataclass
class HandEyeTFState ():
    is_publishing: bool = False
    is_initialised: bool = False
    homogenus_mat: list[list[float]] = None
    quaternion: list[float] = None
    translation: list[float] = None
    # bag mode: if true, the node will publish tf data with the same timestamp as the bag data
    is_bag_mode: bool = True
    
    def dict(self):
        return {
            "is_publishing": self.is_publishing,
            "is_initialized": self.is_initialised,
            "homogenus_mat": self.homogenus_mat,
            "is_bag_mode": self.is_bag_mode,
            "quaternion": self.quaternion,
            "translation": self.translation,
        }
        
        
#################################
# LOGGER                        #
#################################
hand_eye_tf_logger = logging.getLogger("HandEyeProcessor")

class HandEyeTFPublisher ():
    def __init__(self, node: Node, mc: Minio, freq = 15) -> None:
        # State
        self.state_ = HandEyeTFState()
        
        self.br = TransformBroadcaster(node)
        if not self.state_.is_bag_mode:
            self.timer = node.create_timer(1 / freq, self.broadcast_tf)
        else:
            node.create_subscription(TFMessage, "/tf", self._on_bag_tf, 10)
            
        # Dep injection
        self._mc = mc
        self._node = node
        self.logger_ = hand_eye_tf_logger
    
    def _create_homogeneous_matrix(self, rotation, translation):
        """
        Create a 4x4 homogeneous transformation matrix from a 3x3 rotation matrix and a 3x1 translation vector.
        
        :param rotation: 3x3 rotation matrix
        :param translation: 3x1 translation vector
        :return: 4x4 homogeneous transformation matrix
        """
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation
        homogeneous_matrix[:3, 3]  = translation.reshape(3)
        return homogeneous_matrix
    
    def _on_bag_tf (self, msg: TFMessage):
        stamp = msg.transforms[0].header.stamp
        self.broadcast_tf(stamp)
    
    def _load_tf_data(self):
        self._mc.fget_object("handeye", "hand_eye.npz", "hand_eye.npz")
        
        hand_eye = np.load("hand_eye.npz")
        r_cam2gripper = hand_eye["r_cam2gripper"]
        t_cam2gripper = hand_eye["t_cam2gripper"]
        
        hm = self._create_homogeneous_matrix(r_cam2gripper, t_cam2gripper)
        
        # Hm for debugging purposes
        self.state_.homogenus_mat  = hm
        
        # ROS2 tf part 
        self.state_.quaternion     = transforms3d.quaternions.mat2quat(r_cam2gripper)
        self.state_.translation    = t_cam2gripper.reshape(3)
        self.state_.is_initialised = True
        
    def get_tf (self, timestamp = None, frame_id = "link6_1"):    
        if not self.state_.is_initialised:
            self._load_tf_data()
        
        t = TransformStamped()

        # Set the time and frame IDs
        t.header.stamp = timestamp or self._node.get_clock().now().to_msg()
        t.header.frame_id = "link6_1"
        t.child_frame_id = 'depth_camera'

        # Set the translation (example values)
        t.transform.translation.x = self.state_.translation[0]
        t.transform.translation.y = self.state_.translation[1]
        t.transform.translation.z = self.state_.translation[2]

        t.transform.rotation.x = self.state_.quaternion[0]
        t.transform.rotation.y = self.state_.quaternion[1]
        t.transform.rotation.z = self.state_.quaternion[2]
        t.transform.rotation.w = self.state_.quaternion[3]
    
        return t
    
    def broadcast_tf(self, timestamp = None):
        tf = self.get_tf(timestamp)
        # Broadcast the transform
        self.br.sendTransform(tf)