import logging
import json
from dataclasses import dataclass

from rclpy.node import Node
from minio import Minio

import threading
import custom_interfaces.srv as custom_srv
import std_srvs.srv as std_srv
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import camera_intrinsic.utils as camera_utils
import os
import numpy as np

#################################
# STATE                         #
#################################
@dataclass
class CameraIntrinsicState ():
    is_processing: bool = False
    is_saving: bool = False
    
    def dict(self):
        return {
            "is_saving": self.is_saving,
            "is_processing": self.is_processing,
        }
        
        
#################################
# LOGGER                        #
#################################
camera_intrinsic_processor_logger = logging.getLogger("CameraIntrinsicProcessor")


#################################
# PROCESSOR                     #
#################################
class CameraIntrinsicProcessor ():
    def __init__(self, node: Node, mc: Minio, opts ={
            "camera_res" : (640, 480),
            "pattern_size" : (9, 6),
        }) -> None:
        # Deps injection
        self.mc_     = mc
        self.logger_ = camera_intrinsic_processor_logger
        
        # State
        self.state_ = CameraIntrinsicState()
        
        # Options
        self.opts_ = opts
        
        # ROS2 services
        self.process_srv = node.create_service(custom_srv.String, "/intrinsic_processor/process", self._on_process_cb)

        # ROS2 topics
        self.state_pub = node.create_publisher(std_msgs.String, "/intrinsic_processor/state", 10)
        self.state_pub_timer = node.create_timer(0.1, self._on_state_pub)
        
        self.chessboard_pub = node.create_publisher(sensor_msgs.Image, "/intrinsic_processor/chessboard", 10)
        
    
    def _on_state_pub(self) -> None:
        msg = std_msgs.String()
        msg.data = json.dumps(self.state_.dict())
        self.state_pub.publish(msg)
        
        
    def _on_process_cb(self, request: custom_srv.String.Request, response: custom_srv.String.Response) -> std_srv.Trigger.Response:
        threading.Thread(target=self.process, kwargs={"object_name": request.data}).start()
        response.success = True
        response.message = "Processing started"
        return response

    def save_intrinsics (self, mtx, dist, bucket_name = "intrinsic", object_name = ""):
        np.savez(object_name, mtx=mtx, dist=dist)
        self.mc_.fput_object(bucket_name, object_name, object_name)
        os.remove(object_name)

    def process (self, bucket_name = "intrinsic", object_name = ""):
        
        # Download frames
        local_file_path = f".{object_name}"
        self.mc_.fget_object(bucket_name, object_name, local_file_path)
        frames = np.load(local_file_path, allow_pickle=True)
        os.remove(local_file_path)
        
        # Get sharp frames, blurried frames are less likely to have accurate corners
        sharp_frames = camera_utils.get_sharp_frames(frames, 300)
        # Only process sufficiently different frames to reduce computation time and avoid overfitting
        representative_frames_idxs = camera_utils.get_n_representative_frames(sharp_frames, num_frames=30)
        representative_frames = [sharp_frames[i] for i in representative_frames_idxs]
        
        # Process intrinsic
        self.state_.is_processing = True
        mtx, dist ,rvecs, tvecs, objectpoints, imgpoints   = camera_utils.compute_intrinsic(representative_frames, self.opts_["pattern_size"], self.opts_["camera_res"])
        mean_error, errors = camera_utils.compute_reprojection_error(objectpoints, imgpoints, rvecs, tvecs,  mtx, dist)
        self.logger_.info(f"Mean error: {mean_error}")
        self.state_.is_processing = False
        
        # Save intrinsic
        self.state_.is_saving = True
        self.save_intrinsics(mtx, dist, object_name =f"{object_name.split('.npz')[0]}_intrinsics.npz")
        self.state_.is_saving = False