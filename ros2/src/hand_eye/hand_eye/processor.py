import logging
import json
from dataclasses import dataclass

from rclpy.node import Node
from minio import Minio
import camera_intrinsic.utils as camera_utils
import cv2
import threading
import custom_interfaces.srv as custom_srv
import std_srvs.srv as std_srv
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import os
import numpy as np

#################################
# STATE                         #
#################################
@dataclass
class HandEyeState ():
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
hand_eye_processor_logger = logging.getLogger("HandEyeProcessor")


#################################
# PROCESSOR                     #
#################################
class HandEyeProcessor ():
    def __init__(self, node: Node, mc: Minio, opts ={
            "camera_res" : (640, 480),
            "pattern_size" : (9, 6),
        }) -> None:
        # Deps injection
        self.mc_     = mc
        self.logger_ = hand_eye_processor_logger
        
        # State
        self.state_ = HandEyeState()
        
        # Options
        self.opts_ = opts
        
        # ROS2 services
        self.process_srv = node.create_service(custom_srv.String, "/hand_eye/process", self._on_process_cb)

        # ROS2 topics
        self.state_pub = node.create_publisher(std_msgs.String, "/hand_eye/state", 10)
        self.state_pub_timer = node.create_timer(0.1, self._on_state_pub)
        
    
    def _on_state_pub(self) -> None:
        msg = std_msgs.String()
        msg.data = json.dumps(self.state_.dict())
        self.state_pub.publish(msg)
        
        
    def _on_process_cb(self, request: custom_srv.String.Request, response: custom_srv.String.Response) -> std_srv.Trigger.Response:
        threading.Thread(target=self.process, kwargs={"object_name": request.data}).start()
        response.success = True
        response.message = "Processing started"
        return response

    def save_hand_eye (self, mtx, dist, bucket_name = "handeye", object_name = ""):
        np.savez(object_name, mtx=mtx, dist=dist)
        self.mc_.fput_object(bucket_name, object_name, object_name)
        os.remove(object_name)

    def process (self, bucket_name = "hand_eye", object_name = ""):
        
        # Download frames
        local_file_path = f".{object_name}"
        self.mc_.fget_object(bucket_name, object_name, local_file_path)
        
        data1 = np.load("/calibration/ros2/src/hand_eye/hand_eye/2024-07-21_16_38_38.100940.npy", allow_pickle=True)
        data2 = np.load("/calibration/ros2/src/hand_eye/hand_eye/2024-07-21_16_49_35.921496.npy", allow_pickle=True)
        cam_data = np.load("/calibration/ros2/src/hand_eye/hand_eye/intrinsic_2024-07-07_14-56-53_intrinsics (1).npz", allow_pickle=True)
        data  = np.vstack((data1, data2))
        frames, poses = data[:, 0], data[:, 1]
        
        sharp_frames_idxs = camera_utils.get_sharp_frames(frames, 280)
        sharp_poses, sharp_frames  = poses[sharp_frames_idxs], frames[sharp_frames_idxs]

        # Only process sufficiently different frames to reduce computation time and avoid overfitting
        representative_frames_idxs = camera_utils.get_n_representative_frames(frames[sharp_frames_idxs], num_frames=60)

        representative_poses, representative_frames  = sharp_poses[representative_frames_idxs], sharp_frames[representative_frames_idxs]
        
        chessboard_pts = camera_utils.get_chessboard_pts((9, 6))

        R_gripper2base_r = []
        t_gripper2base_r = []
        R_target2cam_r   = []
        t_target2cam_r   = []

        for frame, pose in zip(representative_frames, representative_poses):
            ret, corners = camera_utils.find_chessboard_corners(frame)    
            if ret:
                ret2, rvec, tvec = cv2.solvePnP(chessboard_pts, corners, cam_data["mtx"], cam_data["dist"])
                if ret2:
                    R_gripper2base_r.append(pose[:3, :3])
                    t_gripper2base_r.append(pose[:3, 3].reshape((3, 1)))
                    
                    rot ,jacobian = cv2.Rodrigues(rvec)
                    R_target2cam_r.append(rot)
                    t_target2cam_r.append(tvec)

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_r, 
            t_gripper2base_r, 
            R_target2cam_r, 
            t_target2cam_r, 
            method = cv2.CALIB_HAND_EYE_TSAI
        )
        
        self.state_.is_processing = False
        
        # Save hand_eye
        self.state_.is_saving = True
        self.save_hand_eye(R_cam2gripper, t_cam2gripper, object_name =f"{object_name.split('.npz')[0]}_hand_eye.npz")
        self.state_.is_saving = False