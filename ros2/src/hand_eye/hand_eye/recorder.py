import logging
from minio import Minio
from rclpy.node import Node
import rclpy
import numpy as np
import std_srvs.srv as std_srv
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import json
from dataclasses import dataclass
import tf2_ros
from tf_transformations import quaternion_matrix
from datetime import datetime
import os

@dataclass
class HandEyeRecorderState ():
    is_recording: bool = False
    frames: list = None
    poses: list = None
    
    def dict(self):
        return {
            "is_recording": self.is_recording
        }


#################################
# LOGGER                        #
#################################
hand_eye_logger = logging.getLogger("CameraIntrinsicRecorder")

class HandEyeRecorder ():
    
    def __init__(self, node: Node, mc: Minio, fps: int = 15, frame_w = 640, frame_h = 480) -> None:
        
        # Deps injection
        self.mc_ = mc
        self.logger_ = hand_eye_logger
        
        # ROS2 services
        self.record_srv = node.create_service(std_srv.Trigger, "/hand_eye_recorder/start_record", self._on_start_record_cb)
        self.record_srv = node.create_service(std_srv.Trigger, "/hand_eye_recorder/stop_record", self._on_stop_record_cb)
        
        # ROS2 topics
        self.state_pub = node.create_publisher(std_msgs.String, "/hand_eye_recorder/state", 10)
        self.state_pub_timer = node.create_timer(0.1, self._on_state_pub)
        
        self.img_sub = node.create_subscription(sensor_msgs.Image, "/camera/color/image_raw", self._on_raw_img, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        
        # Options
        self.frame_w_ = frame_w
        self.frame_h_ = frame_h
        self.fps_ = fps
        
        # State
        self._state  = HandEyeRecorderState()
    
    def get_homogeneous_matrix(self, stamp, base_frame = "ur10e_base_link", gripper_frame = "link6_1"):
        try:
            trans = self.tf_buffer.lookup_transform(base_frame, gripper_frame, stamp)
            translation = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ]
            rotation = [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ]
            T = quaternion_matrix(rotation)
            T[0:3, 3] = translation
            return T
        except tf2_ros.LookupException as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return None
    
    def _on_state_pub(self) -> None:
        msg = std_msgs.String()
        msg.data = json.dumps(self._state.dict())
        self.state_pub.publish(msg)
    
    def _ros2img_to_npimg(self, ros2_img: sensor_msgs.Image) -> None:
        np_img = np.frombuffer(ros2_img.data, dtype=np.uint8).reshape(ros2_img.height, ros2_img.width, -1)
        return np_img
    
    def _on_raw_img(self, msg: sensor_msgs.Image) -> None:
        try:
            if self._state.is_recording:
                timestap = msg.header.stamp
                img = self._ros2img_to_npimg(msg)
                pose = self.get_homogeneous_matrix(timestap)
                self._state.poses.append(pose)
                self._state.frames.append(img)                

        except Exception as e:
            self.logger_.error(f"Error on recording frame: {e}") 
    
    def _upload_record (self):
        try:
            object_name =  "_".join(str(datetime.now()).split(" ")) + ".npy"
            # Get the absolute path of the current file
            current_file_path = os.path.abspath(__file__)
            current_file_dir = os.path.dirname(current_file_path)
            record_path = os.path.join(current_file_dir, object_name) 
            
            data = [(frame, pose) for frame, pose in zip(self._state.frames, self._state.poses)]
            
            np.save(record_path, np.array(data, dtype=object))
            
            if not self.mc_.bucket_exists("handeye"):
                self.mc_.make_bucket("handeye")
                
            self.mc_.fput_object("handeye", object_name, record_path)
            self.logger_.info("Upload record to MinIO")
            os.remove(record_path)
        except Exception as e:
            self.logger_.error(f"Error on uploading record: {e}")
    
            
    def start_recording(self) -> None:
        self._state.is_recording = True
        self._state.frames = []
        self._state.poses = []
            
    def _on_start_record_cb(self, request, response) -> std_srv.Trigger.Response:
        self.start_recording()
        return std_srv.Trigger.Response(success=True)
    
    def stop_recording(self) -> None:
        self._state.is_recording = False
        
        self._upload_record()
        
        self._state.frames = []
        self._state.poses = []
        self.logger_.info("Stop recording")
    
    def _on_stop_record_cb(self, request, response) -> std_srv.Trigger.Response:
        self.stop_recording()
        return std_srv.Trigger.Response(success=True)