from rclpy.node import Node
import numpy as np
import std_srvs.srv as std_srv
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from dataclasses import dataclass
from minio import Minio
from datetime import datetime
import json
import logging
import os


#################################
# STATE                         #
#################################
@dataclass
class CameraExtrinsicRecorderState ():
    is_recording: bool = False
    is_saving: bool = False
    frames: list = None
    
    def dict(self):
        return {
            "is_saving": self.is_saving,
            "is_recording": self.is_recording
        }
        
        
#################################
# LOGGER                        #
#################################
camera_intrinsic_logger = logging.getLogger("CameraIntrinsicRecorder")


#################################
# RECORDER                      #
#################################
class CameraIntrinsicRecorder ():
    
    def __init__(self, node: Node, mc: Minio, frame_w = 1280, frame_h = 720) -> None:
        
        # Deps injection
        self.mc_ = mc
        self.logger_ = camera_intrinsic_logger
        
        # ROS2 services
        self.record_srv = node.create_service(std_srv.Trigger, "/intrinsic_recorder/start_record", self._on_start_record_cb)
        self.record_srv = node.create_service(std_srv.Trigger, "/intrinsic_recorder/stop_record", self._on_stop_record_cb)
        
        # ROS2 topics
        self.state_pub = node.create_publisher(std_msgs.String, "/intrisic_recorder/state", 10)
        self.state_pub_timer = node.create_timer(0.1, self._on_state_pub)
        self.img_sub = node.create_subscription(sensor_msgs.Image, "/camera/color/image_raw", self._on_raw_img, 10)
        
        # Options
        self.frame_w_ = frame_w
        self.frame_h_ = frame_h
    
        # State
        self._state  = CameraExtrinsicRecorderState()
    
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
                img = self._ros2img_to_npimg(msg)
                self._state.frames.append(img)                

        except Exception as e:
            self.logger_.error(f"Error on recording frame: {e}")          
    
    def start_recording(self) -> None:
        self._state.is_recording = True
        self._state.frames = []
            
    def _on_start_record_cb(self, request, response) -> std_srv.Trigger.Response:
        self.start_recording()
        return std_srv.Trigger.Response(success=True)
    
    def stop_recording(self) -> None:
        self._state.is_recording = False
        
        self._upload_record()
        
        self._state.frames = []
        self.logger_.info("Stop recording")
    
    def _on_stop_record_cb(self, request, response) -> std_srv.Trigger.Response:
        self.stop_recording()
        return std_srv.Trigger.Response(success=True)
        
    def _upload_record (self):
        try:
            object_name =  "_".join(str(datetime.now()).split(" "))
            # Get the absolute path of the current file
            current_file_path = os.path.abspath(__file__)
            current_file_dir = os.path.dirname(current_file_path)
            record_path = os.path.join(current_file_dir, object_name)
            np.save(record_path, np.array(self._state.frames))
            
            if not self.mc_.bucket_exists("intrinsic"):
                self.mc_.make_bucket("intrinsic")
                
            self.mc_.fput_object("intrinsic", object_name, record_path)
            self.logger_.info("Upload record to MinIO")
            os.remove(record_path)
        except Exception as e:
            self.logger_.error(f"Error on uploading record: {e}")