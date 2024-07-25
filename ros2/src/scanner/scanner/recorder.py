
from datetime import datetime
import os
import traceback
from rclpy.node import Node
from minio import Minio
from dataclasses import dataclass
import camera_intrinsic.utils as camera_utils
import logging
import sensor_msgs.msg as sensor_msgs
import numpy as np
import std_srvs.srv as std_srv
import realsense2_camera_msgs.msg as realsense2_camera_msgs

@dataclass
class ScannerRecorderState ():
    is_recording: bool = False
    intrinsics: dict = None
    color_frames: list = None
    depth_frames: list = None
    depth_info: sensor_msgs.CameraInfo = None
    depth_to_color: realsense2_camera_msgs.Extrinsics = None

    def reset(self):
        self.color_frames = []
        self.depth_frames = []
        self.depth_info = None
        self.depth_to_color = None
        

    def dict(self):
        return {
            "is_recording": self.is_recording,
            "intrinsics": self.intrinsics,
        }

logger = logging.getLogger("ScannerRecorder")

class ScannerRecorder ():
    def __init__(self, node: Node, mc: Minio) -> None:
        
        self.node_   = node
        self._state  = ScannerRecorderState()
        self._logger = logger
        self._mc     = mc      
        
        self.hand_eye_tf_start_srv = self.node_.create_client(std_srv.Trigger, "hand_eye/tf/start_publish")
        self.hand_eye_tf_stop_srv  = self.node_.create_client(std_srv.Trigger, "hand_eye/tf/stop_publish")
        
        # ROS2 services
        self.record_srv = node.create_service(std_srv.Trigger, "/scanning/recorder/start_record", self._on_start_record_cb)
        self.record_srv = node.create_service(std_srv.Trigger, "/scanning/recorder/stop_record", self._on_stop_record_cb)
        
        self._load_color_intrinsics()

    def _load_color_intrinsics(self):
        self._mc.fget_object("intrinsic", "intrinsic_2024-07-07_14-56-53_intrinsics.npz", "intrinsics.npz")
        data = np.load("intrinsics.npz")
        self._state.intrinsics = {
            "mtx": data["mtx"],
            "dist": data["dist"]
        }
        self._logger.info("Loaded color camera intrinsics" + str(self._state.intrinsics))
    
    def subscribe (self):
        self.node_.create_subscription(sensor_msgs.Image, "/camera/color/image_raw", self._on_color_image, 10)
        self.node_.create_subscription(sensor_msgs.Image, "/camera/depth/image_rect_raw", self._on_depth_image, 10)
        self.node_.create_subscription(sensor_msgs.CameraInfo, "/camera/extrinsics/depth_to_color", self._on_depth_to_color, 10)

    def unsubscribe (self):
        self.node_.destroy_subscription("/camera/color/image_raw")
        self.node_.destroy_subscription("/camera/depth/image_rect_raw")
        self.node_.destroy_subscription("/camera/extrinsics/depth_to_color")

    def _on_color_image(self, msg: sensor_msgs.Image) -> None:
        if self._state.is_recording:
            self._state.color_frames.append((self._ros2img_to_npimg(msg), msg.header.stamp))

    def _on_depth_image(self, msg: sensor_msgs.Image) -> None:
        if self._state.is_recording:
            self._state.depth_frames.append((self._ros2img_to_npimg(msg), msg.header.stamp))
            
    def _on_depth_to_color(self, msg: realsense2_camera_msgs.Extrinsics) -> None:
        if self._state.is_recording and self._state.depth_to_color is None:
            h = np.eye(4)
            h[:3, :3] = np.array(msg._rotation).reshape(3, 3)
            h[:3, 3]  = np.array(msg._translation)
            self._state.depth_to_color = h

    def _ros2img_to_npimg(self, ros2_img: sensor_msgs.Image) -> None:
        np_img = np.frombuffer(ros2_img.data, dtype=np.uint8).reshape(ros2_img.height, ros2_img.width, -1)
        return np_img 

    def _get_aligned_frames (self, color_stamped_frames, depth_stamped_frames):
        aligned_color_frames = np.zeros((len(color_stamped_frames), 448, 504,3), dtype=np.uint8)
        aligned_depth_frames = np.zeros((len(depth_stamped_frames), 480, 640, 2), dtype=np.uint8)
        
        color_stamped_it = iter(color_stamped_frames + [None])
        depth_stamped_it = iter(depth_stamped_frames + [None])
        aligned_it = 0
        
        while True:
            
            color_frame, color_stamp = next(color_stamped_it) or (None, None)
            depth_frame, depth_stamp = next(depth_stamped_it) or (None, None)
            
            if color_frame is None or depth_frame is None:
                break
        
            color_frame = camera_utils.undistort_image(color_frame, self._state.intrinsics["mtx"], self._state.intrinsics["dist"])
        
            color_stamp_ns = color_stamp.sec * 1E9 + color_stamp.nanosec
            depth_stamp_ns = depth_stamp.sec * 1E9 + depth_stamp.nanosec 
        
            while color_stamp_ns > depth_stamp_ns and abs(color_stamp_ns - depth_stamp_ns) > 0.05 * 1E9:
                depth_frame, depth_stamp = next(depth_stamped_it) or (None, None)
                if depth_frame is None:
                    break
            
            aligned_color_frames[aligned_it] = color_frame
            aligned_depth_frames[aligned_it] = depth_frame
            aligned_it += 1
            
        return aligned_color_frames, aligned_depth_frames    
                
    
    def _upload_record(self) -> None:
        try:
            object_name =  "_".join(str(datetime.now()).split(" ")) + ".npz"
            # Get the absolute path of the current file
            current_file_path = os.path.abspath(__file__)
            current_file_dir = os.path.dirname(current_file_path)
            record_path = os.path.join(current_file_dir, object_name) 

            self._logger.info(str(self._state.color_frames))
            aligned_color_frames, aligned_depth_frames  = self._get_aligned_frames(self._state.color_frames, self._state.depth_frames)
            
            np.savez(record_path, color_frames=aligned_color_frames, depth_frames=aligned_depth_frames)
            
            if not self._mc.bucket_exists("scans"):
                self._mc.make_bucket("scans")
                
            self._mc.fput_object("scans", object_name, record_path)
            self._logger.info("Upload record to MinIO")
            os.remove(record_path)
        except Exception as e:
            self._logger.error(f"Error on uploading record: {e}")
            self._logger.error(traceback.format_exc())
    
    def start_recording(self) -> None:
        self._state.reset()
        self.subscribe()
        # self.hand_eye_tf_start_srv.call()
        self._state.is_recording = True
        self._logger.info("Start recording")
        
    def _on_start_record_cb(self, request, response) -> std_srv.Trigger.Response:
        self.start_recording()
        return std_srv.Trigger.Response(success=True)
    
    def stop_recording(self) -> None:
        self.unsubscribe()
        # self.hand_eye_tf_stop_srv.call()
        self._state.is_recording = False
        self._upload_record()
        self._state.reset()
        self._logger.info("Stop recording")
    
    def _on_stop_record_cb(self, request, response) -> std_srv.Trigger.Response:
        self.stop_recording()
        return std_srv.Trigger.Response(success=True)