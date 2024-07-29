import os
import json
import rclpy
import struct
import numpy as np

from copy import deepcopy
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from std_msgs.msg import String
from std_srvs.srv import Trigger
from realsense2_camera_msgs.msg import Extrinsics
from rclpy.node import Node
from statemachine import StateMachine, State
from cv_bridge import CvBridge
from intrinsics.utils import is_sharp

bridge = CvBridge()
class ScannerNode (Node, StateMachine):

    UNITIALIZED = State(name="UNITIALIZED", initial = True)
    INITIALIZED = State(name="INITIALIZED")
    RECORDING   = State(name="RECORDING")
    PROCESSING  = State(name="PROCESSING")
    FULFILLED   = State(name="FULFILLED", final= True)

    initialize  = UNITIALIZED.to(INITIALIZED, cond = [
        "intrinsics_are_loaded",
        "hand_eye_tf_are_loaded",
        "depth_camera_info_are_loaded"
    ]) | UNITIALIZED.to(UNITIALIZED)

    start_recording = INITIALIZED.to(RECORDING)
    stop_recording  = RECORDING.to(PROCESSING)
    processing_done = PROCESSING.to(FULFILLED)

    def __init__ (self):
        StateMachine.__init__(self)
        Node.__init__(self, 'scanner_node')

        self.create_subscription(String,     "/intrinsics/coeffs", self._on_intrinsics_coeffs, 10)
        self.create_subscription(Image,      "/camera/color/image_raw", self._on_color_image, 10)
        self.create_subscription(Image,      "/camera/depth/image_rect_raw", self._on_depth_image, 10)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info", self._on_depth_info, 10)
        self.create_subscription(Extrinsics, "/camera/extrinsics/depth_to_color", self._on_depth_to_color, 10)
        self.create_subscription(Extrinsics, "/hand_eye/extrinsics", self._on_hand_eye, 10)

        self.create_service(Trigger, "/scanning/recorder/start_record", self._on_start_record_cb)
        self.create_service(Trigger, "/scanning/recorder/stop_record", self._on_stop_record_cb)
        
        
        self.pc_pub = self.create_publisher(PointCloud2, "/scanning/point_cloud", 10)
        self.create_timer(0.1, self._publish_pc)
        
        # Variables
        self.color_frames = []
        self.depth_frames = [] 
        self.curr_depth_msg: None | Image = None
        self.depth_2_color    = None
        self.hand_eye_tf      = None
        self.color_intrinsics = None
        
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
    def before_stop_recording(self):    
        np.save(os.path.join("/", "calibration", "data", "scanner","color_frames.npy"), self.color_frames)
        np.save(os.path.join("/", "calibration", "data", "scanner","depth_frames.npy"), self.depth_frames)
        np.save(os.path.join("/", "calibration", "data", "scanner","depth_2_color.npy"), self.depth_2_color)
        
        
    def after_stop_recording(self):
        
        self.color_frames   = []
        self.depth_frames   = []
        self.curr_depth_msg = None
        self.depth_2_color  = None
        
    def on_initialized(self):
        self.get_logger().info("Scanner Node initialized!")
                
    def _on_stop_record_cb(self, request, response):
        self.stop_recording()
        return response
    
    def _on_start_record_cb(self, request, response):
        self.start_recording()
        return response
    
    def _on_color_image(self, msg: Image):
        if self.current_state == self.RECORDING:
            color_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if self.curr_depth_msg is not None:
                depth_frame = bridge.imgmsg_to_cv2(self.curr_depth_msg)
                self.depth_frames.append(depth_frame)
                self.color_frames.append(color_frame)
                
                
    def _on_depth_image(self, msg: Image):
        if self.current_state == self.RECORDING:
            self.curr_depth_msg = msg
    
    def _on_depth_to_color(self, msg: Extrinsics):
        if self.current_state == self.RECORDING and self.depth_2_color is None:
            self.depth_2_color = msg.rotation, msg.translation
 
    def _on_intrinsics_coeffs(self, msg: String):
        if self.current_state == self.UNITIALIZED:
            self.color_intrinsics = json.loads(msg.data)
            if self.hand_eye_tf_are_loaded() and self.depth_camera_info_are_loaded():
                self.initialize()
    
    def _on_hand_eye(self, msg: Extrinsics):
        if self.current_state == self.UNITIALIZED:
            self.hand_eye_tf = msg.rotation, msg.translation
            if self.intrinsics_are_loaded() and self.depth_camera_info_are_loaded():
                self.initialize()
            
    def _on_depth_info (self, msg: CameraInfo):
        if self.current_state == self.UNITIALIZED:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            if self.intrinsics_are_loaded() and self.hand_eye_tf_are_loaded():
                self.initialize()
    
    def intrinsics_are_loaded(self):
        return self.color_intrinsics is not None
    
    def hand_eye_tf_are_loaded(self):
        return self.hand_eye_tf is not None
    
    def depth_camera_info_are_loaded(self):
        return self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None
    
    def generate_points(self, depth_msg: Image):
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        height, width = depth_image.shape
        points = []
        
        for v in range(height):
            for u in range(width):
                depth = depth_image[v, u] / 1000.0  # converting depth to meters
                if depth == 0:
                    continue
                x = (u - self.cx) * depth / self.fx
                y = (v - self.cy) * depth / self.fy
                z = depth

                # TODO: HANDLE ROTATION (180 AROUND y AXIS)
                points.append((-x, y, -z))
                
        return points
    
    def _publish_pc (self):
        if self.current_state != self.RECORDING:
            return
        
        if self.curr_depth_msg is None:
            return
        
        msg = self.curr_depth_msg
        points = self.generate_points(msg)
        
        self.get_logger().info(f"Min value: {np.min(points)}, Max value: {np.max(points)}, Mean value: {np.mean(points)}")
        
        pc2_msg = PointCloud2()
        pc2_msg.header.stamp = msg.header.stamp
        pc2_msg.header = msg.header
        pc2_msg.header.frame_id = "depth_camera"
        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.is_dense = False
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 12
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        pc2_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        buffer = []
        for point in points:
            buffer.append(struct.pack('fff', *point))
        
        pc2_msg.data = b''.join(buffer)
        self.pc_pub.publish(pc2_msg)

def main (args=None):
    rclpy.init(args=args)
    node = ScannerNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()