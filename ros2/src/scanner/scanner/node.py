import os
import json
import tf2_ros
import rclpy
import struct
import numpy as np
import threading
import transforms3d as t3d
import open3d as o3d
import copy

from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from std_msgs.msg import String
from std_srvs.srv import Trigger
from realsense2_camera_msgs.msg import Extrinsics
from rclpy.node import Node
from statemachine import StateMachine, State
from cv_bridge import CvBridge
from hand_eye.utils import homogenous_from_rt

bridge = CvBridge()

class ScannerNode (Node, StateMachine):

    UNITIALIZED = State(name="UNITIALIZED", initial = True)
    INITIALIZED = State(name="INITIALIZED")
    RECORDING   = State(name="RECORDING")
    PROCESSING  = State(name="PROCESSING")
    VALIDATED   = State(name="VALIDATED", final= True)

    initialize  = UNITIALIZED.to(INITIALIZED, cond = [
        "intrinsics_are_loaded",
        "hand_eye_tf_are_loaded",
        "depth_camera_info_are_loaded"
    ]) | UNITIALIZED.to(UNITIALIZED)

    start_recording = INITIALIZED.to(RECORDING)
    stop_recording  = RECORDING.to(PROCESSING)
    processing_done = PROCESSING.to(VALIDATED)

    def __init__ (self):
        Node.__init__(self, 'scanner_node')
        StateMachine.__init__(self)

        # Subs
        self.create_subscription(String,     "/intrinsics/coeffs",                self._on_intrinsics_coeffs_cb, 10)
        self.create_subscription(Image,      "/camera/color/image_raw",           self._on_color_image_cb, 10)
        self.create_subscription(Image,      "/camera/depth/image_rect_raw",      self._on_depth_image_cb, 10)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info",         self._on_depth_info_cb, 10)
        self.create_subscription(Extrinsics, "/camera/extrinsics/depth_to_color", self._on_depth_to_color_cb, 10)
        self.create_subscription(Extrinsics, "/hand_eye/extrinsics",              self._on_hand_eye_cb, 10)

        # Services
        self.create_service(Trigger, "/scanning/recorder/start_record", self._on_start_record_cb)
        self.create_service(Trigger, "/scanning/recorder/stop_record", self._on_stop_record_cb)
        
        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, "/scanning/point_cloud", 10)
        self.create_timer(0.1, self._publish_pc)
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        
        # Instance variables
        self.color_frames = []
        self.depth_frames = [] 
        
        self.curr_depth_msg: None | Image = None
        self.curr_color_msg: None | Image = None
        
        self.depth_2_color_h  = None
        self.depth_k          = None
        
        self.hand_eye_h       = None
        self.color_k          = None
        
        self.pc_queue         = []
        self.stitched_pc      = None
        
            
    #################################
    #  PREDICATES
    #################################
        
    def intrinsics_are_loaded(self):
        return self.color_k is not None
    
    def hand_eye_tf_are_loaded(self):
        return self.hand_eye_h is not None
    
    def depth_camera_info_are_loaded(self):
        return self.depth_k is not None
        
    def depth_2_color_is_loaded(self):
        return self.depth_2_color_h is not None
    
    #################################
    #  STATE MACHINE HOOKS
    #################################
    
    def before_start_recording(self):
        self.color_frames = []
        self.depth_frames = []
        self.pc_queue     = []
        self.stitched_pc  = o3d.geometry.PointCloud()

        # threading.Thread(target=self.consume_pc_queue).start()
    
    def before_stop_recording(self):    
        np.save(os.path.join("/", "calibration", "data", "scanner","color_frames.npy"), self.color_frames)
        np.save(os.path.join("/", "calibration", "data", "scanner","depth_frames.npy"), self.depth_frames)
        np.save(os.path.join("/", "calibration", "data", "scanner","depth_2_color_h.npy"), self.depth_2_color_h)

    def after_stop_recording(self):
        self.color_frames   = []
        self.depth_frames   = []
        self.curr_depth_msg = None
        threading.Thread(target=self._process_pc_queue).start()
        
    def after_processing_done(self):
        self.get_logger().info(f"Processing done!")
        o3d.io.write_point_cloud(os.path.join("/", "calibration", "data", "scanner", "stitched_pc.ply"), self.stitched_pc)
        self.get_logger().info(f"Point cloud saved!")
            
    def on_enter_state(self, event, state):
        self.get_logger().info(f"Entering '{state.id}' state from '{event}' event.")
    
    #################################
    #  ROS2 CALLBACKS
    #################################
    def _on_stop_record_cb(self, request, response):
        self.stop_recording()
        return response
    
    def _on_start_record_cb(self, request, response):
        self.start_recording()
        return response
    
    def _on_color_image_cb(self, msg: Image):
        if self.current_state == self.RECORDING:
            color_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.curr_color_msg = msg
            
            if self.curr_depth_msg is not None and  self.curr_color_msg is not None:
                depth_frame = bridge.imgmsg_to_cv2(self.curr_depth_msg)
                self.depth_frames.append(depth_frame)
                self.color_frames.append(color_frame)
                
    def _on_depth_image_cb(self, msg: Image):
        if self.current_state == self.RECORDING:
            self.curr_depth_msg = msg
    
    def _on_depth_to_color_cb(self, msg: Extrinsics):
        if self.current_state == self.UNITIALIZED:
            self.depth_2_color_h = homogenous_from_rt(np.array(msg.rotation).reshape(3,3), msg.translation)
            if self.hand_eye_tf_are_loaded() and self.intrinsics_are_loaded() and self.depth_camera_info_are_loaded():
                self.initialize()
 
    def _on_intrinsics_coeffs_cb(self, msg: String):
        if self.current_state == self.UNITIALIZED:
            self.color_k = np.array(json.loads(msg.data)["mtx"]).flatten()
            if self.hand_eye_tf_are_loaded() and self.depth_camera_info_are_loaded() and self.depth_2_color_is_loaded():
                self.initialize()
    
    def _on_hand_eye_cb(self, msg: Extrinsics):
        if self.current_state == self.UNITIALIZED:
            self.hand_eye_h = homogenous_from_rt(np.array(msg.rotation).reshape(3,3), msg.translation)
            if self.depth_camera_info_are_loaded() and self.intrinsics_are_loaded() and self.depth_2_color_is_loaded():
                self.initialize()
            
    def _on_depth_info_cb (self, msg: CameraInfo):
        if self.current_state == self.UNITIALIZED:
            self.depth_k = np.array(msg.k)
            if self.intrinsics_are_loaded() and self.hand_eye_tf_are_loaded() and self.depth_2_color_is_loaded():
                self.initialize()
    
    def generate_points(self, depth_msg: Image, color_msg: Image, base2eye_h: np.array):
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        color_image = bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')
        
        # Convert depth values from millimeters to meters
        depth_image = depth_image / 1000.0

        # Create a grid of indices corresponding to pixel coordinates
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Calculate x, y, z coordinates
        cx = self.depth_k[2]
        cy = self.depth_k[5]
        fx = self.depth_k[0]
        fy = self.depth_k[4]

        x = (u - cx) * depth_image / fx
        y = (v - cy) * depth_image / fy
        z = depth_image
        
        
        # Combine x, y, z into an (N, 3) array, filtering out points with zero depth, negative depth, or depth greater than 1m
        valid = (z > 0.1) & (z < 1.0)
        points = np.stack((x[valid], y[valid], z[valid], ), axis=-1)
        
        
        # Transform points from depth camera frame to color camera frame
        h_inv = np.linalg.inv(self.depth_2_color_h)
        points = np.matmul(points, h_inv[:3, :3].T) - h_inv[:3, 3]
        
        cx_color = self.color_k[2]
        cy_color = self.color_k[5]
        fx_color = self.color_k[0]
        fy_color = self.color_k[4]
        
        
        # Project points to the color image
        u = np.round((points[:, 0] * fx_color / points[:, 2]) + cx_color).astype(int)
        v = np.round((points[:, 1] * fy_color / points[:, 2]) + cy_color).astype(int)
        valid = (u >= 0) & (u < color_image.shape[1]) & (v >= 0) & (v < color_image.shape[0])
        points = points[valid]

        # Transform points from color camera frame to base frame
        points = np.matmul(points, base2eye_h[:3, :3].T) + base2eye_h[:3, 3]
        
        # Add color information to the points
        rgb = color_image[v[valid], u[valid]]
        
        points = np.concatenate((points, rgb), axis=1)

        return points
    
    
    def _process_pc_queue (self):
        for points in self.pc_queue:
            pc = o3d.geometry.PointCloud()
            # Only consider points within a certain range
            valid = (points[:, 2] > 0.1) & (points[:, 2] < 1.0)
            points = points[valid]
            
            pc.points = o3d.utility.Vector3dVector(points[:, :3])
            pc.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.0)
            self.stitched_pc += pc
        
        self.stitched_pc = self.stitched_pc.voxel_down_sample(voxel_size=0.001)
        self.processing_done()
    
    def _publish_pc (self):
        if self.current_state != self.RECORDING:
            return
        
        if self.curr_depth_msg is None or self.curr_color_msg is None:
            return
        
        depth_msg = copy.deepcopy(self.curr_depth_msg)
        color_msg = copy.deepcopy(self.curr_color_msg)
        
        ret, tf, h = self._get_transform("ur10e_base_link", "rgb_camera", depth_msg.header.stamp)
        
        if not ret:
            self.get_logger().error("Transform lookup failed!")
            return
        else:
            self.get_logger().info("Transform lookup successful!")
        
        points = self.generate_points(depth_msg, color_msg, h)
        self.pc_queue.append(points)
        
        self.get_logger().info(f"Min value: {np.min(points)}, Max value: {np.max(points)}, Mean value: {np.mean(points)}")
        pc2_msg = PointCloud2()
        pc2_msg.header.stamp = depth_msg.header.stamp
        pc2_msg.header = depth_msg.header
        pc2_msg.header.frame_id = "ur10e_base_link"
        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.is_dense = False
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 16 # Number of bytes for each point (3*4 for xyz + 4 for rgb)
        pc2_msg.row_step = pc2_msg.point_step * len(points)
        pc2_msg.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32,  count=1),
        ]

        buffer = []
        for point in points:
            # x, y, z are float
            # rgb is uint32
            x, y, z, r, g, b = point
            rgb = struct.unpack('I', struct.pack('BBBB', int(b), int(g), int(r), 0))[0]
            packed_data = struct.pack('fffI', x, y, z, rgb)
            buffer.append(packed_data)
            
        pc2_msg.data = b''.join(buffer)
        self.pc_pub.publish(pc2_msg)
        
    def _get_transform(self, target_frame: str, source_frame: str, stamp = None):
        try:
            stamp = rclpy.time.Time() if stamp is None else stamp
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)

            translation = [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ]
            
            rotation = [
                tf.transform.rotation.w,
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z
            ]
            
            R = t3d.quaternions.quat2mat(rotation)
            t = np.array(translation)

            h = homogenous_from_rt(R, t)
            
            self.get_logger().info(f"Transform lookup successful!")
            
            return True, tf, h
        except Exception as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return False, None, None



def main (args=None):
    rclpy.init(args=args)
    node = ScannerNode()
    rclpy.spin(node)(node)

    rclpy.shutdown()
    
if __name__ == "__main__":
    main()