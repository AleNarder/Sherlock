import numpy as np
import tf2_ros
import rclpy
import transforms3d
import json
import os
import cv2

from realsense2_camera_msgs.msg import Extrinsics
from intrinsics.utils import is_sharp, get_chessboard_pts, compute_reprojection_error
from hand_eye.utils import compute_hand_eye, find_chessboard_corners
from statemachine import StateMachine, State
from rclpy.node import Node
from tf_transformations import quaternion_matrix
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from threading import Thread
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

bridge = CvBridge()

class HandEyeNode (StateMachine, Node):
    
    UNITIALIZED = State(name="UNITIALIZED", initial = True)
    INITIALIZED = State(name="INITIALIZED") 
    RECORDING   = State(name="RECORDING")
    PROCESSING  = State(name="PROCESSING")
    
    initialize      = UNITIALIZED.to(INITIALIZED, cond = "intrinsics_are_loaded") | UNITIALIZED.to(UNITIALIZED)
    start_recording = INITIALIZED.to(RECORDING) 
    stop_recording  = RECORDING.to(PROCESSING)
    processing_done = PROCESSING.to(INITIALIZED)
    
    def __init__ (self, base_frame_id, hand_frame_id, eye_frame_id):        
        StateMachine.__init__(self)
        Node.__init__(self, 'hand_eye_node')

        # Variables
        self.hand_frame_id_ = hand_frame_id
        self.base_frame_id  = base_frame_id
        self.eye_frame_id   = eye_frame_id
        self.frames = []
        self.poses  = []
        self.intrinsics = { "mtx": None, "dist": None }
        self.hand_eye_h = None

        # Subs
        self.create_subscription(Image,  "/camera/color/image_raw", self._on_image_cb, 10)
        self.create_subscription(String, "/intrinsics/coeffs", self._on_intrinsics_cb, 10)
        
        # Services
        self.create_service(Trigger, '/hand_eye/start_recording', self._start_recording_cb),
        self.create_service(Trigger, '/hand_eye/stop_recording',  self._stop_recording_cb)
        
        # Publishers
        self.state_pub_ = self.create_publisher(String, "/hand_eye/state", 10)
        self.ext_pub_   = self.create_publisher(Extrinsics, "/hand_eye/extrinsics", 10)
        self.create_timer(0.1, self._publish)
    
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = TransformBroadcaster(self)

    def on_initialize (self):
        self.get_logger().info("initialized!")
        self.frames = []
        self.poses  = []
        if os.path.exists(os.path.join("/", "calibration", "data", "hand_eye", "h.npy")):
            self.hand_eye_h = np.load(os.path.join("/", "calibration", "data", "hand_eye", "h.npy"))
            self.get_logger().info("hand eye tf loaded!")
        
    def after_initialize(self):
        self.create_subscription(TFMessage, "/tf", self._on_tf, 10)
            
    def intrinsics_are_loaded(self):
        if self.intrinsics["mtx"] is None or self.intrinsics["dist"] is None:
            return False
        else:
            self.get_logger().info("camera intrinsics are loaded!")
            return True
    
    def after_start_recording(self):
        self.get_logger().info("start recording...")
    
    def after_stop_recording(self):
        self.get_logger().info("stop recording...")
        Thread(target = self._process).start()

    def after_processing_done(self):
        self.get_logger().info("processing done, saving hand eye tf...")
        self._on_save()
        self.get_logger().info("hand eye tf saved!")

    def _start_recording_cb(self, request, response):
        try:
            self.start_recording()
        except Exception as e:
            self.get_logger().error(f"Error starting recording: {e}")
            response.success = False
            
        return response

    
    def _stop_recording_cb(self, request, response):
        try:
            self.stop_recording()
        except Exception as e:
            self.get_logger().error(f"Error stopping recording: {e}")
            response.success = False
        
        return response

    def _on_image_cb(self, msg: Image):
        if self.current_state == self.RECORDING:
            cv_frame  = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            frame_is_sharp = is_sharp(cv_frame, 150)
            
            if not frame_is_sharp:
                self.get_logger().info("Image is not sharp enough")
                return
            
            timestap = msg.header.stamp
            pose = self.get_homogeneous_matrix(timestap)
                
            if pose is None:
                self.get_logger().info("Unable to recover consistent pose for current image")
                return
            
            self.frames.append(cv_frame)          
            self.poses.append(pose)
            
            chessboard_pts = get_chessboard_pts((9, 6), square_size=0.023)
            ret, corners = find_chessboard_corners(cv_frame, (9,6))
            
            if not ret:
                self.get_logger().info("No chessboard found")
                return

            _, rvec, tvec = cv2.solvePnP(
                chessboard_pts, 
                corners, 
                self.intrinsics["mtx"],
                self.intrinsics["dist"],
            )
            
            err, _ = compute_reprojection_error(
                [chessboard_pts],
                [corners],
                [rvec],
                [tvec],
                self.intrinsics["mtx"],
                self.intrinsics["dist"]
            )
            
            self.get_logger().info("Chessboard proj err: " + str(err))
            
            rot, _ = cv2.Rodrigues(rvec)

            quaternion  = transforms3d.quaternions.mat2quat(rot)
            translation = tvec.reshape(3)
            
            t = TransformStamped()
            # Set the time and frame IDs
            t.header.stamp = msg.header.stamp
            t.header.frame_id = "rgb_camera"
            t.child_frame_id  = "chessboard"

            # Set the translation
            t.transform.translation.x = translation[0]
            t.transform.translation.y = translation[1]
            t.transform.translation.z = translation[2]

            # Set the rotation
            t.transform.rotation.w = quaternion[0]
            t.transform.rotation.x = quaternion[1]
            t.transform.rotation.y = quaternion[2]
            t.transform.rotation.z = quaternion[3]
            
            self.br.sendTransform(t)

    def _on_tf (self, msg: TFMessage):
        stamp = msg.transforms[0].header.stamp
        self._broadcast_tf(stamp)
        
    def _on_intrinsics_cb(self, msg: String):
        data = json.loads(msg.data)
        self.intrinsics["mtx"] = np.array(data["mtx"]) if data["mtx"] is not None else None
        self.intrinsics["dist"] = np.array(data["dist"]) if data["dist"] is not None else None
        
        if self.current_state == self.UNITIALIZED:
            self.initialize()

    def _publish(self):
        msg = String()
        data = str(self.current_state)
        msg.data = json.dumps(data)
        self.state_pub_.publish(msg)
        
        if self.hand_eye_h is not None:
            msg = Extrinsics()
            msg.rotation = self.hand_eye_h[:3, :3].flatten().tolist()
            msg.translation = self.hand_eye_h[:3, 3].flatten().tolist()
            self.ext_pub_.publish(msg)

    def get_homogeneous_matrix(self, stamp):
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame_id, self.hand_frame_id_, stamp)
            translation = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ]
            rotation = [
                trans.transform.rotation.w,
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z
            ]
            T = quaternion_matrix(rotation)
            T[0:3, 3] = translation
            return T
        except Exception as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return None  
        
    def _get_hand_eye_tf (self, timestamp):    
        
        t = TransformStamped()
        
        quaternion  = transforms3d.quaternions.mat2quat(self.hand_eye_h[:3, :3])
        translation = self.hand_eye_h[:3, 3].reshape(3)
        
        # Set the time and frame IDs
        t.header.stamp = timestamp
        t.header.frame_id = self.hand_frame_id_
        t.child_frame_id  = self.eye_frame_id

        # Set the translation
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        # Set the rotation
        t.transform.rotation.w = quaternion[0]
        t.transform.rotation.x = quaternion[1]
        t.transform.rotation.y = quaternion[2]
        t.transform.rotation.z = quaternion[3]
    
        return t
    
    def _broadcast_tf(self, timestamp = None):
        if self.hand_eye_h is not None:
            
            tf = self._get_hand_eye_tf(timestamp)
            if tf:
                # Broadcast the transform
                self.br.sendTransform(tf)
    
    def _on_save (self):
        np.save(os.path.join("/", "calibration", "data", "hand_eye", "h.npy"), self.hand_eye_h)
        np.save(os.path.join("/", "calibration", "data", "hand_eye", "poses.npy"), self.poses)
        np.save(os.path.join("/", "calibration", "data", "hand_eye", "frames.npy"), self.frames)

    def _process(self):
        self.get_logger().info("processing...")
        
        R_cam2gripper, t_cam2gripper, R_target2cam_r, t_target2cam_r = compute_hand_eye(
            self.frames,
            self.poses,
            self.intrinsics
        )
        
        # !IMPORTANT: The hand-eye calibration is computed in the camera frame
        #             but we need to express it in the robot base frame
        
        h = np.eye(4)
        h[:3, :3] = R_cam2gripper
        h[:3, 3]  = t_cam2gripper.reshape(3)
        self.hand_eye_h = h
        self.processing_done()
        
def main ():
    rclpy.init()
    node = HandEyeNode(
        base_frame_id = "ur10e_base_link",
        hand_frame_id = "tcp",
        eye_frame_id  = "rgb_camera"
    )
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()