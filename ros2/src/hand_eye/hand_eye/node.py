import traceback
import numpy as np
import tf2_ros
import rclpy
import transforms3d
import json
import os
import cv2
import math
import time
import shutil


from statemachine import StateMachine, State
from rclpy.executors import MultiThreadedExecutor
from realsense2_camera_msgs.msg import Extrinsics
from intrinsics.utils import is_sharp, get_chessboard_pts, compute_reprojection_error, find_chessboard_corners
from hand_eye.utils import compute_eye2hand_hs, homogenous_from_rt, CALIB_HAND_EYE_METHODS
from cv_bridge import CvBridge
from threading import Thread, Lock

from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from sklearn.covariance import EllipticEnvelope

from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

bridge = CvBridge()

class HandEyeNode (StateMachine, Node):
    
    UNITIALIZED = State(name="UNITIALIZED", initial = True)
    INITIALIZED = State(name="INITIALIZED") 
    RECORDING   = State(name="RECORDING")
    PROCESSING  = State(name="PROCESSING")
    
    PENDING_VALIDATION = State(name="PENDING_VALIDATION")
    REPROJECTING = State(name="REPROJECTING")
    VALIDATING  = State(name="VALIDATING")
    VALIDATED   = State(name="VALIDATED", final=True)
    
    initialize  = UNITIALIZED.to(VALIDATED, cond = [
            "is_validated"
        ]) | UNITIALIZED.to(PENDING_VALIDATION, cond=[
            "is_pending_validation",
        ]) | UNITIALIZED.to(INITIALIZED, cond = [
            "is_initialized"
        ])
    
    start_recording  = INITIALIZED.to(RECORDING)
    stop_recording   = RECORDING.to(PROCESSING)
    processing_done  = PROCESSING.to(PENDING_VALIDATION)
    
    start_reprojecting = PENDING_VALIDATION.to(REPROJECTING)
    stop_reprojecting  = REPROJECTING.to(VALIDATING)
    validating_done    = VALIDATING.to(VALIDATED)

    def __init__ (
        self, 
        base_frame_id: str, 
        hand_frame_id: str, 
        eye_frame_id: str, 
        target_frame_id: str,
        data_folder: str = "/calibration/data"
    ):        
        Node.__init__(self, 'hand_eye_node')

        # Variables
        self.hand_frame_id   = hand_frame_id
        self.base_frame_id   = base_frame_id
        self.eye_frame_id    = eye_frame_id
        self.target_frame_id = target_frame_id
        self.data_folder     = data_folder

        self.frames          = []
        self.drifts          = []
        self.hand2base_hs    = []
        
        self.target2eye_hs   = []
        self.target2base_hss = [[] for _ in CALIB_HAND_EYE_METHODS]

        self.intrinsics = { "mtx": None, "dist": None }
        if os.path.exists(os.path.join(data_folder, "intrinsics", "mtx.npy")):
            self.intrinsics["mtx"] = np.load(os.path.join(data_folder, "intrinsics", "mtx.npy"))
            self.get_logger().info("mtx loaded!")
        
        if os.path.exists(os.path.join(data_folder, "intrinsics", "dist.npy")):
            self.intrinsics["dist"] = np.load(os.path.join(data_folder, "intrinsics", "dist.npy"))
            self.get_logger().info("distortion loaded!")
        
        self.eye2hand_hs   = []
        if os.path.exists(os.path.join(self.data_folder, "hand_eye","processing", "eye2hand_hs.npy")):
            self.eye2hand_hs = np.load(os.path.join(data_folder, "hand_eye", "processing", "eye2hand_hs.npy"))
            self.get_logger().info("hand2base_hs loaded!")
        
        self.eye2hand_h    = None
        if os.path.exists(os.path.join(self.data_folder, "hand_eye", "eye2hand_h.npy")):
            self.eye2hand_h = np.load(os.path.join(self.data_folder, "hand_eye", "eye2hand_h.npy"))
            self.get_logger().info("hand2eye_h loaded!")
        
        StateMachine.__init__(self)

        # Subs
        self.processing_lock = Lock()
        self.create_subscription(Image,  "/camera/color/image_raw", self._on_image_cb, QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            # Optionally, you can set reliability and durability according to your requirements.
        ))
        
        # Services
        self.create_service(Trigger, '/hand_eye/initialize', self._initialize_cb),
        self.create_service(Trigger, '/hand_eye/start_recording', self._start_recording_cb),
        self.create_service(Trigger, '/hand_eye/stop_recording',  self._stop_recording_cb)
        self.create_service(Trigger, '/hand_eye/start_reprojecting', self._start_reprojecting_cb)
        self.create_service(Trigger, '/hand_eye/stop_reprojecting',  self._stop_reprojecting_cb)
        
        # Publishers
        self.state_pub_ = self.create_publisher(String, "/hand_eye/state", 10)
        self.ext_pub_   = self.create_publisher(Extrinsics, "/hand_eye/extrinsics", 10)
        self.create_timer(0.1, self._publish_cb)
    
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        
        self.dynamic_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster  = StaticTransformBroadcaster(self)
        
        self.initialize()

        
    #################################
    #  PREDICATES
    #################################
    
    def is_initialized(self):
        return self.intrinsics["mtx"] is not None and self.intrinsics["dist"] is not None
        
    def is_pending_validation(self):
        return len(self.eye2hand_hs) > 0 and self.is_initialized()
    
    def is_validated(self):
        return self.eye2hand_h is not None
        
    #################################
    #  STATE MACHINE HOOKS
    #################################
    
    def on_enter_PROCESSING (self):
        if os.path.exists(os.path.join(self.data_folder, "hand_eye", "processing")):
            shutil.rmtree(os.path.join(self.data_folder, "hand_eye", "processing"), ignore_errors=True)
            os.mkdir(os.path.join(self.data_folder, "hand_eye", "processing"))
        
        th = Thread(target = self._compute_hand_eye_hs).start()

    def before_start_recording(self):
        self.frames = []
        self.hand2base_hs = []
        self.target2eye_hs = []
        self.drifts = []
        os.makedirs(os.path.join(self.data_folder, "hand_eye", "processing"), exist_ok=True)
        self.get_logger().info("Recording started!")
        

    def after_stop_recording(self):
        pass
        # os.makedirs(os.path.join(self.data_folder, "hand_eye", "processing"), exist_ok=True)
        # # Save data for debugging and further processing
        # np.save(os.path.join(self.data_folder,  "hand_eye", "recording", "hand2base_hs.npy"), self.hand2base_hs)
        # np.save(os.path.join(self.data_folder,  "hand_eye", "recording", "target2eye_hs.npy"), self.target2eye_hs)
        # np.save(os.path.join(self.data_folder,  "hand_eye", "recording", "frames.npy"), self.frames)
        # np.save(os.path.join(self.data_folder,  "hand_eye", "recording", "drifts.npy"), self.drifts)
        # self.get_logger().info("Recording data saved!")
        
    def after_processing_done(self):
        os.makedirs(os.path.join(self.data_folder, "hand_eye", "processing"), exist_ok=True)
        # Save data for debugging and further processing
        np.save(os.path.join(self.data_folder, "hand_eye", "processing", "eye2hand_hs.npy"), self.eye2hand_hs)
        np.save(os.path.join(self.data_folder, "hand_eye", "processing", "eye2hand_h.npy"), self.eye2hand_h)
        self.get_logger().info("Processing data saved!")
    
    
    def before_start_reprojecting(self):
        self.frames = []
        self.target2base_hss = [[] for _ in CALIB_HAND_EYE_METHODS]
        self.get_logger().info("Reprojecting started!")
    
    def after_stop_reprojecting(self): 
        os.makedirs(os.path.join(self.data_folder, "hand_eye", "reprojecting"), exist_ok=True)
        # Save data for debugging and further processing
        np.save(os.path.join(self.data_folder, "hand_eye", "reprojecting", "frames.npy"), self.frames)
        np.save(os.path.join(self.data_folder, "hand_eye", "reprojecting", "target2base_hss.npy"), self.target2base_hss)
        self.get_logger().info("Reprojecting data saved!")
        
    def after_validating_done(self):
        # Save data for debugging and further processing
        np.save(os.path.join(self.data_folder, "hand_eye", "eye2hand_h.npy"), self.eye2hand_h)
        self.get_logger().info("Validating data saved!")

    
    def on_enter_PENDING_VALIDATION (self):
        for h, (method_name, _) in zip(self.eye2hand_hs, CALIB_HAND_EYE_METHODS):
            R = h[:3, :3]
            t = h[:3, 3]
            q = transforms3d.quaternions.mat2quat(R)

            static_transform_stamped = TransformStamped()
            static_transform_stamped.header.frame_id = self.hand_frame_id
            static_transform_stamped.child_frame_id = self.eye_frame_id + "_" + method_name
            
            static_transform_stamped.transform.translation.x = t[0]
            static_transform_stamped.transform.translation.y = t[1]
            static_transform_stamped.transform.translation.z = t[2]
            
            static_transform_stamped.transform.rotation.w = q[0]
            static_transform_stamped.transform.rotation.x = q[1]
            static_transform_stamped.transform.rotation.y = q[2]
            static_transform_stamped.transform.rotation.z = q[3]
        
            # Broadcast the static transform
            self.static_broadcaster.sendTransform(static_transform_stamped)

    def on_enter_VALIDATING (self):
        th = Thread(target = self._compute_hand_eye_h).start()

    def on_enter_VALIDATED (self):
        R = self.eye2hand_h[:3, :3]
        t = self.eye2hand_h[:3, 3]
        q = transforms3d.quaternions.mat2quat(R)

        static_transform_stamped = TransformStamped()
        static_transform_stamped.header.frame_id = self.hand_frame_id
        static_transform_stamped.child_frame_id = self.eye_frame_id
        
        static_transform_stamped.transform.translation.x = t[0]
        static_transform_stamped.transform.translation.y = t[1]
        static_transform_stamped.transform.translation.z = t[2]
        
        static_transform_stamped.transform.rotation.w = q[0]
        static_transform_stamped.transform.rotation.x = q[1]
        static_transform_stamped.transform.rotation.y = q[2]
        static_transform_stamped.transform.rotation.z = q[3]
        
        # Broadcast the static transform
        self.static_broadcaster.sendTransform(static_transform_stamped)

    def on_enter_state(self, event, state):
        self.get_logger().info(f"Entering '{state.id}' state from '{event}' event.")
    
    
    #################################
    #  ROS2 CALLBACKS
    #################################

    def _initialize_cb(self, request, response):
        try:
            self.reload()
            self.initialize()
        except Exception as e:
            self.get_logger().error(f"Error initializing: {e}")
            response.success = False
        return response

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
    
    def _start_reprojecting_cb(self, request, response):
        try:
            self.start_reprojecting()
        except Exception as e:
            self.get_logger().error(f"Error validating: {e}")
            response.success = False
        
        return response
    
    def _stop_reprojecting_cb(self, request, response):
        try:
            self.stop_reprojecting()
        except Exception as e:
            self.get_logger().error(f"Error stopping validating: {e}")
            response.success = False
        return response

    def _on_image_cb(self, msg: Image):
        try:
            if self.current_state == self.RECORDING:
                self.process_recording(msg)
            elif self.current_state == self.REPROJECTING:
                self.process_reprojecting(msg)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            traceback.print_exc()
        
    def _publish_cb(self):
        
        msg = String()
        data = str(self.current_state)
        msg.data = json.dumps(data)
        self.state_pub_.publish(msg)
        
        if self.eye2hand_h is not None:
            msg = Extrinsics()
            msg.rotation = self.eye2hand_h[:3, :3].flatten().tolist()
            msg.translation = self.eye2hand_h[:3, 3].flatten().tolist()
            self.ext_pub_.publish(msg)
            
    #####################################
    # PROCESSING
    #####################################
    def process_recording(self, msg: Image):            
        timestamp = msg.header.stamp
        cv_frame  = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        if not is_sharp(cv_frame, 290):
            self.get_logger().info("Image is not sharp enough")
            return
        
        ret, hand2base_tf, hand2base_h = self._get_hand2base(timestamp)
        if not ret:
            return
        
        ret, target2eye_tf, target2eye_h = self._get_target2eye(cv_frame, timestamp)
        if not ret:
            return
        
        img_stamp  = timestamp.sec + timestamp.nanosec * 1e-9
        pose_stamp = hand2base_tf.header.stamp.sec + hand2base_tf.header.stamp.nanosec * 1e-9
        drift =  img_stamp - pose_stamp
        
        if abs(drift) > 0.05:
            self.get_logger().info(f"Time drift too high: {drift}")
            return
        
        self.get_logger().info("FRAME OK!")
        self.drifts.append(drift)
        self.frames.append(cv_frame)
        self.hand2base_hs.append(hand2base_h)          
        self.target2eye_hs.append(target2eye_h)
    
    
    def process_reprojecting (self, msg: Image):
        timestamp = msg.header.stamp
        cv_frame  = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        if not is_sharp(cv_frame, 290):
            self.get_logger().info("Image is not sharp enough")
            return

        ret, target2eye_tf, target2eye_h = self._get_target2eye(cv_frame)
        
        if ret:
                
            for it, (method_name, method) in enumerate(CALIB_HAND_EYE_METHODS):
                eye_frame_id = self.eye_frame_id + "_" + method_name
                ret, eye2base_tf, eye2base_h = self._get_transform(self.base_frame_id, eye_frame_id, stamp=timestamp)
                if not ret:
                    continue
                
                target2base_h = np.dot(eye2base_h, target2eye_h)
                self.target2base_hss[it].append(target2base_h)
                self.frames.append(cv_frame)
                
                target2base_rot = target2base_h[:3, :3]
                target2base_t   = target2base_h[:3, 3]
                target2base_q   = transforms3d.quaternions.mat2quat(target2base_rot)

                
                # Add elapsed time to the timestamp
                tf = TransformStamped()
                tf.header.frame_id = self.base_frame_id
                tf.child_frame_id  = self.target_frame_id + "_" + method_name
                tf.transform.translation.x = target2base_t[0]
                tf.transform.translation.y = target2base_t[1]
                tf.transform.translation.z = target2base_t[2]
                
                tf.transform.rotation.w = target2base_q[0]
                tf.transform.rotation.x = target2base_q[1]
                tf.transform.rotation.y = target2base_q[2]
                tf.transform.rotation.z = target2base_q[3]
                
                self.static_broadcaster.sendTransform(tf)
                
    
    def _get_target2eye(self, cv_frame, stamp = None):
        
        # TODO: add parameters
        chessboard_pts = get_chessboard_pts((9, 6), square_size=0.023)
        ret, corners   = find_chessboard_corners(cv_frame, (9,6))
        
        if not ret:
            self.get_logger().info("No chessboard found")
            return False, None, None

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
        
        # TODO: add parameters
        if err > 0.05:
            self.get_logger().info("Target proj err too high")
            return False, None, None
        else:
            self.get_logger().info("Target proj err: " + str(err))
        
        rot = cv2.Rodrigues(rvec)[0]
        h  = homogenous_from_rt(rot, tvec)
        
        tf = TransformStamped()
        quaternion  = transforms3d.quaternions.mat2quat(rot)
        translation = tvec.reshape(3)
        tf.header.stamp = stamp if stamp is not None else rclpy.time.Time().to_msg()
        tf.header.frame_id = self.eye_frame_id
        tf.child_frame_id  = self.target_frame_id
        tf.transform.translation.x = translation[0]
        tf.transform.translation.y = translation[1]
        tf.transform.translation.z = translation[2]
        tf.transform.rotation.w = quaternion[0]
        tf.transform.rotation.x = quaternion[1]
        tf.transform.rotation.y = quaternion[2]
        tf.transform.rotation.z = quaternion[3]
        
        return True, tf, h
    
    def _get_hand2base (self, stamp = None):
        ret, tf, h = self._get_transform(self.base_frame_id, self.hand_frame_id, stamp)
            
        if not ret:
            self.get_logger().info("Unable to recover consistent pose for current image")
            return False, None, None
        
        if stamp is not None:
            delta_s  = tf.header.stamp.sec - stamp.sec
            delta_ns = tf.header.stamp.nanosec - stamp.nanosec
            delta    = delta_s + delta_ns * 1e-9
        
            if delta > 0.1:
                self.get_logger().info("Time drift too high")
                return False, None, None
        
        return True, tf, h
    
    def _compute_hand_eye_hs (self, min_samples = 130, max_iterations = 5):
        
        hand2base_hs     = self.hand2base_hs
        target2eye_hs    = self.target2eye_hs
        
        best_mse         = math.inf
        best_eye2hand_h  = None
        best_pts         = None
        best_center      = None

        curr_it          = 1
        ok = True
        while len(hand2base_hs) > min_samples or curr_it > max_iterations:
            
            self.get_logger().info(f"Computing hand-eye calibration with {len(hand2base_hs)} samples")
            self.get_logger().info(f"Current MSE: {best_mse}")
            self.get_logger().info(f"Iteration: {curr_it}")
            
            method_mses    = []
            method_pts     = []
            method_centers =  []
            
            try:
            
                methods, eye2hand_hs = compute_eye2hand_hs(hand2base_hs, target2eye_hs)
                
                for _, eye2hand_h in zip(methods, eye2hand_hs):
                    
                    target2base_hs = []
                    for hand2base_h, target2eye_h in zip(hand2base_hs, target2eye_hs):
                        target2hand_h = np.dot(eye2hand_h, target2eye_h)
                        target2base_h = np.dot(hand2base_h, target2hand_h)
                        target2base_hs.append(target2base_h)
                        
                    # Perform outlier rejection
                    pts    = np.array([h[:3, 3] for h in target2base_hs])
                    center = np.mean(pts, axis = 0)
                    mse    = np.mean(np.linalg.norm(pts - center, axis = 1) ** 2) 
                    
                    method_mses.append(mse)
                    method_pts.append(pts)
                    method_centers.append(center)
                    
                self.get_logger().info("=" * 10)
                for i, mse in enumerate(method_mses):
                    self.get_logger().info(CALIB_HAND_EYE_METHODS[i][0] + ": " + str(mse) + "mm^2")
                self.get_logger().info("=" * 10)
                
                best_idx        = np.argmin(method_mses)
                best_mse        = method_mses[best_idx]
                best_pts        = method_pts[best_idx]
                best_center     = method_centers[best_idx]
                best_eye2hand_h = eye2hand_hs[best_idx]

                
                os.mkdir(os.path.join(self.data_folder, "hand_eye", "processing", f"it_{curr_it}"))
                np.save(os.path.join(self.data_folder, "hand_eye", "processing", f"it_{curr_it}", "method_mses.npy"), method_mses)
                np.save(os.path.join(self.data_folder, "hand_eye", "processing", f"it_{curr_it}", "method_pts.npy"), method_pts)
                np.save(os.path.join(self.data_folder, "hand_eye", "processing", f"it_{curr_it}", "method_centers.npy"), method_centers)
                
                np.save(os.path.join(self.data_folder, "hand_eye", "processing", f"it_{curr_it}", "best_mse.npy"), best_mse)
                np.save(os.path.join(self.data_folder, "hand_eye", "processing", f"it_{curr_it}", "best_pts.npy"), best_pts)
                np.save(os.path.join(self.data_folder, "hand_eye", "processing", f"it_{curr_it}", "best_center.npy"), best_center)
                np.save(os.path.join(self.data_folder, "hand_eye", "processing", f"it_{curr_it}", "best_eye2hand_h.npy"), best_eye2hand_h)
                                
                                
                hand2base_hs  = [h for h, pt in zip(hand2base_hs, best_pts) if (np.linalg.norm(pt - best_center) ** 2) <= best_mse]
                target2eye_hs = [h for h, pt in zip(target2eye_hs, best_pts) if (np.linalg.norm(pt - best_center) ** 2) <= best_mse]
                curr_it += 1
            
            except Exception as e:
                self.get_logger().error(f"Error computing hand-eye calibration at iteration {curr_it}: {e}")
                ok  = False

        self.eye2hand_hs = eye2hand_hs
        self.eye2hand_h  = best_eye2hand_h
        
        self.processing_done()
        
    def _compute_hand_eye_h (self):
        self.eye2hand_h = self.eye2hand_hs[0]
        self.validating_done()

    def _publish_transform(self, tf: TransformStamped):
        self.dynamic_broadcaster.sendTransform(tf)

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
            
            R = transforms3d.quaternions.quat2mat(rotation)
            t = np.array(translation)

            h = homogenous_from_rt(R, t)
            
            self.get_logger().info(f"Transform lookup successful!")
            
            return True, tf, h
        except Exception as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return False, None, None

        
def main ():
    rclpy.init()
    node = HandEyeNode(
        base_frame_id   = "ur10e_base_link",
        hand_frame_id   = "tcp",
        eye_frame_id    = "rgb_camera",
        target_frame_id = "chessboard",
    )
    
    # Create a MultiThreadedExecutor with the number of threads you want
    executor = MultiThreadedExecutor(num_threads=4)

    # Add the node to the executor
    executor.add_node(node)
    # Spin the executor (this will block until shutdown)
    executor.spin()
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()