import numpy as np
import rclpy
import json
import os
import cv2

from ament_index_python.packages import get_package_share_directory
from statemachine import StateMachine, State
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from intrinsics.utils import is_sharp, get_n_representative_frames, compute_intrinsics, compute_reprojection_error, find_chessboard_corners
from threading import Thread

bridge = CvBridge()

class IntrinsicsNode(StateMachine, Node):
   
    INITIALIZED = State(name="INITIALIZED", initial = True) 
    RECORDING   = State(name="RECORDING")
    PROCESSING  = State(name="PROCESSING")

    start_recording = INITIALIZED.to(RECORDING) 
    stop_recording  = RECORDING.to(PROCESSING)
    processing_done = PROCESSING.to(INITIALIZED)

    def __init__ (self):        
        StateMachine.__init__(self)
        Node.__init__(self, 'intrinsics_node')

        # Variables
        self.frames = []
        self.intrinsics = { "mtx": None, "dist": None, "err": None }
        
        # Subs
        self.create_subscription(Image, "/camera/color/image_raw", self._on_image_cb, 10)
        
        # Services
        self.create_service(Trigger, '/intrinsics/start_recording', self._start_recording_cb),
        self.create_service(Trigger, '/intrinsics/stop_recording',  self._stop_recording_cb)
        
        # Publishers
        self.coeffs_pub_= self.create_publisher(String, "/intrinsics/coeffs", 10)
        self.state_pub_ = self.create_publisher(String, "/intrinsics/state", 10)
        self.corner_pub_= self.create_publisher(Image, "/intrinsics/corners", 10)
        
        self.create_timer(0.1, self._publish)
        
        if os.path.exists(os.path.join("/", "calibration", "data", "intrinsics", "mtx.npy")):
            self.intrinsics["mtx"] = np.load(os.path.join("/", "calibration", "data", "intrinsics", "mtx.npy")) 
            self._logger.info("mtx loaded!")
        if os.path.exists(os.path.join("/", "calibration", "data", "intrinsics", "dist.npy")):
            self.intrinsics["dist"] = np.load(os.path.join("/", "calibration", "data", "intrinsics", "dist.npy"))
            self._logger.info("dist loaded!")

    def after_start_recording(self):
        self.get_logger().info("start recording...")

    def after_stop_recording(self):
        self.get_logger().info("stop recording...")
        Thread(target = self._process).start()

    def after_processing_done(self):
        self.get_logger().info("processing done, saving intrinsics...")
        self._on_save_intrinsics()
        self.get_logger().info("intrinsics saved!")
    
    def intrinsics_are_loaded(self):
        return self.intrinsics["mtx"] is not None and self.intrinsics["dist"] is not None
    
    def on_initialized (self):
        self.get_logger().info("initialized!")
        self.frames = []
        
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
            if (is_sharp(cv_frame, 280)): 
                self.frames.append(cv_frame)
                ret, corners = find_chessboard_corners(cv_frame, (9, 6))
                
                if not ret:
                    return
                
                corner_frame = cv2.drawChessboardCorners(cv_frame, (9, 6), corners, True)
                self.corner_pub_.publish(bridge.cv2_to_imgmsg(corner_frame))                
       
    def _publish(self):
        msg = String()
        
        data = {}
        data["mtx"]  = self.intrinsics["mtx"].tolist() if self.intrinsics["mtx"] is not None else None
        data["dist"] = self.intrinsics["dist"].tolist() if self.intrinsics["dist"] is not None else None
        data["err"]  = self.intrinsics["err"]
        
        msg.data = json.dumps(data)
        self.coeffs_pub_.publish(msg)
        
        msg = String()
        msg.data = str(self.current_state)
        self.state_pub_.publish(msg)


    def _process(self):
        rep_frames = [self.frames[idx] for idx in get_n_representative_frames(self.frames)]
        mtx, dist ,rvecs, tvecs, objectpoints, imgpoints = compute_intrinsics(rep_frames, (9, 6), (640, 480))
        err, _ = compute_reprojection_error(objectpoints, imgpoints, rvecs, tvecs, mtx, dist)

        self.intrinsics["mtx"]  = mtx
        self.intrinsics["dist"] = dist
        self.intrinsics["err"]  = err
        
        self.processing_done()
        
    def _on_save_intrinsics(self):
        np.save(os.path.join("/", "calibration", "data", "intrinsics", "mtx.npy")   , self.intrinsics["mtx"])
        np.save(os.path.join("/", "calibration", "data", "intrinsics", "dist.npy")  , self.intrinsics["dist"])
        np.save(os.path.join("/", "calibration", "data", "intrinsics", "err.npy")   , np.array(self.intrinsics["err"]))
        np.save(os.path.join("/", "calibration", "data", "intrinsics", "frames.npy"), self.frames)
        
   
def main ():
    rclpy.init()
    node = IntrinsicsNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()