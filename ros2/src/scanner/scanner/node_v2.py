from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from rclpy.node import Node
from statemachine import StateMachine, State

class ScannerNode (Node, StateMachine):

    UNITIALIZED = State(name="UNITIALIZED", initial = True)
    INITIALIZED = State(name="INITIALIZED")
    RECORDING   = State(name="RECORDING")
    PROCESSING  = State(name="PROCESSING")
    FULFILLED   = State(name="FULFILLED", final= True)

    initialize  = UNITIALIZED.to(INITIALIZED, cond = [
        "intrinsics_are_loaded",
        "hand_eye_tf_are_loaded"
    ]) | UNITIALIZED.to(UNITIALIZED)

    start_recording = INITIALIZED.to(RECORDING)
    stop_recording  = RECORDING.to(PROCESSING)
    processing_done = PROCESSING.to(FULFILLED)


    def __init__ (self):
        StateMachine.__init__(self)
        Node.__init__(self, 'scanner_node')

        
        self.create_subscription(Image, "/camera/color/image_raw", self._on_color_image, 10)
        self.create_subscription(Image, "/camera/depth/image_rect_raw", self._on_depth_image, 10)
        self.create_subscription(CameraInfo, "/camera/extrinsics/depth_to_color", self._on_depth_to_color, 10)

        self.create_service(Trigger, "/scanning/recorder/start_record", self._on_start_record_cb)
        self.create_service(Trigger, "/scanning/recorder/stop_record", self._on_stop_record_cb)
        
        # Variables
        
    def _on_stop_record_cb(self, request, response):
        self.stop_recording()
        return response
    
    def _on_start_record_cb(self, request, response):
        self.start_recording()
        return response
    
    def _on_color_image(self, msg):
        pass
        
    def _on_depth_image(self, msg):
        pass
    
    def _on_depth_to_color(self, msg):
        pass