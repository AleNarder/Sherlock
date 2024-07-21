from camera_intrinsic.processor import CameraIntrinsicProcessor
from camera_intrinsic.recorder import CameraIntrinsicRecorder
import rclpy
import logging
from rclpy.node import Node
from minio import Minio

mc = Minio(
    "minio:9000",   # Replace with your MinIO server URL
    access_key="username",  # Replace with your access key
    secret_key="password",  # Replace with your secret key
    secure = False     # Set to False if not using HTTPS
)
 
def init_logger ():
    sh = logging.StreamHandler()
    logging.basicConfig(level = logging.DEBUG, format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s', handlers=[sh])

def main ():
    init_logger()
    
    rclpy.init()
    node = Node("camera_intrinsic")
    
    CameraIntrinsicRecorder(node, mc)
    CameraIntrinsicProcessor(node, mc)
    
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()