import tf2_ros
import rclpy
import numpy as np
import traceback

from rclpy.node import Node
from moving_average.circular_buffer import CircularBuffer
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32

class MovingAverageNode(Node):
    def __init__(self, base_link: str, gripper_link: str):
        super().__init__('moving_average')
        
        self.positions_buffer = CircularBuffer(5, 4, fill_value=np.inf)
        self.base_link = base_link
        self.gripper_link = gripper_link
        
        # Tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        
        self.vec_speed_pb = self.create_publisher(Vector3, '/odom/vec_speed', 10)
        self.speed_pb     = self.create_publisher(Float32, '/odom/speed', 10)
        self.create_timer(1 / 10, self.timer_callback)
        
        
    def timer_callback(self):
        
        vec_speed_msg = Vector3()
        speed_msg     = Float32()
        
        vec_speed_msg.x = np.inf
        vec_speed_msg.y = np.inf
        vec_speed_msg.z = np.inf
        speed_msg.data  = np.inf
        
        try: 
            tf = self.tf_buffer.lookup_transform(self.base_link, self.gripper_link, rclpy.time.Time())

            # Update buffer
            translation = tf.transform.translation
            time        = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            self.positions_buffer.add(np.array([translation.x, translation.y, translation.z, time]))
            
            # Get positions and timestamps
            positions = self.positions_buffer.get()
            timestamps = positions[:, 3]
            positions  = positions[:, :3]

            # Compute partial speeds
            displacements = np.diff(positions, axis=0)
            time_diffs    = np.diff(timestamps).flatten()
            
            self.get_logger().info(f'displacements: {displacements}')
            self.get_logger().info(f'time_diffs: {time_diffs}')
            speeds = displacements / time_diffs[:, np.newaxis]
            
            # Compute average speed
            avg_vec_speed = np.mean(speeds, axis=0)
            
            
            vec_speed_msg.x = avg_vec_speed[0]
            vec_speed_msg.y = avg_vec_speed[1]
            vec_speed_msg.z = avg_vec_speed[2]
            
            speed_msg.data = np.linalg.norm(avg_vec_speed)
    
        except Exception as e:
            self.get_logger().error('error: %s' % e)
            self.get_logger().error(traceback.format_exc())
        
        self.vec_speed_pb.publish(vec_speed_msg)
        self.speed_pb.publish(speed_msg)        
        
def main(args=None):
    rclpy.init(args=args)
    
    node = MovingAverageNode(
        base_link='ur10e_base_link',
        gripper_link='link6_1'
    )
    
    rclpy.spin(node)
    rclpy.shutdown()
        