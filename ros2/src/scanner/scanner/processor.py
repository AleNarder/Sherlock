class ScannerProcessor ():
    def __init__(self) -> None:
        pass

# import rosbag
# import numpy as np
# import cv2
# import pyrealsense2 as rs
# from sensor_msgs.msg import Image, CameraInfo
# from cv_bridge import CvBridge

# # Initialize the CvBridge class
# bridge = CvBridge()

# # Bag file path
# bag_file = 'path/to/your.bag'

# # Variables to store the extracted data
# depth_image_rect = None
# color_image = None
# depth_intrinsics = None
# color_intrinsics = None
# depth_to_color_extrinsics = None

# # Extract data from the bag file
# with rosbag.Bag(bag_file, 'r') as bag:
#     for topic, msg, t in bag.read_messages(topics=['/camera/depth/image_rect_raw',
#                                                    '/camera/color/image_raw',
#                                                    '/camera/depth/camera_info',
#                                                    '/camera/color/camera_info',
#                                                    '/camera/extrinsics/depth_to_color']):
#         if topic == '/camera/depth/image_rect_raw':
#             depth_image_rect = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#         elif topic == '/camera/color/image_raw':
#             color_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
#         elif topic == '/camera/depth/camera_info':
#             depth_intrinsics = rs.intrinsics()
#             depth_intrinsics.width = msg.width
#             depth_intrinsics.height = msg.height
#             depth_intrinsics.ppx = msg.K[2]
#             depth_intrinsics.ppy = msg.K[5]
#             depth_intrinsics.fx = msg.K[0]
#             depth_intrinsics.fy = msg.K[4]
#             depth_intrinsics.model = rs.distortion.none
#             depth_intrinsics.coeffs = msg.D
#         elif topic == '/camera/color/camera_info':
#             color_intrinsics = rs.intrinsics()
#             color_intrinsics.width = msg.width
#             color_intrinsics.height = msg.height
#             color_intrinsics.ppx = msg.K[2]
#             color_intrinsics.ppy = msg.K[5]
#             color_intrinsics.fx = msg.K[0]
#             color_intrinsics.fy = msg.K[4]
#             color_intrinsics.model = rs.distortion.none
#             color_intrinsics.coeffs = msg.D
#         elif topic == '/camera/extrinsics/depth_to_color':
#             depth_to_color_extrinsics = rs.extrinsics()
#             depth_to_color_extrinsics.rotation = msg.rotation
#             depth_to_color_extrinsics.translation = msg.translation

# # Assuming depth scale is known or obtained from the RealSense device
# depth_scale = 0.001  # For example, typical depth scale for RealSense cameras

# # Create pointcloud object
# pc = rs.pointcloud()

# # Generate the pointcloud and texture mappings
# depth_frame = rs.frame(depth_image_rect)
# color_frame = rs.frame(color_image)
# pc.map_to(color_frame)
# points = pc.calculate(depth_frame)

# # Get vertices and texture coordinates
# vertices = np.asanyarray(points.get_vertices())
# tex_coords = np.asanyarray(points.get_texture_coordinates())

# # Transform the depth points to the color camera's coordinate system
# depth_to_color_transform = np.array(depth_to_color_extrinsics.rotation).reshape(3, 3)
# depth_to_color_translation = np.array(depth_to_color_extrinsics.translation).reshape(3, 1)

# aligned_points = []
# for vertex in vertices:
#     point = np.array([vertex.x, vertex.y, vertex.z]).reshape(3, 1)
#     transformed_point = depth_to_color_transform @ point + depth_to_color_translation
#     aligned_points.append(transformed_point)

# # Project transformed points onto the color image plane
# projected_points = []
# for point in aligned_points:
#     x = point[0] / point[2] * color_intrinsics.fx + color_intrinsics.ppx
#     y = point[1] / point[2] * color_intrinsics.fy + color_intrinsics.ppy
#     projected_points.append((x, y))

# # Visualize the alignment
# for (x, y) in projected_points:
#     if 0 <= x < color_image.shape[1] and 0 <= y < color_image.shape[0]:
#         cv2.circle(color_image, (int(x), int(y)), 2, (0, 0, 255), -1)

# cv2.imshow('Aligned Depth to Color', color_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()