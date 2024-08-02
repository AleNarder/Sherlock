import numpy as np
import cv2
import cv2.typing as cv2_t
from datetime import datetime, timedelta
from builtin_interfaces.msg import Time as RosTime
from intrinsics.utils import find_chessboard_corners, get_chessboard_pts, get_n_representative_frames, IntrinsicsDict

P_cam2ros = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])

def subtract_milliseconds(ros_timestamp, milliseconds):
    # Convert ROS2 Time to datetime
    seconds = ros_timestamp.sec
    nanoseconds = ros_timestamp.nanosec
    timestamp = datetime.fromtimestamp(seconds + nanoseconds / 1e9)
    
    # Subtract the specified milliseconds
    new_timestamp = timestamp - timedelta(milliseconds=milliseconds)
    
    # Convert back to ROS2 Time
    new_seconds = int(new_timestamp.timestamp())
    new_nanoseconds = int((new_timestamp.timestamp() - new_seconds) * 1e9)
    
    new_ros_timestamp = RosTime(sec=new_seconds, nanosec=new_nanoseconds)
    return new_ros_timestamp


def homogenous_matrix_from_rt (r: np.array, t: np.array):
    h = np.eye(4)
    h[:3, :3] = r
    h[:3, 3] = t.reshape(3)
    return h

def compute_hand_eye(frames: list[cv2_t.MatLike], poses: list[np.array], cam_data: IntrinsicsDict, logger):
        chessboard_pts = get_chessboard_pts((9, 6), square_size=0.023)
        
        frames = np.array(frames)
        poses = np.array(poses)

        R_gripper2base_r = []
        t_gripper2base_r = []
        R_target2cam_r   = []
        t_target2cam_r   = [] 

        # Only process sufficiently different frames to reduce computation time and avoid overfitting
        # representative_frames_idxs = get_n_representative_frames(frames, num_frames=75)
        # representative_frames, representative_poses  = frames[representative_frames_idxs], poses[representative_frames_idxs]
        
        for frame, pose in zip(frames, poses):
            ret, corners = find_chessboard_corners(frame, (9,6))    
            if ret:
                ret2, rvec_target2cam, tvec_target2cam = cv2.solvePnP(chessboard_pts, corners, cam_data["mtx"], cam_data["dist"])
                if ret2:

                    rot_target2camm, _ = cv2.Rodrigues(rvec_target2cam)

                    R_target2cam_r.append(rot_target2camm)
                    t_target2cam_r.append(tvec_target2cam)
                                        
                    R_gripper2base_r.append(pose[:3, :3])
                    t_gripper2base_r.append((pose[:3, 3]).reshape((3, 1)))


        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_r, 
            t_gripper2base_r,
             
            R_target2cam_r, 
            t_target2cam_r, 
            method = cv2.CALIB_HAND_EYE_TSAI
        )
        
    
        h_cam2gripper   = homogenous_matrix_from_rt(R_cam2gripper, t_cam2gripper)
        hs_gripper2base = [homogenous_matrix_from_rt(R, t) for R, t in zip(R_gripper2base_r, t_gripper2base_r)]
        hs_target2cam   = [homogenous_matrix_from_rt(R, t) for R, t in zip(R_target2cam_r, t_target2cam_r)]
        
        return (
            h_cam2gripper,
            hs_gripper2base,
            hs_target2cam
        ) 
            
