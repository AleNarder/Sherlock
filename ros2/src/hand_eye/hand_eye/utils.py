import numpy as np
import cv2
import cv2.typing as cv2_t
from intrinsics.utils import find_chessboard_corners, get_chessboard_pts, get_n_representative_frames, IntrinsicsDict

def homogenous_matrix_from_rt (r: np.array, t: np.array):
    h = np.eye(4)
    h[:3, :3] = r
    h[:3, 3] = t.reshape(3)
    return h

def compute_hand_eye(frames: list[cv2_t.MatLike], poses: list[np.array], cam_data: IntrinsicsDict):
        chessboard_pts = get_chessboard_pts((9, 6), square_size=0.033)
        
        frames = np.array(frames)
        poses = np.array(poses)

        R_gripper2base_r = []
        t_gripper2base_r = []
        R_target2cam_r   = []
        t_target2cam_r   = [] 

        # Only process sufficiently different frames to reduce computation time and avoid overfitting
        representative_frames_idxs = get_n_representative_frames(frames, num_frames=60)
        representative_frames, representative_poses  = frames[representative_frames_idxs], poses[representative_frames_idxs]
        
        for frame, pose in zip(representative_frames, representative_poses):
            ret, corners = find_chessboard_corners(frame, (9,6))    
            if ret:
                ret2, rvec, tvec = cv2.solvePnP(chessboard_pts, corners, cam_data["mtx"], cam_data["dist"])
                if ret2:
                    R_gripper2base_r.append(pose[:3, :3])
                    t_gripper2base_r.append(pose[:3, 3].reshape((3, 1)))
                    
                    rot, _ = cv2.Rodrigues(rvec)
                    h = homogenous_matrix_from_rt(rot, tvec)
                    
                    rot  = h[:3, :3]
                    tvec = h[:3, 3].reshape(3, 1)
                    
                    R_target2cam_r.append(rot)
                    t_target2cam_r.append(tvec)


        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_r, 
            t_gripper2base_r, 
            R_target2cam_r, 
            t_target2cam_r, 
            method = cv2.CALIB_HAND_EYE_TSAI
        )
        
        return R_cam2gripper, t_cam2gripper
