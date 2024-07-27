import numpy as np
import cv2
import cv2.typing as cv2_t
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import TypedDict

class IntrinsicsDict(TypedDict):
    mtx: np.ndarray
    dist: np.ndarray

def is_sharp ( frame: cv2_t.MatLike, threshold = 300) -> bool:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return sharpness > threshold

def undistort_image (img: cv2_t.MatLike, mtx, dist):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst


def find_chessboard_corners ( img: cv2_t.MatLike, pattern_size=(9, 6)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=cv2.CALIB_CB_FAST_CHECK)
    if ret:
        print(f"Found chessboard corners in image")
        return ret, cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    else:
        return False, None


def get_chessboard_pts (pattern_size=(9, 6), square_size=1.0) -> list[np.array]:
    rows, cols = pattern_size
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2) * square_size
    return objp


def compute_intrinsics ( imgs: list[cv2_t.MatLike], pattern_size, img_size, square_size=1.0, flags = None):    
    objpoints = []
    imgpoints = []  

    for img in imgs:
        ret, img_pts = find_chessboard_corners(img, pattern_size)
        if ret:
            obj_pts = get_chessboard_pts(pattern_size, square_size)
            objpoints.append(obj_pts)
            imgpoints.append(img_pts)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, flags, None)
    return mtx, dist, rvecs, tvecs, objpoints, imgpoints


def compute_reprojection_error (objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    errors     = []    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        errors.append(error)

    return np.sum(errors)/len(objpoints), errors


def get_n_representative_frames ( frames: list[cv2_t.MatLike], num_frames = 75) -> list[int]:
    print(f"Selecting {num_frames} representative frames out of {len(frames)} frames")
    
    if len(frames) < num_frames:
        return frames
    else:
        skip_frames = [frames[i] for i in range(0, len(frames), 2)]
        reduced_frames = [cv2.resize(frames[i], None, fx = 0.6, fy = 0.6, interpolation= cv2.INTER_LINEAR) for i in range(0, len(skip_frames), 1)]
        gray_flattened_frames = []
        for frame in reduced_frames:
            gray_flattened_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten())
        
        # Perform pca on flattened frames to reduce dimensionality
        # Take the first n components that explain 95% of the variance
        expected_variance = 0.95
        pca = PCA(n_components=expected_variance, svd_solver="full")
        pca_reduced_frames = pca.fit_transform(gray_flattened_frames)
        print(f"Reduced frames from {len(gray_flattened_frames[0])} to {len(pca_reduced_frames[0])} dimensions keeping {expected_variance * 100} of variance")
        
        # Perform kmeans clustering on pca reduced frames to obtain representative frames
        kmeans = KMeans(n_clusters=num_frames, random_state=0, n_init="auto").fit(pca_reduced_frames)
        representative_frames_idxs = []
        
        def euclidean_distance (x, y):
            return np.linalg.norm(x - y)

        for cluster_idx, cluster_center in enumerate(kmeans.cluster_centers_):

            cluster_items_idxs = np.where(kmeans.labels_ == cluster_idx)[0]

            nearest_idx = np.argmin([euclidean_distance(pca_reduced_frames[idx], cluster_center) for idx in cluster_items_idxs])

            representative_frames_idxs.append(cluster_items_idxs[nearest_idx] * 2)
        print(f"Selected {len(representative_frames_idxs)} representative frames out of {len(reduced_frames)} frames")
        
        return representative_frames_idxs

