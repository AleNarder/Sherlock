import numpy as np
import cv2
from sklearn.covariance import EllipticEnvelope

CALIB_HAND_EYE_METHODS = [
    ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
    ("PARK", cv2.CALIB_HAND_EYE_PARK),
    ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
    # Online calibration method, not used in this project
    # ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
    ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
]


def get_inliers(points, contamination=0.2):
    cov = EllipticEnvelope(random_state=42, contamination=contamination)
    inliers = cov.fit_predict(points)
    return inliers


def rt_from_homogenous(h: np.array):
    return h[:3, :3], h[:3, 3]


def homogenous_from_rt(r: np.array, t: np.array):
    h = np.eye(4)
    h[:3, :3] = r
    h[:3, 3] = t.reshape(3)
    return h


def compute_eye2hand_hs(hand2base_hs, target2eye_hs):

    target2eye_Rs = np.array([h[:3, :3] for h in target2eye_hs])
    target2eye_ts = np.array([h[:3, 3] for h in target2eye_hs])

    hand2base_Rs = np.array([h[:3, :3] for h in hand2base_hs])
    hand2base_ts = np.array([h[:3, 3] for h in hand2base_hs])

    eye2hand_hs = []
    methods = [method for _, method in CALIB_HAND_EYE_METHODS]

    for method in methods:
        R_eye2hand, t_eye2hand = cv2.calibrateHandEye(
            hand2base_Rs, hand2base_ts, target2eye_Rs, target2eye_ts, method=method
        )

        eye2hand_hs.append(homogenous_from_rt(R_eye2hand, t_eye2hand))

    return methods, eye2hand_hs


def best_eye2hand_h(target2base_hss):
    target2base_hss = [np.array(hs) for hs in target2base_hss]

    mses = []
    for i in range(len(target2base_hss)):
        points_3d = target2base_hss[i, :, :3, 3:]
        cov = EllipticEnvelope(random_state=42)
        cov.fit_predict(points_3d)
        center = np.array(cov.location_).reshape((1, 3, 1))
        errs = np.array([point - center for point in points_3d]).squeeze()
        mse = np.mean(np.linalg.norm(errs, axis=1) ** 2)
        mses.append(mse)

    return np.argmin(mses)
