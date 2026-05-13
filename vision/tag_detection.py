import cv2
import numpy as np
from pupil_apriltags import Detector

TAG_SIZE_M = 0.03  # 0.03 would be 3cm
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FX = 1400.0
FY = 1400.0
CX = FRAME_WIDTH / 2.0
CY = FRAME_HEIGHT / 2.0
CAMERA_PARAMS = (FX, FY, CX, CY)

def wrap_to_pm90(angle_deg):
    # wrap angle to [-90, 90] range (tag visibility not possible outside)
    wrapped = ((angle_deg + 90) % 180) - 90
    return wrapped

def rot_matrix2_yp0r(R):
    # compute yaw/pitch from the tag's forward z-axis expressed in camera coords
    # yaw correspond to rotation about the tag's y-axis as seen by the camera
    # pitch correspond to rotation about the tag's x-axis as seen by the camera
    # assume 0 roll (TODO: may cause error if roll significant? still have to deal with gimbal lock if we try to compute roll)
    z = np.array([R[0, 2], R[1, 2], R[2, 2]])
    yaw = np.arctan2(z[0], z[2])
    pitch = np.arctan2(-z[1], np.hypot(z[0], z[2]))

    return wrap_to_pm90(np.degrees(yaw)), wrap_to_pm90(np.degrees(pitch))

def valid_tag_shape(det, min_area=250, min_margin=30, max_aspect_ratio=4.0):
    corners = det.corners.astype(float)

    if det.decision_margin < min_margin:
        return False

    area = cv2.contourArea(corners.astype(np.float32))
    if area < min_area:
        return False

    side_lengths = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        side_lengths.append(np.linalg.norm(p2 - p1))

    side_lengths = np.array(side_lengths)

    if side_lengths.min() <= 0:
        return False

    if side_lengths.max() / side_lengths.min() > max_aspect_ratio:
        return False

    return True

detector = Detector(
    families="tagCircle21h7",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
)

def detect_apriltags(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    raw_detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=CAMERA_PARAMS,
        tag_size=TAG_SIZE_M,
    )
    detections = [ det for det in raw_detections if valid_tag_shape(det) ] # type: ignore
    
    for det in detections:
        # compute pose
        
        pose_t = np.squeeze(det.pose_t)
        yaw_deg, pitch_deg = rot_matrix2_yp0r(det.pose_R)

        print(
            f"ID{det.tag_id} | "
            f"pose_t =[{pose_t[0]:.3f}, {pose_t[1]:.3f}, {pose_t[2]:.3f}] m | "
            f"yaw = {yaw_deg:.1f} deg | "
            f"pitch = {pitch_deg:.1f} deg"
        )
        
        # drawing detection on frame
        
        corners = det.corners.astype(int)

        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        cx, cy = det.center.astype(int)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"ID {det.tag_id}",
            (cx + 10, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

cv2.destroyAllWindows()