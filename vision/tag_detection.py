import cv2
import numpy as np
from pupil_apriltags import Detector

from autoalign.target_selector import image_err, AlignError

TAG_SIZE_M = 0.10  # this would be 10cm
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FX = 1400.0
FY = 1400.0
CX = FRAME_WIDTH / 2.0
CY = FRAME_HEIGHT / 2.0
CAMERA_PARAMS = (FX, FY, CX, CY)

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
    raw_detections = detector.detect(gray)

    detections = [ det for det in raw_detections if valid_tag_shape(det) ]
    
    for det in detections:
        error = image_err(det, frame.shape[1], frame.shape[0])
        print(
            f"ID {det.tag_id} | "
            f"x_error = {error.x:.1f} px | "
            f"z_error = {error.z:.1f} px | "
            f"yaw_error = {np.degrees(error.yaw):.1f} deg"
            )
        
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