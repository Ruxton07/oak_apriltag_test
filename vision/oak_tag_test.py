import depthai as dai
import cv2
from pupil_apriltags import Detector
import numpy as np

def sq_det(det, min_area=250, side_ratio_tol=0.35, angle_tol_deg=45):
    corners = det.corners.astype(float)
    sides = []

    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        sides.append(np.linalg.norm(p2 - p1))

    sides = np.array(sides)
    area = cv2.contourArea(corners.astype(np.float32))

    if area < min_area:
        return False

    if sides.min() <= 0:
        return False

    if sides.max() / sides.min() > 1.0 + side_ratio_tol:
        return False

    for i in range(4):
        prev_pt = corners[(i - 1) % 4]
        curr_pt = corners[i]
        next_pt = corners[(i + 1) % 4]
        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        
        cos_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )

        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        if abs(angle - 90.0) > angle_tol_deg:
            return False

    return True

pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
q = cam.video.createOutputQueue()

detector = Detector(
    families="tagCircle21h7",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
)

with pipeline:
    pipeline.start()
    
    while pipeline.isRunning():
        frame = q.get().getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw_detections = detector.detect(gray)

        detections = [ det for det in raw_detections if sq_det(det) ]
        
        for det in detections:
            print("id:", det.tag_id, "center:", det.center)
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

        cv2.imshow("OAK-D AprilTags", frame)

        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()