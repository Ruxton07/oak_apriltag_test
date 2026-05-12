from dataclasses import dataclass
import numpy as np

IDEAL_TAG_SIZE = 200.0  # in pixels

@dataclass
class AlignError:
    x: float       # horizontal offset in pixels
    z: float       # distance proxy in pixels
    yaw: float     # angle in radians


def image_err(detection, frame_width, des_tag_size=IDEAL_TAG_SIZE):
    corners = detection.corners.astype(float)

    # X ERROR
    cx, cy = detection.center
    image_center_x = frame_width / 2.0
    x_error = cx - image_center_x

    # Z ERROR (distance proxy from ideal tag size)
    side_lengths = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        side_lengths.append(np.linalg.norm(p2 - p1))

    observed_size = np.mean(side_lengths)
    z_error = des_tag_size - observed_size

    # YAW ERROR (rotation of top edge)
    top_left = corners[0]
    top_right = corners[1]

    dx = top_right[0] - top_left[0]
    dy = top_right[1] - top_left[1]

    yaw_error = np.arctan2(dy, dx)

    return AlignError(
        x=x_error,
        z=z_error,
        yaw=yaw_error,
    )