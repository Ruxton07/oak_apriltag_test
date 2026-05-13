import depthai as dai
import cv2

import vision.tag_detection as tag_detection

def main():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    q = cam.video.createOutputQueue()
    
    with pipeline:
        pipeline.start()
        
        while pipeline.isRunning():
            frame = q.get().getCvFrame() # type: ignore
            tag_detection.detect_apriltags(frame)
            cv2.imshow("OAK-D AprilTags", frame)

            if cv2.waitKey(1) == ord("q"):
                break

if __name__ == "__main__":
    main()