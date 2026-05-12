import depthai as dai

import cv2

def main():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    q = cam.video.createOutputQueue()

    with pipeline:
        pipeline.start()
        while pipeline.isRunning():
            frame = q.get().getCvFrame()
            cv2.imshow("OAK-D", frame)
            if cv2.waitKey(1) == ord("q"):
                break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()