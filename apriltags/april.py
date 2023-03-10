import apriltag
import numpy as np
import cv2 as cv
import time
import util

camMtx = util.readFromFile("newcameramtx.npy")
camMtxDetectorParam = [camMtx[0][0], camMtx[0][2], camMtx[1][1], camMtx[1][2]]
dist = util.readFromFile("dist.npy")

options: apriltag.DetectorOptions = apriltag.DetectorOptions(families="tag16h5", quad_decimate=0, nthreads=4)
detector: apriltag.Detector = apriltag.Detector(options)

detections, img2 = None, None
detect: apriltag.Detection

def callPoseDetection(detection: apriltag.Detection): # NOTE TO SELF: tag_size is in METERS, and is measured NOT INCLUDING the border (so 6 inches in our case)
    return detector.detection_pose(detection=detection, camera_params=camMtxDetectorParam, tag_size=0.15244)

DEBUG = False
# TEST
if DEBUG:
    imagePath = "test.jpg"
    img = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)

    detections, img2 = detector.detect(img, True)
    detect = detections[0]
else:
    cap = cv.VideoCapture('/dev/video0')
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, 90)

while True:
    if not DEBUG:
        start_time = time.time()
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detections, cropped = detector.detect(gray, True)
        # detecte: apriltag.Detection = apriltag.Detection()
        detected = 0
        for idx, detection in enumerate(detections):
            if detection.tag_id < 1 or detection.tag_id > 9 or detection.decision_margin < 20:
                # print("rejected:", detection.tag_id, "dec_mar:", detection.decision_margin)
                continue
            detected += 1
            print(callPoseDetection(detection=detection))
            # print("detected:", detection.tostring())
        print("[INFO]", detected, "total AprilTags detected")
        print("FPS: ", round(1.0 / (time.time() - start_time)))

        cv.imshow("gray", gray)
        cv.imshow("crop", cropped)
    else:
        print(len(detect))
        # cv.imshow("crop", detect)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
if not DEBUG:
    cap.release()
cv.destroyAllWindows()