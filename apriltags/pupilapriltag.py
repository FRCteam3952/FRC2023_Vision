from pupil_apriltags import Detector
import cv2 as cv


at_detector = Detector(
   families="tag16h5",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 90)


while(True):
    ret, frame = cap.read()
    at_detector.detect(True)
    cv.imshow("crop", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break