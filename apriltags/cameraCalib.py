# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2 as cv
import glob
import util

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# test_path = '/home/addison/FRC2023_Vision/apriltags/calibImgs/*.jpg'
test_path = "C:/Users/astro/OneDrive/Documents/FRC2023_Vision/apriltags/calibImgs2/*.jpg" #i was gonna calibrate to my webcam lol
images = glob.glob(test_path)
print(len(images))
for i, fname in enumerate(images):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,7), corners2, ret)
        cv.imshow('img' + str(i), img)
        cv.waitKey(500)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# img = cv.imread('/home/addison/FRC2023_Vision/apriltags/calibImgs/test_1.jpg')
h, w = (720, 1280) # img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult_remap.png', dst)
cv.destroyAllWindows()

util.writeToFile(mtx, "mtx.npy")
util.writeToFile(mapx, "mapx.npy")
util.writeToFile(mapy, "mapy.npy")
util.writeToFile(newcameramtx, "newcameramtx.npy")
util.writeToFile(roi, "roi.npy")
util.writeToFile(dist, "dist.npy")

print(newcameramtx.shape)
print(newcameramtx[0])
print(newcameramtx[1])
print(newcameramtx[2])
print(dist.shape)