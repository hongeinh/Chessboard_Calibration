'''
This program is written to calibrate images from new data in 13/05/2020 chessboard
PROBLEMS:
+ For chess_1: Okay
+ For chess_2: Okay
+ For chess_3: (-215:Assertion failed) nimages > 0 in function 'calibrateCamera'. Due to no chessboard corners found
+ For chess_4: (-215:Assertion failed) nimages > 0 in function 'calibrateCamera'. Due to no chessboard corners found
'''


import numpy as np
import cv2
import glob

filename = "/home/hongeinh/Downloads/chessboard/new_data/13-5-2020chessboard1"
simulation_filename = "/home/hongeinh/Downloads/chessboard/new_data/13-5-2020simulation"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
cbrow = 11
cbcolumn = 7

objp = np.zeros((cbrow*cbcolumn, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbrow, 0:cbcolumn].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 

images = glob.glob(filename + "/distort_image/*.png")

i = 0
count = 0
for fname in images:
    
    i = i + 1
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (11, 7), None, flags= cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH +   cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        count += 1
        print("Found corners: ", count)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)

        img = cv2.drawChessboardCorners(img, (cbrow, cbcolumn), corners2, ret)
        
        
##### CALIBRATION ######

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags =  cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_ASPECT_RATIO)
dist = np.array(dist)


##### UNDISTORTION #####
img1 = cv2.imread(filename + "/distort_image/chessboard64.png")

h, w = img1.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), 5)

resultImg = sorted(glob.glob(simulation_filename + "/distort_image/*.png"))

k = 0

mask = cv2.imread("/home/hongeinh/Downloads/chessboard/new_data/mask.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

for rimg in resultImg:
    k += 1
    img = cv2.imread(rimg)
    #cv2.CV_32FC2
    
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    dst = cv2.bitwise_and(dst, dst, mask=mask)
    path = simulation_filename + "/undistort_image/simulation_calib" + str(k) + ".png"

    cv2.imwrite(path, dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


s = cv2.FileStorage(simulation_filename + "/feature_matching/intrinsic_parameters.xml", cv2.FileStorage_WRITE)

s.write("mtx", mtx)
s.write("dist_coef", dist)
s.write("rvecs", np.asarray(rvecs))
s.write("tvecs", np.asarray(tvecs))
s.write("newcameramtx", np.asarray(newcameramtx))
s.write("mapx", np.asarray(mapx))
s.write("mapy", np.asarray(mapy))



cv2.waitKey(0)
cv2.destroyAllWindows()
