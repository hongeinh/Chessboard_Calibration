'''
This program is written to calibrate images from new data in 13/05/2020 chessboard
PROBLEMS:
+ For chess_1: Okay
+ For chess_2: Okay
+ For chess_3: (-215:Assertion failed) nimages > 0 in function 'calibrateCamera'. Due to no chessboard corners found
+ For chess_4: (-215:Assertion failed) nimages > 0 in function 'calibrateCamera'. Due to no chessboard corners found

---------------
17-06-2020 data: Thu dữ liệu chưa hợp lý, cần phải có viền trắng để thuật toán có thể phân biệt được đâu là rìa bảng đâu là corner points
Đối với dữ liệu mới cần thay đổi tham số để tìm chessboard corner, nếu giữ nguyên thì không tìm được.
+ chess1:
+ chess2: 
+ chess3:
+ chess4:
+ chess5:
'''


import numpy as np
import cv2
import glob

# file information
filename = "/home/hongeinh/Downloads/chessboard/First_data"
mask = cv2.imread("/home/hongeinh/Downloads/chessboard/new_data/mask.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
cbrow = 7
cbcolumn = 11

objp = np.zeros((cbrow*cbcolumn, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbrow, 0:cbcolumn].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = sorted(glob.glob(filename + "/distort_image/*.png"))

i = 0
count = 0
for fname in images:

    i = i + 1
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #masked_image = cv2.bitwise_and(img, img, mask=mask)

    ret, corners = cv2.findChessboardCorners(gray, (11, 7), None, flags=cv2.CALIB_CB_FILTER_QUADS +
                                             cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        count += 1
        print(i, ", Found corners: ", count)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)

        img = cv2.drawChessboardCorners(img, (cbrow, cbcolumn), corners2, ret)
        
        cv2.imwrite(filename +"/drawlines/drawn" + str(i) + ".png", img)

if(count != 0):
    ##### CALIBRATION ######
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_ASPECT_RATIO)
    dist = np.array(dist)

    ##### UNDISTORTION #####
    img1 = cv2.imread(filename + "/distort_image/scene00001.png")

    h, w = img1.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    resultImg = sorted(glob.glob(filename + "/distort_image/*.png"))

    k = 0
    for rimg in resultImg:
        k += 1
        img = cv2.imread(rimg)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        #mask = cv2.threshold(dst, 195, 255, cv2.THRESH_BINARY)[1][:, :, 0]
        path = filename + "/undistort_image/calibresult" + str(k) + ".png"
        cv2.imwrite(path, dst)
        cv2.destroyAllWindows()

    s = cv2.FileStorage(
        filename + "/feature_matching/intrinsic_parameters.xml", cv2.FileStorage_WRITE)

    s.write("mtx", mtx)
    s.write("dist_coef", dist)
    s.write("rvecs", np.asarray(rvecs))
    s.write("tvecs", np.asarray(tvecs))
    s.write("newcameramtx", np.asarray(newcameramtx))
    s.write("mapx", np.asarray(mapx))
    s.write("mapy", np.asarray(mapy))

    print("Done calibration")
else:
    print("No chessboard found")
