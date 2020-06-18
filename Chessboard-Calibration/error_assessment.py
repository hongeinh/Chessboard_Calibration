'''
This program is to assess error of the chessboard images in 13/05/2020
+ chessboard1: OK
+ chessboard2: pending
'''

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

from drawlines import vertical_extraction, horizontal_extraction, drawlines
from calibrateCamera import mapx, mapy, mtx, newcameramtx, dist, rvecs, tvecs

from ImageName import ImageName
'''
s = cv2.FileStorage(
    "/home/hongeinh/Downloads/chessboard/new_data/chess_1/distort_image/intrinsic_parameters.xml", cv2.FileStorage_READ)
mtx = s.getNode("mtx").mat()
dist = s.getNode("dist_coef").mat()
rvecs = s.getNode("rvecs").mat()
tvecs = s.getNode("tvecs").mat()
newcameramtx = s.getNode("newcameramtx").mat()
#mapx = s.getNode("mapx").mat()
#mapy = s.getNode("mapy").mat()
'''
from calibrateCamera import filename
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

row = 11
column = 7

objp = np.zeros((row * column, 3), np.float32)
objp[:, :2] = np.mgrid[0:row, 0:column].T.reshape(-1, 2)

# distort image reading
images = glob.glob( filename + "/distort_image/*.png")

objpoints = []
imgpoints = []
imgpoints_undistort = []

count = 0

# List of corner coordinates (x, y) of distort images
distort_coordinate_list = []
# List of corner coordinates (xcorrected, ycorrected) of undistort images
undistort_coordinate_list = []
# list of errors of corners in distort images
error_list_d = []
# list of errors of corners in undistort images
error_list_u = []
# list of (x, y, error) of corner in distort images
F_distort = []
# list of (x, y, error) of corner in undistort images
F_undistort = []

temp = []

# to write E and F
path3 = filename + "/epipolar_geometry.xml"

s = cv2.FileStorage(path3, cv2.FileStorage_WRITE)


for fname in images:
    img = cv2.imread(fname)
    img1 = cv2.imread(fname)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, (row, column), None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)


    # find chessboard corners
    if ret == True:
        count = count + 1
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

        error_d_h = []                  # total RMSE per horizontal line per image
        error_d_v = []                  # total RMSE per horizontal line per image
        (img, error_d_h, F, distort_coordinate_list, error_list_d) = horizontal_extraction(
            img, error_d_h, corners2, F_distort, distort_coordinate_list, error_list_d)
        (img, error_d_v, F, distort_coordinate_list, error_list_d) = vertical_extraction(
            img, error_d_v, corners2, F_distort, distort_coordinate_list, error_list_d)

        path = filename + "/drawlines/distort/distort" +  str(count) + ".png"

        cv2.imwrite(path, img)

        # UNDISTORT IMAGES
        d = cv2.remap(img1, mapx, mapy, cv2.INTER_LINEAR)
        mask = cv2.threshold(d, 195, 255, cv2.THRESH_BINARY)[1][:,:,0]
        dst = cv2.inpaint(d, mask, 7, cv2.INPAINT_NS)
        corn = cv2.undistortPoints(corners2, mtx, dist, R=None, P=newcameramtx)

        corners_homogenous = cv2.convertPointsToHomogeneous(corners2)
        corn_homogenous = cv2.convertPointsToHomogeneous(corn)
        if(count == 1):
            temp = corn
            img_temp = dst.copy()
        else:
            # FIND FUNDAMENTAL MATRIX AND ESSENTIAL MATRIX
            pts1 = temp
            pts2 = corn

            img_pts1 = img_temp
            img_pts2 = dst.copy()

            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

            lines1 = cv2.computeCorrespondEpilines(
                pts2.reshape(-1, 1, 2), 2, F)
            lines1 = lines1.reshape(-1, 3)
            img5, img6 = drawlines(img_pts1, img_pts2, lines1, pts1, pts2)

            lines2 = cv2.computeCorrespondEpilines(
                pts1.reshape(-1, 1, 2), 2, F)
            lines2 = lines2.reshape(-1, 3)
            img3, img4 = drawlines(img_pts2, img_pts1, lines2, pts2, pts1)

            path1 = filename + "/feature_matching/test_" +  str(count-1) + ".png"
            path2 = filename + "/feature_matching/test_" +  str(count) + ".png"

            cv2.imwrite(path1, img5)
            cv2.imwrite(path2, img3)

            F = np.asarray(F)
            E = mtx.T.dot(F)
            E = E.dot(mtx)

            s.write("F" + str(count), F)
            s.write("E" + str(count), E)

            # found, rv, tv = cv2.solvePnP(objp, corners, mtx, dist, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE )
            # print("Camera calib rvecs: \n", rvecs[count-1])
            # print("SolvePnP rvecs: \n", rv)
            # print("Camera calib tvecs: \n", tvecs[count-1])
            # print("SolvePnP rvecs: \n", tv)
            # print("-------------------------------------------------")
            temp = corn
            img_temp = dst.copy()

        imgpoints_undistort.append(corn)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_d= corners_homogenous.T[0]
        y_d = corners_homogenous.T[1]
        z_d = corners_homogenous.T[2]
        #ax.plot_wireframe(x_d.flatten(), z_d.flatten(), y_d.flatten(), rstride=10, cstride=10)
        ax.plot(x_d.flatten(), z_d.flatten(), y_d.flatten(), '.')
        ax.invert_zaxis()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')
        
        plt.savefig(filename + "/reconstruction/distort/img" + str(count) +".png")


        plt.clf()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = corn_homogenous.T[0]
        y = corn_homogenous.T[1]
        z = corn_homogenous.T[2]
        ax.plot(x.flatten(), z.flatten(), y.flatten(), '.')
        ax.invert_zaxis()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')
    
        plt.savefig(filename + "/reconstruction/undistort/img" + str(count) +".png")

        

        error_u_v = []
        error_u_h = []
        (dst, error_u_h, F_undistort, undistort_coordinate_list, error_list_u) = horizontal_extraction(
            dst, error_u_h, corn, F_undistort, undistort_coordinate_list, error_list_u)
        (dst, error_u_v, F_undistort, undistort_coordinate_list, error_list_u) = vertical_extraction(
            dst, error_u_v, corn, F_undistort, undistort_coordinate_list, error_list_u)

        path = filename + "/drawlines/undistort/undistort" + str(count) + ".png"

        cv2.imwrite(path, dst)

        # plot error
        plt.rcParams["patch.force_edgecolor"] = True
        fig, ax = plt.subplots(2, 2)
        ax[0][0].bar(range(len(error_d_v)), error_d_v)
        ax[0][0].set_title("Distort vertical")

        ax[1][0].bar(range(len(error_d_h)), error_d_h)
        ax[1][0].set_title("Distort horizontal")

        ax[0][1].bar(range(len(error_u_v)), error_u_v)
        ax[0][1].set_title("Undistort vertical")

        ax[1][1].bar(range(len(error_u_h)), error_u_h)
        ax[1][1].set_title("Undistort horizontal")

        for a in ax.flat:
            a.set(xlabel='Line', ylabel='RMSE')

        fig.tight_layout(pad=4.0)
        path = filename +  "/RMSE_figure/Error" + str(count)  + ".png"

        plt.savefig(path)
        plt.clf()

# save (x, y, error) to txt file
np.savetxt(filename + "/point_error_distort.txt", np.asarray(F_distort), fmt='%f', newline="\r\n")
np.savetxt(filename + "/point_error_undistort.txt", np.asarray(F_undistort), fmt='%f', newline="\r\n")

# INTERPOLATE---------------------------------------------------
# Distort
# Remember to change mgrid according to image shape
grid_x, grid_y = np.mgrid[0:1380:1000j, 0:1088:1000j]

grid_z0 = griddata(distort_coordinate_list, error_list_d,
                   (grid_x, grid_y), method='nearest')
grid_z1 = griddata(distort_coordinate_list, error_list_d,
                   (grid_x, grid_y), method='linear')
grid_z2 = griddata(distort_coordinate_list, error_list_d,
                   (grid_x, grid_y), method='cubic')

plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0, 1380, 0, 1088), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1.T,extent=(0, 1380, 0, 1088), origin='lower')
plt.title('Linear')
plt.subplot(224)
plt.imshow(grid_z2.T,extent=(0, 1380, 0, 1088), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(10, 10)
plt.savefig(filename + "/distort.png")
plt.clf()

# Undistort
grid_z0 = griddata(undistort_coordinate_list, error_list_u,
                   (grid_x, grid_y), method='nearest')
grid_z1 = griddata(undistort_coordinate_list, error_list_u,
                   (grid_x, grid_y), method='linear')
grid_z2 = griddata(undistort_coordinate_list, error_list_u,
                   (grid_x, grid_y), method='cubic')

plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0, 1380, 0, 1088), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1.T, extent=(0, 1380, 0, 1088), origin='lower')
plt.title('Linear')
plt.subplot(224)
plt.imshow(grid_z2.T, extent=(0, 1380, 0, 1088), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(10, 10)
plt.savefig(filename + "/undistort.png")
plt.clf()
