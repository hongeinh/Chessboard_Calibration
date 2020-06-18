'''
This file aims to find the disparity map between 2 pictures 
'''
import cv2
import glob

filename = "/home/hongeinh/Downloads/chessboard/new_data/test"

win_size = 5
min_disp = -1
max_disp = 63                                               # min_disp * 9
num_disp = max_disp - min_disp                              # Needs to be divisible by 16

stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=5,
                               uniquenessRatio=5,
                               speckleWindowSize=5,
                               speckleRange=5,
                               disp12MaxDiff=1,
                               P1=8*3*win_size**2,          # 8*3*win_size**2,
                               P2=32*3*win_size**2)         # 32*3*win_size**2)
#disparity = stereo.compute(imgL_downsampled, imgR_downsampled)
count = 1
temp = []
imgL = []
imgR = []
images = sorted(glob.glob(filename + "/undistort_image/*.png"))
for fname in images:
    img = cv2.imread(fname)
    if(count == 1):
        imgL = img.copy()
    else:
        print(count)
        imgR = img.copy()
        disparity = stereo.compute(imgL, imgR)
        cv2.imwrite(filename + "/feature_matching/disparityMap/disparity" + str(count) + ".png", disparity)
        imgL = img.copy()
    count = count + 1
