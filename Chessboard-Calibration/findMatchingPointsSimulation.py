import cv2
import numpy as np 
import glob
from drawlines import drawlines



filename = "/home/hongeinh/Downloads/chessboard/new_data/13-5-2020simulation"
images = sorted(glob.glob(filename + "/undistort_image/*.png"))
img1 = cv2.imread(filename + "/undistort_image/simulation_calib1.png")
count = 0


MIN_MATCH_COUNT = 20

'''
This part reads all the images in the folder determined
Find the matching points and F matrix in each pair of consecutive pictures
'''
for fname in images:
    count += 1
    img = cv2.imread(fname)
    img2 = img.copy()
    if(count != 1):
        
        #Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # Find key points and compute descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)    # find keypoints and compute descriptors
        kp2, des2 = sift.detectAndCompute(img2,None)

        # Match descriptor vectors with FLANN based matcher
        FLANN_INDEX_KDTREE =  1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches
        good = []
        pts1 = []
        pts2 = []
        temp = []
        # distance ratio test to eliminate false match
        ratio_threshold = 0.8
        for i, (m, n) in enumerate(matches):
            if m.distance < ratio_threshold * n.distance:       # 0.8 is treshold --> k nearest neighbours
                good.append(m)
                pts2.append( kp2[m.trainIdx].pt)
                pts1.append( kp1[m.queryIdx].pt)

        # Find fundamental matrix
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        
        # select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
   
        # find epilines corresponding to points in right image (second image) 
        # and draw its line on left image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

        # find epilines corresponding to points in left image
        # and draw its line on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

        cv2.imwrite(filename + "/feature_matching/epilines_" + str(count-1) + "_" + str(count) + ".png", img5)
        cv2.imwrite(filename + "/feature_matching/epilines_" + str(count) + "_" + str(count-1) + ".png", img3)

        '''
        This part below is to write down the fundamental matrix between images
        '''
        path3 = filename + "/feature_matching/matrix.fxml"
        s = cv2.FileStorage(path3, cv2.FileStorage_WRITE)
        s.write("fundmatrix" + "_" + str(count-1) + "_" + str(count),  F)
    img1 = img.copy()
    
# fs = cv2.FileStorage("/home/hongeinh/Downloads/chessboard/feature_matching/intrinsic_parameters_polyp.xml", cv2.FILE_STORAGE_READ)
# mtx = fs.getNode("mtx").mat()
# print("mtx\n", mtx)

# mtx = np.asarray(mtx)   # E = M.T * F * M
# F = np.asarray(F)
# E = mtx.T.dot(F)
# E = E.dot(mtx)
# print("E\n", E)