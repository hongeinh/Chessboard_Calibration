'''
This file aims to find the disparity map between 2 pictures 
'''
import cv2

filename = "/home/hongeinh/Downloads/chessboard/new_data/test"
# find disparity map ----> FOUND. DO NOT DELETE
imgL = cv2.imread(filename + "/undistort_image/Calibresult237.png", 0)
imgR = cv2.imread(filename + "/undistort_image/Calibresult238.png", 0)

win_size = 5
min_disp = -1
max_disp = 63  # min_disp * 9
num_disp = max_disp - min_disp          # Needs to be divisible by 16

stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=5,
                               uniquenessRatio=5,
                               speckleWindowSize=5,
                               speckleRange=5,
                               disp12MaxDiff=1,
                               P1=8*3*win_size**2,  # 8*3*win_size**2,
                               P2=32*3*win_size**2)  # 32*3*win_size**2)
#disparity = stereo.compute(imgL_downsampled, imgR_downsampled)
disparity = stereo.compute(imgL, imgR)

cv2.imwrite(filename + "/feature_matching/disparity.png", disparity)

'''
#Generate  point cloud. 
print ("\nGenerating the 3D map...")
#Get new downsampled width and height 
h,w = imgL.shape[:2]


#Load focal length. 
# This parameter's value is made up for the sake of the algorithm
focal_length = 38.1   
                  # milimet
#Perspective transformation matrix
#This transformation matrix is from the openCV documentation
Q = np.float32([[1, 0, 0, -w/2. 0],
    [0, -1, 0, h/2. 0],
    [0, 0, 0, -focal_length],
    [0, 0, 1, 0]])
#This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
#Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q2 = np.float32([[1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, focal_length * 0.05, 0], #Focal length multiplication obtained experimentally. 
    [0, 0, 0, 1]])
#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity, Q2)
#Get color points
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
#Get rid of points with value 0 (i.e no depth)
mask_map = disparity > disparity.min()
#Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = colors[mask_map]
'''


'''
This function downsamples image x number of times
'''
'''
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

imgL_downsampled = downsample_image(imgL,1)
imgR_downsampled = downsample_image(imgR,1)
'''
