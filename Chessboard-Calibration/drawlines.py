import math
import cv2
import numpy as np 

'''
This function calculates the total RMSE per line in an image. Lines are all vertical lines
@param[in]:     img                 source image to draw on
@param[in]:     error_mtx           an array to save the all the RMSE per line in the image
@param[in]:     corners             chessboard corners used to calculate RMSE per line
@param[in]:     point_error_matrix  each index is an array [x, y, error] where error is the distance from the point to the line plotted
@param[in]:     F                   (x, y, error) list where error is distance from line
@param[in]:     coordinates         (x, y) list 
@param[in]:     error_list          error list according to the above coordinates


@return:        img                 original source image with added line fit to represent errors
                error_mtx           appended error array
                F
                coordinates
                error_list

NOTE: In this function, start, end, or k modulo by 11 are also the row that the points belong to in the matrix.
        To compute the column, simply modulo by 11.  
'''


def vertical_extraction(img, error_mtx, corners, F, coordinates, error_list):
    for i in range(7):
        start = i*11
        end = start + 11

        start_r = start % 11
        start_c = math.floor(start/11)
        end_c = math.floor((end-1)/11)
        end_r = (end-1) % 11

        extract = corners[start:end:]

        [[xstart, ystart]] = extract[0]
        [[xend, yend]] = extract[-1]

        F.append([xstart, ystart, 0])
        coordinates.append([xstart, ystart])
        error_list.append(0)

        cv2.line(img, (xstart, ystart),
                 (xend, yend), (0, 0, 255), 1)
        cv2.line(img, (xstart, ystart), (xstart, ystart),
                 (0, 255, 255), 4)             # starting point
        sum_error = 0
        for j in range(1, len(extract) - 1):
            j_r = j % 11
            [[xk, yk]] = extract[j]
            dv = abs((yend - ystart) * xk - (xend - xstart) * yk + xend *
                     ystart - yend * xstart)/math.sqrt((yend - ystart)**2 + (xend - xstart)**2)
            cv2.line(img, (xk, yk),
                     (xk, yk), (51, 0, 51), 3)
            sum_error = sum_error + dv ** 2

            F.append([xk, yk, dv])
            coordinates.append([xk, yk])
            error_list.append(dv)

        ev = math.sqrt(sum_error/11)
        error_mtx.append(ev)

        F.append([xend, yend, 0])
        coordinates.append([xend, yend])
        error_list.append(0)

        cv2.putText(img, str(round(ev, 1)), (math.floor(xend), math.floor(
            yend+20)), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 255), 1)

    return (img, error_mtx, F, coordinates, error_list)


'''
This function calculates the total RMSE per line in an image. Lines are all horizontal lines
@param[in]:     img                 source image to draw on
@param[in]:     error_mtx           an array to save the all the RMSE per line in the image
@param[in]:     corners             chessboard corners used to calculate RMSE per line
@param[in]:     point_error_matrix  each index is an array [x, y, error] where error is the distance from the point to the line plotted
@param[in]:     F                   (x, y, error) list where error is distance from line
@param[in]:     coordinates         (x, y) list 
@param[in]:     error_list          error list according to the above coordinates


@return:        img                 original source image with added line fit to represent errors
                error_mtx           appended error array
                F
                coordinates
                error_list

NOTE: In this function, start, end, or k modulo by 11 are the row that the points belong to in the matrix.
        To compute the column, simply divide by 11.
'''


def horizontal_extraction(img, error_mtx, corners, F, coordinates, error_list):
    for i in range(11):
        start_h = i
        end_h = start_h + 67

        start_h_r = start_h % 11
        start_h_c = math.floor(start_h / 11)
        end_h_r = (end_h-1) % 11
        end_h_c = math.floor((end_h-1) / 11)

        extract = corners[start_h: end_h: 11]

        [[xstart, ystart]] = extract[0]
        [[xend, yend]] = extract[-1]

        F.append([xstart, ystart, 0])
        coordinates.append([xstart, ystart])
        error_list.append(0)

        cv2.line(img, (xstart, ystart),
                 (xend, yend), (255, 0, 0), 1)
        # starting point                  # end point
        cv2.line(img, (xstart, ystart), (xstart, ystart), (0, 255, 255), 4)
        sum_error = 0
        for j in range(1, len(extract) - 1):
            [[xk, yk]] = extract[j]
            # row is like row of start
            # column is j itself
            dh = abs((yend - ystart) * xk - (xend - xstart) * yk + xend *
                     ystart - yend * xstart)/math.sqrt((yend - ystart)**2 + (xend - xstart)**2)

            F.append([xk, yk, dh])
            coordinates.append([xk, yk])
            error_list.append(dh)

            sum_error = sum_error + dh ** 2
            cv2.line(img, (xk, yk),
                     (xk, yk), (204, 0, 204), 3)
        F.append([xend, yend, 0])
        coordinates.append([xend, yend])
        error_list.append(0)

        eh = math.sqrt(sum_error/7)
        # 11 corner points per line
        error_mtx.append(eh)
        cv2.putText(img, str(round(eh, 1)), (math.floor(xend + 20), math.floor(
            yend)), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
    return (img, error_mtx, F, coordinates, error_list)


'''
find epilines. Epilines corresponding to the points in first image is drawn on the second image
This function draw the lines from the array of lines on the images
'''


def drawlines(img1, img2, lines, pts1, pts2):
    row = img1.shape[0]
    column = img1.shape[1]

    # img1 = img1.astype('uint8')
    # img2 = img2.astype('uint8')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for row, pt1, pt2 in zip(lines, pts1, pts2):
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -row[2] / row[1]])
        x1, y1 = map(int, [column, -(row[2] + row[0] * column) / row[1]])
        print("(x0, y0): {:f}, {:f} \n (x1, y1):{:f}, {:f}".format(x0, y0, x1, y1))
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        
        img1 = cv2.circle(img1, tuple(pt1.flatten()), 3, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.flatten()), 3, color, -1)
    return img1, img2

'''
NOTE: Version not working
def drawlines(img1, img2, lines, pts1, pts2):
    row = img1.shape[0]
    column = img1.shape[1]
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    for row, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -row[2] / row[1] ])
        x1, y1 = map(int, [column, -(row[2] + row[0] * column) / row[1] ])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

'''