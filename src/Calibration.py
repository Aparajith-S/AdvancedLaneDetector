import cv2
import numpy as np
import glob
import pickle
from os import path,mkdir

debug = 0
# @brief a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the camera matrix
def cal_mtx(objpoints, imgpoints,img_size):
    ret, mtx, dist, rvecs, tvecs=cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)
    return mtx, dist

# @brief test function to un-distort an image with the calibrated camera matrix
# and distortion coefficient
# returns the corrected image
def cal_test(img,mtx,dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

'''
get calibration object and image points.
'''
def cal_blind():
    '''
    performs calibration blindly on distorted camera images and returns the object image points
    :return: object points and image points
    '''
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        img_size=(img.shape[1],img.shape[0])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if debug == 1:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)
    if debug == 1:
        cv2.destroyAllWindows()
    return objpoints,imgpoints,img_size

#@brief dump the camera calibration matrix and dist coeff in a file
# for future use.
#
def cal_dump():
    objp, imgp, img_size = cal_blind()
    mtx, dist = cal_mtx(objp, imgp, img_size)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    if not path.exists("../calibration_data"):
        mkdir("../calibration_data")
    pickle.dump(dist_pickle, open("../calibration_data/wide_dist_pickle.p", "wb"))

if __name__ == "__main__":
   objp,imgp,img_size=cal_blind()
   mtx,dist=cal_mtx(objp,imgp,img_size)
   img = cv2.imread("../camera_cal/calibration1.jpg")
   dst=cal_test(img,mtx,dist)
   cv2.imwrite("../output_images/distorted.jpg", img)
   cv2.imwrite("../output_images/undistorted.jpg",dst)
   cv2.imshow('distorted', img)
   cv2.imshow('undistorted',dst)
   cv2.waitKey()
   cv2.destroyAllWindows()
   cal_dump()