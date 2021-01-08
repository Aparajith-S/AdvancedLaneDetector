# @file Transforms.py
# @brief methods to transform and correct image frames from the video
# @author Aparajith Sridharan
#         s.aparajith@live.com
# @date 29.12.2020
import cv2
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
#debug enables printing, image viewing for debugging
debug = 0

def transform_Perspective(img, points):
    # feed the points with order
    (bl, br, tr, tl) = points
    rect = np.array([tl, tr, br, bl], dtype = "float32")
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    maxWidth = img.shape[1]
    # compute the height of the new image
    maxHeight = img.shape[0]
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [100, 0],
        [maxWidth - 250, 0],
        [maxWidth - 250, maxHeight],
        [100, maxHeight]], dtype="float32")
    ''' 
        #uncomment this for challenge video
        dst = np.array([
            [100, 0],
            [maxWidth-150, 0],
            [maxWidth-150, maxHeight],
            [100, maxHeight]], dtype="float32")
    '''
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    Minv = cv2.getPerspectiveTransform(dst, rect)
    warpedimg = cv2.warpPerspective(img, M,(maxWidth,maxHeight))
    if debug==1:
        cv2.imshow("war",warpedimg)
    # return the warped image
    return Minv,warpedimg


def abs_sobel(img, orient='x', sobel_kernel=3):
    gray = img
    if (len(img.shape) > 2):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        return np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel))
    elif orient == 'y':
        return np.absolute(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel))
    return None


def mag_thresh(image, mag_thresh=(0, 255)):
    Sx = abs_sobel(image, sobel_kernel=3)
    Sy = abs_sobel(image, orient='y', sobel_kernel=3)
    gradmag = np.sqrt(Sx ** 2 + Sy ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255
    return binary_output

def dir_thresh(image):
    ga = cv2.GaussianBlur(image,(7,7),0)
    Sx = abs_sobel(ga, sobel_kernel=3)
    Sy = abs_sobel(ga, orient='y', sobel_kernel=3)
    angle = np.arctan(Sy,Sx)
    # Rescale to 8 bit
    anglebin = np.zeros_like(angle)
    anglebin[((angle > 1.558) &(angle < 2.0))]=255
    return anglebin

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

def autobrcrt(image, clip_hist_percent=5):
    cl = clahe(image)
    gray = cv2.cvtColor(cl, cv2.COLOR_RGB2GRAY)
    if debug==1:
        cv2.imshow("sauto",gray)
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    if debug==1:
        # Calculate new histogram with desired range and show histogram
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        #plt.show()
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

def fusionImage(img):
    # convert HLS
    g=autobrcrt(img)
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hslg = cv2.cvtColor(g, cv2.COLOR_RGB2HLS)
    # Split S
    S = hsl[:, :, 2]
    S2 = hslg[:, :, 2]
    # make binary of RGB image using sobel X-Y gradient
    #filteredimg = cv2.GaussianBlur(g, (5, 5),0)
    sobel_bin = mag_thresh(S, (120, 200))
    anglebin = dir_thresh(g)
    anglebin = cv2.medianBlur(anglebin,3)
    # make binary of fusion of S channel of HLS image with Magnitude of sobel XY
    S_bin = np.zeros_like(S)
    S_bin[(anglebin==255)|((S2 >= 100) & (S2 <= 255))|((S >= 130) & (S <= 255)) | ((sobel_bin > 150) & (sobel_bin <= 255))] = 255
    if debug == 1:
        cv2.imshow('debugging4', anglebin)
    return S_bin

#@brief: read the cal data and undistort the image.
def correctImage(img):
    pickfile = open('../calibration_data/wide_dist_pickle.p', "rb")
    camProp = pickle.load(pickfile)
    mtx = camProp['mtx']
    dist = camProp['dist']
    undistorted_img = cv2.undistort(img, mtx, dist)
    return img


def drawPoly(img, poly):
    # to create an image with black color.
    bg = np.zeros_like(img)
    # reshape to opencv format
    poly_new = poly.reshape((-1, 1, 2))
    cv2.polylines(bg, [poly_new], isClosed=True, color=(0, 255, 0), thickness=5)
    blended = cv2.addWeighted(src1=img, alpha=0.9, src2=bg, beta=0.9, gamma=0)
    return blended

#@brief The main pipeline for applying gradients and transforming images to perpective.
# returns the blended binarized image with ROI polygon and warped image with ROI polygon removed
def TransformationPipeline(img):
    S_binary2 = fusionImage(img)
    undistorted_fusion = correctImage(S_binary2)
    if debug == 1:
        cv2.imshow('undist_Fusion_binary', undistorted_fusion)
    # reverse process the final image to draw a polygon on it
    processedImg = cv2.cvtColor(undistorted_fusion, cv2.COLOR_GRAY2RGB)

    # Assigning vertices to polygon
    h = processedImg.shape[0]
    w = processedImg.shape[1]
    poly = np.array([[250, h], [w - 100, h], [(w / 2) + 55, h / 2 + 100], [(w / 2)-55, h / 2 + 100]],
                    dtype=np.int32)
    #for challenge_video uncomment this
    #poly = np.array([[250, h], [w - 200, h], [(w / 2) + 100, h / 2 + 130], [(w / 2) - 30 , h / 2 + 130]],
    #                dtype=np.int32)
    blended=[]
    if debug == 1:
        blended = drawPoly(processedImg, poly)
        cv2.imshow('blended_Fusion_binary', blended)
    Minv, warped = transform_Perspective(processedImg, poly)
    warped = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    return Minv, blended, warped


if __name__ == "__main__":
    # read a sample image
    if debug == 1:
        image = cv2.imread("../output_images/challenge.jpg")
        Minv, blended, warped = TransformationPipeline(image)
        cv2.imshow('blended_Fusion_binary', blended)
        cv2.imshow('new', warped)
        # write output image for documentation ( edited as per needs. not crucial to execution of project)
        #cv2.imwrite('../output_images/BinaryUndistortedturn.jpg', blended)
        #cv2.imwrite('../output_images/PerspectiveCorrectedturn.jpg', warped)
        #cv2.imwrite('../output_images/Fusion_binary.jpg', S_binary2)
        cv2.waitKey()
        cv2.destroyAllWindows()
