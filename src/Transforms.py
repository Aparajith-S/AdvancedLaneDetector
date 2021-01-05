import cv2
import numpy as np
import pickle


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
        [400, 0],
        [maxWidth-200, 0],
        [maxWidth-200, maxHeight],
        [400, maxHeight]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warpedimg = cv2.warpPerspective(img, M,(maxWidth,maxHeight))
    # return the warped image
    return warpedimg


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


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate gradient direction
    # Apply threshold
    dir_binary = None
    return dir_binary


def fusionImage(img):
    # convert HLS
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Split S
    S = hsl[:, :, 2]
    # make binary of RGB image using sobel X-Y gradient
    sobel_bin = mag_thresh(img, (120, 200))
    # make binary of fusion of S channel of HLS image with Magnitude of sobel XY
    S_bin = np.zeros_like(S)
    S_bin[((S >= 130) & (S < 255)) | ((sobel_bin >= 130) & (sobel_bin < 255))] = 255
    return S_bin


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
    cv2.imshow('undist_Fusion_binary', undistorted_fusion)
    # reverse process the final image to draw a polygon on it
    processedImg = cv2.cvtColor(undistorted_fusion, cv2.COLOR_GRAY2RGB)

    # Assigning vertices to polygon
    h = processedImg.shape[0]
    w = processedImg.shape[1]
    # poly=np.array([[h,400],[h,w-200],[h/2,w/2-100],[h/2,w/2+100]],dtype=np.int32)
    poly = np.array([[150, h], [w - 100, h], [(w / 2) + 37, h / 2 + 80], [(w / 2) - 37, h / 2 + 80]],
                    dtype=np.int32)
    blended = drawPoly(processedImg, poly)
    warped = transform_Perspective(processedImg, poly)
    warped = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    return blended, warped


if __name__ == "__main__":
    # read a sample image
    image = cv2.imread("../test_images/straight_lines2.jpg")
    blended, warped = TransformationPipeline(image)
    cv2.imshow('blended_Fusion_binary', blended)
    cv2.imshow('new', warped)
    # write output image for documentation ( edited as per needs. not crucial to execution of project)
    #cv2.imwrite('../output_images/BinaryUndistortedturn.jpg', blended)
    #cv2.imwrite('../output_images/PerspectiveCorrectedturn.jpg', warped)
    #cv2.imwrite('../output_images/Fusion_binary.jpg', S_binary2)
    cv2.waitKey()
    cv2.destroyAllWindows()
