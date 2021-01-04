import cv2
import numpy as np
import pickle

def transform_Perspective(img):
    pass

def abs_sobel(img, orient='x', sobel_kernel=3):
    gray=img
    if(len(img.shape)>2):
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        return np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel))
    elif orient == 'y':
        return np.absolute(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel))
    return None

def mag_thresh(image,mag_thresh=(0, 255)):
    Sx = abs_sobel(image,sobel_kernel=3)
    Sy = abs_sobel(image, orient='y', sobel_kernel=3)
    gradmag = np.sqrt(Sx**2 + Sy**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    dir_binary = None
    return dir_binary

if __name__ == "__main__":
    #read a sample image
    img = cv2.imread("../test_images/straight_lines2.jpg")
    #convert HLS
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #Split S
    S = hsl[:, :, 2]
    #make binary of RGB image using sobel X-Y gradient
    sobel_binary = mag_thresh(img,(120,255))
    cv2.imshow('SobelXYMag_Binary', sobel_binary)
    # make binary of S channel of HLS image
    S_binary1 = np.zeros_like(S)
    S_binary1[(S >= 150)] = 255
    cv2.imshow('S_Channel_binary', S_binary1)
    # make binary of fusion of S channel of HLS image with Magnitude of sobel XY
    S_binary2 = np.zeros_like(S)
    S_binary2[(S >= 150) | (sobel_binary >= 120)] = 255
    cv2.imshow('Fusion_binary', S_binary2)

    #write output image for documentation
    cv2.imwrite('../output_images/S_Channel_binary.jpg', S_binary1)
    cv2.imwrite('../output_images/SobelXYMag_Binary.jpg', sobel_binary)
    cv2.imwrite('../output_images/Fusion_binary.jpg', S_binary2)
    cv2.waitKey()
    cv2.destroyAllWindows()