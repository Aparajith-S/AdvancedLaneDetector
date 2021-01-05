## Advanced Lane Finding
Author:  Aparajith Sridharan(s.aparajith@live.com)  
date: 26/12/2020

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video. 
This README takes the reader through the steps used to develop this project.  

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

# Code

The code was developed in an IDE and not in a jupytr notebook file due to the limitations on `cv2.imshow(...)`. Hence, the code will be found in a folder named `src`  

### Point 1 : Calibration
`Calibration.py` contains the methods used to calibrate the camera using the images in the folder `camera_cal`
 - `cal_dump()` method will be the main method used from this file. this uses the functions `cal_blind(...)` and `cal_mtx(...)` to evaluate the images in `camera_cal` folder and compute the camera matrix and the distortion coefficients. 
   Furthermore, the function will dump the matrix and distortion coefficient for later use in `calibration_data` folder.
 - running this file as a __main__ file will show one example of distortion correction as well as save the source and destination test image in the `calibration_data` folder.  

distorted image | undistorted image  
:-------------: | :---------------:  
![distorted](output_images/distorted.jpg) | ![distorted](output_images/undistorted.jpg) 

### Point 2 : Color spaces, Gradients and Thresholding

`Transforms.py` contains the functions for transformation steps of the images.
 - For the images four types of color spaces were analyzed including the individual R G and B channels that was already covered in depth in the lectures.
   The three other channels are HSV , HLS, YUV*. YUV is the additional channel which was analyzed in addition to the lectures.
         
   YUV was interesting to experiment on because the color space of choice used in the analog television industry follows a unique way of transmitting luminance into Y (brightness) 
   U and V were made from the differences from B and R with Y signals respectively:
   
         Y = 0.299R + 0.587G + 0.114B
         U'= (B-Y)*0.565
         V'= (R-Y)*0.713
   ---
         YUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
         U = YUV[:, :, 1]
         binary = np.zeros_like(U)
         binary[(U > 10) & (U <= 127)] = 255

image used|U-channel | threshold on U-channel  
:-------------:|:-------------: | :---------------:  
![distorted](test_images/straight_lines2.jpg)|![distorted](output_images/ChromaU_Channel_YUV.jpg) | ![distorted](output_images/binarized_ChromaU_Channel_YUV.jpg)
![distorted](test_images/straight_lines1.jpg)|![distorted](output_images/ChromaU_Channel_YUV_st1.jpg) | ![distorted](output_images/binarized_ChromaU_Channel_YUV_st1.jpg) 

As observed, the idea of using YUV color space alone was not particularly great as it worked well for White lines and also could identify Yellow lines but only differently which meant thresholding them would be challenging. 
Hence, Saturation in HLS was tried and...

image used|S-channel | threshold on S-channel  
:-------------:|:-------------: | :---------------:  
![distorted](test_images/test4.jpg)|![distorted](output_images/ChromaU_Channel_HSL_t4.jpg) | ![distorted](output_images/binarized_ChromaU_Channel_HSL_t4.jpg)
![distorted](test_images/straight_lines2.jpg)|![distorted](output_images/S_Channel_HSL_st2.jpg) | ![distorted](output_images/binarized_S_Channel_HSL_st2.jpg)

... it works well consistently with yellow or white lane lines and fairly well under noisy conditions .

#### Sensor fusion:
To improve the binary output a "sensor fusion" approach is taken where magnitude of sobel X and Y together is taken on the grayscale image, thresholded with a different threshold and added with the S-Channel binarized image. 
Thus, providing absent lane details in the S-channel farther away from the Ego vehicle as shown below:

S-channel|Sobel-xy magnitude | Fusioned output  
:-------------:|:-------------: | :---------------:  
![distorted](output_images/S_Channel_binary.jpg)|![distorted](output_images/SobelXYMag_Binary.jpg) | ![distorted](output_images/Fusion_binary.jpg)

### Point 3: Distortion correction and Perspective transform
The following steps were done:
- undistort the binarized image with the previously calculated camera matrix and distortion coefficients. 
- perform a perspective transform. Below illustration shows the output of a straight line lanes, turning lanes. 

Distortion corrected image  (with ROI for illustration) | Perspective transform  
:-------------:|:-------------:   
![pers1](output_images/BinaryUndistorted1.jpg)|![pers2](output_images/PerspectiveCorrected1.jpg)
![pers3](output_images/BinaryUndistortedturn.jpg)|![pers4](output_images/PerspectiveCorrectedturn.jpg)

### Point 4: Lane scanning using windowing method and Polynomial fitting
The steps that are carried after perspective transform is to calculate curvature using two methods.
- first for an initial run to find the lane lines, a polynomial fitting is done by using teh histogram and scanning the likelihood of lanes in the histogram
- to improve on performance, the initially found left and right lane polynomial coefficients would be used as a ballpark to find successive lanes when a video is run frame by frame to detect lanes.  

input image | perspective transformed image    
:-------------:|:-------------:
![pers1](output_images/test4input.jpg)|![pers2](output_images/test4output.jpg)

polynomial fitting using sliding window| Polynomial fitting using fast scanning with known polynomial  
:-------------:|:-------------:
![pers2](output_images/test4polynomfit.png)|![pers2](output_images/test4fpolyfit.png)

- the pipeline is developed in a way so that the scaling of the radius of curvature is done on the polynomial equation  

      x = A(y^2) + By + C 

- substituting for Xr = (1/M_x)*Xpixels and Yr = (1/M_y )*Ypixels  

   
      Xr = (M_x/(M_y^2))*A*Yr + (M_x/M_y)*B*Yr + C  
where, M_x and M_y are meters per pixels scaling factors and Xr, Yr are the real world X and Y coordinates. 

- the factors M_x is taken to be as 3.7 / 600 as seen in the polynomial fitting, the lanes in x direction is around 600 px apart
    M_y is retained at 30 / 720 as it is 720 px long.
- with this setup, curve radius of 529.55 m and 218.85 m were computed. It is obvious from the image as the curve on the right is sharper than on the left. So, the right hand curve has a much smaller radius.     

