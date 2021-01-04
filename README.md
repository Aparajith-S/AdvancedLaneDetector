## Advanced Lane Finding
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


To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

