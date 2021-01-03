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

`Calibration.py` contains the methods used to calibrate the camera using the images in the folder `camera_cal`
 - `cal_dump()` method will be the main method used from this file. this uses the functions `cal_blind(...)` and `cal_mtx(...)` to evaluate the images in `camera_cal` folder and compute the camera matrix and the distortion coefficients. 
   Furthermore, the function will dump the matrix and distortion coefficient for later use in `calibration_data` folder.
 - running this file as a __main__ file will show one example of distortion correction as well as save the source and destination test image in the `calibration_data` folder.  

distorted image | undistorted image  
:-------------: | :---------------:  
![distorted](calibration_data/distorted.jpg) | ![distorted](calibration_data/undistorted.jpg) 



To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

