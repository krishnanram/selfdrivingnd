## Advanced Lane Finding


The objectivce of this project is to display the lane boundaries and numerical estimation of lane curvature and vehicle position.


Advanced_lane_finding.py implements

* Camera calibration matrix and distortion coefficients given a set of chessboard images.
* Distortion correction to the raw image.
* Create a thresholded binary image using color transforms, gradients, etc.,
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


How to run the program

./documents/Advanced_Lane_Finding.docx discusses the directory structure and also how to
run the program in debug or in normal mode. Please refer to this document.

Folder Strucure

* The images for camera calibration are stored in the folder called `camera_cal`.
* The images in `test_images` are for testing your pipeline on single frames.
* videos filder contains test video stream

