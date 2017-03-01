# Vehicle Detection

**Project Description**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


** Documents **

./documents/Vechile Detection.pdf provides instructions, directory structure and descriptions of the project details.


**How to Run  **

python main.py executes

	- Explore.py to explore the input data
	- Train.py trains using SVC and creates the model in ./model directory
	- Predict/Identify the cars and objects using Identify.py
	

** logs **
	
	I have captured the log and saved in logs directory

** Output **

	project_video_output.mp4
