import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Thresholder:

    def __init__(self):

        self.sobel_kernel = 15

        self.thresh_dir_min = 0.7
        self.thresh_dir_max = 1.2

        self.thresh_mag_min = 50
        self.thresh_mag_max = 255

    def applyDirectionThreshold(self, sobelx, sobely):

        print ("Inside Thresholder:applyDirectionThreshold() ")

        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.thresh_dir_min) & (scaled_sobel <= self.thresh_dir_max)] = 1

        return sxbinary

    def applyMangnitudeThreshold(self, sobelx, sobely):

        print ("Inside Thresholder:applyMangnitudeThreshold() ")

        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= self.thresh_mag_min) & (gradmag <= self.thresh_mag_max)] = 1

        return binary_output

    def applyColorThreshold(self, img):

        print ("Inside Thresholder:applyColorThreshold() ")

        '''
        From tutorial, we can see that
          color space like HLS is robust
          S channel is probably  best bet,
            given that it's cleaner than the H channel and
            doing a bit better than the R channel or
            simple grayscaling.

            the S channel is still doing a fairly robust job of picking up the lines under
            very different color and contrast conditions,
            while the other selections look messy.


        '''

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        yellow_min = np.array([15, 100, 120], np.uint8)
        yellow_max = np.array([80, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(img, yellow_min, yellow_max)

        white_min = np.array([0, 0, 200], np.uint8)
        white_max = np.array([255, 30, 255], np.uint8)
        white_mask = cv2.inRange(img, white_min, white_max)

        binary_output = np.zeros_like(img[:, :, 0])
        binary_output[((yellow_mask != 0) | (white_mask != 0))] = 1

        filtered = img
        filtered[((yellow_mask == 0) & (white_mask == 0))] = 0

        return binary_output

    def getCombinedThreshold(self, img):

        print ("Inside Thresholder:getCombinedThreshold() ")

        ### main funciton that applies direction threshold, magnitude threshold followed
        ### by color threshold

        sobelx = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)

        direc = self.applyDirectionThreshold(sobelx, sobely)
        mag = self.applyMangnitudeThreshold(sobelx, sobely)
        color = self.applyColorThreshold(img)

        combined = np.zeros_like(direc)
        combined[((color == 1) & ((mag == 1) | (direc == 1)))] = 1

        return combined




def testColorThreshold() :

    image = mpimg.imread('test6.jpg')
    thresh = (180, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1


if __name__ == '__main__':
    testColorThreshold()