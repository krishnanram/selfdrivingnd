import glob

import cv2
import numpy as np


class Undistorter:

    def __init__(self):
        print ("")

    def run(self, img, caliberator):

        return cv2.undistort(img, caliberator.mtx, caliberator.dist, None, caliberator.mtx)
