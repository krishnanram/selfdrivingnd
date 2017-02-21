import glob

import cv2
import numpy as np


class Calibrator :

    def __init__(self, loadFlag=True):

        self.objpoints = None
        self.imgpoints = None
        self.shape     = None

        self.mtx   = None
        self.dist  = None
        self.rvecs = None
        self.tvecs = None

        self.loadFlag = loadFlag


    def run(self):

        if self.loadFlag == True:
            self.load()
        else :
            self.calibrate()

        ret, self.mtx, self.dist, self.rvecs, self.tvecs \
            = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.shape, None, None)

    def calibrate(self):

        images = glob.glob('camera_cal/calibration*.jpg')
        base_objp = np.zeros((6 * 9, 3), np.float32)
        base_objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        self.objpoints = []
        self.imgpoints = []
        self.shape = None

        for imname in images:
            img = cv2.imread(imname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.shape is None:
                self.shape = gray.shape[::-1]

            print('Finding chessboard corners on {}'.format(imname))
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if ret:
                self.objpoints.append(base_objp)
                self.imgpoints.append(corners)

        if self.loadFlag == False :
            self.save()


    def save(self):

        print ("Inside Caliberator savaing")
        np.save('data/objpoints', self.objpoints)
        np.save('data/imgpoints', self.imgpoints)
        np.save('data/shape', self.shape)


    def load(self):

        print ("Inside Caliberator load")
        try:
            self.objpoints = np.load('data/objpoints.npy')
            self.imgpoints = np.load('data/imgpoints.npy')
            self.shape = tuple(np.load('data/shape.npy'))
        except:
            self.objpoints = None
            self.imgpoints = None
            self.shape = None



    def toString(self):

        print ("Mtx:", self.mtx)
