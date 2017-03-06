debug = False

import pickle
from collections import deque

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from Utils import *
from Explore import *
from Train import *
from Identify import *
import os

VEHICLES_DIR        = "./vehicles/"
NON_VEHICLES_DIR    = "./non-vehicles/"
IMAGES_DIR          = "./images/"
DATA_DIR            = "./data/"
TEST_IMAGES_DIR     = "./testimages/"
OUTPUT_IMAGES       = "./output_images/"
HEATMAP             = "./heatmap/"
VIDEOS              = "./videos/"



def displayImage(fig, i, img, title, cmap=None):

    a = fig.add_subplot(3, 3, i)
    plt.imshow(img, cmap)
    a.set_title(title)
    return i + 1


def processVideoStrem(inputVideo, outputVideo):

    clip1 = VideoFileClip(inputVideo).subclip(30, 51)
    #white_clip = clip1.fl_image(process_image)
    #white_clip.write_videofile(outputVideo, audio=False)



def pipeline(img):
    return img


def runVideo(vidFile, height, width) :

    cap = cv2.VideoCapture(vidFile)
    cap.get(3)  # to display frame rate of video
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break


if __name__ == '__main__':


    explore     = Explore()
    train       = Train()

    print (" Explore the data ...")
    explore.run()

    print (" Model/Classify using training data...")
    train.model()

    os._exit(1)
    model = train.getModel()
    identify    = Identify(train)

    print (" Process the video stream ...")
    # Project Video
    video_output = 'project_video_output.mp4'
    clip1 = VideoFileClip(VIDEOS+"project_video.mp4")

    t0 = time.time()
    clip_output = clip1.fl_image(identify.pipeline)
    clip_output.write_videofile(video_output, audio=False)

    print("Processing Time: ", round(time.time() - t0, 2))
