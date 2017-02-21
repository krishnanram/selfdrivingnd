import matplotlib.image as mpimage
import matplotlib.pyplot as plt

import glob

from Calibrator import Calibrator
from Undistorter import Undistorter
from Thresholder import Thresholder
from PerspectiveTransformer import PerspectiveTransformer
from DetectLanes import DetectLanes

from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc

debug = False

calibrator = Calibrator(True)
undistorter = Undistorter()
thresholder = Thresholder()
perspectiveTransformer = PerspectiveTransformer()
detectLanes = DetectLanes()


def displayImage(fig, i, img, title, cmap=None):

    a = fig.add_subplot(3, 3, i)
    plt.imshow(img, cmap)
    a.set_title(title)
    return i + 1


def processVideoStrem(inputVideo, outputVideo):

    clip1 = VideoFileClip(inputVideo).subclip(30, 51)
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(outputVideo, audio=False)

def pipeline(img):

    fig = plt.figure(figsize=(14, 12))

    i = 1
    i = displayImage(fig, i, img, 'Raw',None)

    # Undistort the image
    undistortedImg = undistorter.run(img, calibrator)
    misc.imsave('tmp/undistortedImg.jpg', undistortedImg)
    i = displayImage(fig, i, undistortedImg, 'Undistorted', 'gray')

    ## Thresholded image
    thresholdedImg = thresholder.getCombinedThreshold(undistortedImg)
    misc.imsave('tmp/thresholdedImg.jpg', img)
    i = displayImage(fig, i, thresholdedImg, 'Thresholded', 'gray')

    ## Transformed and Warped image
    warpedImg = perspectiveTransformer.getWarpedImg(thresholdedImg)
    misc.imsave('tmp/warpedImg.jpg', img)
    i = displayImage(fig, i, warpedImg, 'Warped', 'gray')

    ## Image fitted with polygon
    left_fit, right_fit = detectLanes.detect(warpedImg)
    finalImg = detectLanes.draw(undistortedImg, left_fit, right_fit,
                                perspectiveTransformer.Minv)
    misc.imsave('tmp/finalImg.jpg', finalImg)
    displayImage(fig, i, img, 'Final')

    ## Measure the curvature
    lane_curve, car_pos = detectLanes.measureCurvature(finalImg)
    print ("Lane curvature :", lane_curve)

    if debug == True :
        plt.show()

    return img

if __name__ == '__main__':

    ## Caliberate the camera
    calibrator.run()
    print ( calibrator.toString())

    ### First test the pipeline for some test image
    images = glob.glob('test_images/test1.jpg')
    for idx, fname in enumerate(images):
        img = mpimage.imread(fname)
        pipeline(img)


    ### Run the Video stream over and save the file

    inputVideo  = 'input_videos/project_video.mp4'
    outputVideo = 'ouput_videos/project_video_out1.mp4'
    processVideoStrem(inputVideo,outputVideo)

    #inputVideo  = 'input_videos/harder_challenge_video.mp4'
    #outputVideo = 'ouput_videos/harder_challenge_video_out.mp4'
    #processVideoStrem(inputVideo,outputVideo)