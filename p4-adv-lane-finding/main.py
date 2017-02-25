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
import cv2

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

    fig = plt.figure(figsize=(24, 18))

    i = 1
    i = displayImage(fig, i, img, 'Raw',None)

    # Undistort the image
    undistortedImg = undistorter.run(img, calibrator)
    misc.imsave('tmp/undistortedImg.jpg', undistortedImg)
    i = displayImage(fig, i, undistortedImg, 'Undistorted', 'gray')

    ## Thresholded image
    thresholdedImg = thresholder.getCombinedThreshold(undistortedImg)
    misc.imsave('tmp/thresholdedImg.jpg', thresholdedImg)
    i = displayImage(fig, i, thresholdedImg, 'Thresholded', 'gray')

    ## Transformed and Warped image
    warpedImg = perspectiveTransformer.getWarpedImg(thresholdedImg)
    misc.imsave('tmp/warpedImg.jpg', warpedImg)
    i = displayImage(fig, i, warpedImg, 'Warped', 'gray')

    ## Image fitted with polygon
    left_fit, right_fit = detectLanes.detect(warpedImg)
    finalImg = detectLanes.draw(undistortedImg, left_fit, right_fit,
                                perspectiveTransformer.Minv)
    misc.imsave('tmp/finalImg.jpg', finalImg)
    i=displayImage(fig, i, finalImg, 'Final')

    ## Measure the curvature
    lane_curve, pos = detectLanes.measureCurvature(finalImg)
    print ("Lane curvature :", lane_curve)

    cv2.putText(finalImg, "Radius of Curvature = {}(m)".format(lane_curve.round()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(255, 255, 255), thickness=2)

    cv2.putText(finalImg, "Vehicle is {}m left of center".format(abs(pos)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=2)

    misc.imsave('tmp/finalImgWithLaneCurvature.jpg', finalImg)
    displayImage(fig, i, finalImg, 'Lane Curvature')


    if debug == True :
        plt.show()

    return finalImg




def runVideo(vidFile, height, width) :

    cap = cv2.VideoCapture(vidFile)

    cap.get(5)  # to display frame rate of video
    # print cap.get(cv2.cv.CV_CAP_PROP_FPS)

    while (cap.isOpened()):
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        cv2.imshow('frame', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break

    cap.release()
    cv2.destroyAllWindows()



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
    outputVideo = 'output_videos/project_video_out.mp4'
    processVideoStrem(inputVideo,outputVideo)

    runVideo(outputVideo, 960, 540)
