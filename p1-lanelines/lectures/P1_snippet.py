
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from lectures.drawLib import *

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_line5(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1.0, λ=0.0):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def showImage(image) :

    plt.imshow(image)  # call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    plt.show(block=False)
    #plt.waitforbuttonpress()


def process_image(image):

    print ("Inside.....")
    grayImg = grayscale(image)
    showImage(grayImg)

    print("Inside.....")
    blurredImg = gaussian_blur(grayImg, 3)
    showImage(blurredImg)

    print("Inside.....")
    canImg = canny(blurredImg, 100, 200)
    showImage(canImg)

    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]],
                        dtype=np.int32)
    masked_edges = region_of_interest(canImg, vertices)

    rho = 1
    theta = np.pi / 180
    threshold = 35
    min_line_length = 40
    max_line_gap = 20

    lineImg = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    showImage(lineImg)

    weightedImg = weighted_img(lineImg, image)
    showImage(weightedImg)

    return weightedImg



if __name__ == "__main__":


    os.listdir("test_images/")
    image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
    print('This image is:', type(image), 'with dimesions:', image.shape)
    #showImage(image)
    process_image(image)


    '''

    white_output = 'test_images/white.mp4'
    clip1 = VideoFileClip("test_images/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    HTML("""<video width="960" height="540" controls><source src="{0}"></video>""".format(white_output))

    yellow_output = 'test_images/yellow.mp4'
    clip2 = VideoFileClip('test_images/solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')
    HTML("""<video width="960" height="540" controls> <source src="{0}"> </video>""".format(yellow_output))

    challenge_output = 'test_images/extra.mp4'
    clip2 = VideoFileClip('test_images/challenge.mp4')
    challenge_clip = clip2.fl_image(process_image)
    get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')
    HTML("""<video width="960" height="540" controls><source src="{0}"></video>""".format(challenge_output))

    '''


