import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def showImage(image) :
    plt.imshow(image)  # call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    #plt.show(block=False)
    #plt.waitforbuttonpress()
    plt.draw()
    plt.waitforbuttonpress()

# Read in and grayscale the image
# Note: in the previous example we were reading a .jpg
# Here we read a .png and convert to 0,255 bytescale
image = mpimg.imread('../test_images/solidYellowCurve.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

print ("Showing gray Image...")
showImage(gray)


# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
print ("Showing Blur Gray Image...")
showImage(blur_gray)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
print ("Showing Canny Edges Image...")
showImage(edges)


# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color = 255

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)

#vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)

cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

print ("Showing Masked Edges Image...")
showImage(masked_edges)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

print ("Showing Lined image Image...")
showImage(line_image)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))
print ("Showing Colored Edges Image...")
showImage(color_edges)

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
print ("Showing Lined Edges Image...")
showImage(lines_edges)


