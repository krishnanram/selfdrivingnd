import cv2
import glob
import numpy as np
from math import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import collections
from itertools import chain, repeat
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from scipy.signal import find_peaks_cwt

## Author : Krishnan Ramaswamy


class Lane():

    def __init__(self, base_pt, img_size):

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = collections.deque(maxlen=10)
        self.recent_yfitted = collections.deque(maxlen=10)

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.current_xfit = None
        self.roi_mask = None

        self.img_size = img_size
        self.base_pt = base_pt

        self.yvals = np.linspace(0, img_size[0], 101)
        self.mask = np.ones(img_size, dtype=np.uint8) * 255

        self.dropped_frames = 0

    def add_lane_pixels(self, x, y):
        # Use all pixels from previous detections for curve fit
        x_hist = list(chain(*self.recent_xfitted, x))
        y_hist = list(chain(*self.recent_yfitted, y))

        try:
            p_lane = np.polyfit(y_hist, x_hist, 2)
            self.detected = self.sanity_check_lane(p_lane)
        except:
            #             print('Failed to fit polynomial.')
            self.detected = False

        if self.detected and len(p_lane) == 3:
            self.current_fit = p_lane
            x_fit = p_lane[0] * self.yvals ** 2 + p_lane[1] * self.yvals + p_lane[2]

            self.current_xfit = x_fit  # For drawing

            # Save current detected pixels
            self.allx = np.array(x)
            self.ally = np.array(y)

            self.recent_xfitted.append(x_fit)
            self.recent_yfitted.append(self.yvals)
            if len(self.current_fit) > 1:
                self.diffs = self.current_fit - p_lane

            self.dropped_frames = 0
        else:
            #             print('Sanity check failed!')
            # Use last fit if current one failed
            p_lane = self.current_fit
            x_fit = p_lane[0] * self.yvals ** 2 + p_lane[1] * self.yvals + p_lane[2]
            self.dropped_frames += 1

        # Update ROI mask
        self.mask.fill(0)
        # http://stackoverflow.com/a/35902430/538379
        pts = np.transpose(np.vstack([x_fit, self.yvals])).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(self.mask, pts, -1, (255, 255, 255), thickness=80)

    def sanity_check_lane(self, coeffs):
        # Return true if there is no prior data
        if self.radius_of_curvature is None:
            return True

        A, B, _ = coeffs
        y = self.img_size[0] / 2  # Curvature at middle of image
        R = (1 + (2 * A * y + B) ** 2) ** (3 / 2) / abs(2 * A)

        k = 1 / R  # Curvature is a better measure to track
        R0 = self.radius_of_curvature
        current_k = 1 / self.radius_of_curvature

        return abs(R - R0) / R0 <= 0.01  # Max change from frame to frame is 1%

    def detect_from_mask(self, image):
        mask_lanes = cv2.bitwise_and(image, self.mask)
        all_pts = cv2.findNonZero(mask_lanes)
        if all_pts is not None:
            all_pts = all_pts.reshape((-1, 2))
            self.add_lane_pixels(all_pts[:, 0], all_pts[:, 1])
        else:
            self.detected = False

    def draw_lane(self, image):
        """
        Draws lane on given image
        """
        pts = np.array([np.transpose(np.vstack([self.current_xfit, self.yvals]))])
        cv2.fillPoly(image, np.int_([pts]), (0, 255, 0))
        return image


def findBasePoints(lanes, min_peak = 75.0):

    #Uses histogram to find possible base points for lane lines

    hist = np.sum(lanes[int(lanes.shape[0]*0.5):,:], axis=0)
    idx = find_peaks_cwt(hist, [100, 125, 150], max_distances=[100, 125, 150], noise_perc=50)

    if len(idx) < 2:
        return None

    middle = np.average(idx)
    left_pt = max(i for i in idx if i < middle)   # Closest peak to middle on the left
    right_pt = min(i for i in idx if i >= middle) # Closest peak to middle on the right
    return [left_pt, right_pt]


def calibrateCamera(cal_images, nx, ny):

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

    fname = cal_images[0]
    for fname in cal_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    return mtx, dist

def setupCamera():
    cal_images = glob.glob('camera_cal/calibration*.jpg')
    nx, ny = 9, 6
    cam_mtx, cam_dist = calibrateCamera(cal_images, nx, ny)
    return cam_mtx, cam_dist


def getPerspectiveTransform(image, src_in = None, dst_in = None, display=False):

    img_size = image.shape
    if src_in is None:
        src = np.array([[585. /1280.*img_size[1], 455./720.*img_size[0]],
                        [705. /1280.*img_size[1], 455./720.*img_size[0]],
                        [1130./1280.*img_size[1], 720./720.*img_size[0]],
                        [190. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)
    else:
        src = src_in
        
    if dst_in is None:
        dst = np.array([[300. /1280.*img_size[1], 100./720.*img_size[0]],
                        [1000./1280.*img_size[1], 100./720.*img_size[0]],
                        [1000./1280.*img_size[1], 720./720.*img_size[0]],
                        [300. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)
    else:
        dst = dst_in
        
    warp_m = cv2.getPerspectiveTransform(src, dst)
    warp_minv = cv2.getPerspectiveTransform(dst, src)

    if display:
        plt.subplot(1,2,1)
        plt.hold(True)
        plt.title ("Before Perspective Transform")
        plt.imshow(image, cmap='gray')
        colors = ['r+','g+','b+','w+']
        for i in range(4):
            plt.plot(src[i,0],src[i,1],colors[i])

        im2 = cv2.warpPerspective(image, warp_m, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        plt.subplot(1,2,2)
        plt.hold(True)
        plt.title("After Perspective Transform")
        plt.imshow(im2, cmap='gray')
        for i in range(4):
            plt.plot(dst[i,0],dst[i,1],colors[i])
        plt.show()


    return warp_m, warp_minv



def findPerspectivePoints(edges):


    rho = 2              # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180  # angular resolution in radians of the Hough grid
    threshold = 100       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 # minimum number of pixels making up a line
    max_line_gap = 25    # maximum gap in pixels between connectable line segments

    angle_min_mag = 25*pi/180
    angle_max_mag = 40*pi/180

    lane_markers_x = [[], []]
    lane_markers_y = [[], []]
    
    masked_edges = np.copy(edges)
    masked_edges[:edges.shape[0]*6//10,:] = 0
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            theta = atan2(y1-y2, x2-x1)
            rho = ((x1+x2)*cos(theta) + (y1+y2)*sin(theta))/2

            if (abs(theta) >= angle_min_mag and abs(theta) <= angle_max_mag):
                if theta > 0: # positive theta is downward in image space?
                    i = 0 # Left lane marker
                else:
                    i = 1 # Right lane marker
                lane_markers_x[i].append(x1)
                lane_markers_x[i].append(x2)
                lane_markers_y[i].append(y1)
                lane_markers_y[i].append(y2)

    if len(lane_markers_x[0]) < 1 or len(lane_markers_x[1]) < 1:
        # Failed to find two lane markers
        return None
    
    
    p_left  = np.polyfit(lane_markers_y[0], lane_markers_x[0], 1)
    p_right = np.polyfit(lane_markers_y[1], lane_markers_x[1], 1)
    
    # Find intersection of the two lines
    apex_pt = np.linalg.solve([[p_left[0], -1], [p_right[0], -1]], [-p_left[1], -p_right[1]])
    top_y = ceil(apex_pt[0] + 0.075*edges.shape[0])
    print(top_y)
    
    bl_pt = ceil(np.polyval(p_left, edges.shape[0]))
    tl_pt = ceil(np.polyval(p_left, top_y))
    
    br_pt = ceil(np.polyval(p_right, edges.shape[0]))
    tr_pt = ceil(np.polyval(p_right, top_y))



    src = np.array([[tl_pt, top_y],
                    [tr_pt, top_y],
                    [br_pt, edges.shape[0]],
                    [bl_pt, edges.shape[0]]], np.float32)

    getPerspectiveTransform(edges, src_in = src, dst_in = None, display=True) 
    return src


def sobel(gray, orient='x', sobel_kernel=3):
    return cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y', ksize=sobel_kernel)

def getAbsSobelThresh(sobel, thresh=(0, 255)):
    # Calculate directional gradient
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    retval, output = cv2.threshold(scaled_sobel, thresh[0], thresh[1], cv2.THRESH_BINARY)
    return output

def getGradientMagThreshold(sobelx, sobely, thresh=(0, 255)):
    # Calculate gradient magnitude
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    mag_sobel = (255*mag_sobel/np.max(mag_sobel)).astype(np.uint8)

    retval, output = cv2.threshold(mag_sobel, thresh[0], thresh[1], cv2.THRESH_BINARY)
    return output




def findEdges(image, ksize = 11):
    blur = cv2.GaussianBlur(image, (5, ksize), 0)
    gray = (cv2.cvtColor(blur.astype(np.uint8), cv2.COLOR_RGB2GRAY))

    
    sobelx = sobel(gray, orient='x', sobel_kernel=ksize)
    gradx = getAbsSobelThresh(sobelx, thresh=(60, 70))

    sobely = sobel(gray, orient='y', sobel_kernel=ksize)
    grady = getAbsSobelThresh(sobely, thresh=(50, 120))
    mag_binary = getGradientMagThreshold(sobelx, sobely, thresh=(50, 100))
    _, gray_binary = cv2.threshold(gray.astype('uint8'), 130, 255, cv2.THRESH_BINARY)

    # Check if gradx has sufficient data, else use magnitude
    total_px = gradx.shape[0]*gradx.shape[1]/2
    if cv2.countNonZero(gradx[gradx.shape[0]/2:])/total_px < 0.01:
        mask_two = mag_binary
    else:
        mask_two = gradx
    
    yellow = (np.array([160, 128, 0]), np.array([255, 255, 128]))

    yellow_mask = (cv2.inRange(image, yellow[0], yellow[1])/255.0).astype(np.uint8)

    combined_binary = np.zeros_like(gray_binary)
    combined_binary = np.clip(cv2.bitwise_and(gray_binary, cv2.bitwise_or(mask_two, yellow_mask)), 0, 1).astype('uint8')
    


    return combined_binary

def detectLaneUsingHistogramSlidingWindow(image, left_lane, right_lane, base_pts, num_bands = 15, window_width = 0.2):
    """Uses histogram and sliding window to detect lanes from scratch"""

    height = image.shape[0]
    band_height = int(1./num_bands * height)   # Divide image into horizontal bands
    band_width = int(window_width*image.shape[1])
    
    l_x, l_y, r_x, r_y = [], [], [], []
    
    base_left, base_right = base_pts

    idx_left, idx_right = base_pts
    for i in reversed(range(num_bands)):
        w_left = image[i*band_height:(i+1)*band_height,base_left-band_width//2:base_left+band_width//2]
        w_right = image[i*band_height:(i+1)*band_height,base_right-band_width//2:base_right+band_width//2]
        
        left_y_pt, left_x_pt = np.nonzero(w_left)
        right_y_pt, right_x_pt = np.nonzero(w_right)
        
        l_x.extend(left_x_pt + base_left-band_width//2)
        l_y.extend(left_y_pt + i*band_height)
        r_x.extend(right_x_pt+ base_right-band_width//2)
        r_y.extend(right_y_pt+ i*band_height)

        # Find 'x' with maximum nonzero elements as baseline for next window
        s_left = np.sum(w_left, axis=0)
        s_right = np.sum(w_right, axis=0)
        if np.any(s_left > 0):
            base_left = np.argmax(s_left) + base_left-band_width//2
        if np.any(s_right > 0):
            base_right = np.argmax(s_right) + base_right-band_width//2
    
    left_lane.add_lane_pixels(l_x, l_y)
    right_lane.add_lane_pixels(r_x, r_y)
    return left_lane, right_lane

def processImage(image, key_frame_interval=16, cache_length=10):
    
    global cam_mtx, cam_dist
        
    if processImage.cache is None:
                
        left_lane = Lane(int(0.16*image.shape[0]), image.shape[:2])
        right_lane = Lane(int(0.62*image.shape[0]), image.shape[:2])

        cache = {'cam_mtx': cam_mtx,
                 'cam_dist': cam_dist,
                 'warp_m': None,
                 'warp_minv': None,
                 'last_p_left': collections.deque(maxlen=cache_length),
                 'last_p_right': collections.deque(maxlen=cache_length),
                 'frame_ctr': 0,
                 'persp_found': False,
                 'left': left_lane,
                 'right': right_lane,
                 'base_pts': None}
    else:
        cache = processImage.cache
    

    left_lane = cache['left']
    right_lane = cache['right']

    # Preprocess image and find edges using thresholding
    undist = cv2.undistort(image, cam_mtx, cam_dist, None, cam_mtx)
    edges = findEdges(undist)
    
    if cache['warp_m'] is None:
        src = findPerspectivePoints(edges)
        warp_m, warp_minv = getPerspectiveTransform(image, src_in = src)
        
        if src is not None:
            # Save only if customized perspective transform is found
            cache['warp_m'] = warp_m
            cache['warp_minv'] = warp_minv
    else:
        warp_m, warp_minv = cache['warp_m'], cache['warp_minv']
    
    warp_edges = cv2.warpPerspective(edges, warp_m, (edges.shape[1], edges.shape[0]), flags=cv2.INTER_LINEAR)
    
    base_pts = cache['base_pts']
    if base_pts is None: #or cache['frame_ctr'] % key_frame_interval == 0:
        new_base_pts = findBasePoints(warp_edges)

        if new_base_pts is not None:
            base_pts = new_base_pts
        else:
            # Could not find new base points
            # Re-use previous data if base points could not be found
            cache['frame_ctr'] = cache['frame_ctr'] - 1 # Make sure we try again in the next frame
            return undist

    if ((left_lane is None or not left_lane.dropped_frames > 16) 
            or (right_lane is None or not right_lane.dropped_frames > 16)):
        # Detect from scratch
        detectLaneUsingHistogramSlidingWindow(warp_edges, left_lane, right_lane, base_pts)
    else:
        left_lane.detect_from_mask(warp_edges)
        right_lane.detect_from_mask(warp_edges)

    cache['frame_ctr'] = cache['frame_ctr'] + 1
    cache['base_pts'] = base_pts
    processImage.cache = cache
    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(image).astype(np.uint8)
    
    yvals = left_lane.yvals
    left_fitx = left_lane.current_xfit
    right_fitx = right_lane.current_xfit
    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
#     plt.imshow(color_warp)
#     plt.plot(right_lane.ally, right_lane.allx, '.')
#     plt.figure()
    
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, warp_minv, (image.shape[1], image.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    cache['left'] = left_lane
    cache['right'] = right_lane

    return result

def clearCache():
    processImage.cache = None

def runVideo(vidFile, height, width) :

    cap = cv2.VideoCapture(vidFile)

    cap.get(5)  # to display frame rate of video
    # print cap.get(cv2.cv.CV_CAP_PROP_FPS)

    while (cap.isOpened()):
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    cap.release()
    cv2.destroyAllWindows()


def testAImage(cam_mtx, cam_dist):


    print("Testing the camera calibeation with test_images/test1a.jpg")
    image = mpimage.imread('test_images/test1a.jpg')
    plt.title("Test Image")
    plt.imshow(image)
    plt.show()

    ###
    undist = cv2.undistort(image, cam_mtx, cam_dist, None, cam_mtx)
    yellow = (np.array([160, 128, 0]), np.array([255, 255, 128]))
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HSV)
    # Step 1 : Find the edges
    edges = findEdges(undist)
    src = findPerspectivePoints(edges)
    # Step 2 : Find perspective transform
    warp_m, warp_minv = getPerspectiveTransform(undist, src, display=True)
    lanes = cv2.warpPerspective(edges, warp_m, (edges.shape[1], edges.shape[0]), flags=cv2.INTER_LINEAR)
    plt.rcParams['figure.figsize'] = figsize = (10, 6)
    plt.figure(figsize=(10, 12))
    plt.subplot(1, 3, 1)
    plt.title("Edges")
    plt.imshow(edges, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("Lanes")
    plt.imshow(lanes, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.inRange(hls[:, :, 0], 21, 23), cmap='gray')
    base_pts = findBasePoints(lanes)
    plt.title(" Edges & lanes in a range")
    plt.show()


if __name__ == '__main__':

   # runVideo('project_video.mp4')

    #Step 1: Setup camera
    cam_mtx, cam_dist = setupCamera()

    # Step 2 : Test the camera caliberation with some test image
    testAImage(cam_mtx, cam_dist)

    # Step 3  :Test a pipeline method with some test image
    processImage.clearCache = clearCache
    clearCache()
    image = mpimage.imread('test_images/test2a.jpg')
    out = processImage(image)
    plt.imshow(out)
    plt.title(" Testing pipeline image")
    plt.show()


    # Step 4 : Run the video stream
    clearCache()
    vid_output = './transformed_videos/project_video_out.mp4'
    clip = VideoFileClip('./videos/project_video.mp4')
    vid_clip = clip.fl_image(processImage)
    runVideo(vid_output,960,540)

    clearCache()
    vid_output = './transformed_videos/challenge_video_out.mp4'
    clip2 = VideoFileClip('./videos/challenge_video.mp4')
    vid_clip = clip2.fl_image(processImage)
    runVideo(vid_output, 960, 540)


    clearCache()
    vid_output = './transformed_videos/harder_challenge_video_out.mp4'
    clip2 = VideoFileClip('./videos/harder_challenge_video.mp4')
    vid_clip = clip2.fl_image(processImage)
    runVideo(vid_output, 960, 540)






