import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def draw_lines7(img, lines, color=[255, 0, 0], thickness=5):

    #https://carnd-udacity.atlassian.net/wiki/questions/10322731/having-problems-averaging-the-line-slopes-and-getting-the-desired-output

    # Averaging HoughLines Here
    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:  # left line, negative slope
                yMIN = -slope * x1 + y1
                yMAX = -slope * x2 + y2
            elif slope > 0:  # right line, positive slope
                yMIN = slope * x1 + y1
                yMAX = slope * x2 + y2
            cv2.line(img, (int(x1), int(yMIN)), (int(x2), int(yMAX)), color, thickness)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_select, 0.9, line_image, 1, 0)
    plt.imshow(lines_edges)

    return lines_edges


def draw_lines6(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    right_line = []
    left_line = []
    min_slope = 0.42
    max_slope = 0.78

    if lines is None:
        return

    y_top = 0

    for line in lines:
        for x1, y1, x2, y2 in line:

            slope = ((y2 - y1) / (x2 - x1))
            if (y_top > y1):
                y_top = y1
            if (y_top > y2):
                y_top = y2

            if max_slope >= slope >= min_slope:
                right_line.append([x1, y1])
                right_line.append([x2, y2])

            elif -max_slope <= slope <= -min_slope:
                left_line.append([x1, y1])
                left_line.append([x2, y2])

            # print ('Top y : ',  y_top )
            if (len(right_line) > 0):  # Some image may not have proper  line detected in the selected region


                # Fit line based points on  right line
                r_vx, r_vy, r_cx, r_cy = cv2.fitLine(np.array(right_line), cv2.DIST_L2, 0, 0.01, 0.01)
                # Slope and inception for right line
                r_m = r_vy / r_vx
                r_b = r_cy - r_m * r_cx
                # find top and bottom point for right line


                r_x_top = int((y_top - r_b) / r_m)
                r_x_bottom = int((y_bottom - r_b) / r_m)
                # draw left line
                cv2.line(img, (r_x_top, y_top), (r_x_bottom, y_bottom), color, thickness)

            if (len(left_line) > 0):
                # Fit line based points on left line
                l_vx, l_vy, l_cx, l_cy = cv2.fitLine(np.array(left_line), cv2.DIST_L2, 0, 0.01, 0.01)
                # Slope and inception for left line
                l_m = l_vy / l_vx
                l_b = l_cy - l_m * l_cx

                l_x_top = int((y_top - l_b) / l_m)
                l_x_bottom = int((y_bottom - l_b) / l_m)
                # draw left line
                cv2.line(img, (l_x_top, y_top), (l_x_bottom, y_bottom), color, thickness)


def draw_line5(image, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            print((x1, y1), (x2, y2))
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # Iterate over the output "lines" and draw lines on a blank image
    left_slope = []
    right_slope = []
    left_lines = []
    right_lines = []
    y_min = image.shape[0]
    y_max = image.shape[0] * 0.6

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = float((y2 - y1) / (x2 - x1))
    if slope < 0:  # if less than 0,its left as y axis value is upside down
        left_slope.append(slope)
        left_lines.append(line)

    elif slope > 0:
        right_slope.append(slope)
        right_lines.append(line)

        left_slope_array = np.array(left_slope)
        left_lines_array = np.array(left_lines)

        left_slope_avg = np.mean(left_slope_array)
        left_lines_avg = np.mean(left_lines_array)
        left_y_int = y_min - (left_slope_avg * left_lines_avg)
        left_y_int_max = y_max - (left_slope_avg * left_lines_avg)
        left_x_1 = (y_min - left_y_int) / left_slope_avg
        left_x_1 = left_x_1.astype(int)
        left_x_2 = (y_max - left_y_int_max) / left_slope_avg
        left_x_2 = left_x_2.astype(int)

        right_slope_array = np.array(right_slope)
        right_lines_array = np.array(right_lines)

        right_slope_avg = np.mean(right_slope_array)
        right_lines_avg = np.mean(right_lines_array)
        right_y_int = y_min - (right_slope_avg * right_lines_avg)
        right_y_int_max = y_max - (right_slope_avg * right_lines_avg)
        right_x_1 = (y_min - right_y_int) / right_slope_avg
        right_x_1 = right_x_1.astype(int)
        right_x_2 = (y_max - right_y_int_max) / right_slope_avg
        right_x_2 = right_x_2.astype(int)

    cv2.line(image, (int(left_x_1), int(y_min)), (int(left_x_2), int(y_max)), (255, 0, 0), 10)
    cv2.line(image, (int(right_x_1), int(y_max)), (int(right_x_2), int(y_min)), (255, 0, 0), 10)
    # as you will get many lines on the left and right, then you average the slopes of the lines.


def draw_line4(img, lines, color=[255, 0, 0], thickness=2):
    xMIN = 0
    xMAX = 0
    yMIN = 0
    yMAX = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0:  # left-hand line
                xMIN =  # compute minimum x-value
                xMAX =  # compute maximum x-value
                yMIN =  # compute minimum y-value
                yMAX =  # compute maximum y-value
                cv2.line(img, (xMIN, yMIN), (xMAX, yMAX), color, thickness)
            Else:  # right-hand line
            xMIN =  # compute minimum x-value
            xMAX =  # compute maximum x-value
            yMIN =  # compute minimum y-value
            yMAX =  # compute maximum y-value
            cv2.line(img, (xMIN, yMIN), (xMAX, yMAX), color, thickness)


def draw_line3(img, lines, color=[255, 0, 0], thickness=2):

    y_min = image.shape[0]  # set it to a high value so that first y will be a min
    y_max = image.shape[0]
    # get global y min (highest y)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if y1 < y_min:
                y_min = y1
    # bookkeeping
    right_slope = []
    right_all_x = []
    right_all_y = []
    right_top_y = image.shape[0]
    left_slope = []
    left_all_x = []
    left_all_y = []
    left_top_y = image.shape[0]

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0:  # right
                right_slope.append(slope)
                right_all_x.append(x1)
                right_all_x.append(x2)
                right_all_y.append(y1)
                right_all_y.append(y2)
            if slope < 0:  # left
                left_slope.append(slope)
                left_all_x.append(x1)
                left_all_x.append(x2)
                left_all_y.append(y1)
                left_all_y.append(y2)

    if len(right_slope) > 0:  # I was getting a divide by zero
        avg_right_slope = sum(right_slope) / float(len(right_slope))
        avg_right_x = sum(right_all_x) / float(len(right_all_x))
        avg_right_y = sum(right_all_y) / float(len(right_all_y))
        b_right = avg_right_y - (avg_right_slope * avg_right_x)  # y = ax + b
        x1_right = (y_min - b_right) / avg_right_slope  # find x for highest point
        x2_right = (y_max - b_right) / avg_right_slope  # find x for lowest point
        # draw right line
        cv2.line(img, (int(x1_right), y_min), (int(x2_right), y_max), color, thickness)

    if len(left_slope) > 0:
        avg_left_slope = sum(left_slope) / float(len(left_slope))
        avg_left_x = sum(left_all_x) / float(len(left_all_x))
        avg_left_y = sum(left_all_y) / float(len(left_all_y))
        b_left = avg_left_y - (avg_left_slope * avg_left_x)  # y = ax + b
        x1_left = (y_min - b_left) / avg_left_slope  # find x for highest point
        x2_left = (y_max - b_left) / avg_left_slope  # find x for lowest point
        # draw left line
        cv2.line(img, (int(x1_left), y_min), (int(x2_left), y_max), color, thickness)


def draw_line2(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            print((x1, y1), (x2, y2))
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lines1(img, lines, color=[255, 0, 0], thickness=2):

    left_slope = []
    right_slope = []
    left_lines = []
    right_lines = []
    y_min = image.shape[0]
    y_max = image.shape[0 ] *0.6


    for line in lines:
        for x1 ,y1 ,x2 ,y2 in line:
            slope = float(( y2 -y1 ) /( x2 -x1))

            if slope < 0:
                left_slope.append(slope)
                left_lines.append(line)

            elif slope > 0:
                right_slope.append(slope)
                right_lines.append(line)


        left_slope_array = np.array(left_slope)
        left_lines_array = np.array(left_lines)
        left_slope_avg = np.mean(left_slope_array)
        left_lines_avg = np.average(left_lines_array)
        left_y_int = y_min - (left_slope_avg * left_lines_avg)
        left_y_int_max = y_max - (left_slope_avg * left_lines_avg)
        left_x_1 = (y_min - left_y_int) / left_slope_avg
        left_x_1 = left_x_1.astype(int)
        left_x_2 = (y_max - left_y_int_max) / left_slope_avg
        left_x_2 = left_x_2.astype(int)



        right_slope_array = np.array(right_slope)
        right_lines_array = np.array(right_lines)
        right_slope_avg = np.mean(right_slope_array)
        right_lines_avg = np.mean(right_lines_array)
        right_y_int = y_min - (right_slope_avg * right_lines_avg)
        right_y_int_max = y_max - (right_slope_avg * right_lines_avg)
        right_x_1 = (y_min - right_y_int) / right_slope_avg
        right_x_1 = right_x_1.astype(int)
        right_x_2 = (y_max - right_y_int_max) / right_slope_avg
        right_x_2 = right_x_2.astype(int)



        cv2.line(image, (left_x_1, y_min), (left_x_2, y_max), color, thickness)
        cv2.line(image, (right_x_1, y_max), (right_x_2, y_min), color, thickness)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    # printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)
    plt.imshow(image)  # call as plt.imshow(gray, cmap='gray') to show a grayscaled image

    # Grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)

    gray = cv2.cvtColor(color_select, cv2.COLOR_BGR2GRAY)  # grayscale conversion
    plt.imshow(gray, cmap='gray')

    # Define a kernel size for Gaussian smoothing / blurring
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Define parameters for Canny and run it
    # NOTE: if you try running this code you might want to change these!
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    vertices = np.array([[(0, ysize), (450, 330), (490, 310), (xsize, ysize)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_select, 0.9, line_image, 1, 0)
    plt.imshow(lines_edges)

    (navg, pavg, nbavg, pbavg) = average_slope(lines);

    # y = mx + b
    y_left_1 = -(pavg * 0 + pbavg) + ysize;  # bottom left vertex of trapezoid
    y_right_1 = -(navg * xsize + nbavg) + ysize;  # bottom right vertex of trapezoid

    y_left_2 = -(pavg * 450 + pbavg) + ysize;
    y_right_2 = -(navg * 490 + nbavg) + ysize;

    left = (0, y_left_1, 450, y_left_2);  # left  line of trapezoid
    right = (xsize, y_right_1, 490, y_right_2)  # right line of trapezoid

    l = [left, right];

    return draw_lines(image, l);


####Line avg######################
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    temp = np.copy(img)

    for x1, y1, x2, y2 in lines:
        cv2.line(temp, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return temp;


#  Returns a tuple of:
#    -  Average negative slope
#    -  Average positive slope
#    -  Average negative y-intercept
#    -  Average positive y-intercept
def average_slope(lines):
    def remove_out_of_tolerance(slopes, tol):
        if len(slopes) == 0:
            return;

        for s in slopes:
            # temp = list of slopes in "slopes" that AREN'T s,
            # the current slope
            temp = [x for x in slopes if x is not s];
            tavg = 0.0;

            for ts in temp:
                tavg += ts;
            tavg /= len(temp);

            if abs(s) - abs(tavg) > tol:
                slopes.remove(s);

        print("slopes: ", slopes);

    ######## Start of average_slope ########
    ns = [];  # negative slopes
    ps = [];  # positive slopes

    nbs = [];  # negative intercepts
    pbs = [];  # positive intercepts

    navg = 0.0;  # negative slope average temp variable
    pavg = 0.0;  # positive slope average temp variable

    nbavg = 0.0;  # negative intercept average temp variable
    pbavg = 0.0;  # positive intercept average temp variable

    tol = 0.0005;  # tolerance for comparing the avg slope with test vals

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) * 1. / (x2 - x1 + 1e-6);
            # • multiplying by 1. is to convert rise (numerator) to float
            # • 1e-6 is there to avoid division by zero
            b = y1 - slope * x1;

            if (slope < 0.0):  # and around_avg(ns, slope, tol) == True \
                # and around_avg(nbs, b, tol) == True:
                ns.append(slope);
                print("appended ", str(slope), " to ns");
                nbs.append(b);
            elif (slope > 0.0):  # and around_avg(ps, slope, tol) == True \
                # and around_avg(pbs, b, tol) == True:
                ps.append(slope);
                print("appended ", str(slope), " to ps");
                pbs.append(b);

    remove_out_of_tolerance(ns, tol);
    remove_out_of_tolerance(ps, tol);

    # now you want to average the elements of each list
    for n in ns:
        navg += n;
    if len(ns) > 0:
        navg /= len(ns);

    for p in ps:
        pavg += p;
    if len(ps) > 0:
        pavg /= len(ps);

    for nb in nbs:
        nbavg += nb;
    if len(nbs) > 0:
        nbavg /= len(nbs);

    for pb in pbs:
        pbavg += pb;
    if len(pbs) > 0:
        pbavg /= len(pbs);

    print("(navg, pavg, nbavg, pbavg) = ", str((navg, pavg, nbavg, pbavg)));

    return (navg, pavg, nbavg, pbavg);