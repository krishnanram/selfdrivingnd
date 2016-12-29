import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2




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

        print("adas", left_x_1, y_max, left_x_2, y_min)
        print ("adas", right_x_1, y_max, right_x_2,y_min)

        print ( "(", int(left_x_1), int(y_min)), (int(left_x_2), int(y_max), "),","(", int(right_x_1), int(y_max)), (int(right_x_2), int(y_min), ")"  )
        cv2.line(image, (int(left_x_1), int(y_min)), (int(left_x_2), int(y_max)), (255, 0, 0), 10)
        cv2.line(image, (int(right_x_1), int(y_max)), (int(right_x_2), int(y_min)), (255, 0, 0), 10)
        # as you will get many lines on the left and right, then you average the slopes of the lines.



