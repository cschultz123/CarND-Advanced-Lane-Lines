import cv2
import image_utils
import numpy as np


def process_gradients(img, x_thresh, y_thresh, mag_thresh, dir_thresh):
    """
    This method applies the gradient processing pipeline.

    1. Guassian Blur
    2. Absolute Threshold (X)
    3. Absolute Threshold (Y)
    4. Magnitude Threshold (X and Y)
    5. Direction Threshold (X and Y)

    :param img: (rgb) image
    :param x_thresh (tuple): minimum value, maximum value (int)
    :param y_thresh (tuple): minimum value, maximum value (int)
    :param mag_thresh (tuple): minimum value, maximum value (int)
    :param dir_thresh (tuple): minimum value, maximum value (radians)
    :return: combined binary mask
    """
    blur = image_utils.guassian_blur(img, kernel_size=5)
    gradx = image_utils.abs_sobel_thresh(blur, orient='x', thresh=x_thresh)
    grady = image_utils.abs_sobel_thresh(blur, orient='y', thresh=y_thresh)
    gradmag = image_utils.mag_thresh(blur, sobel_kernel=3, thresh=mag_thresh)
    graddir = image_utils.dir_threshold(blur, sobel_kernel=3, thresh=dir_thresh)

    combined = np.zeros_like(gradx)
    combined[((gradx==1) & (grady==1) | (gradmag==1) & (graddir==1))] = 1

    return combined


def process_color(img, hue_thresh=(0,255), sat_thresh=(0,255)):
    """
    This method will convert the image to HLS and apply a threshold on the
    saturation channel.

    :param img: (rgb) image
    :param thresh (tuple): minimum value, maximum value (int)
    :return: thresholded saturation channel
    """
    # convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    hue = hls[:,:,0]
    hue_mask = np.zeros_like(hue)
    hue_mask[(hue > hue_thresh[0]) & (hue < hue_thresh[1])] = 1

    # threshold saturation channel
    sat = hls[:,:,2]
    sat_mask = np.zeros_like(sat)
    sat_mask[(sat > sat_thresh[0]) & (sat < sat_thresh[1])] = 1

    # combine masks
    combined = np.zeros_like(sat_mask)
    combined[(hue_mask==1) & (sat_mask==1)] = 1

    return combined


def _compute_perspective_src_and_dst(img, wrg=(0.2, 0.9), hrg=(0.66,1.0)):
    """
    This method computes the source and destination points needed to apply the
    perspective tranform. This method uses relative values with respect to the
    images dimensions.

    IMPORTANT: The image should be undistorted.

    :param img (ndarray): (rgb) image
    :param wrg (tuple): min relative offset (top), min relative offset (bottom)
    :param hrg (tuple): max relative offset (top), max relative offset (bottom)
    :return: source coordinates, destination coordinates
    """
    height, width = img.shape[0], img.shape[1]
    x_center = width/2

    # source trapezoid coordinates
    src = np.float32([(x_center + wrg[1] * x_center, height*hrg[1]),
                      (x_center + wrg[0] * x_center, height*hrg[0]),
                      (x_center - wrg[0] * x_center, height*hrg[0]),
                      (x_center - wrg[1] * x_center, height*hrg[1])])

    # destination rectangle coordinates
    dst = np.float32([(0, 0),
                      (width, 0),
                      (0, height),
                      (width, height)])

    return src, dst


def pipeline(img,
             hue_thresh=(0,100),
             sat_thresh=(160,255),
             x_thresh=(10,255),
             y_thresh=(60,255),
             mag_thresh=(40,255),
             dir_thresh=(.65, 1.05)):
    """
    This function applies both gradient thresholding and saturation
    thresholding. A perspective transformation is then applied to the
    image. Returns binary mask of birds eye view of original image.

    :param img (ndarray): (rgb) undistorted image
    :param camera_matrix (ndarray): camera matrix
    :param distortion (ndarray): distortion parameters
    :param sat_thresh (tuple): min value, max value
    :param x_thresh (tuple): min value, max value
    :param y_thresh (tuple): min value, max value
    :param mag_thresh (tuple): min value, max value
    :param dir_thresh (tuple): min value, max value
    :return: (ndarray) birds eye binary mask of original image
    """
    # threshold saturation channel of image
    color_mask = process_color(img, hue_thresh, sat_thresh)

    # threshold gradients
    gradient_mask = process_gradients(img, x_thresh, y_thresh, mag_thresh, dir_thresh)

    # combine masks
    combine = np.zeros_like(color_mask)
    combine[(color_mask > 0) | (gradient_mask > 0)] = 1

    # apply perspective transform to mask
    src, dst = _compute_perspective_src_and_dst(img)
    perspective, M, Minv = image_utils.perspective_transform(combine, src, dst)

    # return perspective transform
    return perspective.astype(np.uint8), M, Minv


def process_image(img,
                  camera_matrix,
                  distortion,
                  hue_thresh=(0,100),
                  sat_thresh=(160,255),
                  x_thresh=(10,255),
                  y_thresh=(60,255),
                  mag_thresh=(40,255),
                  dir_thresh=(.65, 1.05),
                  windows=5,
                  peak_offset=50):
    """
    This is the end to end pipeline that takes a raw image and returns an image
    with the lanes drawn on it.

    :param img: (ndarray) RGB image
    :param camera_matrix: (ndarray) camera matrix
    :param distortion: (ndarray) distortion coefficents
    :param hue_thresh: (tuple) min value, max value
    :param sat_thresh: (tuple) min value, max value
    :param x_thresh: (tuple) min value, max value
    :param y_thresh: (tuple) min value, max value
    :param mag_thresh: (tuple) min value, max value
    :param dir_thresh: (tuple) min value, max value
    :param windows: (int) number of horizontal windows
    :param peak_offset: (int) minimum peak offset value
    :return:
    """
    # undistort the image using the camera matrix and distortion parameters
    undist = image_utils.undistort(img, camera_matrix, distortion)

    # filter and apply perspective tranform for birds eye view
    top, M, Minv = pipeline(undist, hue_thresh, sat_thresh, x_thresh, y_thresh, mag_thresh, dir_thresh)

    # # filter and isolate right and left lanes
    # right, left = find_lanes(top, windows, peak_offset)
    #
    # # fit 2nd order polynomial to left and right lanes
    # rpts, lpts = fit_lanes(right, left)
    left_fit, right_fit = find_lanes(top)

    # draw lanes on original undistorted image
    return draw_lanes(undist, rfit=right_fit, lfit=left_fit, Minv=Minv)


class Lanes(object):
    """
    This class provides methods for finding lanes from image. It also stores
    state that is used for outlier detection.
    """

    def __init__(self):
        pass



def find_lanes(img, num_windows=9, window_margin=100, minimum_pixels=50):
    """
    This function applies a sliding window technique to identify the lanes and
    fit a 2nd order polynomial to them.

    :param img: (binary) thresholded and warped image
    :param num_windows: (int) number of vertical windows
    :param window_margin: (int) window width +/- margin
    :param minimum_pixels: (int) number of pixels required to move window center
    :return: left lane line coefficients, right lane line coefficients
    """
    # Take a histogram of the whole of the image
    histogram = np.sum(img[:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0] / num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - window_margin
        win_xleft_high = leftx_current + window_margin
        win_xright_low = rightx_current - window_margin
        win_xright_high = rightx_current + window_margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minimum_pixels:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minimum_pixels:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def _draw_pw_lines(img, pts, color):
    num_points = pts.shape[0]
    for i in range(num_points):
        if i < num_points-1:
            x1 = pts[i, 0]
            y1 = pts[i, 1]
            x2 = pts[i + 1, 0]
            y2 = pts[i + 1, 1]
            cv2.line(img, (x1, y1), (x2, y2), color, 50)


def draw_lanes(undistorted, rfit, lfit, Minv):
    warp = np.zeros_like(undistorted[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp, warp, warp))

    height, width = undistorted.shape[0], undistorted.shape[1]
    yvals = np.linspace(height/2, height, height/2)

    left_xy_coordinates = np.vstack((np.poly1d(lfit)(yvals), yvals)).T
    right_xy_coordinates = np.vstack((np.poly1d(rfit)(yvals), yvals)).T

    pts = np.vstack((left_xy_coordinates, right_xy_coordinates[::-1]))

    cv2.fillPoly(color_warp, np.uint([pts]), (0, 255, 255))

    left_lane_color = (255, 255, 0)  # yellow
    right_lane_color = (255, 255, 255)  # white

    # draw lane lines using fit
    _draw_pw_lines(color_warp, left_xy_coordinates.astype(np.int_), left_lane_color)
    _draw_pw_lines(color_warp, right_xy_coordinates.astype(np.int_), right_lane_color)

    # revert image back to original perspective
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0]))

    result = cv2.addWeighted(undistorted, 1, newwarp, 0.5, 0)

    return result
