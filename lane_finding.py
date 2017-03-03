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
                  left_line=None,
                  right_line=None,
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
    left_line, right_line = find_lanes(top, left_line, right_line)

    # draw lanes on original undistorted image
    lanes = draw_lanes(undist, left_line.best_fit, right_line.best_fit, Minv)

    # annotate turn radius
    _annotate_image(lanes, left_line.radius_of_curvature)

    return lanes


class Line():
    """
    This class is used to store state from computations on previous frames.
    """
    FRAME_HISTORY = 5

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # all fit values
        self.all_xfitted = []

        # all radiuses
        self.all_radius = []

    @property
    def bestx(self):
        """Returns fitted polynomial"""
        return np.poly1d(self.best_fit)


    def _reset_state(self):
        """
        Reset current state of line. Note, this does not erase the historical
        parameters.
        """
        self.detected = False
        self.current_fit = [np.array([False])]
        self.allx = None
        self.ally = None


    def _update_state(self, x, y):
        """
        Update line attributes given successful line detection.

        :param x: (ndarray) x coordinates of lane pixels
        :param y: (ndarray) y coordinates of lane pixels
        """
        # Compute curvature of lane
        radius_of_curvature = self._compute_turn_radius(x, y, 360)
        if self.is_radius_outlier(radius_of_curvature):
            self._reset_state()
            return
        self.radius_of_curvature = radius_of_curvature

        # Fit second order polynomial to lane
        new_fit = np.polyfit(y, x, 2)
        if self.is_fit_outlier(new_fit):
            self._reset_state()
            return
        self.current_fit = new_fit

        # Update averaged line properties
        self.recent_xfitted.insert(0, self.current_fit)
        if len(self.recent_xfitted) > self.FRAME_HISTORY:
            self.recent_xfitted.pop()

        self.allx = x
        self.ally = y
        self.best_fit = np.average(self.recent_xfitted, axis=0)

        # Temporary complete histories
        self.all_xfitted.append(new_fit)
        self.all_radius.append(self.radius_of_curvature)

    def compute_line_properties(self, x, y):
        """
        Compute line attributes from x and y pixel locations of lane.

        :param x: (ndarray) x coordinate of lane pixels
        :param y: (ndarray) y coordinate of lane pixels
        """
        # Check if lane pixels were found
        if x.size == 0 or y.size == 0:
            self._reset_state()
        else:
            self._update_state(x, y)

    @staticmethod
    def _compute_turn_radius(x, y, image_height):
        """
        This function computes the turn radius of the lane.

        :param leftx: (ndarray) left lane x coordinates
        :param lefty: (ndarray) left lane y coordinates
        :param rightx: (ndarray) right lane x coordinates
        :param righty: (ndarray) right lane y coordinates
        :return: (tuple) left and right lane turn radius in meters
        """
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * image_height * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

        return np.round(curverad, decimals=2)

    def is_radius_outlier(self, radius_of_curvature):
        """Returns True if new radius is outlier."""
        if self.radius_of_curvature:
            return np.abs(radius_of_curvature - self.radius_of_curvature) > 2500
        return False

    def is_fit_outlier(self, new_fit):
        """Returns True if new fit is outlier."""
        if self.detected:
            diff = np.abs(new_fit - self.current_fit)
            if diff[0] > 0.00003 or diff[1] > 0.03:
                return True
        return False


def find_lanes(img, left_line=None, right_line=None, num_windows=9, window_margin=100, minimum_pixels=50):
    """
    This function applies a sliding window technique to identify the lanes and
    fit a 2nd order polynomial to them.

    :param img: (binary) thresholded and warped image
    :param left_line: (Line) left lane line
    :param right_line: (Line) right lane line
    :param num_windows: (int) number of vertical windows
    :param window_margin: (int) window width +/- margin
    :param minimum_pixels: (int) number of pixels required to move window center
    :return: (tuple) left lane line coefficients, right lane line coefficients,
        left lane curve radius (meters), right lane curve radius (meters)
    """
    if left_line is None:
        left_line = Line()
    if right_line is None:
        right_line = Line()

    # Image dimensions
    image_height, image_width = img.shape

    # Take a histogram of the whole of the image
    histogram = np.sum(img[:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(image_width / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(image_height / num_windows)

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
        win_y_low = image_height - (window + 1) * window_height
        win_y_high = image_height - window * window_height
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


    # Extract left and right lane pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Compute line attributes
    left_line.compute_line_properties(leftx, lefty)
    right_line.compute_line_properties(rightx, righty)

    return left_line, right_line


def _draw_pw_lines(img, pts, color):
    num_points = pts.shape[0]
    for i in range(num_points):
        if i < num_points-1:
            x1 = pts[i, 0]
            y1 = pts[i, 1]
            x2 = pts[i + 1, 0]
            y2 = pts[i + 1, 1]
            cv2.line(img, (x1, y1), (x2, y2), color, 50)


def _annotate_image(image, curve_radius):
    """
    Annotate the image with radius of curvature. This procedure augments the
    input image.

    :param image: (ndarray) image to be annotated
    :param curve_radius: (float) radius of curvature (meters)
    """
    radius_text = 'Turn Radius: {} meters'.format(curve_radius)

    # set coordinates for annotated text
    xpos = np.int(image.shape[1] * 0.1)
    ypos = np.int(image.shape[0] * 0.1)

    # set font to use
    font = cv2.FONT_HERSHEY_SIMPLEX

    # add text to image
    cv2.putText(image, radius_text, (xpos, ypos), font, 2, (255, 255, 255), 2,
                cv2.LINE_AA)


def draw_lanes(undistorted, rfit, lfit, Minv):
    warp = np.zeros_like(undistorted[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp, warp, warp))

    height, width = undistorted.shape[0], undistorted.shape[1]
    yvals = np.linspace(height*0.1, height, height*0.9)

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
