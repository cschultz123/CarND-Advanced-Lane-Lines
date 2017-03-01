import cv2
import image_utils
import numpy as np
from scipy.signal import find_peaks_cwt


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

    # filter and isolate right and left lanes
    right, left = find_lanes(top, windows, peak_offset)

    # fit 2nd order polynomial to left and right lanes
    rpts, lpts = fit_lanes(right, left)

    # draw lanes on original undistorted image
    return draw_lanes(undist, rfit=rpts, lfit=lpts, Minv=Minv)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def find_lanes(img, windows=5, peak_offset=50):
    # copy input image
    top_down = np.copy(img).astype(np.uint8)

    # create masks
    l_mask = np.zeros_like(img, dtype=np.uint8)
    r_mask = np.zeros_like(img, dtype=np.uint8)

    # image dimensions
    height, width = top_down.shape[0], top_down.shape[1]

    # window parameters
    window_size = int(height / (windows))

    # compute initial peaks using majority of image
    mean_lane = np.mean(top_down[height / 2:, :], axis=0)
    mean_lane = moving_average(mean_lane, width / 20)
    idx = find_peaks_cwt(mean_lane, [100], max_distances=[800])

    # right and left peak indicies
    r_peak = np.max(idx[:2])
    r_peak_last = r_peak
    l_peak = np.min(idx[:2])
    l_peak_last = l_peak

    # peak location delta
    dr = 0
    dl = 0

    for i in range(windows):

        y_min = i * window_size
        y_max = y_min + window_size

        # generate horizontal slice
        hslice = top_down[y_min:y_max, :]

        # find peaks
        mean_lane = np.mean(hslice, axis=0)
        mean_lane = moving_average(mean_lane, width / 20)
        idx = find_peaks_cwt(mean_lane, [100], max_distances=[800])

        # two peaks found
        if len(idx) >= 2:
            r_peak = np.max(idx[:2])
            l_peak = np.min(idx[:2])

        # one peak found
        elif len(idx) == 1:
            if np.abs(r_peak - idx[0]) < np.abs(l_peak - idx[0]):
                r_peak = idx[0]
                l_peak = l_peak_last + dl
            else:
                l_peak = idx[0]
                r_peak = r_peak_last + dr

        # no peaks found
        else:
            r_peak = r_peak_last + dr
            l_peak = l_peak_last + dl

        # outlier detection
        if np.abs(r_peak - r_peak_last) > 100:
            r_peak = r_peak_last
        if np.abs(l_peak - l_peak_last) > 100:
            l_peak = l_peak_last

        # populate lane mask for slice
        r_min, r_max = np.clip([r_peak - peak_offset, r_peak + peak_offset], 0, width)
        l_min, l_max = np.clip([l_peak - peak_offset, l_peak + peak_offset], 0, width)
        r_mask[y_min:y_max, r_min:r_max] = 1
        l_mask[y_min:y_max, l_min:l_max] = 1

        if i > 0:
            dr = r_peak - r_peak_last
            dl = l_peak - l_peak_last

        r_peak_last = r_peak
        l_peak_last = l_peak

    return np.bitwise_and(top_down, r_mask), np.bitwise_and(top_down, l_mask)


def fit_lanes(right_lane, left_lane):
    # image dimensions
    height, width = left_lane.shape[:2]

    # fit right lane
    r_lane_pts = np.argwhere(right_lane == 1)
    r_fit = np.polyfit(r_lane_pts[:, 0], r_lane_pts[:, 1], deg=2)
    r_y = np.arange(11) * height / 10
    r_x = r_fit[0] * r_y ** 2 + r_fit[1] * r_y + r_fit[2]

    # fit left lane
    l_lane_pts = np.argwhere(left_lane == 1)
    l_fit = np.polyfit(l_lane_pts[:, 0], l_lane_pts[:, 1], deg=2)
    l_y = np.arange(11) * height / 10
    l_x = l_fit[0] * l_y ** 2 + l_fit[1] * l_y + l_fit[2]

    return np.array((r_x, r_y)).T, np.array((l_x, l_y)).T


def _draw_pw_lines(img, pts, color):
    # draw lines
    pts = np.int_(pts)
    for i in range(10):
        x1 = pts[0][i][0]
        y1 = pts[0][i][1]
        x2 = pts[0][i + 1][0]
        y2 = pts[0][i + 1][1]
        cv2.line(img, (x1, y1), (x2, y2), color, 50)


def draw_lanes(undistorted, rfit, lfit, Minv):
    warp = np.zeros_like(undistorted[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp, warp, warp))

    pts = np.vstack((lfit, rfit[::-1, :]))

    cv2.fillPoly(color_warp, np.uint([pts]), (0, 255, 255))

    left_lane_color = (255, 255, 0)  # yellow
    right_lane_color = (255, 255, 255)  # white

    # draw lane lines using fit
    _draw_pw_lines(color_warp, np.int_([lfit]), left_lane_color)
    _draw_pw_lines(color_warp, np.int_([rfit]), right_lane_color)

    # revert image back to original perspective
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0]))

    result = cv2.addWeighted(undistorted, 1, newwarp, 0.5, 0)

    return result
