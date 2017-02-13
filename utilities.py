import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks_cwt


def calibrate_camera(calibration_images, grid=(9,6), plot=False):
    """
    This method will compute the distortion parameters  of the camera using
    the calibration images provided and the grid dimensions.

    :param calibration_images (iterable): calibration images
    :param grid (tuple): chess board dimensions e.g. (10,10)
    :param plot (bool): plot chessboard corners for each calibration image
    :return (tuple): camera matrix and distortion coefficients
    """

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
    objp = np.zeros((grid[0] * grid[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for img in calibration_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if plot:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                plt.figure()
                plt.imshow(img)

    # compute camera matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist


def undistort(image, camera_matrix, distortion):
    """
    This method will undistort an image using the camera matrix and distortion
    coefficients. Use calibrate_camera method to compute camera matrix and
    distortion coefficients.

    :param image: image
    :param camera_matrix (3x3): camera matrix
    :param distortion (1x5): distortion coefficients
    :return (image): returns undistorted image
    """
    return cv2.undistort(image, camera_matrix, distortion, None, camera_matrix)


def warp(image, src, dst):
    """
    Apply perspective transform using source and destination points.

    :param image: image
    :param src (list): points that form a shape in image space
    :param dst (list): points that form the desired shape in the transformed image
    :return (image): warped image
    """
    img_size = (image.shape[1], image.shape[0])

    img_shape = image.shape

    # image dimensions
    height = img_shape[0]
    width = img_shape[1]

    # window parameters
    x_center = width / 2
    y_top = height / 1.5

    # source trapezoid coordinates
    src = np.float32([(x_center + 0.2 * x_center, y_top),
                      (x_center - 0.2 * x_center, y_top),
                      (x_center + x_center, height),
                      (x_center - x_center, height)])

    # destination rectangle coordinates
    dst = np.float32([(0, 0),
                      (width, 0),
                      (0, height),
                      (width, height)])

    # forward transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # inverse transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    return cv2.warpPerspective(image, M, img_size), M, Minv


def guassian_blur(img, kernel_size=3):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def pipeline(img, camera_matrix, distortion):

    # undistort image
    undistorted = undistort(img, camera_matrix, distortion)
    
    blur = guassian_blur(undistorted, kernel_size=5)
    
    # Apply color thresholding
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    sat = hsv[:,:,1]
    sat_binary = np.zeros_like(sat)
    sat_binary[(sat > 160) & (sat < 255)] = 1
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(blur, orient='x', thresh_min=10, thresh_max=255)
    grady = abs_sobel_thresh(blur, orient='y', thresh_min=60, thresh_max=255)
    mag_binary = mag_thresh(blur, mag_thresh=(40, 255))
    dir_binary = dir_threshold(blur, thresh=(.65, 1.05))

    # Combine four masks
    combined_gradient = np.zeros_like(dir_binary)
    combined_gradient[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    # Combine color and gradient thresholds
    color_grad_binary = np.zeros_like(combined_gradient)
    color_grad_binary[(sat_binary > 0) | (combined_gradient > 0)] = 1 
    
    # image dimensions
    img_shape = undistorted.shape
    height = img_shape[0]
    width = img_shape[1]

    # window parameters
    x_center = width / 2
    y_top = height / 1.5

    # source trapezoid coordinates
    src = np.float32([(x_center + 0.9 * x_center, height),
                      (x_center + 0.2 * x_center, y_top),
                      (x_center - 0.2 * x_center, y_top),
                      (x_center - 0.9 * x_center, height)])

    # destination rectangle coordinates
    dst = np.float32([(0, 0),
                      (width, 0),
                      (0, height),
                      (width, height)])
    
    # Apply region of interest
    # final_mask = region_of_interest(color_grad_binary, np.uint([src]))
    
    # Apply perspective transform
    perspective, M, Minv = warp(color_grad_binary, src, dst)

    return perspective.astype(np.uint8), M, Minv


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
    window_size = int(height/(windows))
    
    # compute initial peaks using majority of image
    mean_lane = np.mean(top_down[height/2:, :], axis=0)
    mean_lane = moving_average(mean_lane, width/20)
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
        
        y_min = i*window_size
        y_max = y_min + window_size
        
        # generate horizontal slice
        hslice = top_down[y_min:y_max, :]
        
        # find peaks
        mean_lane = np.mean(hslice, axis=0)
        mean_lane = moving_average(mean_lane, width/20)
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
        r_min, r_max = np.clip([r_peak-peak_offset, r_peak+peak_offset], 0, width)
        l_min, l_max = np.clip([l_peak-peak_offset, l_peak+peak_offset], 0, width)
        r_mask[y_min:y_max, r_min:r_max] = 1
        l_mask[y_min:y_max, l_min:l_max] = 1
        
        if i > 0:
            dr = r_peak - r_peak_last
            dl = l_peak - l_peak_last
            
        r_peak_last = r_peak
        l_peak_last = l_peak
            
    return np.bitwise_and(top_down, r_mask), np.bitwise_and(top_down, l_mask)


def fit_lanes(left_lane, right_lane, point_offset=20):
    
    # image dimensions
    height, width = left_lane.shape[:2]
    
    # fit right lane
    r_lane_pts = np.argwhere(right_lane == 1)
    r_fit = np.polyfit(r_lane_pts[:,0], r_lane_pts[:,1], deg=2)
    r_y = np.arange(11)*height/10
    r_x = r_fit[0]*r_y**2 + r_fit[1]*r_y + r_fit[2]

    # fit left lane
    l_lane_pts = np.argwhere(left_lane == 1)
    l_fit = np.polyfit(l_lane_pts[:,0], l_lane_pts[:,1], deg=2)
    l_y = np.arange(11)*height/10
    l_x = l_fit[0]*l_y**2 + l_fit[1]*l_y + l_fit[2]
    
    return np.array((r_x, r_y)).T, np.array((l_x, l_y)).T
    

def draw_pw_lines(img,pts,color):
    # draw lines
    pts = np.int_(pts)
    for i in range(10):
        x1 = pts[0][i][0]
        y1 = pts[0][i][1]
        x2 = pts[0][i+1][0]
        y2 = pts[0][i+1][1]
        cv2.line(img, (x1, y1), (x2, y2),color,50)


def draw_lanes(undistorted, top, r_fit, l_fit, Minv):
    warp = np.zeros_like(top).astype(np.uint8)
    color_warp = np.dstack((warp, warp, warp))
    
    pts = np.vstack((l_fit, r_fit[::-1,:]))

    cv2.fillPoly(color_warp, np.uint([pts]), (0,255,255))
    
    left_lane_color = (255,255,0)        # yellow
    right_lane_color = (255,255,255)     # white

    # draw lane lines using fit
    draw_pw_lines(color_warp,np.int_([l_fit]),left_lane_color)
    draw_pw_lines(color_warp,np.int_([r_fit]),right_lane_color)

    # revert image back to original perspective
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0])) 

    result = cv2.addWeighted(undistorted, 1, newwarp, 0.5, 0)
    
    return result
