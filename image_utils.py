import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def perspective_transform(image, src, dst):
    """
    Apply perspective transform using source and destination points.

    IMPORTANT: Image must be undistorted!

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
    src = np.float32([(x_center - 0.2 * x_center, y_top),
                      (x_center + 0.2 * x_center, y_top),
                      (x_center - x_center, height),
                      (x_center + x_center, height)])

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
    """
    This method will apply a guassian blur to the image.

    :param img (ndarray): (rgb) image
    :param kernel_size (int): guassian kernel dimensions
    :return: blurred image
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    """
    This method will apply Sobel operator on image along x or y. The resulting
    matrix is then thresholded and a binary mask is returned.

    :param img (ndarray): (rgb) image
    :param orient (str): orientation 'x' or 'y'
    :param thresh (tuple): (int) minimum threshold, (int) maximum threshold
    :return: binary mask
    """
    # convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # appy sobel in specified orientation
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # convert to 8 bit
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # generate binary mask
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """
    This method will apply the Sobel operator in both x and y. The magnitude
    of the two resulting matricies is computed. Finally, the result is
    thresholded and a binary mask is returend.

    :param img (ndarray): (rgb) image
    :param sobel_kernel (int): Sobel kernel dimensions
    :param mag_thresh (tuple): (int) minimum threshold, (int) maximum threshold
    :return: binary mask
    """
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # compute both x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # convert to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # generate binary mask
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    This method will apply the Sobel operator in both x and y orientations.
    The gradient direction will the be computed using the two matricies. The
    gradient direction matrix with then be thresholded and a binary mask will
    be returned.

    :param img (ndarray): (rgb) image
    :param sobel_kernel (int): Sobel kernel dimensions
    :param thresh (tuple): minimum angle, maximum angle (radians)
    :return: binary mask
    """
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # compute both x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

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

