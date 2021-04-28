import matplotlib.pyplot
import numpy as np
import cv2
import glob
import camera_calibration as cc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import pickle
import plotter


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # previous fits
        self.previous_fits = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # x values of the current fit
        self.current_fitx = None
        # polynomial coefficients for the most recent fit
        self.current_fit = np.array([0, 0, 0])  # [np.array([False])]
        # polynomial coefficients of the previous fits
        self.previous_fit = np.array([0, 0, 0])
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # status of the first frame
        self.initialized = False
        # average fit
        self.average_fit = np.array([0, 0, 0])
        # average curvature
        self.average_curvature = 0
        # previous curves
        self.previous_curves = []
        # frame index
        self.frame_ind = 0


class Lane:
    def __init__(self):
        self.bottom_w = 0
        self.top_w = 0
        self.mean_b_w = 0
        self.mean_t_w = 0
        self.recent_b_w = []
        self.recent_t_w = []


# HELPER FUNCTIONS
def fit_poly(leftx, lefty, rightx, righty):
    """
    Fits polynomials to given left and right y vs x values
    """
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def compute_x_from_polyfit(img_shape, left_fit, right_fit):
    """
    Returns x values of the polynomial fit in given image
    """
    # y values
    ploty = np.linspace(0, img_shape[0] -1, img_shape[0])
    # x values
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx


def append_to_previous(current_val, deq, n=3):
    """
    Manages list storage for averaging
    """
    # add the current value to the list if not full
    if len(deq) < n:
        deq.append(current_val)
    # remove the oldest and add the current if full
    if len(deq)==n:
        deq.pop(n - 1)
        deq.insert(0, current_val)


def find_average(deq, n=3):
    average_fit = [0, 0, 0]

    if type(deq[0])=='list':
        if len(deq) > 0:
            for i in range(3):
                tot = 0
                for j in range(0, len(deq)):
                    tot = tot + deq[j][i]
                average_fit[i] = tot / len(deq)
            return average_fit
    else:
        if len(deq) > 0:
            for i in range(0, deq):
                tot = tot + deq[i]
            average = tot / len(deq)
            return average


def calibrate_camera(nx, ny):
    # Create a list of calibration images
    calibration_images = glob.glob("camera_cal/calibration*.jpg")
    # Arrays to store object points and image points from all calibration images
    objpoints = []
    imgpoints = []
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates

    for img_name in calibration_images:
        # Read each image in the directory
        img = mpimg.imread(img_name)
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If corners are detected, add object points and image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    return objpoints, imgpoints


def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100  # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


def camera_calibrate(fig_view=False):
    # Pick a random image to return mtx and dist values
    img = mpimg.imread("camera_cal/calibration8.jpg")
    # Calibration file
    file = "camera_cal/camera_calibration.pickle"
    # number of corners in ChessBoard
    nx = 9
    ny = 6

    # Calibrate only if not previously calibrated
    if not os.path.exists(file):
        print("Calibrating camera...")
        objpoints, imgpoints = calibrate_camera(nx, ny)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)

        data = [objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs]

        with open('camera_cal/camera_calibration.pickle', 'wb') as file:
            pickle.dump(data, file)

        del data
    # load calibration pickle if calibrated
    else:
        print("Calibration exists, loading the calibration file...")
        with open(file, 'rb') as f:
            data = pickle.load(f)
            objpoints = data[0]
            imgpoints = data[1]
            ret = data[2]
            mtx = data[3]
            dist = data[4]
            rvecs = data[5]
            tvecs = data[6]
            del data

    if fig_view:
        print("Plotting unidstorted image...")
        und_img = undistort(img, mtx, dist)
        plotter.plot_images([img, und_img], ["Original Calibration Image", "Undistorted Image"], [1, 2], [0, 0], "Testing_Undistort_on_Calibration_Images")
        print("Plotting warped image...")
        warped_img, M = corners_unwarp(img, nx, ny, mtx, dist)
        plotter.plot_images([und_img, warped_img], ["Undistorted Image", "Unwarped Image with Corners"], [1, 2], [0, 0], "Testing_Warp_on_Calibration_Images")

    return ret, mtx, dist, rvecs, tvecs


# MAIN FUNCTIONS

def undistort(img, mtx, dist):
    undist_image = cv2.undistort(img, mtx, dist, None, mtx)

    return undist_image


def transform_image(img):
    img_size = (img.shape[1], img.shape[0])

    # Define source and destination points based on a straight road section
    src_pts = np.float32([[556, 480], [734, 480], [220, 720], [1105, 720]])
    dst_pts = np.float32([[300, 0], [900, 0], [300, 720], [900, 720]])

    # Calculate transform matrix and inverse transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    transformed = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return transformed, M, Minv


def mask_color(img, fig_view=False):
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask_yellow = cv2.inRange(lab_img, np.array([100, 100, 150], dtype=np.uint8),
                              np.array([220, 180, 255], dtype=np.uint8))

    mask_white = cv2.inRange(hls_img, np.array([0, 200, 0], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))

    kernel = np.ones((5, 5), np.uint8)
    mask_white_enh = cv2.dilate(mask_white, kernel, iterations=2)
    mask_white_enh = cv2.erode(mask_white, kernel, iterations=2)
    mask_white_enh = cv2.dilate(mask_white, kernel, iterations=1)

    mask_combine = mask_white_enh + mask_yellow
    masked_image = np.copy(img)
    masked_image = cv2.dilate(masked_image, kernel=np.ones((5, 5), np.uint8), iterations=2)
    masked_image = cv2.erode(masked_image, kernel=np.ones((5, 5), np.uint8), iterations=2)
    masked_image = cv2.dilate(masked_image, kernel=np.ones((5, 5), np.uint8), iterations=1)
    masked_image[mask_combine != 255] = [0, 0, 0]

    masked_bin = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    masked_bin[masked_bin != 0] = 1

    if fig_view:
        images = [mask_yellow, mask_white, masked_bin]
        titles = ["Yellow lines", 'White lines', 'Combined lines']
        plotter.plot_images(images, titles, [1, 3], [1, 1, 1], "Testing_Lane_Color_Thresholding")

    return masked_bin, masked_image


def threshold_image(img, l_perc=(80, 100), b_thresh=(140, 200), sx_perc=(90, 100), fig_view=False):

    # Make a copy of the image
    img = np.copy(img)

    # Convert to Lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:, :, 2]

    # Convert to LUV color space
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    l_channel = luv[:, :, 0]

    # Create percentile-based thresholds
    l_thresh_min = np.percentile(l_channel, l_perc[0])
    l_thresh_max = np.percentile(l_channel, l_perc[1])

    # Threshold b color channel
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    # Threshold l color channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # Find edges with Sobelx
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Create percentile-based thresholds
    sx_thresh_min = np.percentile(scaled_sobel, sx_perc[0])
    sx_thresh_max = np.percentile(scaled_sobel, sx_perc[1])

    # Threshold edges (x gradient)
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh_min) & (scaled_sobel <= sx_thresh_max)] = 1

    # Get white edges
    sobel_white_binary = np.zeros_like(l_channel)
    sobel_white_binary[(sx_binary == 1) & (l_binary == 1)] = 1

    # Get yellow edges
    sobel_yellow_binary = np.zeros_like(l_channel)
    sobel_yellow_binary[(sx_binary == 1) & (b_binary == 1)] = 1

    # Output image for debugging
    white_sobelx_and_color = np.dstack(
        (sobel_white_binary, sobel_yellow_binary, np.zeros_like(sobel_white_binary))) * 255

    # Output image for pipeline
    combined_binary_sobel = np.zeros_like(b_binary)
    combined_binary_sobel[(sobel_white_binary == 1) | (sobel_yellow_binary == 1)] = 1

    if fig_view:
        print("Plotting image binaries based on thresholds...")
        images = [img, b_binary, l_binary, scaled_sobel, sx_binary, sobel_white_binary, sobel_yellow_binary, combined_binary_sobel]
        labels = ["Original Image", "B-Channel Threshold", "L-Channel Threshold", "Scaled Sobel", "S_x Binary", "Sobel_White", "Sobel_Yellow", "Combined_Binary"]
        cmaps = [0, 1, 1, 1, 1, 1, 1, 1]
        plotter.plot_images(images, labels, [2, 4], cmaps, "Testing_Thresholding")

    return combined_binary_sobel, white_sobelx_and_color


def combine_threshold(thresh_bin, mask_bin):

    combined_bin = np.zeros_like(thresh_bin)
    combined_bin[(thresh_bin == 1) | (mask_bin == 1)] = 1

    return combined_bin


def crop_image(img):
    mask = np.zeros_like(img)
    vertices = np.array([[(0, 720), (500, 450),
                          (780, 450), (1280, 720)]], dtype=np.int32)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    cropped_img = cv2.bitwise_and(img, mask)
    return cropped_img


def sliding_windows(img, margin=80, minpix=50, fig_view=False):
    # Output image initializing
    out_img = np.dstack((img, img, img))
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum across image pixels vertically
    histogram = np.sum(bottom_half, axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 20

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)

    # For sanity check
    if fig_view:
        left_fit_text = "left: %.6f %.6f %.6f" % (left_fit[0], left_fit[1], left_fit[2])
        right_fit_text = "right: %.6f %.6f %.6f" % (right_fit[0], right_fit[1], right_fit[2])
        cv2.putText(out_img, left_fit_text, (380, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), thickness=1)
        cv2.putText(out_img, right_fit_text, (380, 100), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), thickness=1)

    if fig_view:
        print("Plotting lane line curve fitting on warped image...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
        plt.tight_layout()
        ax1.imshow(bottom_half, cmap='gray', aspect='auto')
        ax1.set_title("Bottom Half of Bin Image")
        ax2.imshow(out_img)
        ax2.imshow(img, cmap='gray', aspect='auto')
        ax2.set_title("Binary Image")

        ax3.plot(histogram)
        ax3.set_title("Histogram")

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        ax4.imshow(out_img, aspect='auto')
        ax4.plot(left_fitx, ploty, color='yellow')
        ax4.plot(right_fitx, ploty, color='yellow')
        ax4.set_title("Results from Curve Fitting")

        ax3.set_xlim(0, 1280)
        plt.show()
        fig.savefig("output_images/Test_Image_Results.jpg")

    return left_fit, right_fit, out_img  # leftx, rightx, lefty, righty


def search_around_poly(img, left_fit, right_fit, margin=100, fig_view=False):
    # Grab activated pixels
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    left_fitx, right_fitx = compute_x_from_polyfit(img.shape, left_fit, right_fit)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window

    out_img = np.dstack((img, img, img)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # For Sanity check
    left_fit_text = "left: %.6f %.6f %.6f" % (left_fit[0], left_fit[1], left_fit[2])
    right_fit_text = "right: %.6f %.6f %.6f" % (right_fit[0], right_fit[1], right_fit[2])

    # Text on the output image
    cv2.putText(out_img, left_fit_text, (380, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), thickness=1)
    cv2.putText(out_img, right_fit_text, (380, 100), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), thickness=1)

    return left_fit, right_fit, out_img  # , rightx, lefty, righty


def curvature_image(img_shape, fit):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Generate y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    # Calculate x values using polynomial coeffs
    fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

    # Evaluate at bottom of image
    y_eval = np.max(ploty)

    # Fit curves with corrected axes
    curve_fit = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)

    # Calculate curvature values for left and right lanes
    curvature = ((1 + (2 * curve_fit[0] * y_eval * ym_per_pix + curve_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * curve_fit[0])

    return curvature


def vehicle_offset(img_shape, left_lane_pos, right_lane_pos):
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Calculate position based on midpoint - center of lanes distance
    midpoint = img_shape // 2
    center_of_lanes = (right_lane_pos + left_lane_pos) / 2
    position = midpoint - center_of_lanes

    # Get value in meters
    offset = position * xm_per_pix

    return offset


def check_results(left_lane, right_lane, lane):
    # Compute lane widths at top and bottom of image
    top_w_dif = abs(lane.top_w - lane.mean_t_w)
    bottom_w_dif = abs(lane.bottom_w - lane.mean_b_w)

    # Bound definitions for sanity check
    top_bounds = top_w_dif > 0.2 * lane.mean_t_w or lane.top_w > 1.25 * lane.bottom_w
    bottom_bounds = bottom_w_dif > 0.05 * lane.mean_b_w
    is_intersecting = lane.top_w < 0.0 or lane.bottom_w < 0.0
    curvature_bounds = right_lane.current_fit[0] * left_lane.current_fit[0] < -0.00005 * 0.0001

    # Check bounds
    if (left_lane.frame_ind > 1) and (right_lane.frame_ind > 1):
        if bottom_bounds:
            result = False
        elif top_bounds:
            result = False
        elif is_intersecting:
            result = False
        elif curvature_bounds:
            result = False
        else:
            result = True
    else:
        result = True

    return result


def fit_average(img_shape, lane):

    n = 10
    average_fit = [0, 0, 0]

    # Filling the fits
    if len(lane.previous_fits) < n:
        lane.previous_fits.append(lane.current_fit)
    # Deleting the oldest fit and adding the most recent
    if len(lane.previous_fits) == n:
        lane.previous_fits.pop(n-1)
        lane.previous_fits.insert(0, lane.current_fit)
    # Mean
    if len(lane.previous_fits) > 0:
        for i in range(0, 3):
            tot = 0
            for j in range(0, len(lane.previous_fits)):
                tot = tot + lane.previous_fits[j][i]

            average_fit[i] = tot / len(lane.previous_fits)

    return average_fit


def curvature_average(img_shape, lane):
    tot = 0
    n = 10
    average_curve = 0

    if len(lane.previous_curves) < n:
        lane.previous_curves.append(lane.radius_of_curvature)
    if len(lane.previous_curves) ==n:
        lane.previous_curves.pop(n-1)
        lane.previous_curves.insert(0, lane.radius_of_curvature)

    if len(lane.previous_curves) > 0:
        for i in range(0, len(lane.previous_curves)):
            tot = tot + lane.previous_curves[i]

        average_curve = tot / len(lane.previous_curves)

    return average_curve


def width_average(img_shape, lane):
    tot_bottom = 0
    tot_top = 0
    n = 10
    mean_b_w = 0
    mean_t_w = 0

    if len(lane.recent_b_w) < n:
        lane.recent_b_w.append(lane.bottom_w)
    if len(lane.recent_b_w) == n:
        lane.recent_b_w.pop(n-1)
        lane.recent_b_w.insert(0, lane.bottom_w)
    if len(lane.recent_b_w) > 0:
        for i in range(0, len(lane.recent_b_w)):
            tot_bottom = tot_bottom + lane.recent_b_w[i]
            mean_b_w = tot_bottom / len(lane.recent_b_w)

    if len(lane.recent_t_w) < n:
        lane.recent_t_w.append(lane.top_w)
    if len(lane.recent_t_w) == n:
        lane.recent_t_w.pop(n-1)
        lane.recent_t_w.insert(0, lane.top_w)
    if (len(lane.recent_t_w) > 0):
        for i in range(0, len(lane.recent_t_w)):

            tot_top = tot_top + lane.recent_t_w[i]
            mean_t_w = tot_top / len(lane.recent_t_w)

    return mean_b_w, mean_t_w


def lanes_image(warped, undist, left_fit, right_fit, curvature, position, Minv):

    # Generate y values
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    # Compute x-components
    left_fitx, right_fitx = compute_x_from_polyfit(warped.shape, left_fit, right_fit)

    # Empty image for lanes
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw lane lines
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space by Minv
    new_warp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # Overlay lines with image
    lanes = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)

    # Text on the final result
    curv_text = "Curvature: %.2f meters" % curvature
    pos_text = "Offset: %.2f from center" % position

    # Add text to image
    cv2.putText(lanes, curv_text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
    cv2.putText(lanes, pos_text, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    return lanes


