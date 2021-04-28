import numpy as np
import cv2
import datetime
import line as Ln
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt

# Camera Calibration
import plotter

ret, mtx, dist, rvecs, tvecs = Ln.camera_calibrate(fig_view=False)

# Create a list of test images
test_images = os.listdir("test_images")


def pipeline(pipeline_img):
    # Initialize lane lines and lane objects
    left_lane = Ln.Line()
    right_lane = Ln.Line()
    lane = Ln.Lane()

    # 1) Undistorted image
    undistorted_img = Ln.undistort(pipeline_img, mtx, dist)

    # 2) Threshold image from color transforms, gradients, and masks, etc.
    # Image with Thresholding
    thresh_bin, thresh_col = Ln.threshold_image(undistorted_img)
    mask_bin, mask_col = Ln.mask_color(undistorted_img)
    combined_bin = Ln.combine_threshold(thresh_bin, mask_bin)

    # Crop Image using ROI
    cropped_img = Ln.crop_image(combined_bin)

    # 3) Bird's eye view image
    top_view, M, Minv = Ln.transform_image(cropped_img)



    # 4) Lane pixels on the warped image
    # Sliding window search if no fit is available
    if (left_lane.detected is False) or (right_lane.detected is False):
        try:
            left_fit, right_fit, lanes_colored = Ln.sliding_windows(top_view)
        # Use previous fit if no fit can be found
        except TypeError:
            left_fit = left_lane.previous_fit
            right_fit = right_lane.previous_fit
            lanes_colored = np.zeros_like(pipeline_img)
    # Search around poly if fit is available to decrease run time
    else:
        try:
            left_fit, right_fit, lanes_colored = Ln.search_around_poly(top_view, left_lane.previous_fit,
                                                                       right_lane.previous_fit)
        except TypeError:
            try:
                left_fit, right_fit, lanes_colored = Ln.sliding_windows(top_view)
            # Use previous fit if no fit can be found
            except TypeError:
                left_fit = left_lane.previous_fit
                right_fit = right_lane.previous_fit
                lanes_colored = np.zeros_like(pipeline_img)

    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit

    # Calculate base position of lane lines to get lane distance
    left_lane.line_base_pos = left_fit[0] * (top_view.shape[0] - 1) ** 2 + left_fit[1] * (top_view.shape[0] - 1) + \
                              left_fit[2]
    right_lane.line_base_pos = right_fit[0] * (top_view.shape[0] - 1) ** 2 + right_fit[1] * (top_view.shape[0] - 1) + \
                               right_fit[2]
    left_lane.line_mid_pos = left_fit[0] * (top_view.shape[0] // 2) ** 2 + left_fit[1] * (top_view.shape[0] // 2) + \
                             left_fit[2]
    right_lane.line_mid_pos = right_fit[0] * (top_view.shape[0] // 2) ** 2 + right_fit[1] * (top_view.shape[0] // 2) + \
                              right_fit[2]

    # Sanity check by top and bottom position of lane lines
    lane.top_w = right_fit[2] - left_fit[2]
    lane.bottom_w = right_lane.line_base_pos - left_lane.line_base_pos
    lane.middle_w = right_lane.line_mid_pos - left_lane.line_mid_pos

    # Check if values make sense
    if Ln.check_results(left_lane, right_lane, lane) is False:
        # Use previous values and report no lane found for bad values
        if len(left_lane.previous_fits) == 5:
            diff_left = [0.0, 0.0, 0.0]
            diff_right = [0.0, 0.0, 0.0]
            for i in range(0, 3):
                for j in range(0, 3):
                    diff_left[i] += left_lane.previous_fits[j][i] - left_lane.previous_fits[j + 1][i]
                    diff_right[i] += right_lane.previous_fits[j][i] - right_lane.previous_fits[j + 1][i]

                diff_left[i] /= 4
                diff_right[i] /= 4

            for i in range(0, 3):
                left_lane.current_fit[i] = left_lane.previous_fit[i] + diff_left[i]
                right_lane.current_fit[i] = right_lane.previous_fit[i] + diff_right[i]
            print("prev: ", left_lane.previous_fit)
            print("diff: ", diff_left)
            print("fit: ", left_lane.current_fit)

            left_lane.detected = False
            right_lane.detected = False
        else:
            left_lane.current_fit = left_lane.previous_fit
            right_lane.current_fit = right_lane.previous_fit
            left_lane.detected = False
            right_lane.detected = False

    else:
        # Use current values and report that lanes were found for good values
        if not left_lane.detected or not right_lane.detected:
            left_lane.previous_fits.clear()
            right_lane.previous_fits.clear()
        left_lane.detected = True
        right_lane.detected = True
        left_lane.initialized = True
        right_lane.initialized = True
        left_lane.frame_ind += 1
        right_lane.frame_ind += 1

    # Average of the recent fits  will be set as the current fit
    left_lane.average_fit = Ln.fit_average(top_view.shape, left_lane)
    right_lane.average_fit = Ln.fit_average(top_view.shape, right_lane)

    lane.mean_b_w, lane.mean_t_w = Ln.width_average(top_view.shape, lane)

    # 5) Curvature and vehicle offset
    left_lane.radius_of_curvature = Ln.curvature_image(top_view.shape, left_fit)
    right_lane.radius_of_curvature = Ln.curvature_image(top_view.shape, right_fit)
    curvature = (left_lane.radius_of_curvature + right_lane.radius_of_curvature) / 2

    left_lane.line_base_pos = left_fit[0] * (top_view.shape[0] - 1) ** 2 + left_fit[1] * (top_view.shape[0] - 1) + \
                              left_fit[2]
    right_lane.line_base_pos = right_fit[0] * (top_view.shape[0] - 1) ** 2 + right_fit[1] * (top_view.shape[0] - 1) + \
                               right_fit[2]

    vehicle_offset = Ln.vehicle_offset(top_view.shape[1], left_lane.line_base_pos, right_lane.line_base_pos)

    # 6) Final output with marked lanes on the original image
    final_image = Ln.lanes_image(top_view, undistorted_img, left_lane.average_fit, right_lane.average_fit, curvature,
                                 vehicle_offset, Minv)

    # Set current values as previous values for next frame
    left_lane.previous_fit = left_lane.current_fit
    right_lane.previous_fit = right_lane.current_fit

    # Reset / empty current fit
    left_lane.current_fit = [np.array([False])]
    right_lane.current_fit = [np.array([False])]

    return final_image


# Functions for testing the pipeline
def video_test(video, length):

    if video == "image":
        print("Pipeline is being tested on test images...")
        image_process(test_images)

    elif video == "video1":
        print("Pipeline is being tested on project video...")
        video_process(1, length)

    elif video == "video2":
        print("Pipeline is being tested on challenge video...")
        video_process(2, length)

    elif video == "video3":
        print("Pipeline is being tested on harder challenge project video... Fingers crossed X- -X ")
        video_process(3, length)


def image_process(test_images):
    # Grab all test images and run the pipeline
    for test_img in test_images:
        test_image = mpimg.imread("test_images/" + test_img)
        result = pipeline(test_image)
        # Save final result
        image_name_init = 'output_images/Results_'
        cv2.imwrite(image_name_init + test_img, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def video_process(video, length="long"):
    # Grab all test images and run the pipeline
    test_videos = os.listdir("test_videos")
    if video == 1:
        video_name = test_videos[1]
    elif video == 2:
        video_name = test_videos[0]
    elif video == 3:
        video_name = test_videos[2]

    # Sub-clip the video for shorter run time on tests
    if length == "long":
        test_input = VideoFileClip('test_videos/' + video_name)
    elif length == "short":
        test_input = VideoFileClip('test_videos/' + video_name).subclip(0, 6)

    # Use datetime to record with different file names
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_")
    video_full_name = 'output_videos/output_' + date + video_name

    # Process input video, write to output file
    test_output = test_input.fl_image(pipeline)
    # test_output = test_input.fl_image(lambda inp_img: pipeline(inp_img))
    test_output.write_videofile(video_full_name, audio=False)
