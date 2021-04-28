import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
import line as Ln
import plotter
import pipeline as Pp
from moviepy.editor import VideoFileClip


# Create a list of test images
test_images = os.listdir("test_images")


# Main Function
if __name__ == '__main__':
    ##################################################################
    # Following are the steps for implementing the overall pipeline. #
    ##################################################################
    test = False
    if test:
        # Camera Calibration
        ret, mtx, dist, rvecs, tvecs = Ln.camera_calibrate(fig_view=True)

        # Pick one of test images for testing
        # for test_img in test_images:
        test_img = test_images[2]  # straight_lines1.jpg
        img_in_test = mpimg.imread("test_images/" + test_img)

        # Distorted Image
        undistorted_img = Ln.undistort(img_in_test, mtx, dist)
        # Plot undistorted image
        print('Plotting undistorted image...')
        plotter.plot_images([img_in_test, undistorted_img], ['Original Image', 'Undistorted Image'],
                            [1, 2], [0, 0], 'Testing_Distortion_Correction')
        # Image with Thresholding
        thresh_bin, thresh_col = Ln.threshold_image(undistorted_img, fig_view=True)
        mask_bin, mask_col = Ln.mask_color(undistorted_img, fig_view=True)
        combined_bin = Ln.combine_threshold(thresh_bin, mask_bin)

        # Crop Image using ROI
        cropped_img = Ln.crop_image(combined_bin)


        # Bird's Eye View Image by Perspective Transform
        top_view, M, Minv = Ln.transform_image(cropped_img)

        # Plot sanity check for perspective transform
        bird_view, Mb, Minb = Ln.transform_image(undistorted_img)
        plotter.plot_trapezoid(undistorted_img, bird_view)

        # Plot those three images
        print("Plotting perspective transform image...")
        plotter.plot_images([combined_bin, cropped_img, top_view], ["Binary Edges", "Cropped Image", "Bird's View"], [1, 3], [1, 1, 1], "Test_Perspective_Transform")

        # Test Sliding Window Line Fitting for a single image
        left_fit, right_fit, output_img = Ln.sliding_windows(top_view, fig_view=True)

    #############################################################
    # Testing the overall pipeline with pipeline function call. #
    #############################################################
    # video = input("Please choose testing input: image, video1, video2, or video3\n")
    # length = input("Please choose video length if input is video: long or short")

    # Uncomment the lines of your choice for different inputs
    #
    # Pp.video_test('image', 'long')
    Pp.video_test('video1', 'short')
    # Pp.video_test('video2', 'long')
    # Pp.video_test('video3', 'long')
