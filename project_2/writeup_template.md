<<<<<<< HEAD
# **Advanced Lane Finding** 

#### In this project, lane finding results from the previous project have been improved by additional thresholding and curvature analyses. 


![all text][gif0]

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Below is a figure that represents the overall pipeline which is implemented to complete the goals above. 

![alt text][image0]

Here is a list of scripts that have been implemented for the project. 

* [main.py](./main.py) : This script includes a test case to implement the pipeline process and overall pipeline for all test images and videos. User can uncomment the input source line. 
* [pipeline.py](./pipeline.py): The source code for the pipeline can be found in this file.
* [line.py](./line.py): Here, there are two classes (Lane and Line) and all other helper and main functions to process images in lane line search. 
* [plotter.py](./plotter.py): This is an extra script for display purposes to save space in the actual script files when plotting figures.

[//]: # (Image References)

[gif0]: ./gifs/short_project_video.gif "Project Video"
[image0]: ./gifs/pipeline.png "Pipeline"
[image1]: ./output_images/Testing_Undistort_on_Calibration_Images.jpg "Undistorted Camera"
[image2]: ./output_images/Testing_Warp_on_Calibration_Images.jpg "Warped Camera"
[image3]: ./output_images/Undistorted_Test_Image.png "Undistorted Test"
[image4]: ./output_images/Testing_Lane_Color_Thresholding.jpg "Mask Color"
[image5]: ./output_images/Testing_Thresholding.jpg "Thresholding Image"
[image6]: ./output_images/Sanity_Check_for_PT.jpg "Transformed Image"
[image7]: ./output_images/Test_Perspective_Transform.jpg "Warped Image"
[image8]: ./output_images/Test_Image_Results.jpg "Line Search"
[image9]: ./output_images/Results_straight_lines1.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

## Pipeline Details 

#### *Here you can find the detailed explanations for each box shown in the pipeline.*

---

### (A) Camera Calibration

The pipeline starts with camera calibration. Two functions are implemented for the calibration (See [line.py](./line.py) at [L111](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L111) and [L74](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L174)). The first function, <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue">calibrate_camera()</span>, reads all the calibration Chessboard images and prepares "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here, it is assumed that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners are detected succesfully in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

In the second function, <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> camera_calibrate()</span>, if a camera calibration file is stored it is loaded, otherwise the first function is called to create `objpoints` and `imgpoints` and then, those are used to compute the camera calibration and distortion coefficients using the <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue">cv2.calibrate_camera()</span> function.  With the camera calibration outputs, a distortion correction <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue">cv2.undistort()</span> to a selected calibration image is applied for testing purposes. A perspective transform is also tested on the same image by the detecting the corners on the Chessboard. Here are the results of the distortion correction and warping: 

![alt text][image1]
![alt text][image2]

### (B) Undistorted Image

Similarly, each testing image will be undistorted before any further modifications applied with <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> undistort()</span> (See [line.py](./line.py) at [L222](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L222)). Here is the result for the selected test image ([straigh_lines1.jpg](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/test_images/straight_lines1.jpg)):

![alt text][image3]

### (C) Color Thresholding and Binary Image

We need  a clear edge detection to successfully detect lane lines in the images. Lane line detection techniques from the first project are combined with color and gradient thresholds to produce binary images for this purpose. Firstly, yellow and white lines are separately detected using different color spaces by <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> mask_color()</span> ([line.py](./line.py) at [L243](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L243)) function. LAB images are used to detect yellow lines whereas white lines are extracted from HLS version, similar to the previous project. <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> cv2.inRange()</span> is used with appropriate lower and upper bounds to get the results as follows:

![alt text][image4]

In order to improve the performance, these binary images are combined with another thresholding using gradients. Similar to the previous approach, in <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> threshold_image()</span> ([line.py](./line.py) at [L274](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L274)), yellow and white colors are detected separately. B-channel of LAB space and L-channel of LUV space are thresholded for yellow and white, respectively. These are masked with a `sobel_x` binary image and then combined. See the figure below for the corresponding binaries.

![alt text][image5]

The combination (<span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> combine_threshold()</span>, [line.py](./line.py) at [L338](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L338)) of each resulting binary from those two masking operations are cropped with respect to a trapezodial region of interest and warped before lane line detection.

### (D) Perspective Transform and Crop

A perspective transform is applied to images with <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> transform_image()</span> ([line.py](./line.py) at [L228](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L228)) function. Vertices of a trapezodial region are selected manually for source points (`src`) to be transformed into a rectangular region in the warped image. The `src` and corresponding vertices, i.e. distance points (`dist`), are taken as inputs of <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> cv2.getPerspectiveTransform()</span> function which evaluates the respective transformation matrix. Below is the list of those points and the results are plotted in the figure. 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 556, 480      | 300, 0        | 
| 734, 480      | 900, 0        |
| 220, 720      | 300, 720      |
| 1105, 720     | 900, 0        |

![all text][image6]

One can see that the straight lane lines appear parallel on the Bird's Eye View image. This perspective transform is applied to binary images but to narrow down search windows, binary images are further masked with a <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> crop_image()</span> ([line.py](./line.py) at [L346](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L346)) function that further maskes the binary image with a trapezoidal region of interest before the perspective transformation. Below are the combinary binary image from the previous step, the cropped binary and the warped binary image:

![all text][image7]

### (E) Line Search and Curve Fitting

A sliding window search algorithm, <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> sliding_windows()</span> ([line.py](./line.py) at [L363](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L363)), is applied on the Bird's View image for line search. Since it is closer to the car, histogram of bottom half of the image has been used as a base window for each lane line search. Then, a rectangular window of a selected size in each side is slided to the top and indices of the windows that include white pixels higher than a limit `minpix`  are recorded as lane window index on left and right separately. Finally, coordinates of those indices on the left and right are fitted on a second order polynomial in <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> fit_poly()</span> ([line.py](./line.py) at [L56](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L56)) using <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> np.polyfit()</span> function. The figure below shows the results for line search and curve fitting.

![all text][image8]

Yellow lines on the bottom right image represents the the second order polynomials that are computed with the coefficients provided from the fit_poly() function. Numbers on the image display the corresponding coefficients, <img src="https://latex.codecogs.com/svg.image?Ax^2&plus;Bx&plus;C" title="Ax^2+Bx+C" />, for each side.

It is important to note that we don't need to start repeat this process from the beginning if the test input is a video. Since the changes from image to image are slow and smooth in a video, we can use the results from the previous image and search around those pixels for a faster solution. So, another function, <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> search_around_poly()</span> ([line.py](./line.py) at [L56](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L483)), is implemented in the project that search around polynomials instead.

#### ***Curvature and Vehicle Offset with Lane Line Visualization***

Additionaly, we evaluated curvature of the road and vehicle offset from the lane center using the fitted polynomials. We know that radius of curvature can be computed by

<img src="https://latex.codecogs.com/svg.image?R_{curve}&space;=&space;&space;\dfrac{(1&plus;(2Ay&plus;B)^2)^{3/2}}{\left|&space;2A\right|}" title="R_{curve} = \dfrac{(1+(2Ay+B)^2)^{3/2}}{\left| 2A\right|}" />

Using the formula here, we can evaluate the radius of curvature of the road, however it is important to remember that the previous polynomial fit is in the image space with pixel values. So, we need to convert from pixel to the real world. We implemented a function <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> curvature_image()</span> (See [line.py](./line.py) at [L547](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L547).) for the conversion with coeefficients

```python
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

```

Using those and evaluating the corrected polynomial fit, we get the radius of curvature in the world space. Similarly, we can compute the vehicle offset from the lane center. This is difference between the image center and the lane center (See <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> vehicle_offset()</span>,  [line.py](./line.py) at [L570](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L570).) converted into the world space by multiplying with `xm_per_pix`.

To visualize the final results overlayed with the original image, we can use the inverse perspective transform matrix to transform top view image to the undistorted image view. A function called <span style="font-family:monaco;font-size:1.15em;color:DarkSlateBlue"> lanes_image()</span> (See [line.py](./line.py) at [L688](https://github.com/trkygt/USDC/blob/5a4f4481a252c1a51e332993c81a48b6874582ad/project_2/line.py#L688).) is created to display the image below. 

![all text][image9]

---

#### ***Additional Notes for the Pipeline***

There are additional functions that are implemented in both [pipeline.py](./pipeline.py) and [line.py](./line.py) files. We omitted them here for brevity. However, it is important to note that we created a sanity check function and also recorded some paramters for the lines to improve performance for the videos.  

---

### Pipeline (video)

Below gifs and the gif on the top of this file shows the first 6 seconds of the final results for each video.

Here is the result for the project video.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/VqffdUaCWmg/0.jpg)](https://www.youtube.com/watch?v=VqffdUaCWmg)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
||||||| constructed merge base
=======
## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
>>>>>>> Adding project_2
