##Advanced Lane Finding Project

**Self-Driving Cars Nanodegree - Project 4**

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

[image1]: /output_images/Cal1.PNG "Undistorted Chessboard"
[image2]: ./output_images/Cal2.PNG "Undistorted Images"
[image3]: ./output_images/Cal3.PNG "Radial Distorsion"
[image4]: ./output_images/Pro1(u).PNG "Undistorted Image"
[image5]: ./output_images/Pro4(abs).PNG "Sobel-x (abs) thresholding"
[image6]: ./output_images/Pro5(mag).PNG "Sobel (magnitude) thresholding"
[image7]: ./output_images/Pro6(dir).PNG "Sobel (direction) thresholding"
[image8]: ./output_images/Pro7(hls).PNG "HLS, Saturation thresholding"
[image9]: ./output_images/Pro7(hsv).PNG "HSV, Value thresholding"
[image10]: ./output_images/Pro2(y).PNG "HSV, Yellow Mask"
[image11]: ./output_images/Pro3(w).PNG "HSV, White Mask"
[image12]: ./output_images/Pro8(com).PNG "Combined Thresholding"
[image13]: ./output_images/Per1.PNG "Original Warping"
[image14]: ./output_images/Per2.PNG "Combined Thresholding Warping"
[image15]: ./output_images/Per3.PNG "Combined Thresholding Warping"
[image16]: ./output_images/Per5.PNG "Window Searching"
[image17]: ./output_images/Per6.PNG "Lane Finding"
[image18]: ./output_images/Per7.PNG "Final Image"
[image19]: ./output_images/Per8.PNG "Final Image"
[image20]: ./output_images/Per10.PNG "Fitted Lines"

###Camera Calibration

The code for this step is contained in the IPython notebook named [Calibration](Calibration.ipynb).   

The camera calibration procedure was performed by preparing and finding the coordinates of the chessboard corners from a set of 16 images, using the `cv2.findChessboardCorners()` function from OpenCV, assuming that the chessboards are fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_points` is just a replicated array of coordinates, and `object_points (object points - 3d points in real world space) `will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `image_points (2d points in image plane)` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  Finally, `object_points` and `image_points` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  Also, the coefficients were saved on the follow file `calibration_data/calibration_matrices.p`. 

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following results:

####Chessboards Results
Distorted and undistorted images of a chessboard are shown on the left and right respectively.
![alt text][image1]
####Road Images Results
Distorted and undistorted images of the road are shown on the left and right respectively.
![alt text][image2]
####Distorsion Visualization
It is been shown that original images had a small radial distortion, as shown on the followed image:
![alt text][image3]

###Pipeline (single images)
The pipeline implemented to process the images in order to find the lanes on the road is described as follows:

* Distortion correction to raw image.
* Binary Thresholding.
* Perspective Transform ("birds-eye view").
* Window Searching (Lane-Line Searching).
* Polynomial Fit - Lane Boundary.
* Lane Curvature and Center Deviation.
* Unwarping - Lane Boundary.
* Merge Images and Display Curvature and Lane Deviation.

The code that implements all the steps required for the Project 4 is contained in the IPython notebook named [P4](P4.ipynb).   


####1. Distortion Correction.
Distortion correction to the test images was implemented using the camera calibration and distortion coefficients loaded from `calibration_data/calibration_matrices.p` and using the `cv2.undistort()` function obtaining the following result:
![alt text][image4]

####2. Binary Thresholding - Color transforms, Color Mask and Gradients.
I used a combination of color and gradient thresholds to generate a binary image (The code used to tune the min and max thresholds through trackbars is contained in the IPython notebook named [Processing](Processing.ipynb)).  

![alt text][image12]

For the first video, the combined thresholding method chosen includes the 4 methods proposed on the lectures (abs, mag, dir and hls). Moreover, HSV thresholding was implemented using the V channel, in order to improve the method. The combination of binary images was implemented as follows:

`combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | ((hls_bin == 1)&(hsv_bin == 1))] = 1`

For the challenge video, i decided to include yellow and white color masks, tuning the HSV channels in order to isolate yellow and white colors, The combination of binary images was implemented as follows:

`combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | ((hls_bin == 1)&(hsv_bin == 1)) | ((yellow_bin == 1)|(white_bin == 1))] = 1` 

Examples of different thresholding methods are shown below:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

####3. Perspective Transform.

The code for my perspective transform was tested and verified in the IPython notebook named [Perspective_Lane](Perspective_Lane.ipynb). This step was implemented using the ```cv2.getPerspectiveTransform()``` function to find the transform matrices to warp and unwarp the images and taking as inputs source (`src`) and destination (`dst`) points. Furthermore, the `cv2.warpPerspective()` function implement the warping by taking as inputs an image (`img`), as well as transform matrices (`m | m_inv`) and the interpolation method (`cv2.INTER_LINEAR`).  I chose the hardcode the source and destination points in the following manner:

```sh
src = np.float32(
    [[0, img_size[1]],
    [img_size[0], img_size[1]],
    [510, img_size[1]*2/3],
    [770, img_size[1]*2/3]])

dst = np.float32(
    [[0, img_size[1]],
    [img_size[0], img_size[1]],
    [0, 0],
    [img_size[0], 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 720        | 0, 720        | 
| 1280, 720     | 1280, 720     |
| 510, 480      | 0, 0          |
| 770, 480      | 1280, 0       |

I verified that my perspective transform was working as expected, testing the (`src`) and destination (`dst`) points with the Straight Lane images, as follows: 

![alt text][image13]

####Warped Binary Image Visualization
![alt text][image14]
![alt text][image15]

####4. Lane-Line pixels
The Lale-Line pixels are located using the `find_window_centroids() and window_search()` functions provided in the Udacity lectures. The code used is contained in the cell `no. 3` of the IPython notebook named [P4](P4.ipynb). This approach take advantage from the convolution of the histogram in order to identify the lane-line pixel and also draw a rectangle where the pixels belong to a portion of the lane. The parameters used to perform this operation are listed below.

| Parameter     | Value         | 
|:-------------:|:-------------:| 
| Window Width  | 0, 720        | 
| Window Height | 1280, 720     |
| Margin        | 0, 0          |

![alt text][image16]

####5. Polynomial Fit

A 2nd order Polynomial regression was used in order to fit the Lane Boundary using the `np.polyfit()` function. The code used is contained in the cell `no. 3`, into the function called `pipeline()` of the IPython notebook named [P4](P4.ipynb). To perform this operation left and right lines were isolated using a zero mask. To draw the Lane Boundary, the `cv2.fillConvexPoly()` function was used, taking as inputs the `left_points`computed by the `l_fit` coefficients and by a flipped version of the `right_points` computed by the `r_fit` coefficients.

An example of Lane Boundary is shown below:

![alt text][image17]

####6. Radius of Curvature and Position of the Vehicle.
To calculate the radius of curvature of the lane, i decided to used the equations provided by Udacity in the lectures. The code of this step is contained in the cell `no. 3`, into the function called `curvature()` of the IPython notebook named [P4](P4.ipynb). The position of the vehicle with respect to center, was implemented as follows:
 
`center_dev = (image.shape[1]/2-(l_fit[2]+(r_fit[2]-l_fit[2])/2))*xm_per_pix*100`
 
Where `l_fit[2]` is the intersect with the `x axis` of the fitted left line and `r_fit[2]` is the is the intersect with the `x axis` of the fitted right line.

![alt text][image20]

####7. Results.

Finally, the Lane Boundary was warped by using the inverse transform matrix `m_inv`  previously calculated by ```cv2.getPerspectiveTransform()```, and then  performing the unwarping operatin with the `cv2.warpPerspective()` function. Moreover, Lane Boundary, Undistorted image, Curvatures and Lane Deviation are merged using the `cv2.addWeighted()` and `cv2.putText()` functions. The code of this step is contained at the end of the cell `no. 3`, into the function called `pipeline()` of the IPython notebook named [P4](P4.ipynb).

Examples of `project_video` and `challenge_video` are shown below: 

![alt text][image18]
![alt text][image19]

---

###Pipeline (video)
A youtube video processed with the algorithm are shown below:

[![Alt text for your video](https://img.youtube.com/vi/SJmWCHr21C8/0.jpg)](http://www.youtube.com/watch?v=SJmWCHr21C8)

---

###Discussion
It is been known the amazing power of computer vision tools brings on robotics field, however, in both methods Deep Learning and Computer Vision methods a hard tuning work has to be performed in order to improve this kind of tasks. The pipeline implemented on the project video follows a series of obvious step in order to detect lanes on the road under natural conditions in order to keep the car between the lines detected, also for the challenge video, some adjustments has to be implemented in order to achieve the goal, also, i think that an adaptive method is a good idea in order to get better results on different scenarios. Proper implementation on different ways can achieve the goal for this project, i decided to follow the methods that Udacity proposed. However, when the road have extreme conditions, as shown on the harder video (lot of shadows, very closed curves, extremely light conditions, etc.), a lot of improvements has to be done in order to achieve the goal. The pipeline can be improved by fine tuning of the methods described, also,i guess that the pipeline will fail in case the car changes of lane or some car in the front decides to get into the lane.
    