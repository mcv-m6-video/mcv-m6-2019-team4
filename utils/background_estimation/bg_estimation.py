# -*- coding: utf-8 -*-
from paths import AICITY_DIR
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, Value

# Must test this
SIGMA_THR = 3  # number of standard deviations to define the background/foreground threshold
RHO = 0.01  # memory constant (to update background)
ROI = False
POSTPROC = True
METHOD = 'adaptive'  # adaptive w. a single gaussian (e.g.: the background CAN be updated)


# winStd is a quick way to compute std of an image
# std_mean_welford works very well w. sequences but I cannot manage to get it to match w. images
# Fast solution to compute standard deviation of an image
# def winStd(img, w_len):
#     wmean, wsqrmean = (cv2.boxFilter(x, -1, (w_len, w_len), borderType=cv2.BORDER_REFLECT) for x in (img, img * img))
#
#     return np.sqrt(wsqrmean - wmean * wmean)
#
#
# def std_mean_welford(iterable):
#     M = 0
#     S = 0
#     k = 1
#     for value in iterable:
#         tmpM = M
#         M += (value - tmpM) / k
#         S += (value - tmpM) * (value - M)
#         k += 1
#
#     return np.sqrt(S / (k - 1)), M

# def rapid_variance(samples):  # to remove
#     s0 = sum(1 for x in samples)
#     s1 = sum(x for x in samples)
#     s2 = sum(x * x for x in samples)
#     return s0, s1, s2


class SingleGaussianBackgroundModel(object):

    def __init__(self, im_shape, threshold=SIGMA_THR, rho=RHO, roi=ROI, post_process=True, method='adaptive'):
        if len(im_shape) == 2 or (len(im_shape) == 3 and im_shape[-1] == 3):
            self.sum = np.zeros(im_shape).astype(np.uint64)
            self.mean = np.zeros(im_shape)
            self.var = np.zeros(im_shape)

        else:
            print("FATAL: wrong number of channels, the frame must be either in grayscale or RGB")
            return

        self.threshold = threshold
        self.rho = rho
        self.roi = roi
        self.shape = im_shape
        self.post_process = post_process
        self.method = method

    def running_average(self, curr_frame, detection):  # class_mask is not really needed (?)
        """
        Updates the mean value of the background image
        :param curr_frame:
        :param detection: foreground detection of current frame
        :return:
        """
        back = detection != 255  # mask for background pixels
        # replicate the mask across channels (if channels > 1)
        if len(self.shape) == 3 and self.shape[-1] > 1:
            back = np.repeat(back[:, :, np.newaxis], self.shape[-1], axis=2)

        self.mean[back] = (1 - self.rho) * self.mean[back] + self.rho * curr_frame[back]   # update only background

    def running_variance(self, curr_frame, detection):
        """
        Updates the variance value of the background image
        :param curr_frame:
        :param detection: foreground detection of current frame
        :return:
        """
        back = detection != 255  # mask for background pixels
        # replicate the mask across channels (if channels > 1)
        if len(self.shape) == 3 and self.shape[-1] > 1:
            back = np.repeat(back[:, :, np.newaxis], self.shape[-1], axis=2)

        self.var[back] = (1 - self.rho) * self.var[back] + self.rho * np.square(curr_frame[back] - self.mean[back])

    def update_cumulative_frame(self, image):
        # Add current's frame value to accumulator
        self.sum += image

    def estimate_bg_single_gaussian(self, image_file_names):
        """
        returns an image with channels:
        #   1: mean image
        #   2: stdev image
        #   3: roi image

        :param image_file_names: filenames of the background images
        :return: tuple composed of the mean, stddev and ROI images

        Note: to significantly reduce the memory footprint, we need to loop twice through the frame list:
        Once to compute the mean frame, and the second to compute the stddev image
        """

        # 1. Estimate mean image
        for frame_name in image_file_names:
            # Read frame
            curr_frame = cv2.imread(frame_name, 0)
            # Accumulate image
            self.update_cumulative_frame(curr_frame)

        self.mean = self.sum / len(image_file_names)

        # 2. Estimate stddev image
        for frame_name in image_file_names:
            # Read frame
            curr_frame = cv2.imread(frame_name, 0)
            self.var += np.square(curr_frame - self.mean)

        self.var = self.var / (len(image_file_names) - 1)  # forgot about the -1

        # 3 dim array with all the frames ==> THIS USES A LOOOT OF MEMORY
        # list_images = [cv2.imread(fp,0) for fp in image_file_names]
        # images = np.dstack(list_images)
        #
        # # mean and stdev of each pixel along all frames
        # mean = np.mean(images, axis=2)
        # stdev = np.std(images, axis=2)

        # gaussian_image = np.dstack((mean_image, std_image, roi_image))

        # return gaussian_image

    def apply(self, image, roi_filename=''):
        """
        Segments the current image into two classes: background ('0's) and foreground ('1's).
        :param: image: input image to segment
        :param: roi_filename: path to ROI (if available)
        :return: an image with black on the background, white on the foreground.
        """
        image = image.astype(np.float64)  # maybe is not necessary

        # Define lower and upper threshold 'images' (mean +/- thr*std)
        # Note: for adaptive, we use variance for the running update (just add sqrt here)
        lower_threshold = self.mean - np.sqrt(self.var) * self.threshold
        upper_threshold = self.mean + np.sqrt(self.var) * self.threshold

        detection = ~((image >= lower_threshold) & (image <= upper_threshold))

        # Filter out detection outside ROI (if told so)
        if self.roi and roi_filename:  # ROI and non-empty path
            ROI = cv2.imread(roi_filename, 0).astype(np.uint8)  # not only 0 and 255 but also noisy values
            ROI = ROI > 255/2  # only True or False (i.e.: 1's or 0's)
            detection = detection & ROI  # All pixels outside ROI should be background
            # Only detect if foreground and in ROI
        detection = 255 * (detection.astype(np.uint8))  # map (0,1) to (0,255) as expected for uint8

        # Apply morphological filtering (see 'morphological_filtering')
        if self.post_process:  # try different combinations
            detection = apply_morphological_filters(detection)
            detection = hole_filling(detection)

        if self.method == 'adaptive':  # and np.sum(detection < 255) > 0 , that is, there is background
            # Update variance and mean for pixels classified as background
            self.running_average(image, detection)
            self.running_variance(image, detection)

        return detection


def apply_morphological_filters(image):
    # Threshold
    # Set values
    thr = 130
    max_val = 255
    ret, im_th = cv2.threshold(image, thr, max_val, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    im_open = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    im_out = cv2.morphologyEx(im_open, cv2.MORPH_CLOSE, kernel)

    return im_out


# from: https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
def hole_filling(image):
    """
    Start at (0,0) and fill background and then invert selection
    :param image:
    :return:
    """
    thr = 220
    max_val = 255
    # Threshold (inverted)
    # Set values equal to or above 220 to 0
    # Set values below 220 to 255
    th, im_th = cv2.threshold(image, thr, max_val, cv2.THRESH_BINARY)  # the other way around!

    # Copy the thresholded image
    im_floodfill = im_th.copy()

    # Mask used to flood filling
    # Notice that we need extra rows and columns (2, 2)
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0,0): assumed to be background
    cv2.floodFill(im_floodfill, mask, (0, 0), max_val)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground
    im_out = im_th | im_floodfill_inv

    return im_out


if __name__ == '__main__':  # move this to task 1
    viz = True
    # Define path to video frames
    filepaths = sorted(glob.glob(os.path.join(AICITY_DIR, 'vdo_frames/image-????.png')))  # change folder name (?)
    roi_path = os.path.join(AICITY_DIR, 'roi.jpg')
    percent_back = 25
    num_frames = len(filepaths)
    num_backFrames = int(np.floor((percent_back / 100) * num_frames))
    # Get back frames' names
    back_list = filepaths[:num_backFrames]

    first_frame = cv2.imread(back_list[0], 0)

    # Define background model
    bg_model = SingleGaussianBackgroundModel(first_frame.shape, SIGMA_THR, RHO, ROI, POSTPROC, METHOD)

    print("Estimating background with first '{0}'% of frames".format(percent_back))
    bg_model.estimate_bg_single_gaussian(back_list)  # MUST speed this up, it takes more than a minute

    if viz:
        plt.figure()
        plt.imshow(bg_model.mean, cmap='gray')
        plt.show()

    # Detect foreground (rest of the sequence)
    fore_list = filepaths[num_backFrames:]

    # Define video writer
    video_name = 'background_estimation_single_gaussian_f_ROI_off.avi'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, first_frame.shape)

    for frame in fore_list:
        print("Estimating frame: '{0}'".format(frame))
        segm = bg_model.apply(frame, roi_filename=roi_path)
        image = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
        video.write(image)
