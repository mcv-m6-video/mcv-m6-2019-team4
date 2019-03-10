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

# TODO: write a background_estimation class similar to 'background_subtractor.py'

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


class SingleGaussianBackgroundModel(object):

    def __init__(self, height, width, channels, threshold=SIGMA_THR, rho=RHO, roi=ROI):
        if channels == 1:
            self.sum = np.zeros((height, width)).astype(np.uint64)
            self.mean = np.zeros((height, width))
            self.std = np.zeros((height, width))

        elif channels == 3:
            self.sum = np.zeros((height, width, channels)).astype(np.uint64)
            self.mean = np.zeros((height, width, channels))
            self.std = np.zeros((height, width, channels))
        else:
            print("FATAL: wrong number of channels, the frame must be either in grayscale or RGB")
            return

        self.threshold = threshold
        self.rho = rho
        self.roi = roi

    def running_average(self, curr_frame, mean_old, class_mask):  # class_mask is not really needed (?)
        """
        Updates the mean value of the background image
        :param curr_frame:
        :param mean_old:
        :param class_mask: mask that indicates which pixels should be updated (those on the background)
        :return:
        """
        mean_new = (1 - self.rho) * mean_old + self.rho * curr_frame   # update only background
        return mean_new

    def running_variance(self, curr_frame, var_old, new_mean, class_mask):
        """
        Updates the variance value of the background image
        :param curr_frame:
        :param var_old:
        :param class_mask: mask that indicates which pixels should be updated (those on the background)
        :return:
        """
        # We compute stddev, we should add a square
        var_new = (1 - self.rho) * var_old + self.rho * np.square(curr_frame - new_mean)
        return var_new

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
            self.std += np.square(curr_frame - self.mean)

        self.std = np.sqrt(self.std / len(image_file_names))

        # 3 dim array with all the frames ==> THIS USES A LOOOT OF MEMORY
        # list_images = [cv2.imread(fp,0) for fp in image_file_names]
        # images = np.dstack(list_images)
        #
        # # mean and stdev of each pixel along all frames
        # mean = np.mean(images, axis=2)
        # stdev = np.std(images, axis=2)

        # gaussian_image = np.dstack((mean_image, std_image, roi_image))

        # return gaussian_image

    def bg_segmentation_single_gaussian(self, image_file_name, roi_filename=''):
        """
        Segments the current image into two classes: background ('0's) and foreground ('1's).
        :param image_file_name: path to the frame to segment.
        :param bg_estimation: background estimation model.
        :return: an image with black on the background, white on the foreground.
        """
        image = cv2.imread(image_file_name, 0)
        image = image.astype(np.float64)  # maybe is not necessary

        # mean = # bg_estimation[:, :, 0]
        # std = bg_estimation[:, :, 1] * threshold
        # dist = np.abs(self.mean - image)
        # diff = self.std * self.threshold - dist
        # diff[diff < 0] = 255
        # diff[diff != 255] = 0

        # Define lower and upper threshold 'images' (mean +/- thr*std)
        lower_threshold = self.mean - self.std * self.threshold
        upper_threshold = self.mean + self.std * self.threshold

        detection = ~((image >= lower_threshold) & (image <= upper_threshold))

        # Filter out detection outside ROI (if told so)
        if self.roi:
            ROI = cv2.imread(roi_filename, 0).astype(np.uint8)  # not only 0 and 255 but also noisy values
            ROI = ROI > 255/2  # only True or False (i.e.: 1's or 0's)
            detection = detection & ROI  # All pixels outside ROI should be background
            # Only detect if foreground and in ROI

        return 255 * (detection.astype(np.uint8))  # map 0 to 0 and 1 to 255 (as expected for uint8)


def rapid_variance(samples):
    s0 = sum(1 for x in samples)
    s1 = sum(x for x in samples)
    s2 = sum(x * x for x in samples)
    return s0, s1, s2


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
    height, width = first_frame.shape
    channels = 1

    # Define background model
    bg_model = SingleGaussianBackgroundModel(height, width, channels, SIGMA_THR, RHO, ROI)
    bg_model.estimate_bg_single_gaussian(back_list)  # MUST speed this up, it takes more than a minute

    if viz:
        plt.figure()
        plt.imshow(bg_model.mean, cmap='gray')
        plt.show()

    # Detect foregound (rest of the sequence)
    fore_list = filepaths[num_backFrames:]

    # Define video writer
    video_name = 'background_estimation_single_gaussian_f_ROI_off.avi'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (width, height))

    for frame in fore_list:
        print("Estimating frame: '{0}'".format(frame))
        segm = bg_model.bg_segmentation_single_gaussian(frame, roi_filename=roi_path)
        image = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
        video.write(image)






