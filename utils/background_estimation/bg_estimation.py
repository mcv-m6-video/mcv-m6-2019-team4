# -*- coding: utf-8 -*-
from paths import AICITY_DIR
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt


SIGMA_THR = 2.5  # number of standard deviations to define the background/foreground threshold
RHO = 0.01  # memory constant (to update background)

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


class BackgroundModel(object):

    def __init__(self, height, width, channels, threshold=SIGMA_THR, rho=RHO):
        if channels == 1:
            self.mean = np.zeros((height, width)).astype(np.uint64)
            self.variance = np.zeros((height, width)).astype(np.float64)
        elif channels == 3:
            self.mean = np.zeros((height, width, channels))
            self.variance = np.zeros((height, width, channels))
        else:
            print("FATAL: wrong number of channels, the frame must be either in grayscale or RGB")
            return

        self.threshold = threshold
        self.rho = rho

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


# TODO: move estimation functions into the class

def update_cumulative_frame(image, sum_frame):
    # Add current's frame value to a
    sum_frame += image
    return sum_frame


def estimate_bg_single_gaussian(image_file_names, roi_filename):
    """
    returns an image with channels:
    #   1: mean image
    #   2: stdev image
    #   3: roi image

    :param image_file_names: filenames of the background images
    :param roi_filename: ROI mask
    :return: tuple composed of the mean, stddev and ROI images

    Note: to significantly reduce the memory footprint, we need to loop twice through the frame list:
    Once to compute the mean frame, and the second to compute the stddev image
    """
    roi_image = cv2.imread(roi_filename, 0)  # Take advantage that the ROI has the same dim. as the sequence frames

    height, width = roi_image.shape
    # To avoid overflow, make the accumulator a uint64
    sum_image = np.zeros((height, width)).astype(np.uint64)
    std_image = np.zeros((height, width)).astype(np.float64)

    # 1. Estimate mean image
    for frame in image_file_names:
        # Read frame
        curr_frame = cv2.imread(frame, 0)
        # Accumulate image
        sum_image = update_cumulative_frame(curr_frame, sum_image)

    mean_image = sum_image / len(image_file_names)

    # 2. Estimate stddev image
    for frame in image_file_names:
        # Read frame
        curr_frame = cv2.imread(frame, 0)
        std_image += np.square(curr_frame - mean_image)

    # 3 dim array with all the frames
    # list_images = [cv2.imread(fp,0) for fp in image_file_names]
    # images = np.dstack(list_images)
    #
    # # mean and stdev of each pixel along all frames
    # mean = np.mean(images, axis=2)
    # stdev = np.std(images, axis=2)

    gaussian_image = np.dstack((mean_image, std_image, roi_image))

    return gaussian_image


def bg_segmentation_single_gaussian(image_file_name, bg_estimation, threshold=SIGMA_THR):
    """
    Segments the current image into two classes: background ('0's) and foreground ('1's).
    :param image_file_name: path to the frame to segment.
    :param bg_estimation: background estimation model.
    :param threshold: threshold that determines what is background/foreground.
    :return: an image with black on the background, white on the foreground.
    """
    image = cv2.imread(image_file_name, 0)

    mean = bg_estimation[:, :, 0]
    std = bg_estimation[:, :, 1] * threshold
    dist = np.abs(mean - image)
    diff = std - dist
    diff[diff < 0] = 255
    diff[diff != 255] = 0

    return diff.astype(np.uint8)


if __name__ == '__main__':  # Consider removing this main() as 'Task 1' is almost identical (+ output to video)
    viz = True
    # Define path to video frames
    filepaths = sorted(glob.glob(os.path.join(AICITY_DIR, 'vdo_frames/*.png')))  # change folder name (?)
    roi_path = os.path.join(AICITY_DIR, 'roi.jpg')
    percent_back = 25
    num_frames = len(filepaths)
    num_backFrames = int(np.floor((percent_back / 100) * num_frames))
    # Get back frames' names
    back_list = filepaths[:num_backFrames]

    # Define background model
    bg_est = estimate_bg_single_gaussian(back_list, roi_path)

    # Detect foregound (rest of the sequence)
    fore_list = filepaths[num_backFrames+1:]
    if viz:
        plt.figure()

    for frame in fore_list:
        print("Estimating frame: '{0}'".format(frame))
        segm = bg_segmentation_single_gaussian(frame, bg_est)
        if viz:
            image = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)





