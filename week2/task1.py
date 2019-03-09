# -*- coding: utf-8 -*-
import os
import glob
import math
from utils.background_estimation import bg_estimation
import matplotlib.pyplot as plt

sequence_root_dir = '../data/AICity_data/train/S03/c010/'

def do_bg_estimation_single_gaussian(estimation_frames, roi_file):
    bg_est = bg_estimation.estimate_bg_single_gaussian(estimation_frames, roi_file)

    plt.figure()
    plt.imshow(bg_est[:,:,0], cmap='gray')
    plt.show()

    return bg_est

def do_bg_segmentation_single_gaussian(test_frames, bg_est):
    for f in testing_frames:
        segm = bg_estimation.bg_segmentation_single_gaussian(f, bg_est)

        plt.figure()
        plt.imshow(segm, cmap='gray')
        plt.show()


if __name__ == '__main__':

    """ Task 1 """

    sequence_root_dir = '../' + sequence_root_dir

    # get frame filenames
    filelist = sorted(glob.glob(sequence_root_dir + '*.jpg'))
    # get roi filename
    roi_filename = filelist[-1]
    # remove roi.jpg
    del filelist[-1]
    # split first 25% of frames to estimate background
    num_frames_estimation = math.floor(len(filelist) * 25 / 100)
    training_frames = filelist[0:num_frames_estimation]
    testing_frames = filelist[num_frames_estimation+1:num_frames_estimation+3]
    print("Using ", len(training_frames), " frames to estimate background")


    """ Task 1.1: Gaussian distribution """

    bg_est = do_bg_estimation_single_gaussian(training_frames, roi_filename)
    do_bg_segmentation_single_gaussian(testing_frames, bg_est)

    """ Task 1.2 & 1.3: Evaluate results """
