# -*- coding: utf-8 -*-
import os
import glob
import math
from utils.background_estimation import bg_estimation
import matplotlib.pyplot as plt

def bg_segmentation_single_gaussian():
    sequence_root_dir = '../data/AICity_data/train/S03/c010/'

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

    # background estimation with train set
    bg_est = bg_estimation.estimate_bg_single_gaussian(training_frames, roi_filename)

    plt.figure()
    plt.imshow(bg_est[:,:,0], cmap='gray')
    plt.show()

    # background segmentation of test set
    for f in testing_frames:
        segm = bg_estimation.bg_segmentation_single_gaussian(f, bg_est)

        plt.figure()
        plt.imshow(segm, cmap='gray')
        plt.show()

    """ Task 1.2 & 1.3: Evaluate results """
