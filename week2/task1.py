# -*- coding: utf-8 -*-
from paths import AICITY_DIR
import glob
import math
from utils.background_estimation import bg_estimation
import matplotlib.pyplot as plt
import cv2


def bg_segmentation_single_gaussian():
    sequence_root_dir = AICITY_DIR

    # get frame filenames
    filelist = sorted(glob.glob(sequence_root_dir + '*.jpg'))
    # get roi filename
    roi_filename = filelist[-1]
    # remove roi.jpg
    del filelist[-1]
    # split first 25% of frames to estimate background
    num_frames_estimation = math.floor(len(filelist) * 25 / 100)
    training_frames = filelist[0:num_frames_estimation]
    testing_frames = filelist[num_frames_estimation+1:]
    print("Using ", len(training_frames), " frames to estimate background")

    """ Task 1.1: Gaussian distribution """

    # background estimation with train set
    bg_est = bg_estimation.estimate_bg_single_gaussian(training_frames, roi_filename)

    plt.figure()
    plt.imshow(bg_est[:, :, 0], cmap='gray')
    plt.show()

    # video_name = 'video_single_gaussian.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter(video_name, fourcc, 20.0, (1920, 1080))

    # better to use h264 + mp4 as container
    video_name = 'video_single_gaussian.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    # Original video is only 10fps, may want to keep it at that
    video = cv2.VideoWriter(video_name, fourcc, 20.0, (1920, 1080))

    # background segmentation of test set
    for f in testing_frames:
        print("Estimating frame", f)
        segm = bg_estimation.bg_segmentation_single_gaussian(f, bg_est)

        image = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
        print("Writing frame", f)
        video.write( image )

        # plt.figure()
        # plt.imshow(image)
        # plt.show()

    """ Task 1.2 & 1.3: Evaluate results """
