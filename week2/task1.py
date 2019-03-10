# -*- coding: utf-8 -*-
from paths import AICITY_DIR
import glob
import math
import os
import numpy as np
from utils.background_estimation import bg_estimation
import matplotlib.pyplot as plt
import cv2


def bg_segmentation_single_gaussian():
    viz = True
    # Define path to video frames
    filepaths = sorted(glob.glob(os.path.join(str(AICITY_DIR), 'vdo_frames/image-????.png')))  # change folder name (?)
    roi_path = os.path.join(str(AICITY_DIR), 'roi.jpg')
    percent_back = 25
    num_frames = len(filepaths)
    num_backFrames = int(np.floor((percent_back / 100) * num_frames))
    # Get back frames' names
    back_list = filepaths[:num_backFrames]

    first_frame = cv2.imread(back_list[0], 0)
    height, width = first_frame.shape
    channels = 1

    # Define background model
    SIGMA_THR = 3  # number of standard deviations to define the background/foreground threshold
    RHO = 0.01  # memory constant (to update background)
    ROI = False

    bg_model = bg_estimation.SingleGaussianBackgroundModel(height, width, channels, SIGMA_THR, RHO, ROI)

    print("Estimating background with first '{0}'% of frames".format(percent_back))
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
        print("Segmenting FG/BG frame: {0}".format(frame))
        segm = bg_model.apply( cv2.imread(frame,0), roi_filename=roi_path)
        image = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
        video.write(image)

    print("Video", video_name, "generated")
