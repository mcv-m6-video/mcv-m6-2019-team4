# -*- coding: utf-8 -*-
from paths import AICITY_DIR
import glob
import os
import numpy as np
from utils.background_estimation import bg_estimation
import matplotlib.pyplot as plt
import cv2


def bg_segmentation_single_gaussian(video_name, alpha = 3, preproc = False, postproc = False):
    viz = True
    # Define path to video frames
    filepaths = sorted(glob.glob(os.path.join(str(AICITY_DIR), 'frames/image-????.png')))  # change folder name (?)
    roi_path = os.path.join(str(AICITY_DIR), 'roi.jpg')
    percent_back = 25
    num_frames = len(filepaths)
    num_backFrames = int(np.floor((percent_back / 100) * num_frames))
    # Get back frames' names
    back_list = filepaths[:num_backFrames]

    first_frame = cv2.imread(back_list[0], 0)

    # Define background model
    SIGMA_THR = alpha  # number of standard deviations to define the background/foreground threshold
    RHO = 0.01  # memory constant (to update background) ==> exaggerated to debug
    ROI = True
    PREPROC = preproc  # True
    POSTPROC = postproc  # True
    METHOD = 'non_adaptive'

    bg_model = bg_estimation.SingleGaussianBackgroundModel(first_frame.shape, SIGMA_THR, RHO, ROI, POSTPROC, METHOD)

    print("Estimating background with first {0}% of frames".format(percent_back))
    bg_model.estimate_bg_single_gaussian(back_list)  # MUST speed this up, it takes more than a minute

    if viz:
        plt.figure()
        plt.imshow(bg_model.mean, cmap='gray')
        plt.show()

    # Detect foreground (rest of the sequence)
    fore_list = filepaths[num_backFrames:]

    if len(first_frame.shape) == 2:
        height, width = first_frame.shape

    elif len(first_frame.shape) == 3 and first_frame.shape[-1] == 3:
        height, width, _ = first_frame.shape
    else:
        print("FATAL: unexpected number of channels: must be '1' for grayscale, '3' for RGB")

    # Define video writer
    # video_name = 'background_estimation_single_adaptive_f_ROI_off.avi'
    video_name = "{0}_rho-{1}_sigma_{2}.avi".format(video_name, RHO, SIGMA_THR)
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, four_cc, 10, (width, height))
    # mean_video_name = "{0}_mean.avi".format(video_name.replace('.avi', ''))
    # mean_video = cv2.VideoWriter(mean_video_name, four_cc, 10, (width, height))
    # var_video_name = "{0}_var.avi".format(video_name.replace('.avi', ''))
    # var_video = cv2.VideoWriter(var_video_name, four_cc, 10, (width, height))

    for frame in fore_list:
        print("Segmenting FG/BG frame: {0}".format(frame.split('/')[-1]))
        segm = bg_model.apply(cv2.imread(frame, 0), roi_filename=roi_path)
        image = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
        # print("Mean image range: ({0:.2f}, {1:.2f})\nvar image range: ({2:.2f}, {3:.2f})".format(
        #     np.min(bg_model.mean), np.max(bg_model.mean), np.min(bg_model.var), np.max(bg_model.var)))
        # print("Mean image has mean: {0:.2f}\nvar image has mean: {1:.2f}".format(
        #     np.mean(bg_model.mean), np.mean(bg_model.var)))
        # mean_back = cv2.cvtColor(bg_model.mean.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # var_back = cv2.cvtColor(bg_model.mean.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # mean_video.write(mean_back)
        # var_video.write(var_back)
        video.write(image)

    print("Video '{0}' was successfully generated".format(video_name))


if __name__ == "__main__":
    # Run background subtraction, combinations tested:
    # Edit to match your needs (naming schemes)
    # Note: ROI always in use (not so noticeable anyway with the exception of critical cases)

    # Customise at will (examples below):
    method = 'non_adaptive'
    preproc = True
    postproc = True
    video_name = "BE_1gauss-{0}_pre-{1}_post-{2}".format(method, preproc, postproc)
    bg_segmentation_single_gaussian(video_name, preproc, postproc)

    # 1st: single gaussian, non adaptive, with ROI and NO post-processing
    # method = 'non-adaptive'
    # preproc = True
    # postproc = False
    # video_name = "BE_1gauss-{0}_pre-{1}_post-{2}".format(method, preproc, postproc)
    # bg_segmentation_single_gaussian(video_name, method, preproc, postproc)
    #
    # # 2nd single gaussian, not adaptive, with ROI and post-processing (w.morphological filters)
    # method = 'non-adaptive'
    # preproc = True
    # postproc = True
    # video_name = "BE_1gauss-{0}_pre-{1}_post-{2}".format(method, preproc, postproc)
    # bg_segmentation_single_gaussian(video_name, method, preproc, postproc)
    #
    # # 3rd: single gaussian, adaptive, with ROI and NO post-processing
    # method = 'adaptive'
    # preproc = True
    # postproc = False
    # video_name = "BE_1gauss-{0}_pre-{1}_post-{2}".format(method, preproc, postproc)
    # bg_segmentation_single_gaussian(video_name, method, preproc, postproc)
    #
    # # 4th: single gaussian, adaptive, with ROI and post-processing (w.morphological filters)
    # method = 'adaptive'
    # preproc = True
    # postproc = True
    # video_name = "BE_1gauss-{0}_pre-{1}_post-{2}".format(method, preproc, postproc)
    # bg_segmentation_single_gaussian(video_name, method, preproc, postproc)


