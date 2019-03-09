# -*- coding: utf-8 -*-
import numpy as np
import cv2

def estimate_bg_single_gaussian(image_file_names, roi_filename):
    # returns an image with channels:
    #   1: mean image
    #   2: stdev image
    #   3: roi image

    roi_image = cv2.imread(roi_filename,0)

    # 3 dim array with all the frames
    list_images = [cv2.imread(fp,0) for fp in image_file_names]
    images = np.dstack(list_images)

    # mean and stdev of each pixel along all frames
    mean = np.mean(images, axis=2)
    stdev = np.std(images, axis=2)

    gaussian_image = np.dstack((mean, stdev, roi_image))

    return gaussian_image

def bg_segmentation_single_gaussian(image_file_name, bg_estimation):
    # segment FG/BG wrt estimated BG

    # threshold as number of sigmas
    th = 3.0

    image = cv2.imread(image_file_name,0)

    m = bg_estimation[:, :, 0]
    s = bg_estimation[:, :, 1] * th
    dist = np.abs( m - image )
    diff = s - dist
    diff[diff < 0] = 255
    diff[diff != 255] = 0

    return diff.astype(np.uint8)

    """
    segm_image = np.zeros(image.shape)

    # for each pixel check if it is "inside" gaussian
    h, w = image.shape
    for x in range(1,w-1):
        for y in range(1,h-1):
            m = bg_estimation[y,x,0]
            s = bg_estimation[y,x,1]
            p = image[y,x]

            #print(m, s, p)

            if abs(p - m) > (s * th):
                segm_image[y,x] = 255

    return segm_image
    """