# -*- coding: utf-8 -*-
import numpy as np
import cv2

def estimate_bg_single_gaussian(image_file_names, roi_filename):
    # returns an image with channels:
    #   1: mean image
    #   2: stdev image
    #   3: roi image

    roi_image = cv2.imread(roi_filename,0)
    list_images = [cv2.imread(fp,0) for fp in image_file_names]
    images = np.dstack(list_images)

    mean = np.mean(images, axis=2)
    stdev = np.std(images, axis=2)

    gaussian_image = np.dstack((mean, stdev, roi_image))

    return gaussian_image

def bg_segmentation_single_gaussian(image_file_name, bg_estimation):
    # threshold as number of sigmas
    th = 2.0

    image = cv2.imread(image_file_name,0)

    segm_image = np.zeros(image.shape)

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
