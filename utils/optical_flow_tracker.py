import numpy as np
import pyflow
import time

class OpticalFlowTracker(object):


    def __init__(self, image, roi):
        self.last_image = image
        self.last_roi = roi

        # Flow Options:
        self.alpha = 0.012
        self.ratio = 0.75
        self.minWidth = 20
        self.nOuterFPIterations = 7
        self.nInnerFPIterations = 1
        self.nSORIterations = 30
        self.colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    def predict(self, new_image):

        s = time.time()
        #TODO: crop images at roi for faster optical center computatuion
        im1 = self.last_image
        im2 = new_image
        u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, self.alpha, self.ratio, self.minWidth, self.nOuterFPIterations, self.nInnerFPIterations,
            self.nSORIterations, self.colType)

        new_center = self.estimate_center(u, v, im2W)
        return new_center


    def correct(self, new_image, new_roi):
        self.last_image = new_image
        self.last_roi = new_roi


    def estimate_center(self, u,v,im2W):

        # TODO: Implement center computation from optical flow
        return [[0], [0]]

