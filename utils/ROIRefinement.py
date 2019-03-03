import os
from utils.annotationsParser import annotationsParser
import cv2
import numpy as np


class ROIRefinement():


    def __init__(self, filepath, threshold):

        self.ROI = cv2.imread(filepath)
        self.threshold = threshold


    def refineBBOX (self, BBOXList):
        """ Refine a list of bounding boxes depending on their relative position to the ROI image
        Args:
            BBOXList: List of BBOXes

        """
        RefinedBBOX = []
        for BBOX in BBOXList:
            if not self.discardBBOX(BBOX):
                RefinedBBOX.append(BBOX)
        return RefinedBBOX


    def discardBBOX(self, BBOX):
        """ Check if a BBOX is inside a ROI
        Args:
            BBOX: BBOX to be analyzed

        """
        intBBOX = np.asarray(BBOX, dtype=int)
        ROIBBOX = self.ROI[intBBOX[0]:intBBOX[2], intBBOX[1]:intBBOX[3]]
        BBOXArea = abs((intBBOX[2] - intBBOX[0]) * (intBBOX[3] - intBBOX[1]))
        ROIArea = np.count_nonzero(ROIBBOX)/3

        ratio = ROIArea/BBOXArea

        if ratio < self.threshold:
            return True

        return False

if __name__ == '__main__':

    ROIPath  = '/home/marti/Documents/M6/Lab/code/mcv-m6-2019-team4/datasets/AICity_data/train/S03/c010/roi.jpg'
    refinement = ROIRefinement(ROIPath, 0.5)

    gtExtractor = annotationsParser(
        '/home/marti/Documents/M6/Lab/code/mcv-m6-2019-team4/datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml')


    for i in range(len(gtExtractor.gt)):

        # load the image
        frame_path = 'image-{:07d}.png'.format(gtExtractor.getGTFrame(i)+1)
        frame_path = '/home/marti/Documents/M6/Lab/code/mcv-m6-2019-team4/datasets/AICity_data/train/S03/c010/frames/' + frame_path
        image = cv2.imread(frame_path)

        # Get GT BBOX
        gtBBOX = gtExtractor.getGTBoundingBox(i)

        image = cv2.imread(frame_path)
        # draw the ground-truth bounding box along with the predicted
        # bounding box

        if refinement.discardBBOX(gtBBOX):
            cv2.rectangle(image, (int(gtBBOX[0]), int(gtBBOX[1])),
                          (int(gtBBOX[2]), int(gtBBOX[3])), (0, 0, 255), 2)
        else:

            cv2.rectangle(image, (int(gtBBOX[0]), int(gtBBOX[1])),
                      (int(gtBBOX[2]), int(gtBBOX[3])), (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)