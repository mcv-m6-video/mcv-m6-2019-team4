from utils.annotation_parser import annotationsParser
import cv2
import numpy as np
from paths import AICITY_DIR


class BoundingBoxFilter(object):
    def __init__(self, filepath, threshold):

        self.ROI = cv2.imread(str(filepath))
        self.threshold = threshold

    def refine_bbox(self, bboxes):
        """ Refine a list of bounding boxes depending on their relative position to the ROI image
        Args:
            bboxes: List of BBOXes

        """
        refined_bbox = []
        for bbox in bboxes:
            if not self.discard_bbox_center(bbox):
                refined_bbox.append(bbox)
        return refined_bbox

    def discard_bbox_center(self, bbox):
        """ Check if a BBOX is inside a ROI
        Args:
            bbox: BBOX to be analyzed

        """
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        if self.ROI[center_y, center_x, 0] == 0:
            return True
        return False

    def discard_bb(self, bbox):
        """ Check if a BBOX is inside a ROI
        Args:
            bbox: BBOX to be analyzed

        """
        int_bbox = np.asarray(bbox, dtype=int)
        roi_bbox = self.ROI[int_bbox[0]:int_bbox[2], int_bbox[1]:int_bbox[3]]
        bbox_area = abs(
            (int_bbox[2] - int_bbox[0]) * (int_bbox[3] - int_bbox[1])
        )
        roi_area = np.count_nonzero(roi_bbox) / 3

        ratio = roi_area / bbox_area

        if ratio < self.threshold:
            return True

        return False


def filter_bboxes_out_of_roi():
    roi_path = AICITY_DIR.joinpath('roi.jpg')
    refinement = BoundingBoxFilter(roi_path, 0.5)

    gtExtractor = annotationsParser(
        AICITY_DIR.joinpath('Anotation_40secs_AICITY_S03_C010.xml'))

    for i in range(len(gtExtractor.gt)):

        # load the image
        frame_path = AICITY_DIR.joinpath(
            'frames',
            'image-{:07d}.png'.format(gtExtractor.getGTFrame(i) + 1))
        image = cv2.imread(frame_path)

        # Get GT BBOX
        bbox_gt = gtExtractor.getGTBoundingBox(i)

        image = cv2.imread(frame_path)
        # draw the ground-truth bounding box along with the predicted
        # bounding box

        if refinement.discard_bb(bbox_gt):
            cv2.rectangle(
                image,
                (int(bbox_gt[0]), int(bbox_gt[1])),
                (int(bbox_gt[2]), int(bbox_gt[3])),
                (0, 0, 255),
                2
            )

        else:
            cv2.rectangle(
                image,
                (int(bbox_gt[0]), int(bbox_gt[1])),
                (int(bbox_gt[2]), int(bbox_gt[3])),
                (0, 255, 0),
                2
            )

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
