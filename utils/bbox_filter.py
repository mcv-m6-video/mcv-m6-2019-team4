import cv2
import numpy as np

from aicity_dataset import AICityDataset
from paths import AICITY_DIR, AICITY_ANNOTATIONS

def refine_bbox(bboxes: tuple, roi):
    """ Refine a list of bounding boxes depending on their relative
    position to the ROI image

    Args:
        bboxes: List of BBOXes

    """
    refined_bbox = []
    for bbox in bboxes:
        if not discard_bbox_center(bbox, roi):
            refined_bbox.append(bbox)
    return refined_bbox


def discard_bbox_center(bbox: tuple, roi):
    """ Check if a BBOX is inside a ROI

    Args:
        bbox: BBOX to be analyzed
    """
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)

    if roi[center_y, center_x, 0] == 0:
        return True
    return False


def discard_bb(bbox: tuple, roi, threshold: float):
    """ Check if a BBOX is inside a ROI

    Args:
        bbox: BBOX to be analyzed
    """
    int_bbox = np.asarray(bbox, dtype=int)
    roi_bbox = roi[int_bbox[0]:int_bbox[2], int_bbox[1]:int_bbox[3]]
    bbox_area = abs(
        (int_bbox[2] - int_bbox[0]) * (int_bbox[3] - int_bbox[1])
    )
    roi_area = np.count_nonzero(roi_bbox) / 3

    ratio = roi_area / bbox_area

    if ratio < threshold:
        return True

    return False


def filter_bbox_out_of_roi(image, bbox: tuple, roi, threshold: float):
    """ Uses opencv so colour images images are interepreted as BGR images

    Args:
        image: OpenCV Image related to the bounding box
        bbox: 4-tuple of 2d coordinates of left top and bottom right corner
        roi: OpenCV Image
        threshold: discard bb if its this area ratio is outside the ROI
    """

    if discard_bb(bbox, roi, threshold):
        cv2.rectangle(
            img=image,
            pt1=(int(bbox[0]), int(bbox[1])),
            pt2=(int(bbox[2]), int(bbox[3])),
            color=(0, 0, 255),
            thickness=2,
        )

    else:
        cv2.rectangle(
            img=image,
            pt1=(int(bbox[0]), int(bbox[1])),
            pt2=(int(bbox[2]), int(bbox[3])),
            color=(0, 255, 0),
            thickness=2,
        )
    return image


def test_filter_bboxes_out_or_roi():
    """ Shows images and boundig boxes of all samples of a dataset """
    dataset = AICityDataset(AICITY_DIR, AICITY_ANNOTATIONS)
    roi_image = cv2.imread(str(AICITY_DIR.joinpath('roi.jpg')))

    for image, label in dataset:
        # Convert to CV2 image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        bbox = (
            label[2],
            label[3],
            label[4],
            label[5],
        )
        output_image = filter_bbox_out_of_roi(image,
                                              bbox,
                                              roi_image,
                                              threshold=0.5)

        # show the output image
        cv2.imshow("Image", output_image)
        cv2.waitKey(0)
