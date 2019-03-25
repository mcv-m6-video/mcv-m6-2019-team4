""" Evaluation metrics: mean Average Precision (mAP) """
import numpy as np
from typing import Tuple

from evaluation import intersection_over_union


def find_match_in_gt_from_a_detection(det: np.ndarray,
                                      gt: np.ndarray) -> np.ndarray:
    """  Match a detection with a ground truth detection

    Given a detection and a list of ground truth detections find the one that
    results in the max IoU of them.

    Returns:
        the matched ground truth detection, None if no detection
    """
    # Find matching detections (a detection matches a ground truth when
    # their IoU is the maximum on the same frame)
    ious = np.array(
        [
            intersection_over_union.iou_from_bb(det, get_bbox)
            for get_bbox in gt
        ],
        dtype=float
    )

    # Relate a detection to the detection_gt that results in the
    # highest IoU of them
    matched_gt = gt[ious.argmax()]
    iou = ious.max()
    assert iou == intersection_over_union.iou_from_bb(det, matched_gt), 'BUG'
    return matched_gt


def find_matches_in_gt_from_detections(det: np.ndarray,
                                       gt: np.ndarray) -> np.ndarray:
    """ Matches detections with a ground truth detections

    Given a detection and a list of ground truth detections find the
    one that results in the max IoU of them.

    Returns:
        the matched ground truth detection, None if no detection
    """
    # matches are empty if empty detections
    # match is random but non-relevant when no match is found (iou=0)
    matches = np.empty((0, 4))
    assert det.shape[1] == 4 and gt.shape[1] == 4
    for detection in det:
        match = find_match_in_gt_from_a_detection(detection, gt)
        matches = np.vstack((matches, match))

    return matches


def compute_iou_from_a_frame(det: np.ndarray,
                             gt: np.ndarray) -> np.ndarray:
    """ Computes the Intersection over Union of detections """
    matches = find_matches_in_gt_from_detections(det, gt)
    ious = map(intersection_over_union.iou_from_bb, det, matches)

    return np.array(list(ious))


def count_missed_out_objects(det: np.ndarray,
                             gt: np.ndarray) -> np.ndarray:
    """ Counts missed out objects (GT elements without detections) """
    ious = compute_iou_from_a_frame(det=gt, gt=det)
    return np.count_nonzero(ious == 0.0)


def get_precision_recall(det: np.ndarray,
                         gt: np.ndarray,
                         threshold: float) -> Tuple[float, float]:
    """ Computes precision and recall of detections using IoU scores

    If `IoU >= th` then is a TP, FP otherwise.

    Args:
        det: detections
        gt: ground truth
        threshold: IoU threshold

    Returns:
        precision, recall
    """
    if det.shape[0] == 0:
        # If no detection: Precision = 0/0, and Recall = 0/FN
        return 1, 0

    ious = compute_iou_from_a_frame(det, gt)

    TP = sum(iou >= threshold for iou in ious)
    FP = sum(iou < threshold for iou in ious)
    FN = count_missed_out_objects(det, gt)

    precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 1
    else:
        recall = TP / (TP + FN)

    return precision, recall


def interpolate_precision(pr: np.ndarray, recalls: np.ndarray) -> np.ndarray:
    """ Interpolates precision given a set of recall values

    Interpolated precision takes the maximum precision over all recalls
    greater than r.

    Args:
        pr: (precision, recall) column-table
        recalls: specifics recalls to interpolate a precision value
    """
    recalls.sort()
    return np.array([np.max(pr[pr[:, 1] >= r, 0], initial=0) for r in recalls])


