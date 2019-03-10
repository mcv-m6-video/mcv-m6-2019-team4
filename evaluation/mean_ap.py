""" Evaluation metrics: mean Average Precision (mAP) """
import numpy as np
from typing import Tuple

from evaluation import intersection_over_union

np.set_printoptions(suppress=True)
verbose = False


def binary_test(detections: np.ndarray,
                detections_gt: np.ndarray,
                threshold: float) -> Tuple[float, float]:
    """ Computes precision and recall of detections using IoU scores

    If `IoU >= th` then is a TP, FP otherwise.

    Args:
        threshold: IoU threshold

    Returns:
        precision, recall
    """
    if detections.shape[0] == 0:
        # If no detection: Precision = 0/0, and Recall = 0/FN
        return 1, 0

    ious = compute_ious(detections, detections_gt)

    TP = sum(iou >= threshold for iou in ious)
    FP = sum(iou < threshold for iou in ious)
    FN = count_missed_out_objects(detections, detections_gt)
    precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 1
    else:
        recall = TP / (TP + FN)

    return precision, recall


def compute_ious(detections: np.ndarray,
                 detections_gt: np.ndarray) -> np.ndarray:
    """ Computes the Intersection over Union of detections """
    matches = find_matches(detections, detections_gt)
    ious = map(intersection_over_union.iou_from_bb, detections, matches)

    return np.array(list(ious))


def count_missed_out_objects(detections: np.ndarray,
                             detections_gt: np.ndarray) -> np.ndarray:
    """ Counts missed out objects (GT elements without detections) """
    ious = compute_ious(detections=detections_gt,
                        detections_gt=detections)
    return np.count_nonzero(ious == 0.0)


def delete_frame_info(detection: np.ndarray) -> np.ndarray:
    return detection[:, 1:]


def find_match_gt(detection: np.ndarray,
                  detections_gt_frame_k: np.ndarray) -> np.ndarray:
    """  Match a detection with a ground truth detection

    Given a detection and a list of ground truth detections find the one that
    results in the max IoU of them.

    Returns:
        the matched ground truth detection, None if no detection
    """
    # Find matching detections (a detection matches a ground truth when
    # their IoU is the maximum on the same frame)
    ious_frame_k = np.array(
        [
            intersection_over_union.iou_from_bb(detection, detection_gt)
            for detection_gt in detections_gt_frame_k
        ],
        dtype=float
    )

    # Relate a detection to the detection_gt that results in the
    # highest IoU of them
    matched_gt = detections_gt_frame_k[ious_frame_k.argmax()]
    iou = ious_frame_k.max()
    assert iou == intersection_over_union.iou_from_bb(detection,
                                                      matched_gt), 'BUG'

    return matched_gt


def find_matches(detections: np.ndarray,
                 detections_gt: np.ndarray) -> np.ndarray:
    """  Matches detections with a ground truth detections

    Given a detection and a list of ground truth detections find the
    one that results in the max IoU of them.

    Returns:
        the matched ground truth detection, None if no detection
    """
    # todo: for each detection, find a match
    # matches are empty if no detections_k
    # match is random but non-relevant when no match is found (iou=0)
    matches = np.empty((0, 4))
    for detection in detections:
        match = find_match_gt(detection, detections_gt)
        matches = np.vstack((matches, np.array(match)))

    return matches


def select_frame(detections: np.ndarray, idx: float) -> np.ndarray:
    return detections[detections[:, 0] == idx]


def get_precision_recall(detections,
                         detections_gt,
                         threshold: float) -> Tuple[np.float, np.float]:
    """ Gets precision and recall from detections and the ground truth

    Computes the mean precision and recall for IoU values [0.5, 0.55, .. 1.0],
    of detections which confidence score is higher than a threshold

    Args:
        threshold: discard detections with lower confidence score than th

    Returns:
        precision, recall
    """
    # Discards lower `score` than `confidence_score` from detections
    detections = detections[detections[:, 5] >= threshold]

    print(f'Getting PR for th = {threshold}')

    if verbose:
        print(f'Detections with conf. score over {threshold}: '
              f'{detections.shape}')

    # Discards info not related to bbox coordinates
    detections = np.delete(detections, [5], axis=1)
    detections_gt = delete_frame_info(detections_gt)
    detections = delete_frame_info(detections)
    assert detections.shape[1] == 4, 'BUG: Detections size is not correct'
    assert detections_gt.shape[1] == 4, 'BUG: Detections size is not correct'

    # Computes mean precision-recall in frame over IoU th {0.5, 0.55, ... 0.95}
    pr = np.array([binary_test(detections, detections_gt, th)
                   for th in np.arange(0.5, 1.0, 0.05)])
    mean_pr = pr.mean(axis=0)
    mean_precision, mean_recall = mean_pr[0], mean_pr[1]

    if verbose:
        print(f'Mean Precision: {mean_precision}')
        print(f'Mean Recall: {mean_recall}\n')

    return mean_precision, mean_recall
