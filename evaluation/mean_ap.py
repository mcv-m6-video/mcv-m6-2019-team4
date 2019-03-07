""" Evaluation metrics: mean Average Precision (mAP) """
import numpy as np
from typing import Tuple

from sklearn import metrics
from matplotlib import pyplot as plt

from datasets.aicity_dataset import AICityDataset
from evaluation.intersection_over import bb_intersection_over_union
from detections_loader import load_bounding_boxes
from paths import AICITY_DIR

np.set_printoptions(suppress=True)


def binary_test(detections: np.ndarray,
                detections_gt: np.ndarray,
                threshold: float) -> Tuple[float, float]:
    """ Computes precision and recall of detections using IoU scores

    If `IoU >= th` then is a TP, FP otherwise.

    Args:
        threshold: IoU threshold
    """
    print(f'[binary_test] th: {threshold}')
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
    ious = map(bb_intersection_over_union, detections, matches)

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
            bb_intersection_over_union(detection, detection_gt)
            for detection_gt in detections_gt_frame_k
        ],
        dtype=float
    )

    # Relate a detection to the detection_gt that results in the
    # highest IoU of them
    matched_gt = detections_gt_frame_k[ious_frame_k.argmax()]
    iou = ious_frame_k.max()
    assert iou == bb_intersection_over_union(detection, matched_gt), 'BUG'

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
                         th: float):
    """ Gets precision and recall from detections and the ground truth

    Args:
        th: discard detections with lower confidence score than th
    """
    print(f'[get_precision_recall] th: {th}')

    # Discard lower `score` than `confidence_score` from detections
    detections = detections[detections[:, 5] >= th]

    if verbose:
        print(
            f'Detections with conf. score over {th}: '
            f'{detections.shape}')

    # Discard scores
    if not verbose:
        print(f'Discarding id, class and scores. '
              f'Only [frame, Left, Upper, Right, Lower]')
        print(f'GT detections: {detections_gt.shape}')
        print(f'Detections: {detections.shape}')
    detections = np.delete(detections, [5], axis=1)
    detections_gt = delete_frame_info(detections_gt)
    detections = delete_frame_info(detections)

    if not verbose:
        print(f'GT detections: {detections_gt.shape}')
        print(f'Detections: {detections.shape}')

    # computes mean precision in frame over IoU th {0.5, 0.55, ... 0.95}
    pr_rec = np.array([binary_test(detections, detections_gt, th)
                       for th in np.arange(0.5, 1.0, 0.05)])

    m_pr_rec = pr_rec.mean(axis=0)
    precision, recall = m_pr_rec[0], m_pr_rec[1]

    if not verbose:
        print(f'Precision: {precision}')
        print(f'Recall: {recall}\n')

    return precision, recall


def compute_confidence_precision_recall():
    verbose = False
    detections_path = AICITY_DIR.joinpath('det', 'det_mask_rcnn.txt')
    # detections_path = AICITY_DIR.joinpath('det', 'det_ssd512.txt')
    # detections_path = AICITY_DIR.joinpath('det', 'det_yolo3.txt')
    detections = load_bounding_boxes(detections_path)

    ds = AICityDataset(AICITY_DIR)
    # gt_bb = np.array([
    #     np.hstack((idx, sample[1]))
    #     for idx, sample in enumerate(ds)
    # ])

    detections_gt = ds.get_labels()
    if verbose:
        print(f'GT detections: {detections_gt.shape}')
        print(f'Detections: {detections.shape}\n')

    # Select detections only for some frames (it's time consuming)
    detections = detections[detections[:, 0] >= 1090., :]
    detections = detections[detections[:, 0] <= 1130., :]

    detections_gt = detections_gt[detections_gt[:, 0] >= 1090., :]
    detections_gt = detections_gt[detections_gt[:, 0] <= 1130., :]

    # Ensure all bb belong to the class bicycle
    # assert all(detections_gt[:, 6]) == 0.0

    # Discard `id` and `class` information from gt
    detections_gt = np.delete(detections_gt, [1, 6], axis=1)

    # NOTE: synthetic data for dev
    # detections_k = detections_gt_k
    # detections_gt = np.array([
    #     [0., 100., 100., 200., 200.],
    #     [0., 300., 300., 400., 400.],
    #     [0., 500., 500., 600., 600.],
    # ])
    # detections = np.array([
    #     [0., 100., 100., 200., 200., .9],  # true pos iou = 1
    #     [0., 300., 300., 350., 350., .5],  # true pos iou = 0.25
    #     # [0., 500., 500., 600., 600.],   # missed out (fn) iou = 0
    #     [0., 700., 700., 800., 800., .6],  # false pos iou = 0
    #     [0., 900., 900., 1000., 1000., .4],  # false pos iou = 0
    #     [0., 1100., 1100., 1200., 1200., .2],  # false pos iou = 0
    # ])

    # ¿iterate?: for frame_num in np.arange(0., 40., 1.):
    # def by_frame(frame_num: float):
    # detections_gt_k = select_frame(detections_gt, frame_num)
    # detections_k = select_frame(detections, frame_num)
    # p_and_r_list = np.array([by_frame(frame_num)
    #                          for frame_num in
    #                          np.arange(1., 40., 1.)])
    #
    # precisions, recalls = p_and_r_list[:, 0], p_and_r_list[:, 1]

    # Compute precision as a function of recall [[precision, recall, conf],]
    confidence_precision_recall = np.array([
        (
            *get_precision_recall(detections, detections_gt,
                                  th=confidence),
            confidence,
        )
        for confidence in np.linspace(0.0, 1.0, 31)
    ])

    # Sort table by increasing recall

    print(f'Table: precision, recall, confidence\n'
          f'{confidence_precision_recall}')

    confidence_precision_recall = confidence_precision_recall[
        confidence_precision_recall[:, 1].argsort()]

    print(f'Table (sorted): precision, recall, confidence\n'
          f'{confidence_precision_recall}')

    plt.plot(confidence_precision_recall[:, 1],
             confidence_precision_recall[:, 0])
    #
    # plt.scatter(confidence_precision_recall[:, 2],
    #             confidence_precision_recall[:, 1])

    plt.show()
    return confidence_precision_recall

    # TODO (jon) somehow, iterate selecting only the detections with diff
    #  confidence score to get a list of precision and recalls, so then
    #  somehow average and mean
    # img, gt = ds[0]
    # print(gt)
    # img.crop(gt).show()
    # print(detections[0][1:])
    # img.show()
    # img.crop(detections[0][1:]).show()

    # metrics.average_precision_score()

    # ious = [bb_intersection_over_union(label, label) for _, label in ds]
    # assert sum(ious) == len(ious)

    # mean_avg_precision(ious, .5)
    # mean_avg_precision(ious, .75)

    # for th in np.arange(start=.5, stop=.95, step=.05):
    #     a = mean_avg_precision(ious, th)


def run():
    pre_rec_conf = compute_confidence_precision_recall()

    # pre_rec_conf = np.array([
    #     [0.43071253, 0.88640822, 0.],
    #     [0.43071253, 0.88640822, 0.03333333],
    #     [0.43071253, 0.88640822, 0.06666667],
    #     [0.43071253, 0.88640822, 0.1],
    #     [0.43071253, 0.88640822, 0.13333333],
    #     [0.43071253, 0.88640822, 0.16666667],
    #     [0.43071253, 0.88640822, 0.2],
    #     [0.46346667, 0.6953379, 0.23333333],
    #     [0.47966574, 0.69124287, 0.26666667],
    #     [0.49306358, 0.69020642, 0.3],
    #     [0.5130303, 0.68861168, 0.33333333],
    #     [0.52056075, 0.6873139, 0.36666667],
    #     [0.53173077, 0.68666466, 0.4],
    #     [0.5372093, 0.68368599, 0.43333333],
    #     [0.55400697, 0.68190011, 0.46666667],
    #     [0.56630435, 0.68002819, 0.5],
    #     [0.56792453, 0.67359564, 0.53333333],
    #     [0.56311475, 0.65520325, 0.56666667],
    #     [0.5504386, 0.63464453, 0.6],
    #     [0.53203883, 0.609448, 0.63333333],
    #     [0.53061224, 0.60047557, 0.66666667],
    #     [0.52263158, 0.59243317, 0.7],
    #     [0.51914894, 0.58816615, 0.73333333],
    #     [0.51764706, 0.58668215, 0.76666667],
    #     [0.52108108, 0.5861558, 0.8],
    #     [0.51444444, 0.57622638, 0.83333333],
    #     [0.52215909, 0.57521014, 0.86666667],
    #     [0.52470588, 0.56678326, 0.9],
    #     [0.57320261, 0.56438531, 0.93333333],
    #     [0.79880952, 0.17378778, 0.96666667],
    #     [1., 0., 1., ]])

    # Interpolate data and retrieve Precision for Recall = {0., 0.1, ... 1.0}
    interpol = np.interp(
        x=np.linspace(0., 1.0, 11),
        xp=pre_rec_conf[:, 1],
        fp=pre_rec_conf[:, 0]
    )

    AP = interpol.mean()
    print(f'AP is: {AP}')


if __name__ == '__main__':
    verbose = False
    run()
