""" Task 1: Do inference using Mask RCNN and report mAP metric. """
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from datasets.aicity_dataset import AICityDataset

from evaluation import intersection_over_union, mean_ap
from utils import randomizer, detections_loader, detection_gt_extractor
from paths import AICITY_DIR, AICITY_ANNOTATIONS


def compute_mAP_with_offtheshelf_detections(verbose: bool = False,
                                            plot_interpolations: bool = False):
    """ Compute mAP given some detections and a ground truth.

    It only gets the detections where we have ground truth information
    (regardless there is a bounding box on it or not)
    """

    # Selects detections from available model predictions
    detections_path = Path(__file__).parent.joinpath('det_mask_rcnn.txt')
    # detections_path = AICITY_DIR.joinpath('det', 'det_mask_rcnn.txt')

    detections = detections_loader.load_bounding_boxes(detections_path)

    dataset = AICityDataset(AICITY_DIR, AICITY_ANNOTATIONS)
    ground_truth = dataset.get_labels()

    # Select detections only for frames we have ground truth
    mask = np.zeros(detections.shape[0], dtype=np.bool)
    frame_numbers = np.unique(ground_truth[:, 0])

    print(len(f'Number of frames to analyse: {frame_numbers}'))

    for frame_number in frame_numbers:
        mask |= detections[:, 0] == frame_number

    detections = detections[mask]

    mAP, AP_per_frame = compute_mAP(detections, ground_truth,
                                    verbose, plot_interpolations)

    print(f'Detections: {detections_path.stem} --> mAP: {mAP}')
    plt.plot(AP_per_frame)
    plt.plot(np.repeat(mAP, len(AP_per_frame)))
    plt.ylabel('AP')
    plt.xlabel('frame number')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim((0, 1))
    plt.show()


def compute_mAP(detections, ground_truth,
                verbose: bool = False, plot_interpolations: bool = False):
    """ Compute mAP given some detections and a ground truth.

    Note:
        Algorithm implemented::

            * For every frame
                * For each confidence level
                    * Compute precision-recall (consider TP when IoU >= 0.5)
                * Interpolate precision using 11 ranks r={0.0, 0.1, ... 1.0}
                * AP = Average for all interpolated precisions
            * mAP = Mean of AP for all frames
    """
    # Parameters
    iou_threshold = 0.5
    recalls = np.linspace(start=0, stop=1.0, num=11).round(decimals=1)

    frame_numbers = np.unique(ground_truth[:, 0])

    # Computes AP for every frame
    ap_per_frame = np.empty((0, 2))

    for frame_number in frame_numbers:
        det = detections[detections[:, 0] == frame_number]
        gt = ground_truth[ground_truth[:, 0] == frame_number]

        if verbose:
            print(f'Comparing {det.shape[0]} detections with '
                  f'{gt.shape[0]} ground truth boxes')

        # For each confidence value
        # Gets precision and recall for a frame considering IoU >= 0.5 TP
        confidences = np.unique(det[:, 5])
        confidences.sort()
        prs = np.empty((0, 2))

        for confidence in confidences:
            # Gets detection/s with higher confidence score than a threshold
            det_with_confidence_level = det[det[:, 5] >= confidence, :]

            det_bboxes = det_with_confidence_level[:, 1:5]
            gt_bboxes = gt[:, 2:6]

            if verbose:
                print(
                    f'Comparing {det_bboxes.shape[0]} detections with '
                    f'confidences >= {confidence} against '
                    f'{gt_bboxes.shape[0]} ground truth boxes',
                    end=' --> '
                )

            pr = mean_ap.get_precision_recall(det=det_bboxes,
                                              gt=gt_bboxes,
                                              threshold=iou_threshold)
            if verbose:
                print(f'Precision {pr[0]} Recall: {pr[1]}')

            prs = np.vstack((prs, pr))

        # print(f'Precision/Recall table:\n{prs}')
        # Interpolate p(r) for given r
        precisions_at_specific_recall = mean_ap.interpolate_precision(prs,
                                                                      recalls)
        AP = precisions_at_specific_recall.mean()

        if verbose:
            print(f'Frame {frame_number} AP: {AP}')

        # if verbose:
        #     print(f'precision-recall {prs}')

        if plot_interpolations:
            # Plot AP and its interpolation
            plt.subplot(211)
            plt.plot(prs[:, 1], prs[:, 0])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Mean precision as a function of mean recall')
            plt.axis((0, 1.1, 0, 1.1))

            plt.subplot(212)
            plt.plot(recalls, precisions_at_specific_recall)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Interpolated')
            plt.axis((0, 1.1, 0, 1.1))

            plt.show()

        # [frame_num, AP] column-table
        ap_per_frame = np.vstack((ap_per_frame, (frame_number, AP)))

    mAP = ap_per_frame[:, 1].mean()

    return mAP, ap_per_frame[:, 1]


if __name__ == '__main__':
    compute_mAP_with_offtheshelf_detections(verbose=False,
                                            plot_interpolations=False)
