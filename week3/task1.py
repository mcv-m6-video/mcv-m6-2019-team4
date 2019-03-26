""" Task 1: Do inference using Mask RCNN and report mAP metric. """
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from fonts.ttf import SourceSerifPro
from PIL import Image, ImageDraw, ImageFont

from datasets.aicity_dataset import AICityDataset

from evaluation import intersection_over_union, mean_ap
from utils import randomizer, detections_loader, detection_gt_extractor
from paths import AICITY_DIR, AICITY_ANNOTATIONS


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


def compute_mAP_with_offtheshelf_detections(verbose: bool = False,
                                            plot_interpolations: bool = False):
    """ Compute mAP given some detections and a ground truth.

    It only gets the detections where we have ground truth information
    (regardless there is a bounding box on it or not)
    """

    # Selects detections from available model predictions
    # detections_path = AICITY_DIR.joinpath('det', 'det_mask_rcnn.txt')
    # detections_path = Path(__file__).parent.joinpath('det_mask_rcnn.txt')
    # detections = detections_loader.load_bounding_boxes(detections_path, False)

    detections_path = Path(__file__).parent.joinpath('det_retinanet.txt')
    detections_all = detections_loader.load_bounding_boxes(detections_path, True)

    dataset = AICityDataset(AICITY_DIR, AICITY_ANNOTATIONS)
    ground_truth = dataset.get_labels()

    # Select detections only for frames we have ground truth
    mask = np.zeros(detections_all.shape[0], dtype=np.bool)
    frame_numbers = np.unique(ground_truth[:, 0])

    print(len(f'Number of frames to analyse: {frame_numbers}'))

    for frame_number in frame_numbers:
        mask |= detections_all[:, 0] == frame_number

    detections = detections_all[mask]

    mAP, AP_per_frame = compute_mAP(detections, ground_truth,
                                    verbose, plot_interpolations)

    print(f'Detections: {detections_path.stem} --> mAP: {mAP}')

    plot_AP_per_frame(mAP, AP_per_frame)

    frame_best, frame_worst = np.argmax(AP_per_frame), np.argmin(AP_per_frame)
    print(f'Max AP found in frame: {frame_best} ({AP_per_frame[frame_best]})\n'
          f'Min AP found in frame: {frame_worst} '
          f'({AP_per_frame[frame_worst]})\n'
          f'Frame sorted by best AP:\n{np.argsort(AP_per_frame)}')

    show_example(dataset, detections_all, int(frame_best), 'best')
    show_example(dataset, detections_all, int(frame_worst), 'worst')


def plot_AP_per_frame(mAP, AP_per_frame):
    plt.plot(AP_per_frame)
    plt.plot(np.repeat(mAP, len(AP_per_frame)))
    plt.ylabel('AP')
    plt.xlabel('frame number')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim((0, 1))
    plt.title('AP per frame')
    plt.show()

    return mAP, AP_per_frame


def show_example(dataset, detections, frame_num: int, title: str):
    # Load groundtruth
    # dataset = AICityDataset(AICITY_DIR, AICITY_ANNOTATIONS)
    labels: np.ndarray = dataset.get_labels()
    gt = labels[labels[:, 0] == frame_num]
    image: Image = dataset.get_pil_image(frame_num)
    coord_gt = gt[:, 2:6]

    # Select detections of an specific frame
    detections = detections[detections[:, 0] == frame_num]
    coord_det_and_confidences = detections[:, 1:6]

    # Draw bounding boxes
    draw = ImageDraw.ImageDraw(image)
    colour_gt = (0, 255, 0)
    colour_detection = (255, 0, 0)
    font = ImageFont.truetype(SourceSerifPro, 18)

    for detection in coord_gt:
        bb = detection.astype(np.int).tolist()
        draw.rectangle(bb, outline=colour_gt, width=2)

    for detection in coord_det_and_confidences:
        bb = detection[0:4].astype(np.int).tolist()
        confidence = str(detection[4])

        draw.rectangle(bb, outline=colour_detection, width=2)
        draw.text(
            bb[2:4],
            text=confidence,
            fill=colour_detection,
            font=font,
        )

    image.show(title=title)


if __name__ == '__main__':
    compute_mAP_with_offtheshelf_detections(verbose=False,
                                            plot_interpolations=False)
    # Show best
    # show_example(6, 'best')

    # Show worst
    # show_example(1764, 'worst')
