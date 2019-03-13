""" Task 1: Detection metrics. """
import cv2
import numpy as np
from matplotlib import pyplot as plt

from datasets.aicity_dataset import AICityDataset

from evaluation import intersection_over_union, mean_ap
from utils import randomizer, detections_loader, detection_gt_extractor
from paths import AICITY_DIR, AICITY_ANNOTATIONS

""" T1.1 Detection metrics detection (quantitative) """


def test_iou_with_synth_data():
    # Test function for IoU
    boxA = [0., 0., 10., 10.]
    boxB = [1., 1., 11., 11.]

    result = intersection_over_union.iou_from_bb(boxA, boxB)
    print('Result: {0}\n'
          'Correct Solution: {1}'.format(result, '0.680672268908'))

    print('Normalizing coordinates in a 100x100 coordinate system')
    boxA = [a / 100. for a in boxA]
    boxB = [b / 100. for b in boxB]

    result = intersection_over_union.iou_from_bb(boxA, boxB)

    print('Result:  {0}\n'
          'Correct Solution: {1}'.format(result, '0.680672268908'))

    print('Two boxes with no overlap')
    boxA = [0., 0., 10., 10.]
    boxB = [12., 12., 22., 22.]
    result = intersection_over_union.iou_from_bb(boxA, boxB)

    print('Result:  {0}\n'
          'Correct Solution: {1}'.format(result, '0.0'))

    print('Additional example')
    boxA = [0., 0., 2., 2.]
    boxB = [1., 1., 3., 3.]
    result = intersection_over_union.iou_from_bb(boxA, boxB)

    print('Result:  {0}\n'
          'Correct Solution: {1}'.format(result, '0.142857142857'))


def test_iou_with_noise():
    # Read GT from dataset
    gtExtractor = detection_gt_extractor.detectionExtractorGT(
        AICITY_DIR.joinpath('gt', 'gt.txt'))

    # parameters to randomize detections
    randomNoiseScale = 100
    additionDeletionProbability = 0.0

    TP = 0
    FN = 0
    FP = 0
    threshold = 0.5

    for i in range(gtExtractor.getGTNFrames()):

        # Get GT BBOX
        gt = []
        for j in range(len(gtExtractor.gt)):
            if gtExtractor.getGTFrame(j) == i:
                gtBBOX = gtExtractor.getGTBoundingBox(j)
                gt.append(gtBBOX)

        # Get detection BBOX
        detections = randomizer.randomize_detections(
            additionDeletionProbability,
            randomNoiseScale,
            gt[:]
        )
        BBoxesDetected = []

        for x in range(len(gt)):
            gtBBOX = gt[x]
            detection = []
            maxIoU = 0
            BBoxDetected = -1

            for y in range(len(detections)):
                iou = intersection_over_union.iou_from_bb(gtBBOX,
                                                          detections[
                                                              y])
                if iou >= maxIoU:
                    maxIoU = iou
                    detection = detections[y]
                    BBoxDetected = y

            if maxIoU > threshold:
                TP = TP + 1
                BBoxesDetected.append(BBoxDetected)
            else:
                FN = FN + 1

            # load the image
            frame_path = AICITY_DIR.joinpath('frames',
                                             'image-{:04d}.png'.format(i + 1))
            image = cv2.imread(str(frame_path))

            # draw the ground-truth bounding box along with the predicted
            # bounding box
            cv2.rectangle(
                image,
                (int(detection[0]), int(detection[1])),
                (int(detection[2]), int(detection[3])),
                (0, 0, 255),
                2
            )
            cv2.rectangle(
                image,
                (int(gtBBOX[0]), int(gtBBOX[1])),
                (int(gtBBOX[2]), int(gtBBOX[3])),
                (0, 255, 0),
                2
            )

            # compute the intersection over union and display it

            cv2.putText(
                image,
                "IoU: {:.4f}".format(maxIoU),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            print("{}: {:.4f}".format(frame_path, maxIoU))

            # show the output image
            cv2.imshow("Image", image)
            cv2.waitKey(0)

        for y in range(len(detections)):
            if not BBoxesDetected.__contains__(y):
                FP = FP + 1


""" T1.2 Detection metrics detection (quantitative) """

""" T1.3 Detection metrics detection (quantitative) """


def get_synth_data_perfect_match():
    ground_truth = np.array([
        [0., 100., 100., 200., 200.],
        [0., 300., 300., 400., 400.],
        [0., 500., 500., 600., 600.],
    ])
    detections = np.array([
        [0., 100., 100., 200., 200., .9],  # true pos iou = 1
        [0., 300., 300., 400., 400., .5],  # true pos iou = 1
        [0., 500., 500., 600., 600., .2],  # true pos iou = 1
    ])
    return detections, ground_truth


def get_synth_data():
    """ Synthetic detections and gt for development """
    ground_truth = np.array([
        [0., 100., 100., 200., 200.],
        [0., 300., 300., 400., 400.],
        [0., 500., 500., 600., 600.],
    ])
    detections = np.array([
        [0., 100., 100., 200., 200., .9],  # true pos iou = 1
        [0., 300., 300., 350., 350., .5],  # true pos iou = 0.25
        # [0., 500., 500., 600., 600., .2],   # missed out (fn) iou = 0
        [0., 700., 700., 800., 800., .6],  # false pos iou = 0
        [0., 900., 900., 1000., 1000., .4],  # false pos iou = 0
        [0., 1100., 1100., 1200., 1200., .2],  # false pos iou = 0
    ])

    return detections, ground_truth


def load_precomputed_precision_recall_data():
    """ Code for development.

    Shortcut to get precision recall data while testing
    """
    return np.array([
        [0.43071253, 0.88640822, 0.],
        [0.43071253, 0.88640822, 0.03333333],
        [0.43071253, 0.88640822, 0.06666667],
        [0.43071253, 0.88640822, 0.1],
        [0.43071253, 0.88640822, 0.13333333],
        [0.43071253, 0.88640822, 0.16666667],
        [0.43071253, 0.88640822, 0.2],
        [0.46346667, 0.6953379, 0.23333333],
        [0.47966574, 0.69124287, 0.26666667],
        [0.49306358, 0.69020642, 0.3],
        [0.5130303, 0.68861168, 0.33333333],
        [0.52056075, 0.6873139, 0.36666667],
        [0.53173077, 0.68666466, 0.4],
        [0.5372093, 0.68368599, 0.43333333],
        [0.55400697, 0.68190011, 0.46666667],
        [0.56630435, 0.68002819, 0.5],
        [0.56792453, 0.67359564, 0.53333333],
        [0.56311475, 0.65520325, 0.56666667],
        [0.5504386, 0.63464453, 0.6],
        [0.53203883, 0.609448, 0.63333333],
        [0.53061224, 0.60047557, 0.66666667],
        [0.52263158, 0.59243317, 0.7],
        [0.51914894, 0.58816615, 0.73333333],
        [0.51764706, 0.58668215, 0.76666667],
        [0.52108108, 0.5861558, 0.8],
        [0.51444444, 0.57622638, 0.83333333],
        [0.52215909, 0.57521014, 0.86666667],
        [0.52470588, 0.56678326, 0.9],
        [0.57320261, 0.56438531, 0.93333333],
        [0.79880952, 0.17378778, 0.96666667],
        [1., 0., 1., ],
    ])


def compute_mAP(verbose: bool = False, plot: bool = False):
    """ Compute mAP given some detections and a ground truth.

    It only gets the detections where we have ground truth information
    (regardless there is a bounding box on it or not)

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

    # Selects detections from available model predictions
    detections_path = AICITY_DIR.joinpath('det', 'det_mask_rcnn.txt')
    # detections_path = AICITY_DIR.joinpath('det', 'det_ssd512.txt')
    # detections_path = AICITY_DIR.joinpath('det', 'det_yolo3.txt')

    detections = detections_loader.load_bounding_boxes(detections_path)
    dataset = AICityDataset(AICITY_DIR, AICITY_ANNOTATIONS)
    ground_truth = dataset.get_labels()

    # Select detections only for frames we have ground truth
    mask = np.zeros(detections.shape[0], dtype=np.bool)
    frame_numbers = np.unique(ground_truth[:, 0])
    for frame_number in frame_numbers:
        mask |= detections[:, 0] == frame_number

    detections = detections[mask]

    # Computes AP for every frame
    ap_per_frame = np.empty((0, 2))
    for frame_number in frame_numbers:
        det = detections[detections[:, 0] == frame_number]
        gt = ground_truth[ground_truth[:, 0] == frame_number]

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
            pr = mean_ap.get_precision_recall(det=det_bboxes,
                                              gt=gt_bboxes,
                                              threshold=iou_threshold)
            prs = np.vstack((prs, pr))

        # print(f'Precision/Recall table:\n{prs}')
        # Interpolate p(r) for given r
        precisions_at_specific_recall = mean_ap.interpolate_precision(prs,
                                                                      recalls)
        AP = precisions_at_specific_recall.mean()

        if verbose:
            print(f'Frame {frame_number} precision-recall {prs} AP: {AP}')

        if plot:
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

    print(f'Detections: {detections_path.stem} --> mAP: {mAP}')
    return mAP


""" T1.4 Foreground detection (qualitative) """
