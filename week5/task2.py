import os
from pathlib import Path

import cv2
import motmetrics as mm

from datasets.aicity_mtmc_dataset import AICityMTMCDataset
from utils.object_tracking import ObjectTracker
from week3.task2 import load_detections_txt, print_mot_metrics
from pnca import utils, predict


def make_video_from_tracker(trckr, cam, video_name):
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, four_cc, 10, (1920, 1080))

    idx = 1
    cam.openVideo()
    while (cam.videoIsOpened()):
        ret, image = cam.getNextFrame()
        if image is None:
            break

        image = trckr.draw_frame(idx, image)
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        video.write(image)
        idx += 1

    video.release()
    cam.closeVideo()


def MultiTrackSingleCameraImprovements(distance_metric='hist'):
    ds = AICityMTMCDataset()

    seq = ds.getTrainSeq(3)
    print("Sequence {} has cameras {}".format(seq.getName(), seq.getCameras()))

    trackers = {}

    trackingMethod = "RegionOverlap"
    detectionMethod = "yolo3"
    detectionThreshold = .8
    iniId = 0

    # cams = ['c010', 'c014']
    cams = set(seq.getCameras())
    for c in cams:
        cam = seq.getCamera(c)
        print("Camera {}".format(c))

        # Load detections
        untracked_frames = load_detections_txt(
            cam.getDetectionFile(detectionMethod), "LTWH", detectionThreshold)
        tracker = ObjectTracker(trackingMethod, iniId, distance_metric)
        for id, frame in untracked_frames.items():
            # print("Tracking objects in frame {}".format(id))
            tracker.process_frame(frame)

        # improvements
        tracker.removeStaticObjects()
        tracker.getImagesForROIs(cam.getVideoPath())
        tracker.mergeSimilarObjects()

        # save/view best images for each car
        # for id, obj in tracker.trackedObjects.items():
        # cv2.imwrite("results/{}_{}.png".format(c, id), obj.bestImage)
        # cv2.imshow("Image", obj.bestImage)
        # cv2.waitKey(1)

        trackers[c] = tracker
        # make_video_from_tracker(tracker, cam, "{}.avi".format(c))

        iniId += 1000

    acc = mm.MOTAccumulator(auto_id=True)
    for c in cams:
        cam = seq.getCamera(c)
        tracker = trackers[c]
        # Load ground truth
        gt_frames = load_detections_txt(cam.getGTFile(), "LTWH", .1, isGT=True)
        gt_tracker = ObjectTracker("", distance_metric)
        for id, frame in gt_frames.items():
            gt_tracker.load_annotated_frame(frame)

        # make_video_from_tracker(gt_tracker, cam, "test.avi")

        # Compute metrics
        a = tracker.compute_mot_metrics(gt_tracker)
        print_mot_metrics(a)

        tracker.update_mot_metrics(gt_tracker, acc)

    print_mot_metrics(acc)


def MultiTrackMultiCamera(distance_metric):
    rootpath = Path(__file__).parents[1].joinpath('data', 'AICity_data')
    p = '/'.join(
        os.path.abspath(__file__).split('/')[0:-2] + ['data', 'AICity_data']
    )
    # ds = AICityMTMCDataset(root_dir=os.path.abspath('../data/AICity_data'))
    ds = AICityMTMCDataset(root_dir=p)

    # Get sequence 3 cam 10
    seq = ds.getTrainSeq(3)
    print("Sequence {} has cameras {}".format(seq.getName(), seq.getCameras()))

    trackers = {}

    trackingMethod = "RegionOverlap"
    detectionMethod = "yolo3"
    detectionThreshold = .8
    iniId = 0

    # cams = ['c010', 'c014']
    cams = set(seq.getCameras())
    cams = {'c010', 'c011'}
    print('cams: {}'.format(cams))
    for c in cams:
        cam = seq.getCamera(c)
        print("Camera {}".format(c))

        # Load detections
        untracked_frames = load_detections_txt(
            cam.getDetectionFile(detectionMethod), "LTWH", detectionThreshold)
        tracker = ObjectTracker(trackingMethod, iniId, distance_metric)
        for id, frame in untracked_frames.items():
            # print("Tracking objects in frame {}".format(id))
            tracker.process_frame(frame)

        tracker.removeStaticObjects()
        tracker.getImagesForROIs(cam.getVideoPath())
        tracker.mergeSimilarObjects()
        # for id, obj in tracker.trackedObjects.items():
        # cv2.imwrite("results/{}_{}.png".format(c, id), obj.bestImage)
        # cv2.imshow("Image", obj.bestImage)
        # cv2.waitKey(1)
        trackers[c] = tracker
        # make_video_from_tracker(tracker, cam, "{}.avi".format(c))

        iniId += 1000

    acc = mm.MOTAccumulator(auto_id=True)
    for c in cams:
        cam = seq.getCamera(c)
        tracker = trackers[c]
        # Load ground truth
        gt_frames = load_detections_txt(cam.getGTFile(), "LTWH", .1, isGT=True)
        gt_tracker = ObjectTracker("", distance_metric)
        for id, frame in gt_frames.items():
            gt_tracker.load_annotated_frame(frame)

        # make_video_from_tracker(gt_tracker, cam, "test.avi")

        # Compute metrics
        # a = tracker.compute_mot_metrics(gt_tracker)
        # print_mot_metrics(a)

        tracker.update_mot_metrics(gt_tracker, acc)

    print_mot_metrics(acc)

    # merge tracks from different cameras
    camsToVisit = cams.copy()
    for c1 in cams:
        print('merging tracks')
        tracker = trackers[c1]
        camsToVisit.remove(c1)
        for c2 in camsToVisit:
            if c1 != c2:
                print("Merging {} and {}".format(c1, c2))
                tracker.mergeObjectTrackers(trackers[c2])

    # Compute metrics for merged tracks
    acc2 = mm.MOTAccumulator(auto_id=True)
    for c in cams:
        cam = seq.getCamera(c)
        tracker = trackers[c]
        # Load ground truth
        gt_frames = load_detections_txt(cam.getGTFile(), "LTWH", .1, isGT=True)
        gt_tracker = ObjectTracker("", distance_metric)
        for id, frame in gt_frames.items():
            gt_tracker.load_annotated_frame(frame)

        # make_video_from_tracker(gt_tracker, cam, "test.avi")

        # Compute metrics
        # a = tracker.compute_mot_metrics(gt_tracker)
        # print_mot_metrics(a)

        tracker.update_mot_metrics(gt_tracker, acc2)

    print_mot_metrics(acc2)


if __name__ == '__main__':
    MultiTrackMultiCamera(distance_metric='hist')
    # MultiTrackMultiCamera(distance_metric='pnca')
