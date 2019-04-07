from datasets.aicity_mtmc_dataset import AICityMTMCDataset, AICityMTMCSequence, AICityMTMCCamera
from utils.object_tracking import ObjectTracker
from week3.task2 import load_annotations, load_detections_txt, print_mot_metrics
import cv2

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

def MultiTrackMultiCamera():

    ds = AICityMTMCDataset()

    # Get sequence 3 cam 10
    seq = ds.getTrainSeq(3)
    print("Sequence {} has cameras {}".format(seq.getName(), seq.getCameras()))

    trackers = {}

    for c in seq.getCameras():
        cam = seq.getCamera(c)
        print("Camera {}".format(c))

        # Load detections
        untracked_frames = load_detections_txt(cam.getDetectionFile("yolo3"), "LTWH", .8)
        method = "RegionOverlap"
        tracker = ObjectTracker(method)
        for id, frame in untracked_frames.items():
            #print("Tracking objects in frame {}".format(id))
            tracker.process_frame(frame)

        #make_video_from_tracker(tracker, cam, "test.avi")

        # Load ground truth
        gt_frames = load_detections_txt(cam.getGTFile(), "LTWH", .2, isGT=True)
        gt_tracker = ObjectTracker("")
        for id, frame in gt_frames.items():
            gt_tracker.load_annotated_frame(frame)

        #make_video_from_tracker(gt_tracker, cam, "test.avi")

        # Compute metrics
        acc = tracker.compute_mot_metrics(gt_tracker)
        print_mot_metrics(acc)

        trackers[c] = tracker

    # TODO: play with trackers dict to match tracked objects

