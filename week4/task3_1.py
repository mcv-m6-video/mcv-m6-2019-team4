
from utils.object_tracking import ObjectTracker
import os
from week3.task2 import load_annotations, load_detections_txt, print_mot_metrics, make_video_from_kalman_tracker, make_video_from_tracker

def optical_flow_tracking():
    # Load detections
    #untracked_frames = load_detections_txt(AICITY_DIR.joinpath('det', 'det_yolo3.txt'))
    #untracked_frames = load_detections_txt(AICITY_DIR.joinpath('det', 'det_mask_rcnn.txt'), "LTWH")
    #untracked_frames = load_detections_txt(os.path.join('week3', 'det_mask_rcnn.txt'), "TLBR")
    untracked_frames = load_detections_txt(os.path.join('week3', 'det_retinanet.txt'), "LTWH", .5)
    method = "RegionOverlap"
    method = "Kalman"
    method = "OpticalFlow"
    tracker = ObjectTracker(method)

    for id, frame in untracked_frames.items():
        #print("Tracking objects in frame {}".format(id))
        tracker.process_frame(frame)

    tracker.print_objects()

    video_name = "{0}_{1}.avi".format("Tracking", method)
    #make_video_from_tracker(tracker, video_name)
    make_video_from_kalman_tracker(tracker, video_name)


    #Load annotations
    annotated_frames = load_annotations("m6-full_annotation.xml")
    annotation_tracker = ObjectTracker("")

    for id, frame in annotated_frames.items():
        #print("Loading annotated gt frame {}".format(id))
        annotation_tracker.load_annotated_frame(frame)

    #annotation_tracker.print_objects()
    #annotation_tracker.print_frames()
    #video_name = "{0}.avi".format("Annotations")
    #make_video_from_tracker(annotation_tracker, video_name)

    acc = tracker.compute_mot_metrics(annotation_tracker)
    print_mot_metrics(acc)


if __name__ == "__main__":
    optical_flow_tracking()
