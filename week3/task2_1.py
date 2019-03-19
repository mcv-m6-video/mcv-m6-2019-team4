from utils import detection_gt_extractor
from utils import annotation_parser
from paths import AICITY_DIR
from utils.object_tracking import Frame, ROI, ObjectTracker
import cv2
import glob
import os
import matplotlib.pyplot as plt

def load_detections_txt(detections_file):
    detector = detection_gt_extractor.detectionExtractorGT(detections_file)

    # frame objects creation
    frame_dict = {i: Frame(i) for i in range(1,detector.getGTNFrames()+1)}

    # filling frames with ROIs
    for i in frame_dict:
        for r in detector.getAllFrame(i):
            frame_dict[i].add_ROI( ROI(r[2], r[3], r[4], r[5]) )

    return frame_dict


def load_annotations(annotations_file):
    annotations = annotation_parser.annotationsParser(annotations_file)

    # frame objects creation
    frame_dict = {i: Frame(i) for i in range(1, annotations.getGTNFrames()+1)}

    # filling frames with ROIs
    for i in frame_dict:
        for r in annotations.getAllFrame(i):
            if r[-1] == 1: # only if car
                frame_dict[i].add_ROI( ROI(r[2], r[3], r[4], r[5], r[1]) )

    return frame_dict


def make_video_from_tracker(trckr, video_name):
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, four_cc, 10, (1920, 1080))

    filepaths = sorted(glob.glob(os.path.join(str(AICITY_DIR), 'frames/image-????.png')))
    for idx in range(1,len(filepaths)):
        print(filepaths[idx])
        image = cv2.imread(filepaths[idx])
        image = trckr.draw_frame(idx, image)
        cv2.imshow("Image", image)
        cv2.waitKey(1)

        video.write(image)

    video.release()


def overlap_tracking():

    #Load annotations
    annotated_frames = load_annotations("m6-full_annotation.xml")
    annotation_tracker = ObjectTracker("")

    for id, frame in annotated_frames.items():
        print("Loading annotated gt frame {}".format(id))
        annotation_tracker.load_annotated_frame(frame)

    #annotation_tracker.print_objects()
    #annotation_tracker.print_frames()
    video_name = "{0}.avi".format("Annotations")
    make_video_from_tracker(annotation_tracker, video_name)

    # Load detections
    untracked_frames = load_detections_txt(AICITY_DIR.joinpath('det', 'det_yolo3.txt'))
    method = "RegionOverlap"
    tracker = ObjectTracker(method)

    for id, frame in untracked_frames.items():
        print("Tracking objects in frame {}".format(id))
        tracker.process_frame(frame)

    video_name = "{0}_{1}.avi".format("Tracking", method)
    make_video_from_tracker(tracker, video_name)




if __name__ == "__main__":
    untracked_frames = load_detections_txt(AICITY_DIR.joinpath('det', 'det_yolo3.txt'))
    method = "RegionOverlap"
    tracker = ObjectTracker(method)

    for id, frame in untracked_frames.items():
        print("Tracking objects in frame {}".format(id))
        tracker.process_frame(frame)

    tracker.print_objects()

