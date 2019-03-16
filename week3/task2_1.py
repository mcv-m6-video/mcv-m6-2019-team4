from utils import detection_gt_extractor
from paths import AICITY_DIR
from utils.object_tracking import Frame, ROI, ObjectTracker

def load_data():
    detector = detection_gt_extractor.detectionExtractorGT(
        AICITY_DIR.joinpath('det', 'det_yolo3.txt'))

    # frame objects creation
    frame_dict = {i: Frame(i) for i in range(1,detector.getGTNFrames()+1)}

    # filling frames with ROIs
    for i in frame_dict:
        for r in detector.getAllFrame(i):
            frame_dict[i].add_ROI( ROI(r[2], r[3], r[4], r[5]) )

    return frame_dict

if __name__ == "__main__":
    untracked_frames = load_data()
    tracker = ObjectTracker("RegionOverlap")

    for id, frame in untracked_frames.items():
        print("Tracking objects in frame {}".format(id))
        tracker.process_frame(frame)

        #if id == 3:
            #break

    tracker.print_objects()
