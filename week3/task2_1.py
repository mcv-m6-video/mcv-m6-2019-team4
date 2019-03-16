from utils import detection_gt_extractor
from paths import AICITY_DIR
from utils.object_tracking import Frame, ROI, ObjectTracker
import cv2
import glob
import os
import matplotlib.pyplot as plt

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

def overlap_tracking():
    untracked_frames = load_data()
    method = "RegionOverlap"
    tracker = ObjectTracker(method)

    for id, frame in untracked_frames.items():
        print("Tracking objects in frame {}".format(id))
        tracker.process_frame(frame)

    video_name = "{0}_{1}.avi".format("Tracking", method)
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, four_cc, 10, (1920, 1080))

    filepaths = sorted(glob.glob(os.path.join(str(AICITY_DIR), 'frames/image-????.png')))
    for idx in range(1,len(filepaths)):
        print(filepaths[idx])
        image = cv2.imread(filepaths[idx])
        image = tracker.draw_frame(idx, image)
        cv2.imshow("Image", image)
        cv2.waitKey(1)

        video.write(image)

    video.release()

if __name__ == "__main__":
    untracked_frames = load_data()
    method = "RegionOverlap"
    tracker = ObjectTracker(method)

    for id, frame in untracked_frames.items():
        print("Tracking objects in frame {}".format(id))
        tracker.process_frame(frame)

        #if id == 3:
            #break

    tracker.print_objects()

