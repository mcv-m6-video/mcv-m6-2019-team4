import os
from datasets.aicity_mtmc_dataset import AICityMTMCDataset, AICityMTMCSequence, AICityMTMCCamera
from utils.object_tracking import ObjectTracker
from week3.task2 import load_annotations, load_detections_txt, print_mot_metrics
import cv2
import numpy as np
from matplotlib import pyplot as plt


def make_video_from_tracker(trckr, cam, video_name, plot=False, track_method='RegionOverlap', width=1920, height=1080):
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, four_cc, 10, (width, height))

    idx = 1
    cam.openVideo()
    while cam.videoIsOpened():
        ret, image = cam.getNextFrame()
        if image is None:
            break
        if track_method == 'Kalman':
            image = trckr.draw_frame_kalman(idx, image)
        else:
            image = trckr.draw_frame(idx, image)

        if plot:
            cv2.imshow("Image", image)
            cv2.waitKey(1)
        video.write(image)
        idx += 1

    video.release()
    cam.closeVideo()


def MultiTrackSingleCamera(tested_seqs=[3], track_meth='RegionOverlap', detect_meth='ssd512', make_video_track=True,
                           make_video_gt=True, make_video_unfiltered=True, make_video_mtsc=False, min_conf=.2,
                           parked_threshold=5.0, hasGT=True, mtsc_meth=None):

    ds = AICityMTMCDataset()
    if hasGT:
        avg_idf1 = []
    else:
        avg_idf1 = -1  # meaning there is no ground truth
    # For each sequence, track all cameras
    for sq_id in tested_seqs:
        seq = ds.getTrainSeq(sq_id)
        print("Sequence {} has cameras {}".format(seq.getName(), seq.getCameras()))
        # remove 'rogue' cameras, if any or if we only want to test some
        idf1_cam = dict.fromkeys(seq.getCameras())
        # idf1_cam.pop('c011')
        # idf1_cam.pop('c012')
        # idf1_cam.pop('c013')
        # idf1_cam.pop('c014')
        # idf1_cam.pop('c015')
        for c in sorted(idf1_cam.keys()):  # temporally select only c030-c040
            cam = seq.getCamera(c)
            print("Camera {}".format(c))

            if mtsc_meth is None:
                # Load detections
                untracked_frames = load_detections_txt(cam.getDetectionFile(detect_meth), "LTWH", confidence_th=min_conf)
                tracker = ObjectTracker(track_meth)
                for id, frame in untracked_frames.items():
                    # print("Tracking objects in frame {}".format(id))
                    tracker.process_frame(frame)

                # Changes to remove static objects based on a relative threshold (% of width/height not number of pixels)
                cam.openVideo()
                if cam.videoIsOpened():
                    ret, frame = cam.getNextFrame()
                    height, width, _ = frame.shape

                if make_video_unfiltered:
                    video_name = 'Tracking_{0}_S{1:02d}_{2}_det-{3}_unfiltered.avi'.format(track_meth, sq_id, c,
                                                                                           detect_meth)
                    if not os.path.isfile(video_name):
                        print("Creating video from tracker...")
                        make_video_from_tracker(tracker, cam, video_name, plot=False, track_method=track_meth, width=width,
                                                height=height)
                    else:
                        print("A video with the same name already exists, continuing without overwritting it...")

                tracker.removeStaticObjects(dist_threshold_px=parked_threshold, width=width, height=height)

                if make_video_track:
                    video_name = 'Tracking_{0}_S{1:02d}_{2}_det-{3}.avi'.format(track_meth, sq_id, c, detect_meth)
                    if not os.path.isfile(video_name):
                        print("Creating video from tracker...")
                        make_video_from_tracker(tracker, cam, video_name, plot=False, track_method=track_meth, width=width,
                                                height=height)
                    else:
                        print("A video with the same name already exists, continuing without overwritting it...")

                if hasGT:
                    # Load ground truth
                    gt_frames = load_detections_txt(cam.getGTFile(), "LTWH", .2, isGT=True)
                    gt_tracker = ObjectTracker("")
                    for id, frame in gt_frames.items():
                        gt_tracker.load_annotated_frame(frame)

                    if make_video_gt:
                        video_name = 'Annotations_S{0:02d}_{1}.avi'.format(sq_id, c)
                        if not os.path.isfile(video_name):
                            print("Creating video from gt annotations...")
                            make_video_from_tracker(gt_tracker, cam, video_name, width=width, height=height)
                        else:
                            print("A video with the same name already exists, continuing w/o overwritting it...")

                    # Compute metrics
                    acc = tracker.compute_mot_metrics(gt_tracker)
                    idf1 = print_mot_metrics(acc)
                    idf1_cam[c] = idf1

            else:  # mtsc is not None, use it
                # Load Track Clustering tracks
                mtsc_frames = load_detections_txt(cam.getMTSCtracks(mtsc_meth), gtFormat='LTWH', isGT=True)
                mtsc_tracker = ObjectTracker("")
                for id, frame in mtsc_frames.items():
                    mtsc_tracker.load_annotated_frame(frame)

                mtsc_tracker.removeStaticObjects(dist_threshold_px=parked_threshold, width=width, height=height)

                if make_video_mtsc:
                    video_name = 'MTSC_S{0:02d}_{1}-{2}.avi'.format(sq_id, c, mtsc_meth)
                    if not os.path.isfile(video_name):
                        print("Creating video from MTSC estimated tracks...")
                        make_video_from_tracker(mtsc_tracker, cam, video_name, width=width, height=height)
                    else:
                        print("A video with the same name already exists, continuing w/o overwritting it...")

                if hasGT:
                    acc_mtsc = mtsc_tracker.compute_mot_metrics(gt_tracker, mtsc_tracker=True)
                    idf1_mtsc = print_mot_metrics(acc_mtsc)
                    idf1_cam[c] = idf1_mtsc

        # Compute mean
        if hasGT:
            sum = 0
            for cam_id, idf1 in idf1_cam.items():
                sum += idf1
            avg_idf1_seq = sum / len(idf1_cam.items())
            avg_idf1.append(avg_idf1_seq)

    return np.mean(avg_idf1)  # return mean across training sequences


if __name__ == '__main__':
    # test_seqs = [3, 1, 4]  # only S03 for now
    # test_seqs = [3]  # only S03 for now
    valid_seqs = [3]  #, 4]  # training sequences  avoid seq4 as it contains 25 sequences! (too time-consuming)
    test_seqs = [4]
    #  track_methods = ['RegionOverlap', 'Kalman']
    track_methods = 'RegionOverlap'  # 'RegionOverlap'
    # detect_methods = ['ssd512', 'yolo3', 'mask_rcnn']    
    detect_method = 'ssd512'  # 'retinanet'
    make_vid_track = True
    make_vid_gt = False
    make_vid_unfiltered = False
    plot = False
    valid = False
    test = True
    park_thresh = 12.5
    min_confidence = .2  # selected after testing coarse grid: 0.2, 0.4, 0.5, 0.7, 0.8
    # confidences = [ 0.5, 0.7, 0.8]
    # All tested parked threshs (recorded, at least)
    # Note: non-equally spaced because for some threshs the metrics fail due to all nan's
    if valid:
        parked_threshs = [1.5, 3.5, 6, 8, 10, 12.5, 15, 17.5, 20, 22.5, 25]
        # parked_threshs = np.linspace(1.5, 25, 11)
        # Output results mtx
        # For each threshold and detect method:
        idf1_metrics = np.zeros([len(parked_threshs)])  #, len(detect_methods)])
    else:
        parked_thresh = park_thresh

    # Validation: find best threshold. Test each tracking method separately.
    # idw = 0
    # for detect_method in detect_methods:
    if valid or (valid and test):
        idh = 0
        for parked_thresh in parked_threshs:
            print("Running tracking for tracking method '{0}' and detection method '{1}'".format(
                track_methods, detect_method))
            print("Confidence thresh.: {0:.2f} and parked_thresh.: {1:.2f}".format(min_confidence, parked_thresh))
            aux_idf1 = MultiTrackSingleCamera(tested_seqs=valid_seqs, track_meth=track_methods, detect_meth=detect_method,
                                              make_video_track=make_vid_track, make_video_gt=make_vid_gt,
                                              make_video_unfiltered=make_vid_unfiltered,
                                              min_conf=min_confidence, parked_threshold=parked_thresh)
            idf1_metrics[idh] = aux_idf1
            # idf1_metrics[idh, idw] = aux_idf1
            idh += 1
        # idw += 1

        print("Printing idf1 metrics for validation tests...")
        print(idf1_metrics)

        best_idx = np.argmax(idf1_metrics)
        best_park_thresh = parked_threshs[best_idx]

        if valid and test:
            # Test for one set of parameters (all cameras of one or more sequences)
            idf1_test_metrics = MultiTrackSingleCamera(tested_seqs=test_seqs, track_meth=track_methods,
                                                       detect_meth=detect_method, make_video_track=True,
                                                       make_video_gt=make_vid_gt, make_video_unfiltered=True,
                                                       min_conf=min_confidence, parked_threshold=best_park_thresh)

            print("Printing idf1 test metrics for run tests")
            print(idf1_test_metrics)

    elif not valid and test:
        idf1_metrics = MultiTrackSingleCamera(tested_seqs=test_seqs, track_meth=track_methods,
                                              detect_meth=detect_method, make_video_track=make_vid_track,
                                              make_video_gt=make_vid_gt, make_video_unfiltered=make_vid_unfiltered,
                                              min_conf=min_confidence, parked_threshold=parked_thresh)
    else:  # default values
        idf1_metrics = MultiTrackSingleCamera()

    print("Printing idf1 test metrics for run tests")
    print(idf1_metrics)

    # Read mtsc result, filter it and evaluate it
    # mtsc_method = 'tc_ssd512'
    # idf1_metrics = MultiTrackSingleCamera(mtsc_meth=mtsc_method, hasGT=True, make_video_mtsc=True,
    #                                       parked_threshold=park_thresh, min_conf=min_confidence)

    # Plotting validation figure
    # if plot:
    #     plt.figure(0)
    #     plt.plot(parked_threshs, idf1_metrics[:, 0], 'r', label=detect_methods[0])
    #     plt.plot(parked_threshs, idf1_metrics[:, 1], 'g', label=detect_methods[1])
    #     plt.plot(parked_threshs, idf1_metrics[:, 2], 'b', label=detect_methods[2])
    #
    #     max_maskrcnn_idx = np.argmax(idf1_metrics[:, 2])
    #     max_maskrcnn = np.max(idf1_metrics[:, 2])
    #     max_yolo3_idx = np.argmax(idf1_metrics[:, 1])
    #     max_yolo3 = np.max(idf1_metrics[:, 1])
    #     max_ssd_idx = np.argmax(idf1_metrics[:, 0])
    #     max_ssd = np.max(idf1_metrics[:, 0])
    #
    #     plt.plot(parked_threshs[max_maskrcnn_idx], max_maskrcnn, 'bo', markerfacecolor='w', markeredgewidth=2)
    #     plt.plot(parked_threshs[max_yolo3_idx], max_yolo3, 'go', markerfacecolor='w', markeredgewidth=2)
    #     plt.plot(parked_threshs[max_ssd_idx], max_ssd, 'ro', markerfacecolor='w', markeredgewidth=2)
    #
    #     plt.axvline(x=parked_threshs[max_maskrcnn_idx], color='r', linestyle='--', linewidth=1, alpha=0.75)
    #     plt.axvline(x=parked_threshs[max_yolo3_idx], color='g', linestyle='--', linewidth=1, alpha=0.75)
    #     plt.axvline(x=parked_threshs[max_ssd_idx], color='b', linestyle='--', linewidth=1, alpha=0.75)
    #
    #     plt.grid(True)
    #     plt.ylabel('Average IDF1 (S01 cameras)')
    #     plt.xlabel('Distance threshold (% of width and height)')
    #
    #
