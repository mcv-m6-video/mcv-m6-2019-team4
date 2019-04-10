from paths import AICITY_MC_ROOT
import glob
import os
import cv2

class AICityMTMCSequence:
    def __init__(self, seq_root_dir):
        self.root = seq_root_dir
        cams = glob.glob(os.path.join(self.root, '*'))
        self.cams = {os.path.basename(c): AICityMTMCCamera(c) for c in cams}

    def getName(self):
        return os.path.basename(os.path.normpath(self.root))

    def __str__(self):
        return self.root

    def getCameras(self):
        return self.cams.keys()

    def getCamera(self, camname):
        #camname = "c{:03d}".format(number)
        return self.cams[camname]

class AICityMTMCCamera:
    def __init__(self, cam_root_dir):
        self.root = cam_root_dir
        self.video_path = os.path.join(self.root, 'vdo.avi')

    def getName(self):
        return os.path.basename(os.path.normpath(self.root))

    def getVideoPath(self):
        return self.video_path

    def openVideo(self):
        self.video = cv2.VideoCapture(self.getVideoPath())

    def videoIsOpened(self):
        return self.video.isOpened()

    def getNextFrame(self):
        if (self.video.isOpened()):
            return self.video.read()
        else:
            return (False, None)

    def closeVideo(self):
        self.video.release()

    def getDetectionFile(self, detection_method):
        det = "det_{}.txt".format(detection_method)
        return os.path.join(self.root, 'det', det)

    def getGTFile(self):
        return os.path.join(self.root, 'gt', 'gt.txt')

    def getMTSCtracks(self, mtsc_method):
        mtsc_file = "mtsc_{}.txt".format(mtsc_method)
        return os.path.join(self.root, 'mtsc', mtsc_file)


class AICityMTMCDataset:

    def __init__(self, root_dir = AICITY_MC_ROOT):
        self.root_dir = root_dir

        seqs = glob.glob(os.path.join(self.root_dir, 'train', '*'))
        self.train_seqs = {os.path.basename(s) : AICityMTMCSequence(s) for s in seqs}

    def getTrainSequences(self):
        return self.train_seqs.keys()

    def getTrainSeq(self, number):
        seqname = "S{:02d}".format(number)
        return self.train_seqs[seqname]

