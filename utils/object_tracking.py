
class TrackedObject:
    # a object with its track
    # a track is a dictionary of frame_id and roi of the tracked
    #   object on the frame

    def __init__(self, id):
        self.objectId = id
        self.track = {}

    def add_frame_roi(self, frame_id, roi):
        roi.objectId = self.objectId
        r = ROI(roi.xTopLeft, roi.yTopLeft, roi.xBottomRight, roi.yBottomRight, self.objectId)
        self.track[frame_id] = r

    def get_track(self):
        return self.track

class ROI:
    # region of interest of a detected object
    # may or may not have an object id associated

    def __init__(self, xTopLeft, yTopLeft, xBottomRight, yBottomRight, objectId = -1):
        self.xTopLeft = xTopLeft
        self.yTopLeft = yTopLeft
        self.xBottomRight = xBottomRight
        self.yBottomRight = yBottomRight
        self.objectId = objectId

    def __str__(self):
        return '{} {} {} {}'.format(self.xTopLeft, self.yTopLeft, self.xBottomRight, self.yBottomRight)

    def overlap(self, otherROI):
        return 15

class Frame:
    # collection of ROIs detected on an image frame

    def __init__(self, id):
        self.number = id
        self.ROIs = []

    def get_id(self):
        return  self.number

    def add_ROI(self, r):
        self.ROIs.append(r)

    def get_ROIs(self):
        return self.ROIs

class ObjectTracker:
    # class that tracks objects along a list of frames

    def __init__(self, method):
        self.method = method
        self.firstFrame = True
        self.trackedObjects = {}
        self.trackedFrames = {}
        self.lastObjectId = 0

    def process_frame(self, frame : Frame):

        if self.firstFrame:
            for r in frame.get_ROIs():
                id = self.lastObjectId + 1
                obj = TrackedObject(id)
                obj.add_frame_roi(frame.get_id(), r)
                self.trackedObjects[id] = obj
                self.lastObjectId = id

            self.firstFrame = False

        else:

            if self.method == "RegionOverlap":
                tracked_frame = self._process_frame_overlap(frame)
            else:
                tracked_frame = self._process_frame_kalman(frame)

        self.trackedFrames[frame.get_id()] = tracked_frame

    # gets a frame whose rois have no assigned object id and
    # returns a copy of the frame with assigned oject ids if
    # possible (-1 otherwise)
    def _process_frame_overlap(self, untracked_frame: Frame):
        last_frame = self.trackedFrames[untracked_frame.get_id() - 1]
        pass

    # gets a frame whose rois have no assigned object id and
    # returns a copy of the frame with assigned oject ids if
    # possible (-1 otherwise)
    def _process_frame_kalman(self, untracked_frame: Frame):
        
        pass

    def print_objects(self):
        for obj in self.trackedObjects:
            print('Object {}'.format(obj))
            for frame, roi in self.trackedObjects[obj].get_track().items():
                print('\tin frame {} at position {}'.format(frame, roi))


