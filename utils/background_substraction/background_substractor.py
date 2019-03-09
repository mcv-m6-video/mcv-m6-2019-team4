import cv2
import matplotlib.pyplot as plt

from evaluation import intersection_over_union
from utils.annotation_parser import annotationsParser
from utils.detection_gt_extractor import detectionExtractorGT
from paths import AICITY_DIR
import numpy as np

MIN_AREA =100
MAX_AREA =  500000


class background_substractor():

    def __init__(self, method):

        self.method = method
        if method == 'MOG2':
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        elif method == 'LSBP':
            self.backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()
        elif method == 'GMG':
            self.backSub = cv2.bgsegm.createBackgroundSubtractorGMG()
        elif method == 'KNN':
            self.backSub = cv2.createBackgroundSubtractorKNN()
        elif method == 'GSOC':
            self.backSub = cv2.createBackgroundSubtractorGSOC()
        elif method == 'CNT':
            self.backSub = cv2.createBackgroundSubtractorCNT()
        else:
            self.backSub = cv2.bgsegm.createBackgroundSubtractorMOG()



    def apply(self, image):
        return(self.backSub.apply(image))


    def getBackgroungImage(self, image):
        return(self.backSub.getBackgroundImage())



def process_image (image):

    ret, image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def get_detections(image):

    detections =[]

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    for i in range(len(stats)):
        if stats[i][4] > MIN_AREA and stats[i][4] < MAX_AREA:
            detections.append([stats[i][0], stats[i][1], stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]])

    detections = non_max_suppression_fast(np.array(detections),0)

    return detections


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def add_detections_gt (image, frame_detections, gtExtractor, frame):
        #print detections
        for detection in frame_detections:
            # format detection & GT  [frame, ID, xTopLeft, yTopLeft, xBottomRight, yBottomRight, class]
            cv2.rectangle(
                image,
                (int(detection[0]), int(detection[1])),
                (int(detection[2]), int(detection[3])),
                (0, 0, 255),
                2
            )
        #print gt
        for gtBBOX in gtExtractor.gt:
            if gtBBOX[0] == frame:
                cv2.rectangle(
                    image,
                    (int(gtBBOX[2]), int(gtBBOX[3])),
                    (int(gtBBOX[4]), int(gtBBOX[5])),
                    (0, 255, 0),
                    2
                )
        return image

def compute_iou_precision_recall(gt_detections, detections, method, color_conversion):
    IoUvsFrames = []
    plt.plot(IoUvsFrames)
    plt.ylabel('IoU')
    plt.xlabel('Frames')
    plt.show()
    plt.savefig('figure_' + method + '_' + str(color_conversion) +'.png')


def analyze_sequence(method, color_conversion):

    # Read GT from dataset
    gtExtractor = annotationsParser(AICITY_DIR.joinpath('m6-full_annotation.xml'))

    bckg_subs = background_substractor(method)


    video_name = 'video_' + method + '_' + str(color_conversion) +'.avi'
    video = cv2.VideoWriter(video_name,
                            cv2.VideoWriter_fourcc('M', 'P', '4', 'S'), 10,
                            (1920, 1080))

    detections = []

    #for i in range(gtExtractor.getGTNFrames()):
    for i in range(200):
        # load the image
        frame_path = AICITY_DIR.joinpath('frames',
                                         'image-{:04d}.png'.format(i + 1))
        image = cv2.imread(str(frame_path))

        if color_conversion != None:
            image = cv2.cvtColor(image, color_conversion)

        image = bckg_subs.apply(image)
        image = process_image(image)

        frame_detections = get_detections(image)

        for detection in frame_detections:
            detections.append([i, 0, detection[0], detection[1], detection[2], detection[3], 1])

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = add_detections_gt(image, frame_detections, gtExtractor, i)
        #show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        video.write(image)

    compute_iou_precision_recall(gtExtractor.gt, detections,method, color_conversion)




def find_detections(gtExtractor, detections,threshold, IoUvsFrames, frame_path):


    IoUvsFrame = []

    # Get GT BBOX

    gt = []
    for j in range(len(gtExtractor.gt)):
        if gtExtractor.getGTFrame(j) == j:
            gtBBOX = gtExtractor.getGTBoundingBox(j)
            gt.append(gtBBOX)

    BBoxesDetected = []

    for x in range(len(gt)):
        gtBBOX = gt[x]
        detection = []
        maxIoU = 0
        BBoxDetected = -1

        for y in range(len(detections)):

            iou = intersection_over_union.bb_intersection_over_union(gtBBOX, detections[y])
            if iou >= maxIoU:
                maxIoU = iou
                detection = detections[y]
                BBoxDetected = y

        if maxIoU > threshold:
            TP = TP + 1
            BBoxesDetected.append(BBoxDetected)
        else:
            FN = FN + 1


        print("{}: {:.4f}".format(frame_path, maxIoU))

    if not IoUvsFrame:
        IoUvsFrame = [0]

    IoUvsFrames.append(sum(IoUvsFrame) / len(IoUvsFrame))
    for y in range(len(detections)):
        if not BBoxesDetected.__contains__(y):
            FP = FP + 1
            detection = detections[y]


def run():

    methods =[
        'MOG2',
        'LSBP',
        'GMG',
        'GSOC',
        'CNT',
        'MOG'
    ]


    color_conversions =[
        cv2.COLOR_BGR2HSV,
        cv2.COLOR_BGR2Luv,
        cv2.COLOR_BGR2Lab,
        cv2.COLOR_BGR2YCrCb,
        cv2.COLOR_BGR2HLS,
        None
    ]

    #analyze_sequence('MOG2', None)
    analyze_sequence('MOG2', cv2.COLOR_BGR2Lab)

    #for method in methods:
     #   for color_conversion in color_conversions:
     #   analyze_sequence(method,color_conversion)


if __name__ == '__main__':
    verbose = False
    run()