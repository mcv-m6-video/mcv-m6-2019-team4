import cv2
import matplotlib.pyplot as plt

from evaluation import intersection_over
from utils.annotation_parser import annotationsParser
from utils.detection_gt_extractor import detectionExtractorGT
from utils.roi_refiner import ROIRefiner
from paths import AICITY_DIR


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

    return image


def get_detections(image):

    detections =[]

    return detections

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
                            cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10,
                            (1920, 1080))

    detections = []

    for i in range(gtExtractor.getGTNFrames()):
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
            detections.append [i, 0, detection[0], detection[1], detection[2], detection[3], 1]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = add_detections_gt(image, frame_detections, gtExtractor, i)
        #show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        video.write(image)

    compute_iou_precision_recall(gtExtractor.gt, detections)




def find_detections(gtExtractor, detections,threshold, IoUvsFrames, frame_path):


    IoUvsFrame = []

    # Get GT BBOX

    gt = []
    for j in range(len(gtExtractor.gt)):
        if gtExtractor.getGTFrame(j) == i:
            gtBBOX = gtExtractor.getGTBoundingBox(j)
            gt.append(gtBBOX)

    BBoxesDetected = []

    for x in range(len(gt)):
        gtBBOX = gt[x]
        detection = []
        maxIoU = 0
        BBoxDetected = -1

        for y in range(len(detections)):

            iou = intersection_over.bb_intersection_over_union(gtBBOX, detections[y])
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