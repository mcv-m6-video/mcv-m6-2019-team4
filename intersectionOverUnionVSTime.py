from evaluation import IoU
from utils.annotationsParser import annotationsParser
from utils.randomDetector import randomDetector
from utils.detectionExtractorGT import detectionExtractorGT
from utils.ROIRefinement import  ROIRefinement
import os
import cv2
import matplotlib.pyplot as plt
from time import sleep



if __name__ == '__main__':

    IoU.testIoU()

    # Read GT from dataset
    print(os.getcwd())
    #gtExtractor = annotationsParser(os.getcwd()+'/datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml')
    gtExtractor = annotationsParser(os.getcwd()+'/datasets/AICity_data/train/S03/c010/AICITY_team4.xml')
    #gtExtractor = detectionExtractorGT(os.getcwd() + '/datasets/AICity_data/train/S03/c010/gt/gt.txt')

    #innitialize random detector
    randomNoiseScale = 5
    additionDeletionProbability = 0.01
    #randomDetector = randomDetector(randomNoiseScale,additionDeletionProbability)

    #detector = detectionExtractorGT(os.getcwd() + '/datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    #detector = detectionExtractorGT(os.getcwd() + '/datasets/AICity_data/train/S03/c010/det/det_ssd512.txt')
    detector = detectionExtractorGT(os.getcwd() + '/datasets/AICity_data/train/S03/c010/det/det_yolo3.txt')

    TP = 0
    FN = 0
    FP = 0
    threshold = 0.5

    ROIPath = os.getcwd() + '/datasets/AICity_data/train/S03/c010/roi.jpg'
    refinement = ROIRefinement(ROIPath, 0.2)

    IoUvsFrames = []

    #video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920, 1080))
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M','P','4','2'), 10, (1920, 1080))

    for i in range(gtExtractor.getGTNFrames()):

        # load the image
        frame_path = 'image-{:07d}.png'.format(i + 1)
        frame_path = os.getcwd() + '/datasets/AICity_data/train/S03/c010/frames/' + frame_path
        image = cv2.imread(frame_path)
        IoUvsFrame = []

        #Get GT BBOX

        gt = []
        for j in range(len(gtExtractor.gt)):
            if gtExtractor.getGTFrame(j) == i:
                gtBBOX = gtExtractor.getGTBoundingBox(j)
                gt.append(gtBBOX)

        #gt = refinement.refineBBOX(gt)

        #Get detection BBOX
        detections = []
        for j in range(len(detector.gt)):
            if detector.getGTFrame(j) == i:
                detBBOX = detector.getGTBoundingBox(j)
                detections.append(detBBOX)
        #detections = refinement.refineBBOX(detections)

        BBoxesDetected = []


        for x in range(len(gt)):
            gtBBOX = gt[x]
            detection = []
            maxIoU = 0
            BBoxDetected = -1

            for y in range(len(detections)):

                iou = IoU.bb_intersection_over_union(gtBBOX, detections[y])
                if iou >= maxIoU:
                    maxIoU = iou
                    detection = detections[y]
                    BBoxDetected = y

            if maxIoU > threshold:
                TP = TP + 1
                BBoxesDetected.append(BBoxDetected)
            else:
                FN = FN + 1

            # draw the ground-truth bounding box along with the predicted
            # bounding box
            if detection != []:
                cv2.rectangle(image, (int(detection[0]), int(detection[1])),
                            (int(detection[2]), int(detection[3])), (0, 0, 255), 2)
            cv2.rectangle(image, (int(gtBBOX[0]), int(gtBBOX[1])),
                          (int(gtBBOX[2]), int(gtBBOX[3])), (0, 255, 0), 2)


            # compute the intersection over union and display it
            IoUvsFrame.append(maxIoU)
            cv2.putText(image, "IoU: {:.4f}".format(maxIoU), (10, 30+20*x),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("{}: {:.4f}".format(frame_path, maxIoU))

        if IoUvsFrame == []:
            IoUvsFrame = [0]

        IoUvsFrames.append(sum(IoUvsFrame)/len(IoUvsFrame))
        for y in range(len(detections)):
            if not BBoxesDetected.__contains__(y):
                FP = FP + 1
                detection = detections[y]
                cv2.rectangle(image, (int(detection[0]), int(detection[1])),
                            (int(detection[2]), int(detection[3])), (0, 0, 255), 2)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        video.write(image)

    plt.plot(IoUvsFrames)
    plt.show()
