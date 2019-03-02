from evaluation import IoU
from utils.detectionExtractorGT import detectionExtractorGT
from utils.annotationsParser import annotationsParser
import os
import cv2
import random


def randomizeDetection(BBOX, scale):

    randomBBOX = BBOX
    randomBBOX[0] = randomBBOX[0] + (random.random()-0.5)*scale
    randomBBOX[1] = randomBBOX[1] + (random.random()-0.5)*scale
    randomBBOX[2] = randomBBOX[2] + (random.random()-0.5)*scale
    randomBBOX[3] = randomBBOX[3] + (random.random()-0.5)*scale
    return randomBBOX


if __name__ == '__main__':

    IoU.testIoU()

    # Read GT from dataset
    print(os.getcwd())
    gtExtractor = annotationsParser(os.getcwd()+'/datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml')
    for i in range(len(gtExtractor.gt)):
        gtBBOX = gtExtractor.getGTBoundingBox(i)
        detection = randomizeDetection(gtExtractor.getGTBoundingBox(i), 10)
        frame = gtExtractor.getGTFrame(i)

        frame_path = 'image-{:07d}.png'.format(i+1)
        frame_path = os.getcwd()+'/datasets/AICity_data/train/S03/c010/frames/'+frame_path

        # load the image
        image = cv2.imread(frame_path)

        # draw the ground-truth bounding box along with the predicted
        # bounding box
        cv2.rectangle(image, (int(gtBBOX[0]),int(gtBBOX[1])),
                      (int(gtBBOX[2]),int(gtBBOX[3])), (0, 255, 0), 2)
        cv2.rectangle(image, (int(detection[0]),int(detection[1])),
                      (int(detection[2]),int(detection[3])), (0, 0, 255), 2)

        # compute the intersection over union and display it
        iou = IoU.bb_intersection_over_union(gtBBOX, detection)
        cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("{}: {:.4f}".format(frame_path, iou))

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)