from evaluation import IoU
from utils.annotationsParser import annotationsParser
from utils.randomDetector import randomDetector
from utils.detectionExtractorGT import detectionExtractorGT
import os
import cv2



if __name__ == '__main__':

    IoU.testIoU()

    # Read GT from dataset
    print(os.getcwd())
    #gtExtractor = annotationsParser(os.getcwd()+'/datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml')
    gtExtractor = detectionExtractorGT(os.getcwd() + '/datasets/AICity_data/train/S03/c010/gt/gt.txt')

    #innitialize random detector
    randomNoiseScale = 5
    additionDeletionProbability = 0.01
    randomDetector = randomDetector(5,0.01)

    for i in range(gtExtractor.getGTNFrames()):

        # load the image
        frame_path = 'image-{:07d}.png'.format(i+1)
        frame_path = os.getcwd()+'/datasets/AICity_data/train/S03/c010/frames/'+frame_path
        image = cv2.imread(frame_path)

        #Get GT BBOX

        gt = []
        for j in range(len(gtExtractor.gt)):
            if gtExtractor.getGTFrame(j) == i:
                gtBBOX = gtExtractor.getGTBoundingBox(j)
                gt.append(gtBBOX)



        #Get detection BBOX
        detections = randomDetector.randomizeDetections(gt[:])

        for x in range(len(gt)):
            for y in range(len(detections)):
                image = cv2.imread(frame_path)
                detection = detections[y]
                gtBBOX = gt[x]
                # draw the ground-truth bounding box along with the predicted
                # bounding box
                cv2.rectangle(image, (int(detection[0]), int(detection[1])),
                              (int(detection[2]), int(detection[3])), (0, 0, 255), 2)
                cv2.rectangle(image, (int(gtBBOX[0]), int(gtBBOX[1])),
                              (int(gtBBOX[2]), int(gtBBOX[3])), (0, 255, 0), 2)


                # compute the intersection over union and display it
                iou = IoU.bb_intersection_over_union(gtBBOX, detection)
                cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print("{}: {:.4f}".format(frame_path, iou))

                # show the output image
                cv2.imshow("Image", image)
                cv2.waitKey(0)