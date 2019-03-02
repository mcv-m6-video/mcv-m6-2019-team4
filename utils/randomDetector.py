
import random

class randomDetector():

    def __init__(self, randomScale, probability):

        self.randomScale = randomScale
        self.probability = probability


    def randomizeDetections (self, initializer):
        detections = []
        for element in initializer:

            if random.random() > self.probability:
                detections.append(self.randomizeBBOX(element[:]))

            if random.random() < self.probability:
                detections.append(self.randomBBOX())
        return detections



    def randomizeBBOX(self, BBOX):

        BBOX = BBOX[:]
        BBOX[0] = BBOX[0] + (random.random() - 0.5) * self.randomScale
        BBOX[1] = BBOX[1] + (random.random() - 0.5) * self.randomScale
        BBOX[2] = BBOX[2] + (random.random() - 0.5) * self.randomScale
        BBOX[3] = BBOX[3] + (random.random() - 0.5) * self.randomScale
        return BBOX

    def randomBBOX(self):

        BBOX = []

        for i in range(2):
            BBOX.append ((random.random()) * 1920)
            BBOX.append((random.random()) *  1080)

        return BBOX
