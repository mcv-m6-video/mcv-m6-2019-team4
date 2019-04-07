import numpy as np

class detectionExtractorGT():

    def __init__(self, filePath, gtFormat="LTWH", confidence_th=.2):
        self.file = filePath
        # format detection & GT  [frame, ID, xTopLeft, yTopLeft, xBottomRight, yBottomRight, class]
        self.gt = []
        self.nFrames = 0
        self.gtFormat = gtFormat
        self.confidence_th = confidence_th
        self.extractGT()

    def extractGT(self):
        self.gt = []
        with open(self.file) as fp:
            for line in fp:
                data = [float(elt.strip()) for elt in line.split(',')]
                if self.gtFormat == "LTWH":
                    # format data: [frame, ID, left, top, width, height, 1, -1, -1, -1]
                    data = [data[0], int(data[1]), data[2], data[3], data[2] + data[4],
                            data[3] + data[5], data[6]]
                elif self.gtFormat == "TLBR":
                    # format data: [frame, ID, left, top, right, bottom, 1, -1, -1, -1]
                    data = [data[0], int(data[1]), data[2], data[3], data[4],
                            data[5], data[6]]

                if data[-1] >= self.confidence_th:
                    self.gt.append(data)

        for gtElement in self.gt:
            if int(gtElement[0]) > self.nFrames:
                self.nFrames = int(gtElement[0])
        return

    def setFile(self, filePath):
        self.file = filePath

    def getGTFrame(self, i):
        gtElement = self.gt[i]
        return int(gtElement[0])

    def getGTNFrames(self):
        return self.nFrames

    def getGTID(self, i):
        gtElement = self.gt[i]
        return gtElement[1]

    def getGTBoundingBox(self, i):
        # BBformat [xA,yA, xB, yB]
        gtElement = self.gt[i]
        BBox = [gtElement[2], gtElement[3], gtElement[4], gtElement[5]]
        return BBox

    def getGTList(self):
        return self.gt

    def getAllFrame(self, i):
        return [f for f in self.gt if f[0] == i]

    def getFirstFrame(self):
        return int(np.min(np.array([x[0] for x in self.gt])))

    def getLastFrame(self):
        return int(np.max(np.array([x[0] for x in self.gt])))
