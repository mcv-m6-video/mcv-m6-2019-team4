class detectionExtractorGT():

    def __init__(self, filePath):
        self.file = filePath
        # format detection & GT  [frame, ID, xTopLeft, yTopLeft, xBottomRight, yBottomRight, class]
        self.gt = []
        self.nFrames = 0
        self.extractGT()

    def extractGT(self):
        self.gt = []
        with open(self.file) as fp:
            for line in fp:
                data = [float(elt.strip()) for elt in line.split(',')]
                # format data: [frame, ID, left, top, width, height, 1, -1, -1, -1]
                data = [data[0], data[1], data[2], data[3], data[2] + data[4],
                        data[3] + data[5], data[6]]
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
