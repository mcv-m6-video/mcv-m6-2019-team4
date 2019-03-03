
class detectionExtractorGT():

    def __init__(self, filePath):
        self.file = filePath
        self.gt = []
        self.nFrames = 0
        self.extractGT()

    def extractGT(self):
        self.gt = []
        with open(self.file) as fp:
            for line in fp:
                data = [float(elt.strip()) for elt in line.split(',')]
                self.gt.append(data)

        for gtElement in self.gt:
            if int(gtElement[0]) > self.nFrames:
                self.nFrames = int(gtElement[0])
        return

    def setFile (self, filePath):
        self.file = filePath


    def getGTFrame (self, i):
        gtElement = self.gt[i]
        return int(gtElement[0])


    def getGTNFrames (self):
        return self.nFrames

    def getGTID (self,i ):
        gtElement = self.gt[i]
        return gtElement[1]


    def getGTBoundingBox (self, i):
        #BBformat [xA,yA, xB, yB]
        gtElement = self.gt[i]
        BBox = [gtElement[2], gtElement[3], gtElement[2]+gtElement[4], gtElement[3]+gtElement[5]]
        return BBox


