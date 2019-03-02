
class detectionExtractorGT():

    def __init__(self, filePath):
        self.file = filePath
        self.gt = []
        self.extractGT()

    def extractGT(self):
        self.gt = []
        with open(self.file) as fp:
            for line in fp:
                data = [float(elt.strip()) for elt in line.split(',')]
                self.gt.append(data)

        return

    def setFile (self, filePath):
        self.file = filePath


    def getGTFrame (self, i):
        gtElement = self.gt[i]
        return gtElement[0]


    def getGTID (self,i ):
        gtElement = self.gt[i]
        return gtElement[1]


    def getGTBoundingBox (self, i):
        #BBformat [xA,yA, xB, yB]
        gtElement = self.gt[i]
        BBox = [gtElement[2], gtElement[3], gtElement[2]+gtElement[4], gtElement[3]+gtElement[5]]
        return BBox


