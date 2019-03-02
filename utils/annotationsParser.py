
import xmltodict


class annotationsParser():

    def __init__(self, filePath):
        self.file = filePath
        self.gt = []
        self.extractGT()

    def extractGT(self):
        with open(self.file) as fd:
            doc = xmltodict.parse(fd.read())
        self.gt = doc['annotations']['track']['box']

        return

    def setFile (self, filePath):
        self.file = filePath


    def getGTFrame (self, i):
        gtElement = self.gt[i]
        return gtElement['@frame']


    def getGTID (self,i ):
        gtElement = self.gt[i]
        return gtElement['@id']


    def getGTBoundingBox (self, i):
        #BBformat [xA,yA, xB, yB]
        gtElement = self.gt[i]
        BBox = [float(gtElement['@xtl']), float(gtElement['@ytl']), float(gtElement['@xbr']), float(gtElement['@ybr'])]
        return BBox