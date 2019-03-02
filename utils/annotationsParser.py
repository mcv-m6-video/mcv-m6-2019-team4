
import xmltodict
import random

class annotationsParser():

    def __init__(self, filePath):
        self.file = filePath
        self.gt = []
        self.nFrames = 0
        self.extractGT()


    def extractGT(self):
        self.gt = []
        with open(self.file) as fd:
            doc = xmltodict.parse(fd.read())
        self.gt = doc['annotations']['track']['box']

        for gtElement in self.gt:
            if int(gtElement['@frame']) > self.nFrames:
                self.nFrames = int(gtElement['@frame'])
        return

    def setFile (self, filePath):
        self.file = filePath


    def getGTFrame (self, i):
        gtElement = self.gt[i]
        return int(gtElement['@frame'])


    def getGTNFrames (self):
        return self.nFrames


    def getGTID (self,i ):
        gtElement = self.gt[i]
        return gtElement['@id']


    def getGTBoundingBox (self, i):
        #BBformat [xA,yA, xB, yB]
        gtElement = self.gt[i]
        BBox = [float(gtElement['@xtl']), float(gtElement['@ytl']), float(gtElement['@xbr']), float(gtElement['@ybr'])]
        return BBox
