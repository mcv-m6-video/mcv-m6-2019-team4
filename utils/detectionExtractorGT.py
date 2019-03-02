def extractGT(file):
    gt = []
    with open(file) as fp:
        for line in fp:
            data = [float(elt.strip()) for elt in line.split(',')]
            gt.append(data)

    return gt


def getGTFrame (gtElement):
    return gtElement[0]


def getGTID (gtElement):
    return gtElement[1]


def getGTBoundingBox (gtElement):
    #BBformat [xA,yA, xB, yB]
    BBox = [gtElement[2], gtElement[3], gtElement[2]+gtElement[4], gtElement[3]+gtElement[5]]
    return BBox


