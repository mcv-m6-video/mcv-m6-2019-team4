def detectionExtractGT(file):
    gt = []
    with open(file) as fp:
        for line in fp:
            data = [elt.strip() for elt in line.split(',')]
            gt.append(data)

    return gt
