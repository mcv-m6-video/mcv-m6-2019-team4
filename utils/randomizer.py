import random


def randomize_detections(probability, randomScale, initializer):
    """ Randomize a list of bpunding box as detections
    Args:
        initializer: initial list of boundin boxes to randomize

    """
    detections = []
    for element in initializer:

        if random.random() > probability:
            detections.append(randomize_bb(element[:], randomScale))

        if random.random() < probability:
            detections.append(random_bb())
    return detections


def randomize_bb(bb, randomScale):
    bb = bb[:]
    bb[0] = bb[0] + (random.random() - 0.5) * randomScale
    bb[1] = bb[1] + (random.random() - 0.5) * randomScale
    bb[2] = bb[2] + (random.random() - 0.5) * randomScale
    bb[3] = bb[3] + (random.random() - 0.5) * randomScale
    return bb


def random_bb():
    bb = []

    for i in range(2):
        bb.append((random.random()) * 1920)
        bb.append((random.random()) * 1080)

    return bb
