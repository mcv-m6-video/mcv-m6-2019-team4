from pathlib import Path

import numpy as np

from paths import AICITY_DIR


def detections_to_bounding_boxes(detections: np.ndarray):
    """ Convert detections to bounding boxes.

    Transforms from
    ```
    [frame, left, top, width, height, score]
    ```
    to
    ```
    [frame-1, left, upper, right, lower, score]

    Frame number is modified to match annotations' XML as they start at 0
    ```
     """
    return np.stack(
        (
            detections[:, 0] - np.ones(detections.shape[0]),
            detections[:, 1],
            detections[:, 2],
            detections[:, 1] + detections[:, 3],
            detections[:, 2] + detections[:, 4],
            detections[:, 5],
        ),
        axis=1
    )


def load_detections(path: Path):
    """ Load detections given a path.

    Detections are formatted as

    ```
    [frame, left, top, width, height, score]
    ```

    where frame starts at 1.0 and all elements are float types
    """
    with path.open('r') as file:
        data = [
            line.strip().split(',')
            for line in file.readlines()
        ]

    data = np.array(data, dtype=float)
    # Delete unused columns (they are filled with '-1')
    return np.delete(data, [1, 7, 8, 9], axis=1)


def load_bounding_boxes(path: Path):
    """ Load bounding boxes given a path.

    Detections are formatted as

    ```
    [frame, left, upper, right, lower, score]
    ```

    where frame starts at 0.0 and all elements are float types
    """
    return detections_to_bounding_boxes(load_detections(path))


if __name__ == '__main__':
    filepath = AICITY_DIR.joinpath('det', 'det_mask_rcnn.txt')
    detections = load_detections(filepath)
    print(detections.shape)
    bb = detections_to_bounding_boxes(detections)
    print(bb.shape)

    # print(detections[0,:])
    # print(bb[0,:])

    filepath = AICITY_DIR.joinpath('det', 'det_ssd512.txt')
    detections = load_detections(filepath)
    print(detections.shape)

    filepath = AICITY_DIR.joinpath('det', 'det_yolo3.txt')
    detections = load_detections(filepath)
    print(detections.shape)
