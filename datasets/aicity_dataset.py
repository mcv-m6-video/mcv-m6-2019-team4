from pathlib import Path
from typing import Tuple

import PIL.Image
import numpy as np
import xmltodict

from paths import AICITY_DIR, AICITY_ANNOTATIONS


class AICityDataset(object):
    """ AICity dataset containing manually annotated labels """

    def __init__(self, root: Path):
        self.root = root
        images_dir = root.joinpath('frames')
        labels_path = root.joinpath('Anotation_40secs_AICITY_S03_C010.xml')
        self.labels = self._load_gt(labels_path)

        # labels_path = AICITY_ANNOTATIONS

        self.images_paths = sorted(images_dir.glob('*.png'))
        self.labels = self._load_gt(labels_path)

        self.label_to_class = {'bicycle': 0, 'car': 1}

    def __getitem__(self, idx) -> Tuple[PIL.Image.Image, np.ndarray]:
        """ Returns a sample (image, label).

        The box is a 4-tuple defining the left, upper, right, and lower pixel
        coordinates.
        """
        label = self.labels[idx]
        image = PIL.Image.open(self.images_paths[idx], mode='r')

        return image, label

    def __len__(self):
        return self.labels.shape[0]

    def get_labels(self) -> np.ndarray:
        """ Read XML and parse data

        Output is formatted as:
        ```
        [frame, id, xTopLeft, yTopLeft, xBottomRight, yBottomRight, class]
        ```
        """
        with AICITY_ANNOTATIONS.open('r') as file:
            labels = xmltodict.parse(file.read())

        tracks = labels['annotations']['track']

        def tracks_to_bboxes(track):
            track_id = float(track['@id'])
            track_label = self.label_to_class.get(str(track['@label']))

            def box_to_coord(track):
                return (
                    float(track['@frame']),
                    float(track['@xtl']),
                    float(track['@ytl']),
                    float(track['@xbr']),
                    float(track['@ybr']),
                )

            coordinates = np.array(list(map(box_to_coord, track['box'])))
            bboxes = np.stack(
                (
                    coordinates[:, 0],
                    np.repeat(track_id, coordinates.shape[0]),
                    coordinates[:, 1],
                    coordinates[:, 2],
                    coordinates[:, 3],
                    coordinates[:, 4],
                    np.repeat(track_label, coordinates.shape[0]),
                ),
                axis=1
            )
            return bboxes

        bbs = map(tracks_to_bboxes, tracks)
        bbs = np.vstack(list(bbs))
        return bbs

    def _load_gt(self, path) -> np.ndarray:
        """ Deprecated as only returns bycicle's labels """
        with path.open('r') as file:
            labels = xmltodict.parse(file.read())
        annotations = labels['annotations']
        bycicle_bbs = annotations['track']['box']

        def anno_to_bboxes(anno):
            return (
                float(anno['@xtl']),
                float(anno['@ytl']),
                float(anno['@xbr']),
                float(anno['@ybr']),
            )

        bbs = map(anno_to_bboxes, bycicle_bbs)
        bbs = np.array(list(bbs))
        return bbs


if __name__ == '__main__':
    dataset = AICityDataset(AICITY_DIR)

    # Show first image and a crop of a detection
    image, label = dataset[0]
    crop = image.crop(label)
    image.show()
    crop.show()

    # Get extended labels
    bbs = dataset.get_labels()
    print(bbs.shape)
