from pathlib import Path
from typing import Tuple

import PIL.Image
import numpy as np
import xmltodict

from paths import AICITY_DIR


class AICityDataset(object):
    """ AICity dataset containing manually annotated labels """

    def __init__(self, root: Path):
        self.root = root
        images_dir = root.joinpath('frames')
        labels_path = root.joinpath('Anotation_40secs_AICITY_S03_C010.xml')

        self.images_paths = sorted(images_dir.glob('*.png'))
        self.labels = self._load_gt(labels_path)

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

    def _load_gt(self, path) -> np.ndarray:
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
