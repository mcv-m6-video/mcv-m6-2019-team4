from pathlib import Path
from typing import Tuple

import PIL.Image
import numpy as np

from paths import AICITY_DIR


class AicityTxtDataset(object):
    def __init__(self, root: Path):
        self.root = root
        images_dir = root.joinpath('frames')
        labels_path = root.joinpath('gt', 'gt.txt')

        self.images_paths = sorted(images_dir.glob('*.png'))
        self.labels = self._load_gt(labels_path)

    def __getitem__(self, idx) -> Tuple[PIL.Image.Image, Tuple[int, ...]]:
        label = self.labels[idx]
        frame_id = label[0]

        image_path = self.images_paths[frame_id]
        image = PIL.Image.open(image_path, mode='r')
        label = self._ground_truth_to_bounding_box(label)

        return image, label

    def __len__(self):
        return self.labels.shape[0]

    def _load_gt(self, path) -> np.ndarray:
        with path.open('r') as file:
            ground_truth = [line.strip().split(',') for line in
                            file.readlines()]
            return np.array(ground_truth, dtype=np.int)

    def _ground_truth_to_bounding_box(self, gt: np.array) -> Tuple[int, ...]:
        """" Extracts points to apply PIL.Image.crop()

        Converts to a (left, upper, right, and lower) tuple
        """
        return (
            gt[2],
            gt[3],
            gt[2] + gt[4],
            gt[2] + gt[5],
        )


if __name__ == '__main__':
    dataset = AicityTxtDataset(AICITY_DIR)
    image, label = dataset[0]

    crop = image.crop(label)
    image.show()
    crop.show()
