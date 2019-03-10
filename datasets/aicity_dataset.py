from cv2 import cv2
from pathlib import Path
from typing import Tuple

import PIL.Image
import numpy as np
import xmltodict

from paths import AICITY_DIR, AICITY_ANNOTATIONS


class AICityDataset(object):
    """ AICity dataset containing manually annotated labels """

    def __init__(self, root: Path, annotations_path: Path):
        self.root = root
        self.labels = self._load_gt(annotations_path)
        self.images_paths = sorted(root.joinpath('frames').glob('*.png'))
        self.label_to_class = {'bicycle': 0, 'car': 1}

    def __getitem__(self, idx) -> Tuple[PIL.Image.Image, np.ndarray]:
        """ Returns a sample (image, label).

        The image is a PIL.Image (RGB)

        Label is a 4-tuple defined as
        `[frame, track_id, (coord), track_label]` where coord is a
        4-tuple defined as two corners: top left, bottom right:
        `[x_tl, y_tl, x_br, y_br]` pixel
        coordinates.
        """
        # Reads image using opencv
        # image = cv2.imread(str(self.images_paths[idx]))

        # Reads image using PIL
        return (
            PIL.Image.open(self.images_paths[idx], mode='r'),
            self.labels[idx]
        )

    def __len__(self):
        return self.labels.shape[0]

    def get_labels(self) -> np.ndarray:
        """ Returns all labels

        A label is a 4-tuple defined as
        `[frame, track_id, (coord), track_label]` where coord is a
        4-tuple defined as two corners: top left, bottom right:
        `[x_tl, y_tl, x_br, y_br]` pixel
        coordinates.
        """
        return self.labels

    def _load_gt(self, path) -> np.ndarray:
        """ Return all car bounding boxes """
        with path.open('r') as file:
            labels = xmltodict.parse(file.read())

        annotations = labels['annotations']
        tracks = annotations['track']
        print(f'Tracks found: {len(list(tracks))}')

        def is_a_car_track(track):
            return track['@label'] == 'car'

        def is_visible(box):
            return box['@occluded'] == '0'

        def remove_occluded_bb(track):
            track['box'] = list(filter(is_visible, track['box']))
            return track

        def anno_to_bboxes(anno):
            return (
                float(anno['@xtl']),
                float(anno['@ytl']),
                float(anno['@xbr']),
                float(anno['@ybr']),
            )

        car_tracks = filter(is_a_car_track, tracks)
        car_tracks_clean = map(remove_occluded_bb, car_tracks)

        # car_annotations = [track['box'] for track in car_tracks_clean]
        #
        # # Flatten list of lists
        # car_annotations = list(itertools.chain.from_iterable(car_annotations))
        # car_bboxes = np.array(list(map(anno_to_bboxes, car_annotations)))

        label_to_class = {'bicycle': 0, 'car': 1}

        def tracks_to_bboxes(track):
            track_id = float(track['@id'])
            track_label = label_to_class.get(str(track['@label']))

            def box_to_coord(track):
                return (
                    float(track['@frame']),
                    float(track['@xtl']),
                    float(track['@ytl']),
                    float(track['@xbr']),
                    float(track['@ybr']),
                )

            frame_and_coordinates = np.array(
                list(map(box_to_coord, track['box'])))
            bboxes = np.stack(
                (
                    frame_and_coordinates[:, 0],
                    np.repeat(track_id, frame_and_coordinates.shape[0]),
                    frame_and_coordinates[:, 1],
                    frame_and_coordinates[:, 2],
                    frame_and_coordinates[:, 3],
                    frame_and_coordinates[:, 4],
                    np.repeat(track_label, frame_and_coordinates.shape[0]),
                ),
                axis=1
            )
            return bboxes

        car_bboxes = map(tracks_to_bboxes, car_tracks_clean)
        car_bboxes = np.vstack(list(car_bboxes))
        return car_bboxes


if __name__ == '__main__':
    dataset = AICityDataset(AICITY_DIR, AICITY_ANNOTATIONS)

    # Show first image and a crop of a detection
    image, label = dataset[0]
    crop = image.crop(label)
    image.show()
    crop.show()

    # Get extended labels
    bbs = dataset.get_labels()
    print(bbs.shape)
