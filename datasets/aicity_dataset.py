import itertools

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
        self.labels = self._load_gt(AICITY_ANNOTATIONS)
        self.images_paths = sorted(root.joinpath('frames').glob('*.png'))
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
        return self.labels

    # def get_labels(self) -> np.ndarray:
    #     """ Read XML and parse data
    #
    #     Output is formatted as:
    #     ```
    #     [frame, id, xTopLeft, yTopLeft, xBottomRight, yBottomRight, class]
    #     ```
    #     """
    #     with AICITY_ANNOTATIONS.open('r') as file:
    #         labels = xmltodict.parse(file.read())
    #
    #     tracks = labels['annotations']['track']
    #
    #     def tracks_to_bboxes(track):
    #         track_id = float(track['@id'])
    #         track_label = self.label_to_class.get(str(track['@label']))
    #
    #         def box_to_coord(track):
    #             return (
    #                 float(track['@frame']),
    #                 float(track['@xtl']),
    #                 float(track['@ytl']),
    #                 float(track['@xbr']),
    #                 float(track['@ybr']),
    #             )
    #
    #         coordinates = np.array(list(map(box_to_coord, track['box'])))
    #         bboxes = np.stack(
    #             (
    #                 coordinates[:, 0],
    #                 np.repeat(track_id, coordinates.shape[0]),
    #                 coordinates[:, 1],
    #                 coordinates[:, 2],
    #                 coordinates[:, 3],
    #                 coordinates[:, 4],
    #                 np.repeat(track_label, coordinates.shape[0]),
    #             ),
    #             axis=1
    #         )
    #         return bboxes
    #
    #     bbs = map(tracks_to_bboxes, tracks)
    #     bbs = np.vstack(list(bbs))
    #     return bbs

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

        car_bboxes = map(tracks_to_bboxes, car_tracks_clean)
        car_bboxes = np.vstack(list(car_bboxes))
        return car_bboxes


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
