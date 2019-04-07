from cv2 import cv2
from pathlib import Path
from typing import Tuple

import PIL.Image
import numpy as np
import xmltodict

import utils.detections_loader
from paths import AICITY_DIR, AICITY_ANNOTATIONS, PROJECT_ROOT


class AICityDataset(object):
    """ AICity dataset containing manually annotated labels """

    def __init__(self, root: Path, annotations_path: Path):
        self.root = root
        self.labels = self._load_gt(annotations_path)
        self.images_paths = sorted(root.joinpath('frames').glob('*.png'))
        self.label_to_class = {'bike': 0, 'car': 1}

    def __getitem__(self, idx) -> Tuple[PIL.Image.Image, np.ndarray]:
        """ Returns a sample (image, label).

        The image is a PIL.Image (RGB)

        Label is a 4-tuple defined as
        `[frame, track_id, (coord), track_label]` where coord is a
        4-tuple defined as two corners: top left, bottom right:
        `[x_tl, y_tl, x_br, y_br]` pixel
        coordinates.

        Note:
            the frame number retrieved is idx+1,
            so idx=0 retrieves image-0001.png
        """
        # Frame numbers at annotations' XML start at 0 but at 1 on det*.txt as
        # well as filenames start on `image-0001.png`.
        # So idx=0 (first element) retrieve a label which frame number is
        # [0,F-1], where F is the total number of frames,
        # then images_paths[frame_id] is retrieving from [1,F] in detections
        # indices, but as they are sorted in a python list:
        # frame_id=0 -> frame=0001.png -> first element -> images_paths[0],
        # frame_id=1 -> frame=0002.png -> second element -> images_paths[1] ...
        # frame_id=n -> frame={n+1}.png -> n-th element -> images_paths[n] ...

        # Reads image using PIL
        label = self.labels[idx]
        frame_id = int(label[0])
        return (
            self.get_pil_image(frame_id),
            label
        )

    def get_pil_image(self, frame_id):
        return PIL.Image.open(self.images_paths[frame_id], mode='r')

    def get_cv2_image(self, frame_id):
        return cv2.imread(str(self.images_paths[frame_id]))

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
        label_to_class = {'bike': 0, 'car': 1}

        print(f'Reading this shit: {path}')
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
            """ Retrieves bounding box information from an annotation """
            return (
                float(anno['@xtl']),
                float(anno['@ytl']),
                float(anno['@xbr']),
                float(anno['@ybr']),
            )

        def tracks_to_bboxes(track):
            """ Maps between formats

            Returns a matrix with columns:
            `[frame, track_id, xtl, ytl, xbr, br, label]`
            """
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

            frame_and_coordinates = np.array(list(
                map(box_to_coord, track['box'])))

            if frame_and_coordinates.shape[0] == 0:
                return np.empty((0, 7))

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

        # car_annotations = [track['box'] for track in car_tracks_clean]
        #
        # # Flatten list of lists
        # car_annotations = list(itertools.chain.from_iterable(car_annotations))
        # car_bboxes = np.array(list(map(anno_to_bboxes, car_annotations)))

        car_tracks = filter(is_a_car_track, tracks)
        car_tracks_clean = map(remove_occluded_bb, car_tracks)
        car_bboxes = map(tracks_to_bboxes, car_tracks_clean)

        car_bboxes = np.concatenate(list(car_bboxes), axis=0)
        return car_bboxes


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = AICityDataset(AICITY_DIR, AICITY_ANNOTATIONS)

    print(f'Dataset size: {len(dataset)}')

    # Show an image and a crop of a detection
    frame_num = 1900
    image, label = dataset[frame_num]
    plt.imshow(image)
    plt.show()

    crop = image.crop((label[2:6]))
    plt.imshow(crop)
    plt.show()

    # Get extended labels
    bbs = dataset.get_labels()
    print(bbs.shape)

    # Test detections
    # Selects detections from available model predictions
    from paths import PROJECT_ROOT

    detections_path = PROJECT_ROOT.joinpath('week3', 'det_mask_rcnn.txt')
    detections = detections_loader.load_bounding_boxes(detections_path, False)

    detections = detections[detections[:, 0] == frame_num]
    detections = detections[np.argsort(detections[5, :])]
    for idx in range(detections.shape[0]):
        bb = detections[idx, 1:5]
        crop = image.crop(bb)
        plt.imshow(crop)
        plt.show()
