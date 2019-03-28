import xmltodict
from collections import defaultdict
import numpy as np
from PIL import Image
import random
import glob
import os
import json
from shutil import copyfile


class annotationsParser():

    def __init__(self, filePath):
        self.file = filePath
        self.gt_nc = []
        self.train = []
        self.val = []
        self.test = []
        self.super_dict_train = defaultdict()
        self.super_dict_valid = defaultdict()
        self.super_dict_test = defaultdict()
        self.nFrames = 0
        self.extractGT()

    def extractGT(self):
        # self.gt_nc = []
        # self.train = []
        # self.val = []
        absID = 0
        # Create categories dictionary
        info_dict = {"year": int(2019), "version": "1.0", "description": "AICity challenge dataset",
                     "contributor": "NVIDIA", "url": "https://www.aicitychallenge.org", "date_created": "2019/03/19"
                     }

        licenses_list = [
            {"url": "http://www.aicitychallenge.org/wp-content/uploads/2019/01/DataLicenseAgreement_AICityChallenge2019.pdf",
             "id": 1, "name": "AICity Challenge data license agreement (non-commercial)"}
        ]

        # categories_list = [
        #     {'id': int(2), 'name': 'bicycle', 'supercategory': 'vehicle'},
        #     {'id': int(3), 'name': 'car', 'supercategory': 'vehicle'},
        #     {'id': int(4), 'name': 'motorcycle', 'supercategory': 'vehicle'}
        # ]

        categories_list = [
            {"id": int(3), "name": "car", "supercategory": "vehicle"}
        ]

        segm_empty = [[]]  # polygon format, a list of vertices [x,y]
        # Names in annot.
        annNames = ["id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd"]

        # Allocate space for final dictionary

        with open(self.file) as fd:
            doc = xmltodict.parse(fd.read())
        tracks = doc['annotations']['track']

        labelType = {
            1: 'car',
            2: 'bike',
            3: 'bicycle',
        }

        for track in tracks:
            BBoxes = track['box']
            id = track['@id']
            label = track['@label']
            if label == 'car':
                dataClass = 1
                if COCO:
                    dataClass = 3  # car is id = 3
            elif label == 'bike':
                dataClass = 2
                if COCO:
                    dataClass = 4  # motorcycle has id = 4
            elif label == 'bicycle':
                dataClass = 3
                if COCO:
                    dataClass = 2  # bicycle is id = 2
            else:
                dataClass = 0  # else background class

            for BBox in BBoxes:
                # format detection & GT  [frame, ID, xTopLeft, yTopLeft, xBottomRight, yBottomRight, class]
                if 'attribute' in BBox.keys():
                    add_BBOX = BBox['attribute']['#text'] == 'false'
                else:
                    add_BBOX = True
                if add_BBOX:
                    if COCO and dataClass == 3:  # we only want to detect cars
                        absID += 1
                        xtl = round(float(BBox['@xtl']), 2)
                        ytl = round(float(BBox['@ytl']), 2)
                        xbr = round(float(BBox['@xbr']), 2)
                        ybr = round(float(BBox['@ybr']), 2)

                        # Compute width and height
                        width = round(xbr - xtl, 2)
                        height = round(ybr - ytl, 2)
                        area_bb = width * height  # not rounded in annotations
                        iscrowd = 0  # by default, we assume independent instances (not overlapped)
                        # we do not have a pixel-wise segmentation
                        data = [int(absID), int(BBox['@frame']), int(dataClass), segm_empty, float(area_bb),
                                [xtl, ytl, width, height], iscrowd]

                        tmpZip = zip(annNames, data)
                        tmp_dict = dict(tmpZip)

                        if MODE == 'first':
                            if int(BBox['@frame']) <= num_training_frames:  # to training list
                                self.train.append(tmp_dict)
                            else:
                                self.test.append(tmp_dict)
                        else:
                            if MODE == 'random_tt' or MODE == 'random_tvt':
                                if int(BBox['@frame']) in train_ids:
                                    self.train.append(tmp_dict)
                                elif int(BBox['@frame']) in test_ids:
                                    self.test.append(tmp_dict)
                                else:
                                    if MODE == 'random_tvt' and int(BBox['@frame']) in val_ids:
                                        self.val.append(tmp_dict)
                                    else:
                                        print("FATAL: 'random_tt' should not have a validation split!")




                        # print(absID)
                        # print(len(self.train))

                    else:
                        data = [int(BBox['@frame']),
                                int(id),
                                float(BBox['@xtl']),
                                float(BBox['@ytl']),
                                float(BBox['@xbr']),
                                float(BBox['@ybr']),
                                dataClass]

                        self.gt_nc.append(data)

        # Embed list of dictionaries into super-dictionary
        self.super_dict_train = {'info': info_dict, 'images': image_list_train, 'annotations': self.train,
                                 'licenses': licenses_list, 'categories': categories_list}

        self.super_dict_test = {'info': info_dict, 'images': image_list_test, 'annotations': self.test,
                                'licenses': licenses_list, 'categories': categories_list}

        if MODE == 'random_tvt':
            self.super_dict_valid = {'info': info_dict, 'images': image_list_val, 'annotations': self.val,
                                     'licenses': licenses_list, 'categories': categories_list}

        if not COCO:
            for gtElement in self.gt_nc:
                if int(gtElement[0]) > self.nFrames:
                    self.nFrames = int(gtElement[0])
            return

    def setFile(self, filePath):
        self.file = filePath

    def getGTFrame(self, i):
        gtElement = self.gt_nc[i]
        return int(gtElement[0])

    def getGTNFrames(self):
        return self.nFrames

    def getGTID(self, i):
        gtElement = self.gt_nc[i]
        return gtElement[1]

    def getGTBoundingBox(self, i):
        # BBformat [xA,yA, xB, yB]
        gtElement = self.gt_nc[i]
        BBox = [gtElement[2], gtElement[3], gtElement[4], gtElement[5]]
        return BBox

    def getGTList(self):
        return self.gt_nc

    def getSuperDict_train(self):
        return self.super_dict_train

    def getSuperDict_val(self):
        return self.super_dict_valid

    def getSuperDict_test(self):
        return self.super_dict_test


if __name__ == '__main__':
    COCO = True
    # For simplicity, only take into account cars
    # quick and dirty way to extract list of image names with license, etc.

    FRAMES_DIR = '/home/fperez/Documents/Master-in-Computer-vision/Module 6/Project/mcv-m6-2019-team4/data/AICity_data/train/S03/c010/'
    fnames_list = sorted(glob.glob(os.path.join(str(FRAMES_DIR), 'frames_jpg_tvt/image-????.jpg')))
    MODE = 'random_tvt'  # or 'random_tt', 'random_tvt' (for train/val/test)
    LIC_NUM = 1
    train_perc = 50
    val_perc = 20
    test_perc = 30
    in_dir = '/home/fperez/Desktop'
    out_dir = '/home/fperez/Desktop/tmp_def/d/dd/ddd'
    train_json_file = os.path.join(out_dir, 'aicity_instances_train.json')
    val_json_file = os.path.join(out_dir, 'aicity_instances_val.json')
    test_json_file = os.path.join(out_dir, 'aicity_instances_test.json')

    # Create image list ('images' field in Coco)
    # TODO: put inside function of its own, etc.
    images_fields = ["id", "width", "height", "file_name", "license"]  # added width and height as they are used
    image_list_train = []
    image_list_val = []
    image_list_test = []
    train_count = 0
    val_count = 0
    test_count = 0
    assert (train_perc + val_perc + test_perc == 100)

    val_samples = int(np.floor((val_perc / 100) * len(fnames_list)))
    test_samples = int(np.floor((test_perc / 100) * len(fnames_list)))
    train_samples = len(fnames_list) - (val_samples + test_samples)

    if MODE == 'first':
        # num_training_frames = np.floor((train_perc / 100 * len(fnames_list)))

        for image in fnames_list:
            file_name = image.split('/')[-1]  # only image-0001.jpg ==> field 'file_name'
            # Read width and height
            im = Image.open(image)
            im_width, im_height = im.size

            img_id = int(file_name[6:10])
            license_id = LIC_NUM  # only 1 applies

            data = [img_id, im_width, im_height, file_name, license_id]
            tmpZip = zip(images_fields, data)
            tmpDict = dict(tmpZip)
            if img_id < len(train_samples):  # append to training image_list
                image_list_train.append(tmpDict)
                train_count += 1

            elif img_id >= len(train_samples) and img_id < (len(train_samples) + len(test_samples)):
                image_list_test.append(tmpDict)
                test_count += 1
            else:  # append to validation image_list
                image_list_val.append(tmpDict)
                val_count += 1
            # Add elif for random partition, etc.

    else:
        # Create a serious train/val/test split
        # Given that we will fine-tune from a relatively similar dataset, 40% to train should be fine
        img_num_list = [int(a.split('/')[-1][6:10]) for a in fnames_list]

        if MODE == 'random_tt':
            # Pick random samples from population w/o repetition
            train_ids = random.sample(img_num_list, train_samples)
            # The rest to test
            test_ids = [idx for idx in img_num_list not in train_ids]

        elif MODE == 'random_tvt':
            # Pick random samples from population w/o repetition
            train_ids = random.sample(img_num_list, train_samples + val_samples)
            # The rest to test
            test_ids = [idx for idx in img_num_list if idx not in train_ids]
            # And now split training again into validation + train
            val_ids = random.sample(train_ids, val_samples)
            # Remove val_ids from train
            train_ids = [idx for idx in train_ids if idx not in val_ids]

        else:
            print("FATAL: incorrect 'MODE' ('first', 'random_tt' and 'random_tvt'")

        # assert((len(val_ids) + len(train_ids) + len(test_ids)) == len(img_num_list),
        #        "FATAL: final splits should add to total number of images but do not")
        # Do as before but randomly picking samples
        # Pick random indices for train, validation and test WITHOUT repetition
        for idx in range(len(fnames_list)):
            file_name = fnames_list[idx].split('/')[-1]  # only image-0001.jpg ==> field 'file_name'
            # Read width and height
            im = Image.open(fnames_list[idx])
            im_width, im_height = im.size

            img_id = int(file_name[6:10])
            license_id = LIC_NUM  # only 1 applies

            data = [img_id, im_width, im_height, file_name, license_id]
            tmpZip = zip(images_fields, data)
            tmpDict = dict(tmpZip)

            src_file = fnames_list[img_id - 1]  # idxs start at 1 so we need to account for that
            src_dir = '/'.join(src_file.split('/')[:-1])
            if MODE == 'random_tt':
                if img_id in train_ids:
                    image_list_train.append(tmpDict)
                    # (copy) to destination folder (by default: FRAMES_DIR/<set_name>)
                    dest_file = os.path.join(src_dir, 'train', file_name)
                    copyfile(src_file, dest_file)
                    train_count += 1
                else:
                    image_list_val.append(tmpDict)
                    dest_file = os.path.join(src_dir, 'val', file_name)
                    copyfile(src_file, dest_file)
                    val_count += 1

            elif MODE == 'random_tvt':
                if img_id in train_ids:
                    image_list_train.append(tmpDict)
                    dest_file = os.path.join(src_dir, 'train', file_name)
                    copyfile(src_file, dest_file)
                    train_count += 1
                elif img_id in val_ids:
                    image_list_val.append(tmpDict)
                    dest_file = os.path.join(src_dir, 'val', file_name)
                    copyfile(src_file, dest_file)
                    val_count += 1
                else:
                    image_list_test.append(tmpDict)
                    dest_file = os.path.join(src_dir, 'test', file_name)
                    copyfile(src_file, dest_file)
                    test_count += 1

            # Add elif for random partition, etc.

    print("Number of training frames: {0}\nNumber of validation frames: {1}\nNumber of test frames: {2}".format(
        train_count, val_count, test_count))
    print("OG number of images was: {0}, total length of splits lists: {1} (should match)".format(
        len(fnames_list), train_count + val_count + test_count))


    # Generate annotations
    anPars = annotationsParser(os.path.join(in_dir, 'm6-full_annotation_AICITY.xml'))  # change to correct path
    # Get dictionaries for train, validation and test (for this MODE!)
    trainDict = anPars.getSuperDict_train()
    valDict = anPars.getSuperDict_val()
    testDict = anPars.getSuperDict_test()

    # Save to json file
    with open(train_json_file, 'w') as outfile:
        json.dump(trainDict, outfile)

    with open(val_json_file, 'w') as outfile:
        json.dump(valDict, outfile)

    with open(test_json_file, 'w') as outfile:
        json.dump(testDict, outfile)

