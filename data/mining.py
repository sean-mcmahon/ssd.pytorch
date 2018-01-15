"""
My Mining Dataset Classes for parsing through SSD in pytorch.

Based on VOC0712.py (https://github.com/amdegroot/ssd.pytorch)

Sean McMahon - 15 Jan 2017
"""

import os
from os.path import expanduser
import sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2

MINING_CLASSES = ('mine_vehicle', 'car', 'signs', 'pole', 'person', 'slips')
mining_root = '/home/sean/hpc-home/Mining_Site/MM_Car_Cam'

work_dir = '/home/sean/hpc-home' if os.path.isdir(
    '/home/sean') else expanduser("~")


class MiningAnnotationTransform(object):
    """
    Transforms my mining data into a Tensor of bbox coords and label index.
    Takes bbox coord in image pixels and converts to percentage, and converts
    label string to an index.
    """

    def __init__(self, class_to_ind=None):
        if class_to_ind is None:
            self.class_to_ind = dict(
                zip(MINING_CLASSES, range(len(MINING_CLASSES))))
        else:
            self.class_to_ind = class_to_ind

    def __call__(self, target, width, height):
        return [anno2bbox(bb, width, height) for bb in target]

    def anno2bbox(self, sloth_anno, w, h):
        assert sloth_anno['type'] == 'rect', 'invalid label type "{}"'.format(
            sloth_anno['type'])
        x1 = sloth_anno['x'] / w
        y1 = sloth_anno['y'] / h
        x2 = sloth_anno['x'] + sloth_anno['width']
        x2 /= w
        y2 = sloth_anno['y'] + sloth_anno['height']
        y2 /= h
        return [x1, y1, x2, y2, self.class_to_ind[sloth_anno['class']]]


class MiningDataset(data.Dataset):
    '''

    '''

    def __init__(self, root=mining_root,
                 json_set='train_gopro1_scraped',
                 transform=None, target_transform=MiningAnnotationTransform(),
                 dataset_name='MiningToy'):
        self.root = root
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', '%s.json')
        self._imgpath = os.path.join(('%s', ))
        json_set = json_set if json.endswith('.json') else json_set + '.json'
        json_data = json.load(open(os.path.join(root, json_set)))
        self.im_names = []
        self.targets = []

        for datum in json_data:
            if os.isdir('/home/n8307628'):
                n_name = datum['filename'].replace(
                    '/home/sean/hpc-home', expanduser("~"))
            else:
                n_name = datum['filename']
            self.im_names.append(str(n_name))
            self.targets = datum['annotations']
            # self.targets += [self.anno2bbox(bb) for bb in
            # datum['annotations']]
        assert len(ids) == len(targets), 'len ids: {}\n len targets {}'.format(
            len(ids, len(targets)))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.im_names)

    def pull_anno(self, index):
        img_name = self.im_names[index]
        targets = self.targets[index]
        return [img_name, targets]

    def pull_image(self, index):
        img_name = self.im_names(index)
        return Image.open(img_name)

    def pull_item(self, index):
        img_name = self.im_names[index]
        # Beause I don't trust opencv to handle invalid file names.
        assert os.path.isfile(img_name), 'Invalid file "{}"'.format(img_name)
        img = cv2.read(img_name)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        else:
            print 'WARNING: Raw target values used.'

        if self.transform is not None:
            target = np.array(target)
            imgs, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


if __name__ == '__main__':
    # TODO write toy examples for testing classes
