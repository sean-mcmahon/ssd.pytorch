#!/usr/env python3.5
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
import json
try:
    from data.voc0712 import VOCDetection, AnnotationTransform
except ImportError:
    from voc0712 import VOCDetection, AnnotationTransform

sys.path.append('/home/sean/Dropbox/Uni/Code/ssd_pytorch/utils')
from augmentations import SSDAugmentation, SSDMiningAugmentation
# from ..utils.augmentations import SSDAugmentation, SSDMiningAugmentation


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
        return [self.anno2bbox(bb, width, height) for bb in target]

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


class MiningDataset(VOCDetection):
    '''

    '''

    def __init__(self, root=mining_root,
                 json_set='train_gopro1_scraped',
                 transform=None, target_transform=MiningAnnotationTransform(),
                 dataset_name='MiningToy'):
        self.root = root
        self.target_transform = target_transform
        self.transform = transform
        self.name = dataset_name
        json_set = json_set if json_set.endswith(
            '.json') else json_set + '.json'
        json_data = json.load(open(os.path.join(root, json_set)))
        self.im_names = []
        self.targets = []
        print('Loading from {}.'.format(json_set))

        for datum in json_data:
            if os.path.isdir('/home/n8307628'):
                n_name = datum['filename'].replace(
                    '/home/sean/hpc-home', expanduser("~"))
            else:
                n_name = datum['filename']
            self.im_names.append(str(n_name))
            self.targets.append(datum['annotations'])
            # self.targets += [self.anno2bbox(bb) for bb in
            # datum['annotations']]
        assert len(self.im_names) == len(self.targets), '\nlen img names: {}\n len targets {}'.format(
            len(self.im_names), len(self.targets))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.im_names)

    def pull_anno(self, index):
        img_name = self.im_names[index]
        targets = self.target_transform(self.targets[index], 1, 1)
        return [img_name, targets]

    def pull_image(self, index):
        img_name = self.im_names[index]
        assert os.path.isfile(
            img_name), 'Invalid name {} - "{}"'.format(index, img_name)
        return cv2.imread(img_name, cv2.IMREAD_COLOR)

    def pull_item(self, index):
        img_name = self.im_names[index]
        target = self.targets[index]
        # Beause I don't trust opencv to handle invalid file names.
        assert os.path.isfile(img_name), 'Invalid file "{}"'.format(img_name)
        img = cv2.imread(img_name)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        else:
            print('WARNING: Raw target values used.')

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


def getMiningMean(root, json_files):
    if isinstance(json_files, list):
        njson_file = os.path.join(root, 'combined.json')
        jdict = {}
        with open(njson_file, 'r+') as f:
            for j in json_files:
                jdict.update(json.load(open(os.path.join(root, j))))
            f.seek(0)
            json.dump(jdict, f)
        json_files = os.path.basename(njson_file)

    mdata = MiningDataset(root, transform=None,
                          target_transform=MiningAnnotationTransform(),
                          json_set=json_files)
    mean = np.zeros(3)
    print('Calulating mean from {} ({} images)'.format(
        json_files, mdata.__len__()))
    N = 0
    for idx in range(mdata.__len__()):
        img = mdata.pull_image(idx)
        height, width, channels = img.shape
        # blue
        mean[2] += np.sum(img[:, :, 0])
        # green
        mean[1] += np.sum(img[:, :, 1])
        # red
        mean[0] += np.sum(img[:, :, 2])

        N += (height * width)

    return mean / N


def rmNonLabelledInstances(root, json_file, new_name=None):
    json_file = json_file if json_file.endswith(
        '.json') else json_file + '.json'
    jdata = json.load(open(os.path.join(root, json_file)))

    all_labelled_ins = [d for d in jdata if any(d['annotations'])]

    print('removed {} labels'.format(len(jdata) - len(all_labelled_ins)))
    if new_name is not None:
        newjson = os.path.join(root, new_name)
        newjson = newjson if newjson.endswith(
            '.json') else newjson + '.json'
    else:
        newjson = json_file

    if len(jdata) - len(all_labelled_ins) == 0:
        return os.path.basename(newjson)
    else:
        with open(newjson, 'w', encoding='utf-8') as f:
            json.dump(all_labelled_ins, f)
        return os.path.basename(newjson)


if __name__ == '__main__':
    voc_root = '/home/sean/data/VOCdevkit'
    voc_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    ssd_dim = 300
    voc_means = (104, 117, 123)

    mining_root = '/home/sean/hpc-home/Mining_Site/MM_Car_Cam'
    json_file = 'train_gopro1_scraped_all_labelled'
    # json_file = rmNonLabelledInstances(mining_root, json_file, json_file + '_all_labelled')
    m_mean = getMiningMean(mining_root, json_file)
    print('mining mean = {}'.format(m_mean))
    m_mean = [round(m) for m in m_mean]
    print('mining mean rounded = {}'.format(m_mean))
    m_save = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'toy_mining_means.txt')
    with open(m_save, 'w') as f:
        f.write(str(m_mean))

    print('Testing Constructors')
    voc_dataset = VOCDetection(voc_root, voc_sets, SSDAugmentation(
        ssd_dim, voc_means), AnnotationTransform())

    mdataset = MiningDataset(
        mining_root, transform=SSDMiningAugmentation(ssd_dim, m_mean),
        target_transform=MiningAnnotationTransform(), json_set=json_file)

    data_iterators = [voc_dataset, mdataset]

    for it in data_iterators:
        print('-' * 10)
        print('Testing "{}":'.format(it.__class__.__name__))
        print('-' * 10)
        print('Test pull_anno:\n', it.pull_anno(4))
        print('Test __len__:  ', it.__len__())
        print('Test pull_image:')
        img = it.pull_image(4)
        print('Img shape {}'.format(np.shape(img)))
        print('Test pull_tensor: ', it.pull_tensor(4).size())

        print('Test pull_item')
        im, gt, h, w = it.pull_item(4)
        print('Image shape: {} (w={},h={})\nBboxes {}:\n{}'.format(
            im.size(), w, h, len(gt), gt))
        assert im.size()[1] == ssd_dim and im.size()[
            2] == ssd_dim, 'Height and width do not match ssd_dim'

        print('Test __getitem__')
        im, gt = it.__getitem__(4)
        print('Img dimensions {}. Range {} - {}'.format(im.size(),
                                                        im.min(), im.max()))
        print('Number of bboxes {}'.format(len(gt)))

        if 'VOCDetection' not in it.__class__.__name__:
            print('Iterating over entire dataset...')
            for idx in range(it.__len__()):
                it.pull_item(idx)

        print('~' * 20, '\n')
