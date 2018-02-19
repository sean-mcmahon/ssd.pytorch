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
try:
    from utils.augmentations import SSDAugmentation, SSDMiningAugmentation
except ImportError:
    sys.path.append('/home/sean/Dropbox/Uni/Code/ssd_pytorch/utils')
    from augmentations import SSDAugmentation, SSDMiningAugmentation
# from ..utils.augmentations import SSDAugmentation, SSDMiningAugmentation


MINING_CLASSES = ('mine_vehicle', 'car', 'signs', 'pole', 'person', 'slips')
mining_root = '/home/sean/hpc-home/Mining_Site/MM_Car_Cam'
PUDDLE_CLASSES = ('puddle', 'non_puddle_img')


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
        if sloth_anno:
            x1 = sloth_anno['x'] / w
            y1 = sloth_anno['y'] / h
            x2 = sloth_anno['x'] + sloth_anno['width']
            x2 /= w
            y2 = sloth_anno['y'] + sloth_anno['height']
            y2 /= h
            return [x1, y1, x2, y2, self.class_to_ind[sloth_anno['class']]]
        else:
            return []


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
        json_filename = os.path.join(root, json_set)
        if not os.path.isfile(json_filename):
            raise IOError('Invalid json file "%s"' % json_filename)
        json_data = json.load(open(json_filename))
        self.im_names = []
        self.targets = []
        self.classes = MINING_CLASSES
        print('Loading from {}.'.format(json_set))

        for datum in json_data:
            if os.path.isdir('/home/n8307628'):
                n_name = datum['filename'].replace(
                    '/home/sean/hpc-home', expanduser("~"))
                n_name = n_name.replace(
                    '/home/sean//hpc-home', expanduser("~"))
            else:
                n_name = datum['filename']
            n_name = n_name.replace('//', '/')
            if self.root not in n_name:
                n_name = os.path.join(self.root, n_name)
            assert os.path.isfile(n_name), 'Invalid img "{}"'.format(n_name)
            self.im_names.append(str(n_name))
            self.targets.append(datum['annotations'])
            # self.targets += [self.anno2bbox(bb) for bb in
            # datum['annotations']]
        assert len(self.im_names) == len(
            self.targets), '\nlen img names: {}\n len targets {}'.format(
            len(self.im_names), len(self.targets))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.im_names)

    def num_classes(self):
        # Pytorch crashed with weird cuda error without this + 1.
        # It's also present when returning the VOC num classes, so must be needed.
        return len(MINING_CLASSES) + 1

    def pull_anno(self, index):
        img_name = self.im_names[index]
        targets = self.target_transform(self.targets[index], 1, 1)
        return [img_name, targets]

    def pull_image(self, index):
        img_name = self.im_names[index]
        return cv2.imread(img_name, cv2.IMREAD_COLOR)

    def pull_image_name(self, index):
        return self.im_names[index]

    def pull_item(self, index):
        img_name = self.im_names[index]
        target = self.targets[index]
        # Beause I don't trust opencv to handle invalid file names.
        # assert os.path.isfile(img_name), 'Invalid file "{}"'.format(img_name)
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)

        if img is None:
            print('Image read was "Nonetype" - "{}"'.format(img_name))
            img = np.asarray(Image.open(img_name)).astype(np.uint8)
            img = img[:, :, (2, 1, 0)]  # convert to bgr
            # img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            # if img is None:
            #     print('image is still none!')
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        else:
            print('WARNING: Raw target values used.')

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # to rgb
            img = img[:, :, (2, 1, 0)]

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


class PuddleDataset(MiningDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # puddle + blank img label + background
        self.number_classes = 3
        self.classes = PUDDLE_CLASSES
        if self.name == 'MiningToy':
            self.name = 'PUDDLES'

    def num_classes(self):
        return self.number_classes

    def pull_item(self, index):
        img_name = self.im_names[index]
        target = self.targets[index]
        # Beause I don't trust opencv to handle invalid file names.
        # assert os.path.isfile(img_name), 'Invalid file "{}"'.format(img_name)
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)

        if img is None:
            print('Image read was "Nonetype" - "{}"'.format(img_name))
            img = np.asarray(Image.open(img_name)).astype(np.uint8)
            img = img[:, :, (2, 1, 0)]  # convert to bgr
            # img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            # if img is None:
            #     print('image is still none!')
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        else:
            print('WARNING: Raw target values used.')

        if self.transform is not None:
            target = np.array(target)
            if target.size > 0:
                img, boxes, labels = self.transform(
                    img, target[:, :4], target[:, 4])
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                # no label, use dummy label so augmentators still work.
                dummy_target = np.array([[0.01, 0.01, 0.99, 0.99, 1]])
                img, boxes, labels = self.transform(
                    img, dummy_target[..., :4], dummy_target[..., 4])
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # to rgb
            img = img[:, :, (2, 1, 0)]

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_anno(self, index):
        img_name = self.im_names[index]
        if np.array(self.targets[index]).size < 1:
            img = self.pull_image(index)
            img = self.pull_image(index) if img is None else img
            h, w, channels = img.shape
            dummy_target = np.array(
                [[0.01 * w, 0.01 * h, 0.99 * w, 0.99 * h, 1]])
            return [img_name, dummy_target]
        targets = self.target_transform(self.targets[index], 1, 1)
        return [img_name, targets]


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
                          target_transform=None,
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

    mining_root = '/media/sean/mydrive/Mining_Site/MM_Car_Cam'
    json_file = 'train_gopro1_scraped_all_labelled'
    # json_file = rmNonLabelledInstances(mining_root, json_file, json_file + '_all_labelled')
    m_save = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'toy_mining_means.txt')
    if not os.path.isfile(m_save):
        m_mean = getMiningMean(mining_root, json_file)
        print('mining mean = {}'.format(m_mean))
        m_mean = [round(m) for m in m_mean]
        print('mining mean rounded = {}'.format(m_mean))
        with open(m_save, 'w') as f:
            f.write(str(m_mean))
    else:
        m_mean = [65.0, 69.0, 76.0]

    # Puddle Detection Dataset
    puddle_root = '/home/sean/Downloads/road_slip_hazards'
    pjson_file = 'mine_and_car_puddles'
    p_save = os.path.join(os.path.dirname(m_save),
                          'all_puddle_dataset_means.txt')
    if not os.path.isfile(p_save):
        p_mean = getMiningMean(puddle_root, pjson_file)
        with open(p_save, 'w') as f:
            f.write(str(p_mean))
    else:
        p_mean = np.array([107.86730173, 106.28081276, 107.2159824])

    print('Testing Constructors')
    voc_dataset = VOCDetection(voc_root, voc_sets, SSDAugmentation(
        ssd_dim, voc_means), AnnotationTransform())

    mdataset = MiningDataset(
        mining_root, transform=SSDMiningAugmentation(ssd_dim, m_mean),
        target_transform=MiningAnnotationTransform(), json_set=json_file)

    pud_dataset = PuddleDataset(
        puddle_root, transform=SSDMiningAugmentation(ssd_dim, p_mean),
        target_transform=MiningAnnotationTransform(class_to_ind={'puddle': 0}),
        json_set=pjson_file, dataset_name='PUDDLES')

    data_iterators = [pud_dataset, voc_dataset, mdataset]

    for it in data_iterators:
        print('-' * 10)
        print('Testing "{} - {}":'.format(it.__class__.__name__, it.name))
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
        print('Num classes {}'.format(it.num_classes()))
        class_instances = np.zeros(it.num_classes())
        class_per_img = np.zeros(it.num_classes())

        if 'VOCDetection' not in it.__class__.__name__:
            print('Iterating over entire dataset...')
            for idx in range(it.__len__()):
                im, gt, h, w = it.pull_item(idx)
                labels = gt[:, -1].astype(np.int)
                class_per_img[np.unique(labels)] += 1
                for ll in labels:
                    class_instances[ll] += 1
            cls_per = class_instances*100 / np.sum(class_instances)
            cls_per_image = class_per_img*100 / len(it)
            for i in range(it.num_classes()-1):
                print('{}% of labels are "{}"'.format(
                    round(cls_per[i], 3), it.classes[i]))
                print('{}% '.format(round(cls_per_image[i], 3)) +
                      'of images have at least one instance of ' +
                      '"{}"'.format(it.classes[i]))
                print(' ')

        print('~' * 20, '\n')
