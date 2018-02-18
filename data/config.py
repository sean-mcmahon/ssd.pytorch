# config.py
import os.path
import sys
from data.mining import MiningAnnotationTransform, MiningDataset, PuddleDataset
from data.voc0712 import AnnotationTransform, VOCDetection
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))
from utils.augmentations import SSDAugmentation, SSDMiningAugmentation
# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home, "data/VOCdevkit/")

# Dataset dicts for multi-dataset initializing
train_sets = {'voc': [('2007', 'trainval'), ('2012', 'trainval')],
              'mining': 'train_gopro1_scraped_all_labelled.json',
              'puddles': 'mine_and_car_puddles.json'}
test_sets = {'voc': [('2007', 'test')],
             'mining': 'test_gopro2_all_labelled.json',
             'puddles': 'mine_and_car_puddles.json'}
rgb_means = {'voc': (104, 117, 123), 'mining': (65, 69, 76),
             'puddles': (107.9, 106.3, 107.2)}
data_iters = {'voc': VOCDetection, 'mining': MiningDataset,
              'puddles': PuddleDataset}
augmentators = {'voc': SSDAugmentation,
                'mining': SSDMiningAugmentation,
                'puddles': SSDMiningAugmentation}
target_transforms = {'voc': AnnotationTransform(),
                     'mining': MiningAnnotationTransform(),
                     'puddles': MiningAnnotationTransform(
                     class_to_ind={'puddle': 0})}

# note: if you used our download scripts, this should be right
VOCroot = ddir  # path to VOCdevkit root dir

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4


# SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
v2 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [30, 60, 111, 162, 213, 264],

    'max_sizes': [60, 111, 162, 213, 264, 315],

    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'name': 'v2',
}

# use average pooling layer as last layer before multibox layers
v1 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [30, 60, 114, 168, 222, 276],

    'max_sizes': [-1, 114, 168, 222, 276, 330],

    # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'aspect_ratios': [[1, 1, 2, 1 / 2], [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3],
                      [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3]],

    'variance': [0.1, 0.2],

    'clip': True,

    'name': 'v1',
}
