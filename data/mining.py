"""
My Mining Dataset Classes for parsing through SSD in pytorch.

Based on VOC0712.py (https://github.com/amdegroot/ssd.pytorch)

Sean McMahon - 15 Jan 2017
"""

import os
import sys
import torch
from PIL import Image
import numpy as np
import cv2
import voc0712

MINING_CLASSES = ('')

class AnnotationTransform(object):
    """
    Transforms my mining data into a Tensor of bbox coords and label index.
    """
    def __init__(self):
        raise NotImplementedError

    def __call__(self, target, width, height):
        '''

        '''
        raise NotImplementedError

class MiningDataset(voc0712.VOCDetection):
    '''

    '''
    def __init__(self):
        raise NotImplementedError

    def pull_anno(self):
        raise NotImplementedError

    def pull_item(self):
        raise NotImplementedError
