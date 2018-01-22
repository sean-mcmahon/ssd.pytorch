'''
A re-write of eval.py but designed to work with custom datasets
'''
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))
try:
    from eval import Timer, voc_ap
    from ssd import build_ssd
    from data import VOC_CLASSES as labelmap
    from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
except ImportError:
    from .eval import Timer, voc_ap
    from .ssd import build_ssd
    from .data import VOC_CLASSES as labelmap
    from .data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES


def write_class_wise_results(bboxes, dataset, save_path='eval'):
    def getpath(save_path, cls):
        res_dir = os.path.join(save_path, 'class_wise_dets')
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)
        return os.path.join(res_dir, 'det_{}.txt'.format(cls))

    for cls_ind, cls in enumerate(dataset.classes):
        print 'Writing {:s} results file'.format(cls)
        filename = getpath(save_path, cls)
        with open(filename, 'wt') as f:
            for im_id, index in enumerate(dataset.)


def eval_ssd(data_iter, network, save_path, cuda=True, use_voc_07=False):
    print('VOC07 AP metric? ' + (
        'Yes' if use_voc_07 else 'No, using VOC 2010-2012'))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    det_filename = os.path.join(save_path, 'detections.pkl')
    gt_filename = os.path.join(save_path, 'ground_truth_labels.pkl')
    if not os.path.isfile(det_filename) or not os.path.isfile(gt_filename):
        # might as well generate/load and save both.
        print('Generating network predictions and reading annotation GT...')
        all_bboxes, all_targets = getDetandGT(
            data_iter, network, use_cuda=cuda)
        print('Saving to path "{}"'.format(save_path))
        with open(det_filename, 'wb') as f:
            pickle.dump(all_bboxes, f, pickle.HIGHEST_PROTOCOL)
        with open(gt_filename, 'wb') as g:
            pickle.dump(all_targets, g, pickle.HIGHEST_PROTOCOL)
    else:
        print('Loading network predictions and' +
              ' ground truth annotations from "{}"'.format(save_path))
        with open(det_filename, 'rb') as f:
            all_bboxes = pickle.load(f)
        with open(gt_filename, 'rb') as g:
            all_targets = pickle.load(g)

    write_class_wise_results(all_bboxes, data_iter, save_path)
    aps = []
    for i, cls in enumerate(data_iter.classes):
        prec, rec, ap = calcMetrics(
            all_bboxes, all_targets, data_iter, cls, thresh=0.5, use_07_metric=use_voc_07)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        fname = os.path.join(save_path, cls + '_pr.txt')
        with open(fname, 'w') as f:
            f.write('Class: {}\nrec: {}\nprec: {}\nap: {}'.format(
                cls, rec, prec, ap))
    print('Metrics saves to "{}"'.format(save_path))


def calcMetrics(dets, gt, cls, dataset, thresh=0.5, use_07_metric=False):
    cls_id = dataset.classes.index(cls)
    class_recs = []
    num_pos = 0
    for idx in range(dataset.num_images):
        # Get Gts for class
        imname = dataset.pull_image_name(idx)
        cls_r = [r for r in gt[imname] if dataset.classes(r[4]) == cls]
        if cls_r != []:
            if 'voc' in dataset.__class__.__name__.lower():
                # VOC has 'difficult' bboxes which are not included in the
                # metrics
                anname = os.path.splitext(imname)[0]
                anname = anname.replace('JPEGImages', 'Annotations')
                annotation = parse_rec(anname + '.xml')
                difficult = [[x['difficult'] for x in R]
                             for R in annotation if R['name'] == cls]
                difficult = np.array(difficult).astype(np.bool)
                assert len(difficult) == len(cls_r), (
                    'len(difficult) = {}. len(cls_r) = {}'.format(
                        len(difficult), len(cls_r)))
            else:
                difficult = []

            det = [False] * len(cls_r)
            # should be equivalent to 'npos = npos + sum(~difficult)'
            num_pos += len(cls_r) - sum(difficult)
            assert len(cls_r) - sum(difficult) == sum(~difficult), (
                'diff={}. cls_r={}'.format(difficult, cls_r))
            class_recs.append({'bbox': gt[imname][:, :4], 'det': det,
                               'difficult': difficult, 'imname': imname})

    cls_dets_arr = dets[cls_id][:]
    dets_count = [len(k) for k in cls_dets_arr]
    num_dets = np.sum(dets_count)
    tp = np.zeros(num_dets)
    fp = np.zeros(num_dets)
    # Loop over detections, compute overlap with GT
    # If overlap (IOU) greater than threshold its a true positive
    # if the GT has been detected before of IOU below threshold its a false
    # positive
    for idx, dd in enumerate(cls_dets_arr):
        # if nothing detected, skip
        if dd == []:
            continue
        imname = dataset.pull_image_name(idx)
        confidences = dd[:, 4]
        bboxes = dd[:, :4]

        # sort by confidences (not sure why, but Girshick did it!)
        sorted_ind = np.argsort(-confidences)
        bboxes = bboxes[sorted_ind, :]

        gt_for_img = class_recs[idx]  # this is R in SSD eval.py
        # make sure we have the same image
        assert gt_for_img['imname'] == imname, '%s != %s' % (
            gt_for_img['imname'], imname)
        bbgt = gt_for_img['bbox']
        num_dets_so_far = np.sum(dets_count[:idk])
        for b_num, bb in enumerate(bboxes, 1):
            ovmax = -np.inf
            if bbgt.size > 0:
                overlaps, ovmax, jmax, = calcOverlap(bb, bbgt)
            if ovmax > ovthresh:
                if gt_for_img['difficult'] == []:
                    if not gt_for_img['det'][jmax]:
                        tp[num_dets_so_far + b_num] = 1.
                        gt_for_img['det'][jmax] = 1
                    else:
                        fp[num_dets_so_far + b_num] = 1.


    return -1, -1, -1


def calcOverlap(bb, bbgt):
    ixmin = np.maximum(bbgt[:, 0], bb[0])
    iymin = np.maximum(bbgt[:, 1], bb[1])
    ixmax = np.minimum(bbgt[:, 2], bb[2])
    iymax = np.minimum(bbgt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw * ih
    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
           (bbgt[:, 2] - bbgt[:, 0]) *
           (bbgt[:, 3] - bbgt[:, 1]) - inters)
    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return overlaps, ovmax, jmax


def getDetandGT(dataset, net, use_cuda=True):
    '''
    Perform a forward pass of the network over every image in dataset.
    Save results as pickle and return.
    '''
    _t = {'im_detect': Timer(), 'misc': Timer()}
    num_images = dataset.__len__()
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(dataset.num_classes())]
    all_targets = {}
    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if use_cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets
        all_targets[dataset.pull_image_name(i)] = gt

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
    return all_boxes, all_targets


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detection')
    parser.add_argument('--trained_model',
                        default='/home/sean/src/ssd_pytorch/weights/ssd300_mAP_77.43_v2.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='myeval/', type=str,
                        help='File path to save results')
    parser.add_argument('--save_path', default='eval', type=str,
                        help='Dir to save results')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=str2bool,
                        help='Use cuda to train model')
    parser.add_argument('--data_root', default='/home/sean/data/VOCdevkit',
                        help='Location of VOC root directory')
    args = parser.parse_args()
    return args


def main(args):
    img_dim = 300
    set_type = 'test'
    use_voc_07_ap_metric = True

    data_iter = VOCDetection(args.data_root, [('2007', set_type)], BaseTransform(
        img_dim, (104, 117, 123)), AnnotationTransform())
    print('Using data iterator "{}"'.format(data_iter.__class__.__name__))
    num_classes = data_iter.num_classes()

    net = build_ssd('test', img_dim, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model! {} Cuda'.format(
        'Using' if args.cuda else 'No'))

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    eval_ssd(data_iter, net, args.save_path, cuda=args.cuda,
             use_voc_07=use_voc_07_ap_metric)

if __name__ == '__main__':
    args = get_args()
    main(args)
