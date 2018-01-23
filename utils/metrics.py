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
import math
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))
try:
    from eval import Timer, voc_ap, parse_rec
    from ssd import build_ssd
    from data import VOC_CLASSES as labelmap
    from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
except ImportError:
    from .eval import Timer, voc_ap, parse_rec
    from .ssd import build_ssd
    from .data import VOC_CLASSES as labelmap
    from .data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES

# def write_class_wise_results(bboxes, dataset, save_path='eval'):
#     def getpath(save_path, cls):
#         res_dir = os.path.join(save_path, 'class_wise_dets')
#         if not os.path.isdir(res_dir):
#             os.mkdir(res_dir)
#         return os.path.join(res_dir, 'det_{}.txt'.format(cls))
#
#     for cls_ind, cls in enumerate(dataset.classes):
#         print 'Writing {:s} results file'.format(cls)
#         filename = getpath(save_path, cls)
    # with open(filename, 'wt') as f:
    # for im_id, index in enumerate(dataset.)


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

    aps = []
    for i, cls in enumerate(data_iter.classes):
        prec, rec, ap = calcMetrics(
            all_bboxes, all_targets, cls, data_iter, thresh=0.5, use_07_metric=use_voc_07)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        fname = os.path.join(save_path, cls + '_pr.txt')
        with open(fname, 'w') as f:
            f.write('Class: {}\nrec: {}\nprec: {}\nap: {}'.format(
                cls, rec, prec, ap))
    print('Metrics saves to "{}"'.format(save_path))


def calcMetrics(dets, gt, classname, dataset, thresh=0.5, use_07_metric=False):
    cls_id = dataset.classes.index(classname) + 1
    class_recs = {}
    num_pos = 0
    for idx in range(len(dataset)):
        # Get Gts for class
        imname = dataset.pull_image_name(idx)
        cls_r = [rr for rr in gt[imname]
                 if dataset.classes[int(rr[4])] == classname]
        if cls_r != []:
            if 'voc' in dataset.__class__.__name__.lower() and dataset.target_transform.keep_difficult:
                # VOC has 'difficult' bboxes which are not included in the
                # metrics
                anname = os.path.splitext(imname)[0]
                anname = anname.replace('JPEGImages', 'Annotations')
                annotation = parse_rec(anname + '.xml')
                difficult = [R['difficult']
                             for R in annotation if R['name'] == classname]
                difficult = np.array(difficult).astype(np.bool)
                assert len(difficult) == len(cls_r), (
                    'len(difficult) = {}. len(cls_r) = {}'.format(
                        len(difficult), len(cls_r)))
                assert len(cls_r) - sum(difficult) == sum(~difficult), (
                    'diff={}. cls_r={}'.format(difficult, cls_r))
            else:
                difficult = []

            det = [False] * len(cls_r)
            # should be equivalent to 'npos = npos + sum(~difficult)'
            num_pos += len(cls_r) - sum(difficult)
            if np.any(gt[imname][:, :4] < 1.0):
                height, width, channels = dataset.pull_image(idx).shape
                gt[imname][:, 0] *= width
                gt[imname][:, 2] *= width
                gt[imname][:, 1] *= height
                gt[imname][:, 3] *= height
            bbox = np.array([bb[0:4] for bb in cls_r])
            class_recs[imname] = {'bbox': bbox, 'det': det,
                                  'difficult': difficult}
        else:
            # image has no instances of class 'classname'
            class_recs[imname] = {'bbox': np.array(
                []), 'det': [], 'difficult': []}

    cls_dets_arr = dets[cls_id][:]
    dets_count = []
    dets_count = [len(k[0]) if k != [] else 0 for k in cls_dets_arr]
    dets_count = np.cumsum(dets_count)
    num_dets = dets_count[-1]
    assert num_dets > 1, 'num_dets = {}'.format(num_dets)
    assert len(dets_count) == len(cls_dets_arr), (
        'len(dets_count) %d; len(cls_dets_arr) %d' % (len(dets_count),
                                                      len(cls_dets_arr)))
    tp = np.zeros(num_dets)
    fp = np.zeros(num_dets)
    # Loop over detections, compute overlap with GT
    # If overlap (IOU) greater than threshold its a true positive
    # if the GT has been detected before of IOU below threshold its a false
    # positive
    if classname == 'aeroplane':
        plane_resfile = '/home/sean/hpc-home/SSD_detection/' + \
            'results/eval_voc/ssd300_120000/test/aeroplane_pr.pkl'
        pl_pr_file = plane_resfile.replace('aeroplane_pr', 'voc_aeroplane_fp_tp')
        with open(plane_resfile, 'rb') as f:
            pl_res = pickle.load(f)
        with open(pl_pr_file, 'rb') as f:
            pl_fp_tp = pickle.load(f)
    for idx, dd in enumerate(cls_dets_arr):
        # if nothing detected, skip
        if dd == []:
            continue
        imname = dd[1]
        confidences = dd[0][:, 4]
        bboxes = dd[0][:, :4]

        # sort by confidences (not sure why, but Girshick did it!)
        sorted_ind = np.argsort(-confidences)
        bboxes = bboxes[sorted_ind, :]

        gt_for_img = class_recs[imname]  # this is R in SSD eval.py

        # make sure we have the same image
        # assert gt_for_img['imname'] == imname, '%s != %s' % (
        #     gt_for_img['imname'], imname)
        bbgt = gt_for_img['bbox']
        num_dets_so_far = dets_count[idx] - len(bboxes)
        for b_num, bb in enumerate(bboxes, 0):
            assert num_dets_so_far + b_num < dets_count[-1], (
                '%d < %d' % (num_dets_so_far + b_num, dets_count[-1]))
            ovmax = -np.inf
            import pdb; pdb.set_trace()
            if bbgt.size > 0:
                overlaps, ovmax, jmax, = calcOverlap(bb, bbgt)
            if ovmax > thresh:
                if gt_for_img['difficult'] == []:
                    if not gt_for_img['det'][jmax]:
                        tp[num_dets_so_far + b_num] = 1.
                        gt_for_img['det'][jmax] = 1
                    else:
                        fp[num_dets_so_far + b_num] = 1.
                elif not gt_for_img['difficult'][jmax]:
                    if not gt_for_img['det'][jmax]:
                        tp[num_dets_so_far + b_num] = 1.
                        gt_for_img['det'][jmax] = 1
                    else:
                        fp[num_dets_so_far + b_num] = 1.
            else:
                fp[num_dets_so_far + b_num] = 1.
        if classname == 'aeroplane':
            d_idx = dets_count[idx]
            assert len(fp) == len(pl_fp_tp['fp'])
            assert np.array_equal(fp[:d_idx], pl_fp_tp['fp'][:d_idx])
            assert np.array_equal(tp[:d_idx], pl_fp_tp['tp'][:d_idx])
    assert np.sum(fp) + np.sum(tp) == num_dets, (
        'sum(fp)={}; sum(tp)={}; num_dets={}'.format(np.sum(fp), np.sum(tp), num_dets))
    # compute precision recall
    fp_ = fp.copy()
    tp_ = tp.copy()
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_pos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    if classname == 'aeroplane':
        assert math.isclose(
            ap, 0.8484757496817481, abs_tol=1e-12), 'ap={}'.format(ap)
    elif classname == 'cow':
        cow_resfile = '/home/sean/hpc-home/SSD_detection/' + \
            'results/eval_voc/ssd300_120000/test/cow_pr.pkl'
        with open(cow_resfile, 'rb') as f:
            cow_res = pickle.load(f)
        assert math.isclose(
            ap, 0.8243, abs_tol=1e-3), 'ap={}'.format(ap)
    # ap = 0.8484757496817481 for an 'aeroplane'
    return rec, prec, ap


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
        imname = dataset.pull_image_name(i)

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
            all_boxes[j][i] = [cls_dets, imname]
        gt[:, 0] *= w
        gt[:, 2] *= w
        gt[:, 1] *= h
        gt[:, 3] *= h
        all_targets[imname] = gt

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
    return all_boxes, all_targets


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Single Shot MultiBox Detection')
    parser.add_argument('--trained_model',
                        default='/home/sean/src/ssd_pytorch/weights/ssd300_mAP_77.43_v2.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_path', default='myeval/', type=str,
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
