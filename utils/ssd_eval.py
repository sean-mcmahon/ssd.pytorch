"""
A re-re-write of the eval.py code.

Sean McMahon
"""
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
import sys
import argparse
import pickle
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))
# from .eval import Timer, voc_ap, parse_rec
from metrics import getDetandGT
from ssd import build_ssd
from data import MiningDataset, MINING_CLASSES, MiningAnnotationTransform
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES


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


def check_dets(all_bboxes, dets_dir, dataset, out_dir):
    def bb_str(name, box_arr):
        return '{} array has shape {}; type {}'.format(name, np.shape(box_arr),
                                                       type(box_arr))

    def cmp_bboxes(all_bboxes, voc_bb):
        for cls_id, (mbb, vbb) in enumerate(zip(all_bboxes, voc_bb)):
            for kk in range(np.shape(voc_bb)[1]):
                assert np.shape(mbb[kk]) == np.shape(vbb[kk])
                cond = np.isclose(mbb[kk], vbb[kk], atol=1e-3)
                if cond.size > 0:
                    if not np.all(cond):
                        fid = np.where(cond == False)
                        m_e = mbb[kk][fid]
                        v_e = vbb[kk][fid]
                        e_str = '[{}][{}]\n{}\nis not close to\n{}\nloc {}'.format(
                            cls_id, kk, m_e, v_e, fid)
                        if np.all(vbb[kk][np.unique(fid[0]), -1] < 0.1):
                            print('Warning: ' + e_str)
                        else:
                            raise AssertionError('Error at ' + e_str)
                else:
                    assert mbb[kk] == vbb[
                        kk], 'Error at [{}][{}]'.format(cls_id, kk)
    bsdir = '/home/sean/hpc-home/'
    voc_eval_path = bsdir + 'SSD_detection/src/eval/ssd300_120000/test/'
    vbb1name = voc_eval_path + 'detections.pkl'
    assert os.path.isdir(voc_eval_path), '"{}"'.format(voc_eval_path)
    with open(vbb1name, 'rb') as f:
        voc_bb = pickle.load(f)
    # vbb2name = os.path.join(voc_eval_path, 'detections.pkl').replace(
    #         '/eval/', '/eval_t1/')
    # with open(vbb2name, 'rb') as g:
    #     voc_bb2 = pickle.load(g)
    # vbb3name = '/home/sean/Dropbox/Uni/Code/ssd_pytorch/eval_tlocal/ssd300_120000/test/'
    # vbb3name += 'detections.pkl'
    # with open(vbb3name, 'rb') as h:
    #     voc_bb3 = pickle.load(h)
    vbb_orign = '/home/sean/src/ssd_pytorch/ssd300_120000/test/detections.pkl'
    with open(vbb_orign, 'rb') as j:
        vbb_orig = pickle.load(j)
    bb_s = '{} array has shape {}; type {}'
    e_str = 'BBoxes not equal\n{}\n{}'.format(
        bb_s.format('mybbox', np.shape(all_bboxes), type(all_bboxes)),
        bb_s.format('vocbbox', np.shape(voc_bb), type(voc_bb)))
    assert np.shape(voc_bb) == np.shape(all_bboxes), e_str
    # cmp_bboxes(voc_bb, voc_bb2)
    # print('cmp bb with bb3')
    # cmp_bboxes(all_bboxes, voc_bb3)
    cmp_bboxes(all_bboxes, vbb_orig)
    # print('\n' + '='*50 + '\n')
    # raise Exception(' ')
    # cmp_bboxes(all_bboxes, voc_bb)
    print('BBox array check passed.')

    # cls_voc_dets = os.path.join(voc_eval_path, 'results', 'det_test_{}.txt')
    orig_detsdir = '/home/sean/data/VOCdevkit/VOC2007/results'
    cls_voc_dets = os.path.join(orig_detsdir, 'det_test_{}.txt')
    mydets = os.path.join(out_dir, 'class_detections', '{}_dets.txt')

    for cls_id, classname in enumerate(dataset.classes):
        print('Checking {} dets text file {}/{}.'.format(classname, cls_id,
                                                         len(dataset.classes)))
        with open(cls_voc_dets.format(classname), 'r') as f:
            voc_lines = f.readlines()
        with open(mydets.format(classname), 'r') as f:
            mylines = f.readlines()
        myconf, my_s_scores, myBB, myim_ids = process_dets_txt(mylines)
        myim_ids = [os.path.basename(os.path.splitext(x)[0]) for x in myim_ids]

        vocconf, voc_s_scores, vocBB, vocim_ids = process_dets_txt(voc_lines)

        assert np.array_equal(myconf, vocconf), 'Confs not equal\n{}\n{}'.format(
            bb_str('myconf', myconf), bb_str('vocconf', vocconf))
        assert np.array_equal(
            my_s_scores, voc_s_scores), 'Sorted Scores not equal\n{}\n{}'.format(
            bb_str('my_s_scores', my_s_scores), bb_str('voc_s_scores', voc_s_scores))

        assert np.array_equal(
            myBB, vocBB), 'BBoxes not equal\n{}\n{}'.format(
            bb_str('myBB', myBB), bb_str('vocBB', vocBB))

        diffs = [a for a in myim_ids +
                 vocim_ids if (a not in myim_ids) or (a not in vocim_ids)]
        assert set(myim_ids) == set(vocim_ids), 'Im ids not equal\n{}\n{}\nNum diffs: {}'.format(
            bb_str('myim_ids', myim_ids), bb_str('vocim_ids', vocim_ids), len(diffs))
        assert myim_ids == vocim_ids, 'Im ids ordering is different'

    print('VOC Detections check passed!!!')
    return True


def process_dets_txt(dets_lines):
    '''
    get useful information from detection text files
    Code from https://github.com/amdegroot/ssd.pytorch/blob/master/eval.py
    (voc_eval)
    '''
    if any(dets_lines) == 1:
        splitlines = [x.strip().split(' ') for x in dets_lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        return confidence, sorted_scores, BB, image_ids
    else:
        raise Exception('No lines found.')


def write_class_dets(bboxes, dataset, out_dir):
    assert os.path.isdir(out_dir), 'Invalid: "%s"' % out_dir
    dets_dir = os.path.join(out_dir, 'class_detections')
    tempname = os.path.join(dets_dir, '{}_dets.txt')
    imnames = [dataset.pull_image_name(id_) for id_ in range(len(dataset))]
    if 'voc' in dataset.__class__.__name__.lower():
        # do some checks
        es = 'len(dataset.ids) = {}. len(imnames) = {}'.format(
            len(dataset.ids), len(imnames))
        assert len(dataset.ids) == len(imnames), es
        for idx, imid in enumerate(dataset.ids):
            my_id = os.path.basename(os.path.splitext(imnames[idx])[0])
            image_id = imid[1]
            es = 'Different ids: \nmy_id = {}\nimid  = {}'.format(
                my_id, image_id)
            assert my_id == image_id, es
        print('VOC checks passed.')

    if not os.path.isdir(dets_dir):
        os.mkdir(dets_dir)
    for cls_id, classname in enumerate(dataset.classes):
        fname = tempname.format(classname)
        if os.path.isfile(fname):
            print('Reading detections for class {}'.format(classname))
            continue
        print('Writing detections for class {}'.format(classname))
        with open(fname, 'wt') as f:
            for im_id, name in enumerate(imnames):
                dets = bboxes[cls_id + 1][im_id]
                if dets == []:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(name, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
    return dets_dir


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

    print('Reformatting Detections and Labels...')
    det_dir = write_class_dets(all_bboxes, data_iter, save_path)
    # label_dir = write_class_labels(all_targets, data_iter, save_path)
    if 'voc' in data_iter.__class__.__name__.lower():
        # check detections and labels against those created from eval.py
        check_dets(all_bboxes, det_dir, data_iter, save_path)

    print('Calulating Metrics...')

    aps = []
    # for i, cls in enumerate(data_iter.classes):
    #     prec, rec, ap = calcMetrics(
    #         all_bboxes, all_targets, cls, data_iter, thresh=0.5, use_07_metric=use_voc_07)
    #     aps += [ap]
    #     print('AP for {} = {:.4f}'.format(cls, ap))
    #     fname = os.path.join(save_path, cls + '_pr.txt')
    #     with open(fname, 'w') as f:
    #         f.write('Class: {}\nrec: {}\nprec: {}\nap: {}'.format(
    #             cls, rec, prec, ap))
    # print('Metrics saves to "{}"'.format(save_path))


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
