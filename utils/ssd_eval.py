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
import json
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))
# from .eval import Timer, voc_ap, parse_rec
from eval import voc_ap
from metrics import getDetandGT
from ssd import build_ssd
from data import MiningDataset, MINING_CLASSES, MiningAnnotationTransform
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Single Shot MultiBox Detection')
    parser.add_argument('--trained_model',
                        default='/home/sean/src/ssd_pytorch/weights/ssd300_mAP_77.43_v2.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_path', default='myeval2/', type=str,
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
    bsdir = '/home/sean/Dropbox/Uni/Code/ssd_pytorch'
    voc_eval_path = bsdir + '/eval/ssd300_120000/test/'
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
    # Convert back to original format
    # nbb = []
    # for ii in range(len(all_bboxes)):
    #     for jj in range(len(all_bboxes[0])):
    #         if all_bboxes[ii][jj] == []:
    #             all_bboxes[ii][jj] = []
    #         else:
    #             nbb.append(all_bboxes[ii][jj][0])
    nbb = [[[] if all_bboxes[ii][jj] == [] else all_bboxes[ii][jj][0] for jj in range(len(all_bboxes[0]))]
           for ii in range(len(all_bboxes))]
    assert isinstance(nbb[5][999], np.ndarray)
    assert np.shape(nbb) == np.shape(all_bboxes)
    assert np.array_equal(nbb[5][999], all_bboxes[5][999][0])
    cmp_bboxes(nbb, vbb_orig)
    # print('\n' + '='*50 + '\n')
    # raise Exception(' ')
    # cmp_bboxes(all_bboxes, voc_bb)
    print('BBox array check passed.')

    # cls_voc_dets = os.path.join(voc_eval_path, 'results', 'det_test_{}.txt')
    orig_detsdir = '/home/sean/data/VOCdevkit/VOC2007/results'
    cls_voc_dets = os.path.join(orig_detsdir, 'det_test_{}.txt')
    mydets = os.path.join(dets_dir, '{}_dets.txt')

    for cls_id, classname in enumerate(dataset.classes):
        print('Checking {} dets text file {}/{}.'.format(classname, cls_id,
                                                         len(dataset.classes)))
        with open(cls_voc_dets.format(classname), 'r') as f:
            voc_lines = f.readlines()
        with open(mydets.format(classname), 'r') as f:
            mylines = f.readlines()
        myconf, my_s_scores, myBB, myim_ids = process_dets_txt(
            mylines, classname)
        myim_ids = [os.path.basename(os.path.splitext(x)[0]) for x in myim_ids]

        vocconf, voc_s_scores, vocBB, vocim_ids = process_dets_txt(
            voc_lines, classname)

        assert np.allclose(myconf, vocconf), 'Confs not equal\n{}\n{}'.format(
            bb_str('myconf', myconf), bb_str('vocconf', vocconf))
        assert np.allclose(
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


def process_dets_txt(dets_lines, classname):
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
        raise Exception(
            'No lines found. Check text file for class %s' % classname)


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
                if isinstance(dets, list):
                    assert os.path.isfile(dets[1])
                    assert name == dets[1]
                    dets = dets[0]
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(name, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
    return dets_dir


def get_voc_label_recs(out_dir, annopath, image_ids):
    def parse_rec(filename):
        """
        Parse a PASCAL VOC xml file. From:
        https://github.com/amdegroot/ssd.pytorch/blob/master/eval.py
        """
        assert os.path.isfile(filename), 'Invalid: "{}"'.format(filename)
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                                  int(bbox.find('ymin').text) - 1,
                                  int(bbox.find('xmax').text) - 1,
                                  int(bbox.find('ymax').text) - 1]
            objects.append(obj_struct)
        return objects

    cachedir = os.path.join(out_dir, 'labels')
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(image_ids):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(image_ids)))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
    return recs


def check_annots(all_targets, dataset, out_dir):
    voc_root = '/home/sean/data/VOCdevkit'
    annopath = os.path.join(voc_root, 'VOC2007', 'Annotations', '%s.xml')
    imagenames = [dataset.pull_image_name(id_) for id_ in range(len(dataset))]
    image_ids = [os.path.basename(os.path.splitext(name)[0])
                 for name in imagenames]
    # cachefile = '/home/sean/data/VOCdevkit/VOC2007/annotations_cache/annots.pkl'
    recs = get_voc_label_recs(out_dir, annopath, image_ids)

    for cls_id, classname in enumerate(dataset.classes):
        print('Checking "{}" annotations {}/{}'.format(classname, cls_id + 1,
                                                       len(dataset.classes)))
        voc_cls_rec, voc_npos = get_voc_class_dict(classname, image_ids, recs)
        my_cls_rec, m_npos = get_class_dict(cls_id, imagenames, all_targets)

        e_str = 'Different number of class instances. m_npos = {}. voc_npos = {}'
        assert m_npos == voc_npos, e_str.format(m_npos, voc_npos)

        e_str = 'len(my_cls_rec) = {}. len(voc_cls_rec) = {}'
        assert len(my_cls_rec) == len(voc_cls_rec), e_str.format(
            len(my_cls_rec), len(voc_cls_rec))

        for imname, img_id in zip(imagenames, image_ids):
            voc_inst = voc_cls_rec[img_id]
            diff_ind = np.where(voc_inst['difficult'] == False)[0]
            voc_det = [voc_inst['det'][i] for i in diff_ind]
            my_inst = my_cls_rec[imname]

            if voc_inst['det'] == [] and my_inst['det'] == []:
                continue
            if np.all(voc_inst['difficult']) and my_inst['det'] == []:
                continue

            e = 'Dets mismatch. \nmy_inst={}\nvoc_inst={}'.format(
                my_inst['det'], voc_det)
            info = '\nvoc_inst["det"]={}  diff_ind={}'.format(
                voc_inst['det'], diff_ind)
            assert np.array_equal(
                my_inst['det'], voc_det), e + info

            e = 'my bboxes ({}). vox boxes ({})'.format(
                np.shape(my_inst['bbox']), np.shape(voc_inst['bbox'][diff_ind]))
            assert my_inst['bbox'].shape == voc_inst['bbox'][diff_ind].shape, e

            e = '\nmy_inst[bboxes] = {}\nvoc_inst[bboxes][diff_ind] = {}'.format(
                my_inst['bbox'], voc_inst['bbox'][diff_ind])
            vocbb = '\nvoc bboxes = {}'.format(voc_inst['bbox'])
            assert np.any(abs(my_inst['bbox'] - voc_inst['bbox']
                              [diff_ind]) <= 1.0), e + vocbb
    return True


def write_class_labels(all_targets, dataset, out_dir):
    assert os.path.isdir(out_dir), 'Invalid dir: "%s"' % out_dir
    cls_label_path = os.path.join(out_dir, 'class_labels')
    imagenames = [dataset.pull_image_name(id_) for id_ in range(len(dataset))]
    if not os.path.isdir(cls_label_path):
        os.mkdir(cls_label_path)
    fname = os.path.join(cls_label_path, '{}_labels.txt')
    for cls_id, classname in enumerate(dataset.classes):
        my_cls_rec, m_npos = get_class_dict(cls_id, imagenames, all_targets)
        with open(fname.format(classname), 'rb') as j:
            json.dump(m_npos, j)
            json.dumps(my_cls_rec, j, indent=3)
    return cls_label_path


def get_class_dict(cls_id, imagenames, targets):
    '''
    Get class-wise GT label. GT originally from my Extended Pytorch Data iterators
    '''
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        cls_labels = [obj for obj in targets[imagename] if obj[4] == cls_id]
        bbox = np.array([obj[:4] for obj in cls_labels])
        det = [False] * len(cls_labels)
        npos += len(cls_labels)
        class_recs[imagename] = {'bbox': bbox, 'det': det}
    return class_recs, npos


def get_voc_class_dict(classname, imagenames, recs, rm_diff=False):
    '''
    Get class-wise GT VOC labels. From dict loading VOC data directly from .xml
    From:
    https://github.com/amdegroot/ssd.pytorch/blob/master/eval.py
    '''
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        if rm_diff:
            if np.size(difficult) > 0 and np.any(difficult):
                diff_in = np.where(difficult)[0]
                bbox = np.delete(bbox, diff_in, axis=0)
                det = np.array(det)
                det = np.delete(det, diff_in)
                det = det.tolist()
            class_recs[imagename] = {'bbox': bbox,
                                     'det': det}
        else:
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}
    return class_recs, npos


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
        # det_pass = check_dets(all_bboxes, det_dir, data_iter, save_path)
        # anno_pass = check_annots(all_targets, data_iter, save_path)
        # pass_s = '-' * 20 + '\nAll Checks Passed!\n' + '-' * 20
        # fail_s = '*' * 20 + \
        #     '\nChecks FAILED! det_pass={}, anno_pass={}\n'.format(det_pass,
        #                                                           anno_pass)\
        #     + '*' * 20
        # print((pass_s if det_pass and anno_pass else fail_s))
        pass

    print('Calulating Metrics...')

    aps = []
    for i, cls in enumerate(data_iter.classes):
        prec, rec, ap = calcMetrics(
            det_dir, all_targets, cls, data_iter, ovthresh=0.5,
            use_07_metric=use_voc_07)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        fname = os.path.join(save_path, cls + '_pr.txt')
        with open(fname, 'w') as f:
            f.write('Class: {}\nrec: {}\nprec: {}\nap: {}'.format(
                cls, rec, prec, ap))
    print('~~~~~~~~')
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print(' ')
    print('--------------------------------------------------------------')
    print('Results computed with my NEW **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')
    fname = os.path.join(save_path, 'mean_ap.txt')
    with open(fname, 'w') as f:
        f.write('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('Metrics saves to "{}"'.format(save_path))


def calcMetrics(dets_file_dir, gt_labels, classname, dataset,
                ovthresh=0.5, use_07_metric=False):
    cls_id = dataset.classes.index(classname)
    detsfile = os.path.join(dets_file_dir, '{}_dets.txt'.format(classname))
    imagenames = [dataset.pull_image_name(id_) for id_ in range(len(dataset))]

    with open(detsfile, 'r') as f:
        detlines = f.readlines()
    conf, sorted_scores, BB, image_per_det = process_dets_txt(
        detlines, classname)

    pred_ans = os.path.join(dets_file_dir, 'tps_fps_{}.pkl'.format(classname))
    if os.path.isfile(pred_ans):
        with open(pred_ans, 'rb') as f:
            tp_dict = pickle.load(f)
        voc_tp = tp_dict['tps']
        voc_fp = tp_dict['fps']

    voc_root = '/home/sean/data/VOCdevkit'
    annopath = os.path.join(voc_root, 'VOC2007', 'Annotations', '%s.xml')
    recs = get_voc_label_recs(args.save_path, annopath, imagenames)

    im_is = [os.path.splitext(os.path.basename(name))[0]
             for name in imagenames]
    # class_recs, npos = get_voc_class_dict(classname, im_is, recs, rm_diff=False)
    class_recs, npos = get_class_dict(cls_id, imagenames, gt_labels)

    num_dets = len(image_per_det)
    tp = np.zeros(num_dets)
    fp = np.zeros(num_dets)

    for d in range(num_dets):
        if image_per_det[d] in class_recs:
            R = class_recs[image_per_det[d]]
            BBGT = R['bbox'].astype(float)
        else:
            bname = os.path.splitext(os.path.basename(image_per_det[d]))[0]
            R = class_recs[bname]
            BBGT = R['bbox'].astype(float)
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (BBGT[:, 2] - BBGT[:, 0]) *
                   (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if 'difficult' not in R:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            elif not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
        # if voc_fp[d] != fp[d]:
        #     import pdb; pdb.set_trace()
        #     print('fp different!')
        # if voc_tp[d] != tp[d]:
        #     import pdb; pdb.set_trace()
        #     print('tp differnt!')

    if not os.path.isfile(pred_ans):
        # print('saving to {}'.format(pred_ans))
        with open(pred_ans, 'wb') as f:
            p_infos = {'fps': fp, 'tps': tp}
            pickle.dump(p_infos, f)
    # if 'difficult' not in R:
    #     print('No difficult images (not VOC dataset?)')
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap


def main(args):
    img_dim = 300
    set_type = 'test'
    use_voc_07_ap_metric = False

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
