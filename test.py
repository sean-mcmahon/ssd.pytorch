from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot, VOC_CLASSES
from PIL import Image, ImageDraw, ImageFont
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data.mining import MiningDataset, MiningAnnotationTransform, MINING_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    else:
        return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--data_root', default=VOCroot,
                    help='Location of Dataset root directory')
parser.add_argument('--dataset', default='voc')
parser.add_argument('--test_split', default='test_gopro2.json',
                    help='The data split to use, train, val or test.' +
                    ' Not used with VOC dataset')
parser.add_argument('--vis_preds', default=True, type=str2bool,
                    help='If true save images with network detections')

args = parser.parse_args()
labelmaps = {'voc': VOC_CLASSES, 'mining': MINING_CLASSES}
labelmap = labelmaps[args.dataset]


def test_net(save_folder, net, cuda, testset, transform, thresh,
             save_pred_img=False):
    # dump predictions and assoc. ground truth to text file for now
    filename = os.path.join(save_folder, 'test1.txt')
    if os.path.isfile(filename):
        os.remove(filename)
    num_images = len(testset)
    print('\nPredictions based on threshold of ' +
          '{}. {} saving prediction images...'.format(
              thresh, 'Will be' if save_pred_img else 'NOT'))
    print('-'*50)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: ' + img_id + '\n')
            for box in annotation:
                f.write('label: ' + ' || '.join(str(b) for b in box) + '\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        pred_num = 0
        bboxes = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: ' + '\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i - 1]
                # if 'mine_vehicle' in label_name:
                #     continue
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = [pt[0], pt[1], pt[2], pt[3]]
                pred_num += 1
                bboxes.append(coords + [label_name] + [score])
                # print('Prediction! Theres a {} at {}'.format(label_name, coords))
                with open(filename, mode='a') as f:
                    f.write(str(pred_num) + ' label: ' + label_name + ' score: ' +
                            str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
                j += 1
        if bboxes and bboxes[0] is not None:
            im_name = os.path.splitext(os.path.basename(img_id))[0]
            while im_name.endswith('.png'):
                im_name = os.path.splitext(im_name)[0]
            if save_pred_img:
                visRes(bboxes, img, os.path.join(
                    save_folder, 'preds'), im_name, conf=True)
        else:
            print('No predictions for this image ({})'.format(img_id))


def visResults(detections, images, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    assert len(detections) == len(images)
    print('Visualising {} Images'.format(len(images)))
    for count, det, img in zip(range(len(images)), detections, images):
        visRes(det, img, save_path, str(count))
        if count % 10 == 0:
            print('Saved %d/%d images' % (count, len(images)))


def visRes(det, img, save_path, name, conf=False):
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 25)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    assert isinstance(det, list)
    if isinstance(img, np.ndarray):
        img = img[:, :, (2, 1, 0)]
        img = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for bbox in det:
        draw.rectangle(bbox[0:4], outline=(255, 0, 0))
        if conf:
            txt = " ".join((bbox[4], str(round(bbox[5], 3))))
            draw.text(bbox[0:2], txt, font=fnt, fill=(0, 200, 0))
        else:
            draw.text(bbox[0:2], bbox[4], font=fnt, fill=(0, 200, 0))
    img.save(os.path.join(save_path, 'img_%s.png' % name))

if __name__ == '__main__':
    save_folder = args.save_folder + '_%s' % args.dataset
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # load net
     # +1 background
    num_classes = {'voc': len(VOC_CLASSES) + 1,
                   'mining': len(MINING_CLASSES) + 1}
    means = {'voc': (104, 117, 123), 'mining': (65, 69, 76)}
    net = build_ssd('test', 300, num_classes[args.dataset])  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    split = {'voc': [('2007', 'test')], 'mining': args.test_split}
    anno = {'voc': AnnotationTransform(), 'mining': MiningAnnotationTransform()}
    datasets = {
        'voc': VOCDetection, 'mining': MiningDataset}
    testset = datasets[args.dataset](args.data_root, split[args.dataset],
                                     None, anno[args.dataset])
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(save_folder, net, args.cuda, testset,
             BaseTransform(net.size, means[args.dataset]),
             thresh=args.visual_threshold, save_pred_img=args.vis_preds)
