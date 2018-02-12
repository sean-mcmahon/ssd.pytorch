import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision.utils as vutils
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from data.mining import MiningDataset, MiningAnnotationTransform
from utils.augmentations import SSDAugmentation, SSDMiningAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time
import subprocess
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
import atexit


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def writeHyperparams(writer, args, others=None):
    fam = '/params'
    for a in vars(args):
        litfam = os.path.join(fam, a)
        writer.add_text(litfam, str(getattr(args, a)))
    if others:
        for key, item in others.items():
            litfam = os.path.join(fam, key)
            writer.add_text(litfam, str(item))


def writeParamsTxt(filename, args, others=None):
    with open(filename, 'w') as f:
        for a in vars(args):
            arg_val = str(getattr(args, a))
            f.write('{}: {}\n'.format(a, arg_val))
        if others is not None:
            for key, item in others.items():
                f.write('{}: {}\n'.format(key, str(item)))


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0

    print('Using "{}" Data'.format(dataset.__class__.__name__))

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    batch_iterator = None
    data_loader = data.DataLoader(
        dataset, batch_size, num_workers=args.num_workers,
        shuffle=True, collate_fn=detection_collate, pin_memory=True)
    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            new_lr = optimizer.param_groups[0]['lr']
            l_d = {'total_loss': loc_loss + conf_loss, 'loc_loss': loc_loss,
                   'conf_loss': conf_loss, 'learning_rate': new_lr}
            writer.add_scalars('loss/l_per_stepval', l_d, epoch)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True)
                       for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (
                loss.data[0]), end=' ')
        if iteration % epoch_size == 0:
            epoch_n = iteration / epoch_size
            print('\n--> Epoch ' + repr(epoch_n) + ' ++ Loss: %.4f ++' % (
                loss.data[0]) + ' ({} iterations)'.format(iteration), end=' ')
            total_loss = loss_l.data[0] + loss_c.data[0]
            losses_d = {'total_loss': total_loss, 'loc_loss': loss_l.data[0],
                        'conf_loss': loss_c.data[0]}
            writer.add_scalars('loss/l_per_epoch', losses_d, epoch_n)

        total_loss = loss_l.data[0] + loss_c.data[0]
        losses_d = {'total_loss': total_loss, 'loc_loss': loss_l.data[0],
                    'conf_loss': loss_c.data[0]}
        writer.add_scalars('loss/l_per_iter', losses_d, iteration)
        if iteration % 1000 == 0 and iteration > 0:
            for name, param in net.named_parameters():
                writer.add_histogram(
                    name.replace('.', '/'),
                    param.clone().cpu().data.numpy(), iteration)
            if args.send_images_to_tb:
                if batch_size > 10:
                    imgx = vutils.make_grid(images.data.cpu()[:10, :, :],
                                            normalize=True, scale_each=True)
                else:
                    imgx = vutils.make_grid(images.data.cpu(),
                                            normalize=True, scale_each=True)
                writer.add_image(
                    'aug_image', imgx, iteration)
            print('\nSaving state, iter:', iteration, end=' ')
            sstr = os.path.join(
                save_weights, 'ssd{}_0712_{}_{}.pth'.format(
                    str(ssd_dim), repr(iteration), args.dataset))
            torch.save(ssd_net.state_dict(), sstr)
            # eval_network(net.eval(), eval_iter, writer)
            # net.train()
    torch.save(ssd_net.state_dict(), save_weights +
               'ssd' + str(ssd_dim) + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training')
    parser.add_argument('--version', default='v2',
                        help='conv11_2(v2) or pool6(v1) as last layer')
    parser.add_argument('--ssd_dim', default=300, type=int)
    parser.add_argument('--basenet', default='/home/sean/src/ssd_pytorch/weights/vgg16_reducedfc.pth',
                        help='pretrained base model')
    parser.add_argument('--jaccard_threshold', default=0.5,
                        type=float, help='Min Jaccard index for matching')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Resume from checkpoint')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--iterations', default=120000, type=int,
                        help='Number of training iterations')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Begin counting iterations starting from this value (should be used with resume)')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-3,
                        type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--tb', '--tensorboard', default=False, type=str2bool,
                        help='Startup Tensorboard thread for visualisation')
    parser.add_argument('--send_images_to_tb', type=str2bool, default=False,
                        help='Sample a random image from each 10th batch,' +
                        ' send it to tensorboard after augmentations step')
    parser.add_argument('--save_folder', default='/home/sean/Documents/ssd/',
                        help='Location to save checkpoint models')
    parser.add_argument('--data_root', default=VOCroot,
                        help='Location of VOC root directory')
    parser.add_argument('--dataset', default='voc', type=str)
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    cfg = (v1, v2)[args.version == 'v2']

    ssd_dim = args.ssd_dim  # only support 300 now
    assert ssd_dim == 300
    batch_size = args.batch_size
    accum_batch_size = 32
    iter_size = accum_batch_size / batch_size
    max_iter = args.iterations
    if args.dataset == 'mining':
        stepvalues = (150, 300, 450, 600)
    else:
        stepvalues = (80000, 100000, 120000)

    train_sets = {'voc': [('2007', 'trainval'), ('2012', 'trainval')],
                  'mining': 'train_gopro1_scraped_all_labelled.json'}
    rgb_means = {'voc': (104, 117, 123), 'mining': (65, 69, 76)}
    data_iters = {'voc': VOCDetection, 'mining': MiningDataset}
    augmentators = {'voc': SSDAugmentation(ssd_dim, rgb_means['voc']),
                    'mining': SSDMiningAugmentation(ssd_dim,
                                                    rgb_means['mining'])}
    target_transforms = {'voc': AnnotationTransform,
                         'mining': MiningAnnotationTransform}

    print('Loading Dataset...')
    assert os.path.exists(args.data_root), 'root invalid "%s"' % args.data_root
    dataset = data_iters[args.dataset](
        args.data_root, train_sets[args.dataset], augmentators[args.dataset],
        target_transforms[args.dataset]())

    num_classes = dataset.num_classes()

    if os.path.isdir('/home/sean'):
        h_dir = '/home/sean'
    else:
        h_dir = '/home/n8307628'

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    jobname = "ssd{}_{}_{}_{}".format(
        ssd_dim, args.dataset, current_time, socket.gethostname())
    job_path = os.path.join(args.save_folder, jobname)
    save_weights = os.path.join(job_path, 'weights')
    if not os.path.isdir(save_weights):
        os.makedirs(save_weights)

    logpath = os.path.join(job_path, 'runs')
    if not os.path.isdir(logpath):
        os.makedirs(logpath)

    print('Logging run to "%s"' % logpath)
    # if not os.path.isdir(logpath):
    writer = SummaryWriter(log_dir=logpath)
    if args.tb:
        cmd = ['tensorboard', '--logdir', logpath]
        process = subprocess.Popen(cmd)
        # Kill subprocess on script error
        atexit.register(process.terminate)
        pid = process.pid
        # else:
    writeHyperparams(writer, args, {'stepvalues': stepvalues})
    writeParamsTxt(os.path.join(job_path, 'params.txt'), args,
                   {'stepvalues': stepvalues})

    ssd_net = build_ssd('train', 300, num_classes)
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        assert os.path.isfile(args.resume), 'Invalid "%s"' % args.resume
        ssd_net.load_weights(args.resume)
    else:
        assert os.path.isfile(args.basenet), 'Invalid "%s"' % args.basenet
        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    # dummy_in = Variable(torch.rand(batch_size, 3, ssd_dim, ssd_dim))
    # writer.add_graph(net, (dummy_in, ))
    # with SummaryWriter(comment='ssd300') as w:
    #     w.add_graph(net, (dummy_in, ), verbose=True)
    # with SummaryWriter(comment='ssd300onix') as w:
    #     torch.onnx.export(net, dummy_in, "test.proto", verbose=True)
    #     w.add_graph_onnx("test.proto")

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0,
                             True, 3, 0.5, False, args.cuda)

    train()
