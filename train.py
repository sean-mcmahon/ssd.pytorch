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


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2',
                    help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int,
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
parser.add_argument('--log_iters', default=True, type=bool,
                    help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='/home/sean/Documents/ssd/',
                    help='Location to save checkpoint models')
parser.add_argument('--weights_folder',
                    default='/home/sean/src/ssd_pytorch/weights/')
parser.add_argument('--data_root', default=VOCroot,
                    help='Location of VOC root directory')
parser.add_argument('--dataset', default='voc', type=str)
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = (v1, v2)[args.version == 'v2']

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
save_weights = os.path.join(args.save_folder, 'weights')
if not os.path.isdir(save_weights):
    os.mkdir(save_weights)

ssd_dim = 300  # only support 300 now
batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = args.iterations
if args.dataset == 'mining':
    stepvalues = (150, 300, 450, 600)
else:
    stepvalues = (80000, 100000, 120000)

train_sets = {'voc': [('2007', 'trainval'), ('2012', 'trainval')],
              'mining': 'split2/train_gopro2_scraped.json'}
rgb_means = {'voc': (104, 117, 123), 'mining': (65, 69, 76)}
data_iters = {'voc': VOCDetection, 'mining': MiningDataset}
augmentators = {'voc': SSDAugmentation(ssd_dim, rgb_means['voc']),
                'mining': SSDMiningAugmentation(ssd_dim, rgb_means['mining'])}
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

if args.visdom:
    path = os.path.join(args.save_folder, 'runs')
    if not os.path.isdir(path):
        os.mkdir(path)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logname = "{}_{}_ssd{}_{}".format(current_time, socket.gethostname(),
                                      ssd_dim, args.dataset)
    logdir = os.path.join(path, logname)
    print('Logging run to "%s"' % logdir)
    # if not os.path.isdir(logdir):
    writer = SummaryWriter(log_dir=logdir)
    # else:
    # writer = SummaryWriter(log_dir=logdir, comment='ssd{}_{}'.format(ssd_dim, args.dataset))

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
    weights_n = os.path.join(args.weights_folder, args.basenet)
    assert os.path.isfile(weights_n), 'Invalid "%s"' % weights_n
    vgg_weights = torch.load(weights_n)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net = net.cuda()

if args.visdom:
    dummy_in = Variable(torch.rand(batch_size, 3, ssd_dim, ssd_dim))
    writer.add_graph(net, (dummy_in, ))
    # with SummaryWriter(comment='ssd300') as w:
    #     w.add_graph(net, (dummy_in, ), verbose=True)
    # with SummaryWriter(comment='ssd300onix') as w:
    #     torch.onnx.export(net, dummy_in, "test.proto", verbose=True)
    #     w.add_graph_onnx("test.proto")


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


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
            if args.visdom:
                l_d = {'total_loss': loc_loss + conf_loss, 'loc_loss': loc_loss,
                       'conf_loss': conf_loss}
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
            if args.visdom:
                for name, param in net.named_parameters():
                    writer.add_histogram(
                        name.replace('.', '/'), param.clone().cpu().data.numpy(), iteration)

            if args.visdom:
                if args.send_images_to_visdom:
                    if batch_size > 10:
                        imgx = vutils.make_grid(images.data.cpu()[:10, :, :],
                                                normalize=True, scale_each=True)
                    else:
                        imgx = vutils.make_grid(images.data.cpu(),
                                                normalize=True, scale_each=True)
                    random_batch_index = np.random.randint(images.size(0))

                    writer.add_image(
                        'aug_image', imgx, iteration)
        if args.visdom:
            total_loss = loss_l.data[0] + loss_c.data[0]
            losses_d = {'total_loss': total_loss, 'loc_loss': loss_l.data[0],
                        'conf_loss': loss_c.data[0]}
            writer.add_scalars('loss/l_per_iter', losses_d, iteration)
        if iteration % 100 == 0:
            print('Saving state, iter:', iteration)
            sstr = os.path.join(
                save_weights, 'ssd{}_0712_{}_{}.pth'.format(
                    str(ssd_dim), repr(iteration), args.dataset))
            torch.save(ssd_net.state_dict(), sstr)
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
    train()
