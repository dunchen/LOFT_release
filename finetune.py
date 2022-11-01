import argparse
import os
import numpy as np
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from data import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tin_data,
    get_svhn_loaders,
    get_imagenet_loaders,
    imagenet_data_prefetcher as data_prefetcher,
)

from models import *
# from model import *

import warnings
warnings.filterwarnings("ignore", message="The given NumPy array")
warnings.filterwarnings("ignore", message="Corrupt EXIF data")

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=25, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--grad-accu', '-ga', default=4, type=int,
                    metavar='N', help='gradient accumulation (default: 1)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--s', type=float, default=0,
                    help='scale sparse rate (default: 0)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='the PATH to pruned model')

best_prec1 = 0

# =================================== Training ===============================

def main():
    global args, best_prec1
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic=True
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    # args.distributed = args.world_size > 1

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # if args.distributed:
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size)

    if args.refine:
        if os.path.isfile(args.refine):
            checkpoint = torch.load(args.refine)
            model = WideResNet34()#out_size=512, num_classes=1000)
            # model = ResNet101(out_size=512, num_classes=1000)
            pruned_filter = checkpoint['prune']
            print("successful loaded " + args.refine)
        else:
            print("ERROR: cannot load file!")

    # if not args.distributed:
    #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         model = torch.nn.DataParallel(model).cuda()
    # else:
    #     model.cuda()
    #     model = torch.nn.parallel.DistributedDataParallel(model)

    if args.refine:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("starting from new model")
        model = WideResNet34()#out_size=512, num_classes=1000)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=False)
        # model = ResNet101(out_size=512, num_classes=1000)
    
    if args.cuda:
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    trn_dl, test_dl = get_cifar10_loaders()

    if args.evaluate:
        validate(test_dl, model, criterion)
        return

    history_score = np.zeros((args.epochs + 1, 1))
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
    for epoch in range(args.start_epoch, args.epochs):
        print("starting epoch " + str(epoch))

        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(trn_dl, model, criterion, optimizer, epoch, pruned_filter)

        # evaluate on validation set
        prec1 = validate(test_dl, model, criterion)
        # prec1 = prec1.cpu()
        history_score[epoch] = prec1
        np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save, epoch)

    history_score[-1] = best_prec1
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')

def train(trn_dl, model, criterion, optimizer, epoch, pruned_filter):
    print("training epoch " + str(epoch))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    prefetcher = data_prefetcher(trn_dl)
    model_in, labels = prefetcher.next()

    # switch to train mode
    model.train()

    t = time.time()
    end = time.time()
    agg_val_loss = 0.
    num_correct = 0.
    total_ex = 0.
    while model_in is not None:
    # for i, (model_in, labels) in enumerate(trn_dl):
        # if(i % 100 == 0):
        #     print(str(i) + "/" + str(len(trn_dl)))
        #     print(time.time() - t)
        # # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            model_in = model_in.cuda()
            labels = labels.cuda()

        # compute output
        model_output = model(model_in)
        val_loss = criterion(model_output, labels)

        # measure accuracy and record loss
        agg_val_loss += val_loss.item()
        _, preds = model_output.max(1)
        total_ex += labels.size(0)
        num_correct += preds.eq(labels).sum().item()

        # compute gradient and do SGD step
        val_loss = val_loss / args.grad_accu
        val_loss.backward()
        # if (i + 1) % args.grad_accu == 0:
        optimizer.step()
        dropout_site_filters(model, pruned_filter)
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        model_in, labels = prefetcher.next()

    agg_val_loss /= len(trn_dl)
    val_acc = num_correct/total_ex
    print(f'Epoch {epoch}: Full Model Valid Acc. {num_correct/total_ex:.4f}, time {time.time() - t}')

def validate(test_dl, model, criterion):
    model.eval()
    prefetcher = data_prefetcher(test_dl)
    model_in, labels = prefetcher.next()

    agg_val_loss = 0.
    num_correct = 0.
    total_ex = 0.
    while model_in is not None:
        if args.cuda:
            model_in = model_in.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            model_output = model(model_in)
            val_loss = criterion(model_output, labels)
        agg_val_loss += val_loss.item()
        _, preds = model_output.max(1)
        total_ex += labels.size(0)
        num_correct += preds.eq(labels).sum().item()
        model_in, labels = prefetcher.next()
    # model.cpu() # clear space on the GPU -- don't want to leave all models there
    agg_val_loss /= len(test_dl)
    val_acc = num_correct/total_ex
    print(f'Full Model Valid Acc. {num_correct/total_ex:.4f}')
    return val_acc

def save_checkpoint(state, is_best, filepath, epoch):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if epoch > 30:
        lr /= 10
    if epoch > 60:
        lr /= 10
    if epoch > 80:
        lr /= 10
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def dropout_site_filters(orig_model, partition):
    with torch.no_grad():
        # nothing from first convolution or final batch norm/linear layer
        # needs to be dropped/zeroed out
        # layer1
        for block_idx, block in enumerate(orig_model.layer1[1:]):
            if len(partition[0]):
                # assert len(block.shortcut) == 0. # should not be strided
                block.downsample == None
                part_vec = partition[0][block_idx]
                if args.cuda:
                    part_vec = part_vec.cuda()
                # print(block.conv1.weight.shape)
                # first batch norm layer should not have anything removed
                # first conv layer has output channels pruned
                block.conv1.weight.mul_(
                    part_vec[:, None, None, None])
                # second batch norm is pruned along with output channels
                block.bn2.weight.mul_(part_vec)
                block.bn2.bias.mul_(part_vec)
                # the input channels of the second conv are pruned
                block.conv2.weight.mul_(
                    part_vec[None, :, None, None])
        # layer2
        for block_idx, block in enumerate(orig_model.layer2[1:]):
            if len(partition[1]):
                # assert len(block.shortcut) == 0. # should not be strided
                assert block.downsample == None # should not be strided
                part_vec = partition[1][block_idx]
                if args.cuda:
                    part_vec = part_vec.cuda()
                # first batch norm layer should not have anything removed
                # first conv layer has output channels pruned
                block.conv1.weight.mul_(
                    part_vec[:, None, None, None])
                # second batch norm is pruned along with output channels
                block.bn2.weight.mul_(part_vec)
                block.bn2.bias.mul_(part_vec)
                # the input channels of the second conv are pruned
                block.conv2.weight.mul_(
                    part_vec[None, :, None, None])
        
        # layer3
        for block_idx, block in enumerate(orig_model.layer3[1:]):
            if len(partition[2]):
                # assert len(block.shortcut) == 0. # should not be strided
                block.downsample == None
                part_vec = partition[2][block_idx]
                if args.cuda:
                    part_vec = part_vec.cuda()
                # first batch norm layer should not have anything removed
                # first conv layer has output channels pruned
                block.conv1.weight.mul_(
                    part_vec[:, None, None, None])
                # second batch norm is pruned along with output channels
                block.bn2.weight.mul_(part_vec)
                block.bn2.bias.mul_(part_vec)
                # the input channels of the second conv are pruned
                block.conv2.weight.mul_(
                    part_vec[None, :, None, None])


if __name__ == '__main__':
    main()
