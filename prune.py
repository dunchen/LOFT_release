import argparse
import numpy as np
import os
import time
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms

from data import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tin_data,
    get_svhn_loaders,
    get_imagenet_loaders,
    imagenet_data_prefetcher as data_prefetcher,
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

from models import *

import warnings
warnings.filterwarnings("ignore", message="The given NumPy array")
warnings.filterwarnings("ignore", message="Corrupt EXIF data")

# Prune settings
parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
parser.add_argument('--data', type=str, default='./data/tiny-imagenet-200',
                    help='Path to imagenet validation data')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
# parser.add_argument('-v', default='A', type=str, 
#                     help='version of the pruned model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

# model = WideResNet(28, 20, 0.3, num_classes=10)\
# model = WideResNet18()#out_size=512, num_classes=1000)
model = WideResNet34()#out_size=512, num_classes=1000)
if args.model:
    if os.path.isfile(args.model):
        chkpt = torch.load(args.model)
        # model.load_state_dict(chkpt['state_dict'])
        model.load_state_dict(chkpt['model'].state_dict())
        print("=> loaded state dict '{}'"
              .format(args.model))

if args.cuda:
    model.cuda()

# model = torch.nn.DataParallel(model).cuda()
# cudnn.benchmark = True

print('Pre-processing Successful!')

# test(specs,args, ist_model, device, test_loader, epoch, num_sync, test_loss_log, test_acc_log)
# def test(specs,args, ist_model: BaselineResNetModel, device, test_loader, epoch, num_sync, test_loss_log, test_acc_log):
def test(model):
    # ist_model.base_model.eval()
    model.eval()
    trn_dl, test_dl = get_cifar10_loaders()
    prefetcher = data_prefetcher(test_dl)
    model_in, labels = prefetcher.next()
    agg_val_loss = 0.
    num_correct = 0.
    total_ex = 0.
    criterion = torch.nn.CrossEntropyLoss()
    while model_in is not None:
        #print(labels)
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
    agg_val_loss /= len(test_dl)
    val_acc = num_correct/total_ex
    print("Test Loss: {:.6f}; Test Accuracy: {:.4f}.\n".format(agg_val_loss, val_acc))

test(model)

prune_prob = [0.5, 0.5, 0.5, 0.5]

def prune_filter_indices(layer, block_idx, prune_perc):
    # assert layer[block_idx].conv1.out_channels == layer[block_idx].conv1.in_channels
    num_filters = layer[block_idx].conv1.out_channels

    with torch.no_grad():
        l2_vals = torch.sum((layer[block_idx].conv1.weight)**2, dim=[1, 2, 3])
        sorted_filt_inds = torch.argsort(l2_vals).cpu().numpy()

    assert sorted_filt_inds.shape[0] == num_filters
    site_filters = []
    new_len = int(num_filters * (1 - prune_perc))

    site_filters = sorted_filt_inds[:new_len]
    # random.shuffle(site_filters)  # DO NOT UNCOMMENT! THIS IS FOR RANDOM PRUNING
    site_filters = torch.LongTensor(site_filters)
    return site_filters

def create_partitions(orig_model):
    # create the site models if they do not exist yet
    # if not len(self.site_models):
    #     self.instantiate_site_models()

    # I decided not to partition the first layer of convs because these are
    # likely more sensitive to pruning -- they are also more narrow so it
    # is not a big deal to not prune them
    layer1_partition_vectors = []
    for block_idx in range(1, len(orig_model.layer1)):
        # assert (
        #         orig_model.layer1[block_idx].conv1.in_channels ==
        #         orig_model.layer1[block_idx].conv1.out_channels)
        num_filt = orig_model.layer1[block_idx].conv1.out_channels
        conv_partition = prune_filter_indices(
            orig_model.layer1, block_idx, prune_prob[0])

        partition_tensor = torch.zeros(num_filt)
        partition_tensor[conv_partition] = 1.
        # if args.cuda:
        #     partition_tensor = partition_tensor.cuda()
        layer1_partition_vectors.append(partition_tensor)

    # layer2 partition
    layer2_partition_vectors = []
    for block_idx in range(1, len(orig_model.layer2)):
        # assert (
        #         orig_model.layer2[block_idx].conv1.in_channels ==
        #         orig_model.layer2[block_idx].conv1.out_channels)
        num_filt = orig_model.layer2[block_idx].conv1.out_channels
        conv_partition = prune_filter_indices(
            orig_model.layer2, block_idx, prune_prob[1])

        partition_tensor = torch.zeros(num_filt)
        partition_tensor[conv_partition] = 1.
        # if args.cuda:
        #     partition_tensor = partition_tensor.cuda()
        layer2_partition_vectors.append(partition_tensor)

    # layer3 partition
    layer3_partition_vectors = []
    for block_idx in range(1, len(orig_model.layer3)):
        # assert (
        #         orig_model.layer3[block_idx].conv1.in_channels ==
        #         orig_model.layer3[block_idx].conv1.out_channels)
        num_filt = orig_model.layer3[block_idx].conv1.out_channels
        conv_partitions = prune_filter_indices(
            orig_model.layer3, block_idx, prune_prob[2])
            
        partition_tensor = torch.zeros(num_filt)
        partition_tensor[conv_partition] = 1.
        # if args.cuda:
        #     partition_tensor = partition_tensor.cuda()
        layer3_partition_vectors.append(partition_tensor)

    # layer4 partition
    layer4_partition_vectors = []
    for block_idx in range(1, len(orig_model.layer4)):
        # assert (
        #         orig_model.layer4[block_idx].conv1.in_channels ==
        #         orig_model.layer4[block_idx].conv1.out_channels)
        num_filt = orig_model.layer4[block_idx].conv1.out_channels
        conv_partitions = prune_filter_indices(
            orig_model.layer4, block_idx, prune_prob[3])
            
        partition_tensor = torch.zeros(num_filt)
        partition_tensor[conv_partition] = 1.
        # if args.cuda:
        #     partition_tensor = partition_tensor.cuda()
        layer4_partition_vectors.append(partition_tensor)

    part = [layer1_partition_vectors, layer2_partition_vectors, layer3_partition_vectors, layer4_partition_vectors]
    return part

def dropout_site_filters(orig_model, partition):
    with torch.no_grad():
        # nothing from first convolution or final batch norm/linear layer
        # needs to be dropped/zeroed out
        # layer1
        for block_idx, block in enumerate(orig_model.layer1[1:]):
            if len(partition[0]):
                # assert len(block.downsample) == 0. # should not be strided
                assert block.downsample is None
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
                # assert len(block.downsample) == 0. # should not be strided
                assert block.downsample is None
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
                # assert len(block.downsample) == 0. # should not be strided
                assert block.downsample is None
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
        
        # layer4
        for block_idx, block in enumerate(orig_model.layer4[1:]):
            if len(partition[3]):
                # assert len(block.downsample) == 0. # should not be strided
                assert block.downsample is None
                part_vec = partition[3][block_idx]
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

part = create_partitions(model)
dropout_site_filters(model, part)
test(model)

torch.save({'prune': part, 'state_dict': model.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))