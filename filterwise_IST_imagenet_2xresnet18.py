import torch.nn.functional as functional
import torch.distributed as dist
import numpy as np
import argparse
import torch
import time
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10
from random import shuffle, choice, seed
import random
import torch.nn as nn
from data import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tin_data,
    get_svhn_loaders,
    get_imagenet_loaders,
    imagenet_data_prefetcher as data_prefetcher,
)
from utils import get_demon_momentum, aggregate_resnet_optimizer_statistics
print(torch.__version__)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="6, 7"
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform_(m.weight, gain=np.sqrt(2))
#         #init.constant_(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         init.constant_(m.weight, 1)
#         #init.constant_(m.bias, 0)

# class wide_basic(nn.Module):
#     def __init__(self, in_planes, planes, dropout_rate, stride=1, partition_num_sites=1):
#         super(wide_basic, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes, affine=True, track_running_stats=False)
#         self.conv1 = nn.Conv2d(in_planes, int(planes/partition_num_sites), kernel_size=3, padding=1, bias=False)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.bn2 = nn.BatchNorm2d(int(planes/partition_num_sites), track_running_stats=False)
#         self.conv2 = nn.Conv2d(int(planes/partition_num_sites), planes, kernel_size=3, stride=stride, padding=1, bias=False)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
#             )

#     def forward(self, x):
#         out = self.dropout(self.conv1(F.relu(self.bn1(x))))
#         out = self.conv2(F.relu(self.bn2(out)))
#         out += self.shortcut(x)
#         return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=dilation, groups=groups, bias=False, dilation=dilation)

class wide_basic(torch.nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, widen_factor=1,partition_num_sites=1):
        super(wide_basic, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes, affine=True, track_running_stats=True)
        self.conv1 = torch.nn.Conv2d(in_planes, int(planes*widen_factor/partition_num_sites), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(int(planes*widen_factor/partition_num_sites), affine=True, track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(int(planes*widen_factor/partition_num_sites), planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = functional.relu(self.bn1(x))
        shortcut = self.downsample(out) if self.downsample is not None else x
        out = self.conv1(out)
        out = self.conv2(functional.relu(self.bn2(out)))
        out += shortcut
        return out

class WideResNet(torch.nn.Module):
    # taken from https://github.com/kuangliu/pytorch-cifar

    def __init__(self, num_blocks, widen_factors, out_size=512, num_classes=10, block=wide_basic):
        super(WideResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        # self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,widen_factor=widen_factors[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,widen_factor=widen_factors[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,widen_factor=widen_factors[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,widen_factor=widen_factors[3])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = torch.nn.Linear(out_size*block.expansion, num_classes)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, widen_factor):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,widen_factor=widen_factor))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
       # out = self.conv1(x)
       # out = self.layer1(out)
       # out = self.layer2(out)
       # out = self.layer3(out)
       # out = self.layer4(out)
       # out = functional.avg_pool2d(out, 4)
       # out = out.view(out.size(0), -1)
       # out = self.fc(out)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class subWideResNet(torch.nn.Module):
    # taken from https://github.com/kuangliu/pytorch-cifar

    def __init__(self, num_blocks, widen_factors, partition_num_sites,out_size=512, num_classes=10,block=wide_basic):
        super(subWideResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,widen_factor=widen_factors[0], partition_num_sites=partition_num_sites[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,widen_factor=widen_factors[1], partition_num_sites=partition_num_sites[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,widen_factor=widen_factors[2], partition_num_sites=partition_num_sites[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,widen_factor=widen_factors[3], partition_num_sites=partition_num_sites[3])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = torch.nn.Linear(out_size*block.expansion, num_classes)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, widen_factor,partition_num_sites):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if stride==1 and partition_num_sites>1:
                layers.append(block(self.in_planes, planes, stride,widen_factor=widen_factor,partition_num_sites=partition_num_sites))
            else:
                layers.append(block(self.in_planes, planes, stride,widen_factor=widen_factor,partition_num_sites=1))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
       # out = self.conv1(x)
       # out = self.layer1(out)
       # out = self.layer2(out)
       # out = self.layer3(out)
       # out = self.layer4(out)
       # out = functional.avg_pool2d(out, 4)
       # out = out.view(out.size(0), -1)
       # out = self.fc(out)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def WideResNet18(blocks=[2, 2, 2, 2], widen_factors=[2,2,2,2], out_size=512, num_classes=1000):
    return WideResNet(num_blocks=blocks, widen_factors=widen_factors, out_size=512, num_classes=1000, block=wide_basic)

def subWideResNet18(num_sites,blocks=[2, 2, 2, 2], widen_factors=[2,2,2,2],out_size=512, num_classes=1000):
    partition_num_sites=[1,num_sites,num_sites,num_sites]
    return subWideResNet(num_blocks=blocks, widen_factors=widen_factors, partition_num_sites=partition_num_sites,out_size=512, num_classes=1000, block=wide_basic)



test_rank=0
test_total_time=0


def broadcast_weight(para, rank_list=None,source=0):
    if rank_list is None:
        group = dist.group.WORLD
    else:
        group = dist.new_group(rank_list)
    #group = dist.group.WORLD
    

    #test_start_time=time.time()
    dist.broadcast(para, src=source, group=group, async_op=False)
    #test_end_time=time.time()
    #global test_total_time
    #test_total_time+=local_test_end-local_test_start

    if rank_list is not None:
       dist.destroy_process_group(group)

def broadcast_module_itr(module:torch.nn.Module, rank_list=None,source=0):
    # if rank_list is None:
    #     group = dist.group.WORLD
    # else:
    #     group = dist.new_group(rank_list)
    group = dist.new_group([0,1])

    #test_start_time=time.time()
    for para in module.parameters():
        dist.broadcast(para.data, src=source, group=group, async_op=False)
    #test_end_time=time.time()
    #global test_total_time
    #test_total_time+=local_test_end-local_test_start

    # if rank_list is not None:
    #     dist.destroy_process_group(group)
    dist.destroy_process_group(group)

def reduce_module(specs, args, module:torch.nn.Module, rank_list=None):
    if rank_list is None:
        raise 'error'
    else:
        group = dist.new_group(rank_list)
    for para in module.parameters():
        dist.reduce(para.data, dst=min(rank_list), op=dist.ReduceOp.SUM, group=group)
        if args.rank == min(rank_list): # compute average
            if rank_list is None:
                para.data = para.data.div_(specs['world_size'])
            else:
                para.data = para.data.div_(len(rank_list))
    if rank_list is not None:
        dist.destroy_process_group(group)

def all_reduce_module(specs, args, module:torch.nn.Module, rank_list=None):
    
    group = dist.group.WORLD
    for para in module.parameters():
        dist.all_reduce(para.data, op=dist.ReduceOp.SUM, group=group)
        # if args.rank == 0: # compute average
        #     if rank_list is None:
        #         para.data = para.data.div_(specs['world_size'])
        #     else:
        #         para.data = para.data.div_(len(rank_list))
        if rank_list is None:
            para.data = para.data.div_(specs['world_size'])
        else:
            para.data = para.data.div_(len(rank_list))
    if rank_list is not None:
        dist.destroy_process_group(group)

class ISTResNetModel():
    def __init__(self, args, device, num_sites=4, min_blocks_per_site=0):

        self.base_model = subWideResNet18(num_sites=num_sites).to(device)
        if args.rank==0:
            self.whole_model = WideResNet18().to(device)
        self.device=device
        self.min_filter_ratio = None
        self.smart = False

        self.min_blocks_per_site = min_blocks_per_site
        self.num_sites = num_sites
        self.site_indices = None
        if min_blocks_per_site == 0:
            self.scale_constant=1.0/num_sites
        else:
            # dropout prob becomes total blocks per site / total blocks in layer3

            self.scale_constant = max(1.0/num_sites, min_blocks_per_site/22)

        # self.base_parameter_list=[]
        # self.base_parameter_list.extend(list(self.base_model.conv1.parameters()))
        # self.base_parameter_list.extend(list(self.base_model.layer1.parameters()))
        # self.base_parameter_list.extend(list(self.base_model.layer2.parameters()))
        # self.base_parameter_list.extend(list(self.base_model.layer4.parameters()))
        # self.base_parameter_list.extend(list(self.base_model.fc.parameters()))
        # self.base_parameter_list.extend(list(self.base_model.layer3[0].parameters()))
        self.layer_server_list=[]

    def sample_filter_indices(self, layer, block_idx, min_filter_ratio=None):
        #assert layer[block_idx].conv1.out_channels == layer[block_idx].conv1.in_channels
        num_filters = layer[block_idx].conv1.out_channels
        if self.smart:
            with torch.no_grad():
                l2_vals = torch.sum((layer[block_idx].conv1.weight)**2, dim=[1, 2, 3])
                sorted_filt_inds = torch.argsort(l2_vals).cpu().numpy()
        else:
            sorted_filt_inds = [x for x in range(num_filters)]
            random.shuffle(sorted_filt_inds)
            sorted_filt_inds = torch.tensor(sorted_filt_inds)
        assert sorted_filt_inds.shape[0] == num_filters
        site_filters = [[] for x in range(self.num_sites)]
        if min_filter_ratio is None or not min_filter_ratio > 0.:
            site_idx = 0
            for ind in sorted_filt_inds:
                #site_idx = np.random.choice([x for x in range(self.num_sites)])
                site_idx = site_idx % self.num_sites
                site_filters[site_idx].append(ind)
                site_idx += 1
        else:
            # assign filters until the minimum ratio is satisfied for all sites
            site_idx = 0
            filt_ind = 0
            total_filters = int(min_filter_ratio*num_filters)*self.num_sites
            while sum([len(x) for x in site_filters]) < total_filters:
                filt_ind = filt_ind % num_filters
                site_idx = site_idx % self.num_sites
                curr_filt_idx = sorted_filt_inds[filt_ind]

                # if a filter is already given to a site, find a new one
                if not curr_filt_idx in site_filters[site_idx]:
                    site_filters[site_idx].append(curr_filt_idx)
                else:
                    while sorted_filt_inds[filt_ind] in site_filters[site_idx]:
                        filt_ind += 1
                        filt_ind = filt_ind % num_filters
                    site_filters[site_idx].append(sorted_filt_inds[filt_ind])
                site_idx += 1
                filt_ind += 1
        site_filters = torch.stack([torch.LongTensor(x) for x in site_filters])
        return site_filters

    def create_partitions(self,args):
        if args.rank==0:
            # create the site models if they do not exist yet
            # if not len(self.site_models):
            #     self.instantiate_site_models()

            # I decided not to partition the first layer of convs because these are
            # likely more sensitive to pruning -- they are also more narrow so it
            # is not a big deal to not prune them
            self.layer1_partition_vectors = []
            self.layer1_filter_partition = []

            # layer2 partition
            self.layer2_partition_vectors = []
            self.layer2_filter_partition = []
            for block_idx in range(1, len(self.whole_model.layer2)):
                #assert (
                #        self.whole_model.layer2[block_idx].conv1.in_channels ==
                #        self.whole_model.layer2[block_idx].conv1.out_channels)
                num_filt = self.whole_model.layer2[block_idx].conv1.out_channels
                conv_partitions = self.sample_filter_indices(
                    self.whole_model.layer2, block_idx,
                    min_filter_ratio=self.min_filter_ratio)
                # tmp_conv_partition_vectors = []
                # for filt_ind in conv_partitions:
                #     partition_tensor = torch.zeros(num_filt)
                #     partition_tensor[filt_ind] = 1.
                #     partition_tensor = partition_tensor.to(self.device)
                #     tmp_conv_partition_vectors.append(partition_tensor)
                #self.layer2_partition_vectors.append(tmp_conv_partition_vectors)                
                self.layer2_filter_partition.append(conv_partitions)
            self.layer2_filter_partition=torch.stack(self.layer2_filter_partition).to(self.device)
            #print(self.layer2_filter_partition)

            # layer3 partition
            self.layer3_partition_vectors = []
            self.layer3_filter_partition = []
            for block_idx in range(1, len(self.whole_model.layer3)):
                #assert (
                #        self.whole_model.layer3[block_idx].conv1.in_channels ==
                #        self.whole_model.layer3[block_idx].conv1.out_channels)
                num_filt = self.whole_model.layer3[block_idx].conv1.out_channels
                conv_partitions = self.sample_filter_indices(
                    self.whole_model.layer3, block_idx,
                    min_filter_ratio=self.min_filter_ratio)
                # tmp_conv_partition_vectors = []
                # for filt_ind in conv_partitions:
                #     partition_tensor = torch.zeros(num_filt)
                #     partition_tensor[filt_ind] = 1.
                #     partition_tensor = partition_tensor.to(self.device)
                #     tmp_conv_partition_vectors.append(partition_tensor)
                # self.layer3_partition_vectors.append(tmp_conv_partition_vectors)
                self.layer3_filter_partition.append(conv_partitions)
            self.layer3_filter_partition=torch.stack(self.layer3_filter_partition).to(self.device)


            # layer4 partition
            self.layer4_partition_vectors = []
            self.layer4_filter_partition = []
            for block_idx in range(1, len(self.whole_model.layer4)):
                #assert (
                #        self.whole_model.layer4[block_idx].conv1.in_channels ==
                #        self.whole_model.layer4[block_idx].conv1.out_channels)
                num_filt = self.whole_model.layer4[block_idx].conv1.out_channels
                conv_partitions = self.sample_filter_indices(
                    self.whole_model.layer4, block_idx,
                    min_filter_ratio=self.min_filter_ratio)
                # tmp_conv_partition_vectors = []
                # for filt_ind in conv_partitions:
                #     partition_tensor = torch.zeros(num_filt)
                #     partition_tensor[filt_ind] = 1.
                #     partition_tensor = partition_tensor.to(self.device)
                #     tmp_conv_partition_vectors.append(partition_tensor)
                # self.layer3_partition_vectors.append(tmp_conv_partition_vectors)
                self.layer4_filter_partition.append(conv_partitions)
            self.layer4_filter_partition=torch.stack(self.layer4_filter_partition).to(self.device)
            #self.layer1_filter_partition = torch.FloatTensor(self.layer1_filter_partition)
            # print(self.layer2_filter_partition)
            # self.layer2_filter_partition = torch.FloatTensor(self.layer2_filter_partition)
            # self.layer3_filter_partition = torch.FloatTensor(self.layer3_filter_partition)

            #print(self.layer2_filter_partition.shape)
            #print(self.layer3_filter_partition.shape)
            #print(self.layer4_filter_partition.shape)
            broadcast_weight(self.layer2_filter_partition,rank_list=list(range(self.num_sites)),source=0)
            broadcast_weight(self.layer3_filter_partition,rank_list=list(range(self.num_sites)), source=0)
            broadcast_weight(self.layer4_filter_partition,rank_list=list(range(self.num_sites)), source=0)
        else:
            self.layer1_partition_vectors = []
            self.layer1_filter_partition = []

            self.layer2_filter_partition=torch.zeros([1,2,128]).to(self.device)
            self.layer3_filter_partition=torch.zeros([1,2,256]).to(self.device)
            self.layer4_filter_partition=torch.zeros([1,2,512]).to(self.device)
            broadcast_weight(self.layer2_filter_partition,rank_list=list(range(self.num_sites)),source=0)
            broadcast_weight(self.layer3_filter_partition,rank_list=list(range(self.num_sites)),source=0)
            broadcast_weight(self.layer4_filter_partition,rank_list=list(range(self.num_sites)),source=0)
        print("partition sucess")

    def prepare_eval(self):
        # for i in range(1,self.base_model.num_blocks[2]):
        #     self.base_model.layer3[i].active_flag = True
        #     self.base_model.layer3[i].scale_constant = self.scale_constant
        pass


    def prepare_train(self):
        # for i in range(1,self.base_model.num_blocks[2]):
        #     self.base_model.layer3[i].active_flag = i in self.site_indices[args.rank]
        #     self.base_model.layer3[i].scale_constant = 1.0
        pass

    def dispatch_model(self, specs, args):
        with torch.no_grad():
            # nothing from first convolution or final batch norm/linear layer
            # needs to be dropped/zeroed out

            # layer1
            # for block_idx, block in enumerate(self.site_models[site_idx].layer1[1:]):
            #     if len(self.layer1_partition_vectors):
            #         assert len(block.shortcut) == 0. # should not be strided
            #         part_vec = self.layer1_partition_vectors[block_idx][site_idx]

            #         # first batch norm layer should not have anything removed

            #         current_group = []
            #         for site_i in range(self.num_sites):
            #             if block_idx in self.site_indices[site_i]:
            #                 current_group.append(site_i)

            #         # first conv layer has output channels pruned
            #         block.conv1.weight.mul_(
            #             part_vec[:, None, None, None])

            #         # second batch norm is pruned along with output channels
            #         block.bn2.weight.mul_(part_vec)
            #         block.bn2.bias.mul_(part_vec)

            #         # the input channels of the second conv are pruned
            #         block.conv2.weight.mul_(
            #             part_vec[None, :, None, None])

            for block_idx, block in enumerate(self.base_model.layer2[1:]):
                #print(block_idx)
                if len(self.layer2_filter_partition):
                    #assert len(block.shortcut) == 0. # should not be strided
                    # print(self.layer2_filter_partition[0][0].shape)
                    # print(block.conv1.weight.shape)
                    # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                    if args.rank==0:
                        # print(self.layer2_filter_partition[0])
                        # print(block.conv1.weight.shape)
                        # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                        block.conv1.weight.data=self.whole_model.layer2[block_idx+1].conv1.weight.data.index_select(0,self.layer2_filter_partition[block_idx][0])

                        block.bn2.weight.data=self.whole_model.layer2[block_idx+1].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][0])
                        block.bn2.bias.data=self.whole_model.layer2[block_idx+1].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][0])

                        block.conv2.weight.data= self.whole_model.layer2[block_idx+1].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][0])
                    for site_i in range(1,self.num_sites):
                        current_group=[0,site_i]
                        if args.rank==0:
                            broadcast_weight(self.whole_model.layer2[block_idx+1].conv1.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer2[block_idx+1].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer2[block_idx+1].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer2[block_idx+1].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                        else:
                            broadcast_weight(block.conv1.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.bias.data, rank_list=current_group,source=0)
                            broadcast_weight(block.conv2.weight.data, rank_list=current_group,source=0)

            for block_idx, block in enumerate(self.base_model.layer3[1:]):
                #print(block_idx)
                if len(self.layer3_filter_partition):
                    #assert len(block.shortcut) == 0. # should not be strided
                    # print(self.layer2_filter_partition[0][0].shape)
                    # print(block.conv1.weight.shape)
                    # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                    if args.rank==0:
                        # print(self.layer2_filter_partition[0])
                        # print(block.conv1.weight.shape)
                        # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                        block.conv1.weight.data=self.whole_model.layer3[block_idx+1].conv1.weight.data.index_select(0,self.layer3_filter_partition[block_idx][0])

                        block.bn2.weight.data=self.whole_model.layer3[block_idx+1].bn2.weight.data.index_select(0,self.layer3_filter_partition[block_idx][0])
                        block.bn2.bias.data=self.whole_model.layer3[block_idx+1].bn2.bias.data.index_select(0,self.layer3_filter_partition[block_idx][0])

                        block.conv2.weight.data= self.whole_model.layer3[block_idx+1].conv2.weight.data.index_select(1,self.layer3_filter_partition[block_idx][0])
                        #print("sucess rank 0")
                    for site_i in range(1,self.num_sites):
                        current_group=[0,site_i]
                        if args.rank==0:
                            broadcast_weight(self.whole_model.layer3[block_idx+1].conv1.weight.data.index_select(0,self.layer3_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer3[block_idx+1].bn2.weight.data.index_select(0,self.layer3_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer3[block_idx+1].bn2.bias.data.index_select(0,self.layer3_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer3[block_idx+1].conv2.weight.data.index_select(1,self.layer3_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                        else:
                            broadcast_weight(block.conv1.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.bias.data, rank_list=current_group,source=0)
                            broadcast_weight(block.conv2.weight.data, rank_list=current_group,source=0)

            
            for block_idx, block in enumerate(self.base_model.layer4[1:]):
                #print(block_idx)
                if len(self.layer4_filter_partition):
                    #assert len(block.shortcut) == 0. # should not be strided
                    # print(self.layer2_filter_partition[0][0].shape)
                    # print(block.conv1.weight.shape)
                    # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                    if args.rank==0:
                        # print(self.layer2_filter_partition[0])
                        # print(block.conv1.weight.shape)
                        # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                        block.conv1.weight.data=self.whole_model.layer4[block_idx+1].conv1.weight.data.index_select(0,self.layer4_filter_partition[block_idx][0])

                        block.bn2.weight.data=self.whole_model.layer4[block_idx+1].bn2.weight.data.index_select(0,self.layer4_filter_partition[block_idx][0])
                        block.bn2.bias.data=self.whole_model.layer4[block_idx+1].bn2.bias.data.index_select(0,self.layer4_filter_partition[block_idx][0])

                        block.conv2.weight.data= self.whole_model.layer4[block_idx+1].conv2.weight.data.index_select(1,self.layer4_filter_partition[block_idx][0])
                        #print("sucess rank 0")
                    for site_i in range(1,self.num_sites):
                        current_group=[0,site_i]
                        if args.rank==0:
                            broadcast_weight(self.whole_model.layer4[block_idx+1].conv1.weight.data.index_select(0,self.layer4_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer4[block_idx+1].bn2.weight.data.index_select(0,self.layer4_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer4[block_idx+1].bn2.bias.data.index_select(0,self.layer4_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer4[block_idx+1].conv2.weight.data.index_select(1,self.layer4_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                        else:
                            broadcast_weight(block.conv1.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.bias.data, rank_list=current_group,source=0)
                            broadcast_weight(block.conv2.weight.data, rank_list=current_group,source=0)

        print('sucess dispatch')

    def sync_model(self, specs, args):
        print('start sync')
        with torch.no_grad():
            # aggregate conv1
            all_reduce_module(specs, args, self.base_model.conv1)
            # aggregate layer 1 & 2 & 4
            all_reduce_module(specs, args, self.base_model.bn1)
            # aggregate FC layer
            all_reduce_module(specs, args, self.base_model.linear)

            # layer1
            all_reduce_module(specs, args, self.base_model.layer1[0])
            if not len(self.layer1_filter_partition):
                # for block_idx, global_block in enumerate(self.base_model.layer1[1:]):
                #     site_blocks = [x.layer1[block_idx + 1] for x in self.site_models]
                #     self.set_block_average(global_block, site_blocks)

                for i in range(1,len(self.base_model.layer1)):
                    all_reduce_module(specs, args, self.base_model.layer1[i])
                    if args.rank==0:
                        self.whole_model.layer1[i]=self.base_model.layer1[i]

            else:
                raise 'error, should not partition layer 1'
                # # for block_idx, global_block in enumerate(self.base_model.layer1[1:]):
                # #     part_vecs = self.layer1_partition_vectors[block_idx]
                # #     site_blocks = [x.layer1[block_idx + 1] for x in self.site_models]
                # #     self.agg_block_partition(global_block, site_blocks, part_vecs)

                # for i in range(1,len(self.base_model.layer1)):
                #     filters_idx=self.layer1_partition_vectors[i-1]
                #     current_group = []
                #     for site_i in range(1, self.num_sites):
                #         reduce_module(specs, args,self.base_model.layer1[i], rank_list=[0,site_i], source=site_i)


            all_reduce_module(specs, args, self.base_model.layer2[0])
            if not len(self.layer2_filter_partition):
                # for block_idx, global_block in enumerate(self.base_model.layer1[1:]):
                #     site_blocks = [x.layer1[block_idx + 1] for x in self.site_models]
                #     self.set_block_average(global_block, site_blocks)

                for i in range(1,len(self.base_model.layer2)):
                    all_reduce_module(specs, args, self.base_model.layer2[i])

                    if args.rank==0:
                        self.whole_model.layer2[i]=self.base_model.layer2[i]
            else:
                for block_idx, block in enumerate(self.base_model.layer2[1:]):
                    #assert len(block.shortcut) == 0. # should not be strided
                    
                    all_reduce_module(specs, args, self.base_model.layer2[i].bn1)
                    if args.rank==0:
                        self.whole_model.layer2[i].bn1=self.base_model.layer2[i].bn1
                        # self.whole_model.layer2[block_idx+1].conv1.weight.data.index_select(0,self.layer2_filter_partition[block_idx][0])=block.conv1.weight

                        # self.whole_model.layer2[block_idx+1].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i])=block.bn2.weight
                        # self.whole_model.layer2[block_idx+1].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][site_i])=block.bn2.bias

                        # self.whole_model.layer2[block_idx+1].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][site_i])=block.conv2.weight

                        self.whole_model.layer2[block_idx+1].conv1.weight.data[self.layer2_filter_partition[block_idx][0],:,:,:]=block.conv1.weight.data

                        self.whole_model.layer2[block_idx+1].bn2.weight.data[self.layer2_filter_partition[block_idx][0]]=block.bn2.weight.data
                        self.whole_model.layer2[block_idx+1].bn2.bias.data[self.layer2_filter_partition[block_idx][0]]=block.bn2.bias.data

                        self.whole_model.layer2[block_idx+1].conv2.weight.data[:,self.layer2_filter_partition[block_idx][0],:,:]=block.conv2.weight.data
                        #print('rank 0 sync sucess')


                    for site_i in range(1,self.num_sites):
                        current_group=[0,site_i]
                        # if args.rank==0:
                        #     reduce_module(self.whole_model.layer2[block_idx].data.index_select(0,self.layer2_filter_partition[block_idx][0]), rank_list=current_group,dst=0)
                        #     reduce_module(self.whole_model.layer2[block_idx].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,dst=0)
                        #     reduce_module(self.whole_model.layer2[block_idx].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,dst=0)
                        #     reduce_module(self.whole_model.layer2[block_idx].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,dst=0)
                        # else:
                        #     reduce_module(block.conv1.weight, rank_list=current_group,dst=0)
                        #     reduce_module(block.bn2.weight, rank_list=current_group,dst=0)
                        #     reduce_module(block.bn2.bias, rank_list=current_group,dst=0)
                        #     reduce_module(block.conv2.weight, rank_list=current_group,dst=0)
                        
                        #for now only considering non-overlapping case
                        if args.rank==0:
                            broadcast_weight(self.whole_model.layer2[block_idx+1].conv1.weight.data[self.layer2_filter_partition[block_idx][site_i],:,:,:], rank_list=current_group,source=site_i)
                            broadcast_weight(self.whole_model.layer2[block_idx+1].bn2.weight.data[self.layer2_filter_partition[block_idx][site_i]], rank_list=current_group,source=site_i)
                            broadcast_weight(self.whole_model.layer2[block_idx+1].bn2.bias.data[self.layer2_filter_partition[block_idx][site_i]], rank_list=current_group,source=site_i)
                            broadcast_weight(self.whole_model.layer2[block_idx+1].conv2.weight.data[:,self.layer2_filter_partition[block_idx][site_i],:,:], rank_list=current_group,source=site_i)
                        else:
                            broadcast_weight(block.conv1.weight.data, rank_list=current_group,source=site_i)
                            broadcast_weight(block.bn2.weight.data, rank_list=current_group,source=site_i)
                            broadcast_weight(block.bn2.bias.data, rank_list=current_group,source=site_i)
                            broadcast_weight(block.conv2.weight.data, rank_list=current_group,source=site_i)

            print('layer 2 sync sucess')
            all_reduce_module(specs, args, self.base_model.layer3[0])
            if not len(self.layer3_filter_partition):
                # for block_idx, global_block in enumerate(self.base_model.layer1[1:]):
                #     site_blocks = [x.layer1[block_idx + 1] for x in self.site_models]
                #     self.set_block_average(global_block, site_blocks)

                for i in range(1,len(self.base_model.layer3)):
                    all_reduce_module(specs, args, self.base_model.layer3[i])

                    if args.rank==0:
                        self.whole_model.layer3[i]=self.base_model.layer3[i]
            else:
                for block_idx, block in enumerate(self.base_model.layer3[1:]):
                    #assert len(block.shortcut) == 0. # should not be strided
                    
                    all_reduce_module(specs, args, self.base_model.layer3[i].bn1)
                    if args.rank==0:
                        self.whole_model.layer3[i].bn1=self.base_model.layer3[i].bn1
                        # self.whole_model.layer2[block_idx+1].conv1.weight.data.index_select(0,self.layer2_filter_partition[block_idx][0])=block.conv1.weight

                        # self.whole_model.layer2[block_idx+1].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i])=block.bn2.weight
                        # self.whole_model.layer2[block_idx+1].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][site_i])=block.bn2.bias

                        # self.whole_model.layer2[block_idx+1].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][site_i])=block.conv2.weight

                        self.whole_model.layer3[block_idx+1].conv1.weight.data[self.layer3_filter_partition[block_idx][0],:,:,:]=block.conv1.weight.data

                        self.whole_model.layer3[block_idx+1].bn2.weight.data[self.layer3_filter_partition[block_idx][0]]=block.bn2.weight.data
                        self.whole_model.layer3[block_idx+1].bn2.bias.data[self.layer3_filter_partition[block_idx][0]]=block.bn2.bias.data

                        self.whole_model.layer3[block_idx+1].conv2.weight.data[:,self.layer3_filter_partition[block_idx][0],:,:]=block.conv2.weight.data
                        #print('rank 0 sync sucess')


                    for site_i in range(1,self.num_sites):
                        current_group=[0,site_i]
                        # if args.rank==0:
                        #     reduce_module(self.whole_model.layer2[block_idx].data.index_select(0,self.layer2_filter_partition[block_idx][0]), rank_list=current_group,dst=0)
                        #     reduce_module(self.whole_model.layer2[block_idx].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,dst=0)
                        #     reduce_module(self.whole_model.layer2[block_idx].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,dst=0)
                        #     reduce_module(self.whole_model.layer2[block_idx].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,dst=0)
                        # else:
                        #     reduce_module(block.conv1.weight, rank_list=current_group,dst=0)
                        #     reduce_module(block.bn2.weight, rank_list=current_group,dst=0)
                        #     reduce_module(block.bn2.bias, rank_list=current_group,dst=0)
                        #     reduce_module(block.conv2.weight, rank_list=current_group,dst=0)
                        
                        #for now only considering non-overlapping case
                        if args.rank==0:
                            broadcast_weight(self.whole_model.layer3[block_idx+1].conv1.weight.data[self.layer3_filter_partition[block_idx][site_i],:,:,:], rank_list=current_group,source=site_i)
                            broadcast_weight(self.whole_model.layer3[block_idx+1].bn2.weight.data[self.layer3_filter_partition[block_idx][site_i]], rank_list=current_group,source=site_i)
                            broadcast_weight(self.whole_model.layer3[block_idx+1].bn2.bias.data[self.layer3_filter_partition[block_idx][site_i]], rank_list=current_group,source=site_i)
                            broadcast_weight(self.whole_model.layer3[block_idx+1].conv2.weight.data[:,self.layer3_filter_partition[block_idx][site_i],:,:], rank_list=current_group,source=site_i)
                        else:
                            broadcast_weight(block.conv1.weight.data, rank_list=current_group,source=site_i)
                            broadcast_weight(block.bn2.weight.data, rank_list=current_group,source=site_i)
                            broadcast_weight(block.bn2.bias.data, rank_list=current_group,source=site_i)
                            broadcast_weight(block.conv2.weight.data, rank_list=current_group,source=site_i)
            print('layer 3 sync sucess')

            all_reduce_module(specs, args, self.base_model.layer4[0])
            if not len(self.layer4_filter_partition):
                # for block_idx, global_block in enumerate(self.base_model.layer1[1:]):
                #     site_blocks = [x.layer1[block_idx + 1] for x in self.site_models]
                #     self.set_block_average(global_block, site_blocks)

                for i in range(1,len(self.base_model.layer4)):
                    all_reduce_module(specs, args, self.base_model.layer4[i])

                    if args.rank==0:
                        self.whole_model.layer4[i]=self.base_model.layer4[i]
            else:
                for block_idx, block in enumerate(self.base_model.layer4[1:]):
                    #assert len(block.shortcut) == 0. # should not be strided
                    
                    all_reduce_module(specs, args, self.base_model.layer4[i].bn1)
                    if args.rank==0:
                        self.whole_model.layer4[i].bn1=self.base_model.layer4[i].bn1
                        # self.whole_model.layer2[block_idx+1].conv1.weight.data.index_select(0,self.layer2_filter_partition[block_idx][0])=block.conv1.weight

                        # self.whole_model.layer2[block_idx+1].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i])=block.bn2.weight
                        # self.whole_model.layer2[block_idx+1].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][site_i])=block.bn2.bias

                        # self.whole_model.layer2[block_idx+1].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][site_i])=block.conv2.weight

                        self.whole_model.layer4[block_idx+1].conv1.weight.data[self.layer4_filter_partition[block_idx][0],:,:,:]=block.conv1.weight.data

                        self.whole_model.layer4[block_idx+1].bn2.weight.data[self.layer4_filter_partition[block_idx][0]]=block.bn2.weight.data
                        self.whole_model.layer4[block_idx+1].bn2.bias.data[self.layer4_filter_partition[block_idx][0]]=block.bn2.bias.data

                        self.whole_model.layer4[block_idx+1].conv2.weight.data[:,self.layer4_filter_partition[block_idx][0],:,:]=block.conv2.weight.data
                        #print('rank 0 sync sucess')


                    for site_i in range(1,self.num_sites):
                        current_group=[0,site_i]
                        # if args.rank==0:
                        #     reduce_module(self.whole_model.layer2[block_idx].data.index_select(0,self.layer2_filter_partition[block_idx][0]), rank_list=current_group,dst=0)
                        #     reduce_module(self.whole_model.layer2[block_idx].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,dst=0)
                        #     reduce_module(self.whole_model.layer2[block_idx].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,dst=0)
                        #     reduce_module(self.whole_model.layer2[block_idx].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,dst=0)
                        # else:
                        #     reduce_module(block.conv1.weight, rank_list=current_group,dst=0)
                        #     reduce_module(block.bn2.weight, rank_list=current_group,dst=0)
                        #     reduce_module(block.bn2.bias, rank_list=current_group,dst=0)
                        #     reduce_module(block.conv2.weight, rank_list=current_group,dst=0)
                        
                        #for now only considering non-overlapping case
                        if args.rank==0:
                            broadcast_weight(self.whole_model.layer4[block_idx+1].conv1.weight.data[self.layer4_filter_partition[block_idx][site_i],:,:,:], rank_list=current_group,source=site_i)
                            broadcast_weight(self.whole_model.layer4[block_idx+1].bn2.weight.data[self.layer4_filter_partition[block_idx][site_i]], rank_list=current_group,source=site_i)
                            broadcast_weight(self.whole_model.layer4[block_idx+1].bn2.bias.data[self.layer4_filter_partition[block_idx][site_i]], rank_list=current_group,source=site_i)
                            broadcast_weight(self.whole_model.layer4[block_idx+1].conv2.weight.data[:,self.layer4_filter_partition[block_idx][site_i],:,:], rank_list=current_group,source=site_i)
                        else:
                            broadcast_weight(block.conv1.weight.data, rank_list=current_group,source=site_i)
                            broadcast_weight(block.bn2.weight.data, rank_list=current_group,source=site_i)
                            broadcast_weight(block.bn2.bias.data, rank_list=current_group,source=site_i)
                            broadcast_weight(block.conv2.weight.data, rank_list=current_group,source=site_i)
            print('layer 4 sync sucess')
            if args.rank==0:
                self.whole_model.conv1.weight=self.base_model.conv1.weight
                #self.whole_model.bn1.weight=self.base_model.bn1.weight
                #self.whole_model.bn1.bias=self.base_model.bn1.bias
                self.whole_model.bn1=self.base_model.bn1
                self.whole_model.linear.weight=self.base_model.linear.weight
                self.whole_model.linear.bias=self.base_model.linear.bias
                self.whole_model.layer1[0]=self.base_model.layer1[0]
                self.whole_model.layer2[0]=self.base_model.layer2[0]
                self.whole_model.layer3[0]=self.base_model.layer3[0]
                self.whole_model.layer4[0]=self.base_model.layer4[0]
            print('all sync sucess')
            self.create_partitions(args)



    def ini_sync_dispatch_model(self,specs,args):
        #self.parameter_list=self.base_parameter_list


        self.create_partitions(args)

        with torch.no_grad():

                    # aggregate conv1
            if args.rank==0:
                self.base_model.conv1.weight=self.whole_model.conv1.weight
                self.base_model.bn1.weight=self.whole_model.bn1.weight
                self.base_model.bn1.bias=self.whole_model.bn1.bias
                self.base_model.linear.weight=self.whole_model.linear.weight
                self.base_model.linear.bias=self.whole_model.linear.bias
                self.base_model.layer1[0]=self.whole_model.layer1[0]
                self.base_model.layer2[0]=self.whole_model.layer2[0]
                self.base_model.layer3[0]=self.whole_model.layer3[0]
                self.base_model.layer4[0]=self.whole_model.layer4[0]

            broadcast_module_itr(self.base_model.conv1,source=0)
            # broadcast conv1 
            broadcast_module_itr(self.base_model.bn1,source=0)

            # # broadcast FC layer
            broadcast_module_itr(self.base_model.linear,source=0)

            broadcast_module_itr(self.base_model.layer1[0],source=0)
            broadcast_module_itr(self.base_model.layer2[0],source=0)
            broadcast_module_itr(self.base_model.layer3[0],source=0)
            broadcast_module_itr(self.base_model.layer4[0],source=0)
            print('sucess itr')
            # nothing from first convolution or final batch norm/linear layer
            # needs to be dropped/zeroed out

            # layer1
            # for block_idx, block in enumerate(self.site_models[site_idx].layer1[1:]):
            #     if len(self.layer1_partition_vectors):
            #         assert len(block.shortcut) == 0. # should not be strided
            #         part_vec = self.layer1_partition_vectors[block_idx][site_idx]

            #         # first batch norm layer should not have anything removed

            #         current_group = []
            #         for site_i in range(self.num_sites):
            #             if block_idx in self.site_indices[site_i]:
            #                 current_group.append(site_i)

            #         # first conv layer has output channels pruned
            #         block.conv1.weight.mul_(
            #             part_vec[:, None, None, None])

            #         # second batch norm is pruned along with output channels
            #         block.bn2.weight.mul_(part_vec)
            #         block.bn2.bias.mul_(part_vec)

            #         # the input channels of the second conv are pruned
            #         block.conv2.weight.mul_(
            #             part_vec[None, :, None, None])


            # layer1
            if len(self.layer1_filter_partition):
                raise 'error, should not partition layer 1'
            else:
                for block_idx, block in enumerate(self.base_model.layer1[1:]):
                    if args.rank==0:
                        block=self.whole_model.layer1[block_idx+1]
                    broadcast_module_itr(self.base_model.layer1[block_idx+1],source=0)


            # layer2
            for block_idx, block in enumerate(self.base_model.layer2[1:]):
                #print(block_idx)
                if len(self.layer2_filter_partition):
                    #assert len(block.shortcut) == 0. # should not be strided
                    # print(self.layer2_filter_partition[0][0].shape)
                    # print(block.conv1.weight.shape)
                    # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                    if args.rank==0:
                        # print(self.layer2_filter_partition[0])
                        # print(block.conv1.weight.shape)
                        # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                        block.conv1.weight.data=self.whole_model.layer2[block_idx+1].conv1.weight.data.index_select(0,self.layer2_filter_partition[block_idx][0])

                        block.bn2.weight.data=self.whole_model.layer2[block_idx+1].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][0])
                        block.bn2.bias.data=self.whole_model.layer2[block_idx+1].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][0])

                        block.conv2.weight.data= self.whole_model.layer2[block_idx+1].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][0])
                        #print("sucess rank 0")
                    for site_i in range(1,self.num_sites):
                        current_group=[0,site_i]
                        if args.rank==0:
                            broadcast_weight(self.whole_model.layer2[block_idx+1].conv1.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer2[block_idx+1].bn2.weight.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer2[block_idx+1].bn2.bias.data.index_select(0,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer2[block_idx+1].conv2.weight.data.index_select(1,self.layer2_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                        else:
                            broadcast_weight(block.conv1.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.bias.data, rank_list=current_group,source=0)
                            broadcast_weight(block.conv2.weight.data, rank_list=current_group,source=0)

            for block_idx, block in enumerate(self.base_model.layer3[1:]):
                if len(self.layer3_filter_partition):
                    #assert len(block.shortcut) == 0. # should not be strided
                    # print(self.layer2_filter_partition[0][0].shape)
                    # print(block.conv1.weight.shape)
                    # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                    if args.rank==0:
                        # print(self.layer2_filter_partition[0])
                        # print(block.conv1.weight.shape)
                        # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                        block.conv1.weight.data=self.whole_model.layer3[block_idx+1].conv1.weight.data.index_select(0,self.layer3_filter_partition[block_idx][0])

                        block.bn2.weight.data=self.whole_model.layer3[block_idx+1].bn2.weight.data.index_select(0,self.layer3_filter_partition[block_idx][0])
                        block.bn2.bias.data=self.whole_model.layer3[block_idx+1].bn2.bias.data.index_select(0,self.layer3_filter_partition[block_idx][0])

                        block.conv2.weight.data= self.whole_model.layer3[block_idx+1].conv2.weight.data.index_select(1,self.layer3_filter_partition[block_idx][0])
                        #print("sucess rank 0")
                    for site_i in range(1,self.num_sites):
                        current_group=[0,site_i]
                        if args.rank==0:
                            broadcast_weight(self.whole_model.layer3[block_idx+1].conv1.weight.data.index_select(0,self.layer3_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer3[block_idx+1].bn2.weight.data.index_select(0,self.layer3_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer3[block_idx+1].bn2.bias.data.index_select(0,self.layer3_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer3[block_idx+1].conv2.weight.data.index_select(1,self.layer3_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                        else:
                            broadcast_weight(block.conv1.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.bias.data, rank_list=current_group,source=0)
                            broadcast_weight(block.conv2.weight.data, rank_list=current_group,source=0)     

            for block_idx, block in enumerate(self.base_model.layer4[1:]):
                if len(self.layer4_filter_partition):
                    #assert len(block.shortcut) == 0. # should not be strided
                    # print(self.layer2_filter_partition[0][0].shape)
                    # print(block.conv1.weight.shape)
                    # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                    if args.rank==0:
                        # print(self.layer2_filter_partition[0])
                        # print(block.conv1.weight.shape)
                        # print(self.whole_model.layer2[block_idx].conv1.weight.shape)
                        block.conv1.weight.data=self.whole_model.layer4[block_idx+1].conv1.weight.data.index_select(0,self.layer4_filter_partition[block_idx][0])

                        block.bn2.weight.data=self.whole_model.layer4[block_idx+1].bn2.weight.data.index_select(0,self.layer4_filter_partition[block_idx][0])
                        block.bn2.bias.data=self.whole_model.layer4[block_idx+1].bn2.bias.data.index_select(0,self.layer4_filter_partition[block_idx][0])

                        block.conv2.weight.data= self.whole_model.layer4[block_idx+1].conv2.weight.data.index_select(1,self.layer4_filter_partition[block_idx][0])
                        #print("sucess rank 0")
                    for site_i in range(1,self.num_sites):
                        current_group=[0,site_i]
                        if args.rank==0:
                            broadcast_weight(self.whole_model.layer4[block_idx+1].conv1.weight.data.index_select(0,self.layer4_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer4[block_idx+1].bn2.weight.data.index_select(0,self.layer4_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer4[block_idx+1].bn2.bias.data.index_select(0,self.layer4_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                            broadcast_weight(self.whole_model.layer4[block_idx+1].conv2.weight.data.index_select(1,self.layer4_filter_partition[block_idx][site_i]), rank_list=current_group,source=0)
                        else:
                            broadcast_weight(block.conv1.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.weight.data, rank_list=current_group,source=0)
                            broadcast_weight(block.bn2.bias.data, rank_list=current_group,source=0)
                            broadcast_weight(block.conv2.weight.data, rank_list=current_group,source=0)   

            print('ini_sync_dispatch sucess')



    def prepare_whole_model(self, specs, args):
        # aggregate conv1
        # all_reduce_module(specs, args, self.base_model.conv1)
        # # aggregate layer 1 & 2 & 4
        # all_reduce_module(specs, args, self.base_model.layer1)
        # all_reduce_module(specs, args, self.base_model.layer2)
        # all_reduce_module(specs, args, self.base_model.layer4)
        # # aggregate FC layer
        # all_reduce_module(specs, args, self.base_model.fc)
        # # apply IST aggregation here
        # all_reduce_module(specs, args, self.base_model.layer3[0])
        #self.layer_server_list=[-1]
        # for i in range(1,self.base_model.num_blocks[2]):

        #     current_group = []
        #     for site_i in range(self.num_sites):
        #         if i in self.site_indices[site_i]:
        #             current_group.append(site_i)
        #     #self.layer_server_list.append(min(current_group))

        #     if not(current_group[0]==0):
        #         broadcast_weight(self.base_model.layer3[i], rank_list=[0,min(current_group)], source=min(current_group))
        pass


def train(specs, args, start_time, model_name, ist_model: ISTResNetModel, optimizer, device, train_loader, test_loader, epoch, num_sync, num_iter, train_time_log,test_loss_log, test_acc_log):

    #lr = specs.get('lr', 1e-2)
    # employ a step schedule for the sub nets

    # if num_sync > int(78):
    #     lr /= 10
    # if num_sync > int(78+78/2):

    #     lr /= 10
    if specs.get('momentum_decay', False):
        delay_ratio = specs.get('delay_ratio', 0.75)
        momentum = get_demon_momentum(
                shuffle_idx, specs.get('num_shuffles', 500),
                specs.get('momentum', 0.9), delay_ratio)
    else:
        momentum = specs.get('momentum', 0.9)

    prefetcher = data_prefetcher(train_loader)
    data, target = prefetcher.next()
    i = 0
    while data is not None:
        data = data.to(device)
        target = target.to(device)
        if num_iter % specs['repartition_iter'] == 0:
            if num_iter>0:
                ist_model.dispatch_model(specs, args)
            
            lr = specs.get('lr', 0.1)
            # employ a step schedule for the sub nets
            if num_sync > 3800:
                 lr /= 10
            if num_sync > 3800*2:
                 lr /= 10 
            if num_sync > 3800*3:
                 lr /= 10
            optimizer = torch.optim.SGD(
                    ist_model.base_model.parameters(), lr=lr,
                    momentum=momentum, weight_decay=specs.get('wd', 5e-4))

        optimizer.zero_grad()
        output = ist_model.base_model(data)
        loss = functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
        if num_iter % specs['log_interval'] == 0:
            #pass
            print('Train Epoch {} num_sync {} iter {} <Loss: {:.6f}, Accuracy: {:.4f}%>'.format(
                epoch, num_sync, num_iter%specs['repartition_iter'], loss.item(), 100. * train_correct / target.shape[0]))
                # optionally sync model parameters
        if (
                ((num_iter + 1) % specs['repartition_iter'] == 0) or
                (i == len(train_loader) - 1 and epoch == specs['epochs'])):
            old_partition = [ist_model.layer2_filter_partition, ist_model.layer3_filter_partition, ist_model.layer4_filter_partition]
            ist_model.sync_model(specs, args)
            num_sync = num_sync+1
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Node {}: Train Num sync {} total time {:3.2f}s'.format(args.rank, num_sync, elapsed_time))
            if args.rank == 0:
                if num_sync == 1:
                    train_time_log[num_sync - 1] = elapsed_time
                else:
                    train_time_log[num_sync - 1] = train_time_log[num_sync - 2] + elapsed_time
                print('total time {:3.2f}s'.format(train_time_log[num_sync - 1]))
            if ((num_sync - 1) % 150) == 0:
                ist_model.prepare_whole_model(specs,args)
                test(specs,args, ist_model, device, test_loader, epoch, num_sync, test_loss_log, test_acc_log)

                # always save model with best valid accuracy
                model_fn = os.path.join(
                    specs.get('save_dir', './'),
                    specs.get('save_name', str(args.rank) + 'sub' + str(epoch) + '.pth'))

                # must save site optimizers to restart momentum tracking from checkpoint
                checkpoint = {
                    'num_iter': num_iter,
                    'num_sync':num_sync,
                    'epoch':epoch,
                    'model': ist_model.base_model,
                    'partition': old_partition
                }
                print(model_fn)
                torch.save(checkpoint, model_fn) # save the global model

                if args.rank == 0:
                    np.savetxt('./log/' + model_name + '_train_time.log', train_time_log, fmt='%1.4f', newline=' ')
                    np.savetxt('./log/' + model_name + '_test_loss.log', test_loss_log, fmt='%1.4f', newline=' ')
                    np.savetxt('./log/' + model_name + '_test_acc.log', test_acc_log, fmt='%1.4f', newline=' ')
                    
                    # always save model with best valid accuracy
                    model_fn = os.path.join(
                        specs.get('save_dir', './'),
                        specs.get('save_name', model_name+'.pth'))

                    # must save site optimizers to restart momentum tracking from checkpoint
                    checkpoint = {
                        'num_iter': num_iter,
                        'num_sync':num_sync,
                        'epoch':epoch,
                        'model': ist_model.whole_model
                    }
                    print(model_fn)
                    torch.save(checkpoint, model_fn) # save the global model
                    
                    if (num_sync-1)==3800:
                        model_fn = os.path.join(
                            specs.get('save_dir', './'),
                            specs.get('save_name', model_name+'_3800.pth'))

                        torch.save(checkpoint, model_fn) # save the global model

            start_time = time.time()
        num_iter = num_iter +1
        data, target = prefetcher.next()
        i += 1


    return num_sync, num_iter, start_time, optimizer

def test(specs,args, ist_model: ISTResNetModel, device, test_loader, epoch, num_sync, test_loss_log, test_acc_log):
    # Do validation only on prime node in cluster.
    if args.rank == 0:
        #ist_model.prepare_eval()
        ist_model.whole_model.eval()
        # test_loss = 0
        # test_correct = 0
        # test_total =0
        # # with torch.no_grad():
        # #     for i, (data, target) in enumerate(test_loader):
        # #         data = data.to(device)
        # #         target = target.to(device)
        # #         output = ist_model.base_model(data)
        # #         test_loss += functional.cross_entropy(output, target, reduction='sum')  # sum up batch loss
        # #         test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # #         test_correct += test_pred.eq(target.view_as(test_pred)).sum()
        # #         test_total += target.shape[0]
        # loss = test_loss / test_total
        # accuracy = test_correct / test_total

        prefetcher = data_prefetcher(test_loader)
        model_in, labels = prefetcher.next()

        agg_val_loss = 0.
        num_correct = 0.
        total_ex = 0.
        criterion = torch.nn.CrossEntropyLoss()
        count=0
        while (model_in is not None):
            model_in = model_in.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                model_output = ist_model.whole_model(model_in)
                val_loss = criterion(model_output, labels)
            agg_val_loss += val_loss.item()
            _, preds = model_output.max(1)

            #print(preds)
            total_ex += labels.size(0)
            num_correct += preds.eq(labels).sum().item()
            model_in, labels = prefetcher.next()
            count=count+1
        agg_val_loss /= len(test_loader)
        val_acc = num_correct/total_ex
        print("Epoch {} Number of Sync {} Local Test Loss: {:.6f}; Test Accuracy: {:.4f}.\n".format(epoch, num_sync, agg_val_loss, val_acc))
        test_loss_log[num_sync - 1] = agg_val_loss
        test_acc_log[num_sync - 1] = val_acc
        #ist_model.prepare_train()
        ist_model.whole_model.train()


def main():


    specs = {
        'test_type': 'ist_resnet', # should be either ist or baseline
        'model_type': 'preact_resnet', # use to specify type of resnet to use in baseline
        'use_valid_set': False,
        'model_version': 'v1', # only used for the mobilenet tests
        'dataset': 'imagenet',
        'repartition_iter': 15, # number of iterations to perform before re-sampling subnets
        'epochs': 20,
        'world_size': 2, # number of subnets to use during training
        'layer_sizes': [3, 4, 23, 3], # used for resnet baseline, number of blocks in each section
        'expansion': 1.,
        'lr': .1,
        'momentum': 0.9,
        'wd': 1e-5,
        'log_interval': 5,
        #'start_from_checkpoint': False,
        # 'save_dir': './test_results/',
        # 'save_name': 'tin_resnet_large.pth',
        # 'results_dir': './test_results/',
        # 'results_name': 'tin_resnet_large.pckl',
        # 'plot_results': True, # whether or not to display visualizations of available metrics after training
        # 'custom_init': True, # whether to use a fancy weight initialization or not (only for MobileNet baseline)
        'momentum_decay': False, # only used for IST tests, whether or not to apply momentum decay during training
        'delay_ratio': 0.75, # when to start momentum decay during training
        'track_momentum': True, # only use for resnet_ist, ensures momentum is maintained when you reshuffle sites
        'min_blocks_per_site': 0, # used for the resnet ist, allow overlapping block partitions to occur
        'head_agg_iter': None, # determines how often you will aggregate classification head weights between sites
        'data_dir': '/scratch0/cd46/imagenet/'
    }

    parser = argparse.ArgumentParser(description='PyTorch ResNet (IST distributed)')
    # parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dist-backend', type=str, default='nccl', metavar='S',
                        help='backend type for distributed PyTorch')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9030', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for distributed PyTorch')
    # parser.add_argument('--world-size', type=int, default=2, metavar='D',
    #                     help='partition group (default: 2)')
    # parser.add_argument('--model-size', type=int, default=256, metavar='N',
    #                     help='model size for intermediate layers (default: 4096)')
    # parser.add_argument('--batch-size', type=int, default=32, metavar='N',
    #                     help='input batch size for training (default: 100)')
    # parser.add_argument('--epochs', type=int, default=40, metavar='N',
    #                      help='number of epochs to train (default: 100)')
    # parser.add_argument('--repartition-iter', type=int, default=50, metavar='N',
    #                     help='keep model in local update mode for how many iteration (default: 5)')
    # parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
    #                     help='learning rate (default: 1.0 for BN)')
    parser.add_argument('--pytorch-seed', type=int, default=-1, metavar='S',
                        help='random seed (default: -1)')
    # parser.add_argument('--log-interval', type=int, default=5, metavar='N',
    #                     help='how many batches to wait before logging training status')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--resume', type=int, default=0)

    args = parser.parse_args()
    if args.pytorch_seed == -1:
        torch.manual_seed(args.rank+(args.resume+1)**2)
    else:
        torch.manual_seed(args.pytorch_seed)
    seed(0)  # This makes sure, node use the same random key so that they does not need to sync partition info.
    if args.use_cuda:
        assert args.cuda_id < torch.cuda.device_count()
        device = torch.device('cuda',args.cuda_id)
    else:
        device = torch.device('cpu')
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size=specs['world_size'])
    global test_rank
    test_rank=args.rank
    if specs['dataset'] == 'cifar10':
        out_size = 512
        for_cifar = True
        num_classes = 10
        input_size = 32
        trn_dl, test_dl = get_cifar10_loaders(specs.get('use_valid_set', False))
        criterion = torch.nn.CrossEntropyLoss()
    elif specs['dataset'] == 'cifar100':
        out_size = 512
        for_cifar = True
        num_classes = 100
        input_size = 32
        trn_dl, test_dl = get_cifar100_loaders(specs.get('use_valid_set', False))
        criterion = torch.nn.CrossEntropyLoss()
    elif specs['dataset'] == 'imagenet':
        for_cifar = False
        num_classes = 1000
        out_size=512
        trn_dl, test_dl = get_imagenet_loaders(specs)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'{specs["dataset"]} dataset not supported')

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    model_name = 'wideresnet_IST_'+ str(specs['dataset'])+'_'+str(specs['world_size']) + '_' + str(specs['repartition_iter'])
    # train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    ist_model = ISTResNetModel(args, device, num_sites=specs['world_size'], min_blocks_per_site=specs['min_blocks_per_site'])

    if args.resume>0:
        checkpoint = torch.load(os.path.join(
                        specs.get('save_dir', './'),
                        specs.get('save_name', model_name+'.pth')))
        if args.rank == 0:
            ist_model.whole_model.load_state_dict(checkpoint['model'].state_dict())
        epoch=checkpoint['epoch']
        #num_iter=checkpoint['num_iter']
        num_iter=0
        num_sync=checkpoint['num_sync']-1
        train_time_log=np.loadtxt('./log/' + model_name + '_train_time.log') if args.rank == 0 else None
        test_loss_log=np.loadtxt('./log/' + model_name + '_test_loss.log') if args.rank == 0 else None
        test_acc_log=np.loadtxt('./log/' + model_name + '_test_acc.log') if args.rank == 0 else None
    else:
        num_sync = 0
        num_iter = 0
        train_time_log = np.zeros(10000) if args.rank == 0 else None
        test_loss_log = np.zeros(10000) if args.rank == 0 else None
        test_acc_log = np.zeros(10000) if args.rank == 0 else None
        epoch=1

    optimizer = torch.optim.SGD(ist_model.base_model.parameters(), lr=specs['lr'])
    ist_model.ini_sync_dispatch_model(specs, args)
    epochs = specs['epochs']

 
    start_time = time.time()
    while epoch<(epochs+1):
        num_sync, num_iter, start_time, optimizer = train(specs, args, start_time, model_name, ist_model, optimizer, device, trn_dl, test_dl, epoch, num_sync, num_iter, train_time_log, test_loss_log, test_acc_log)
        epoch=epoch+1

if __name__ == '__main__':
    main()

