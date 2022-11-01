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

print(torch.__version__)
import os

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
        self.bn2 = torch.nn.BatchNorm2d(int(planes*widen_factor/partition_num_sites), affine=True, track_running_stats=True)
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
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,widen_factor=widen_factors[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,widen_factor=widen_factors[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,widen_factor=widen_factors[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,widen_factor=widen_factors[3])
        self.linear = torch.nn.Linear(out_size*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, widen_factor):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,widen_factor=widen_factor))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class subWideResNet(torch.nn.Module):
    # taken from https://github.com/kuangliu/pytorch-cifar

    def __init__(self, num_blocks, widen_factors, partition_num_sites,out_size=512, num_classes=10,block=wide_basic):
        super(subWideResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,widen_factor=widen_factors[0], partition_num_sites=partition_num_sites[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,widen_factor=widen_factors[1], partition_num_sites=partition_num_sites[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,widen_factor=widen_factors[2], partition_num_sites=partition_num_sites[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,widen_factor=widen_factors[3], partition_num_sites=partition_num_sites[3])
        self.linear = torch.nn.Linear(out_size*block.expansion, num_classes)

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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def WideResNet18(blocks=[2, 2, 2, 2], widen_factors=[2,2,2,2], out_size=512, num_classes=100):
    return WideResNet(num_blocks=blocks, widen_factors=widen_factors, out_size=512, num_classes=100, block=wide_basic)

def subWideResNet18(num_sites,blocks=[2, 2, 2, 2], widen_factors=[2,2,2,2],out_size=512, num_classes=100):
    partition_num_sites=[1,num_sites,num_sites,num_sites]
    return subWideResNet(num_blocks=blocks, widen_factors=widen_factors, partition_num_sites=partition_num_sites,out_size=512, num_classes=100, block=wide_basic)

def WideResNet34(blocks=[3, 4, 6, 3], widen_factors=[2,2,2,2], out_size=512, num_classes=10):
    return WideResNet(num_blocks=blocks, widen_factors=widen_factors, out_size=512, num_classes=10, block=wide_basic)

def subWideResNet34(num_sites,blocks=[3, 4, 6, 3], widen_factors=[2,2,2,2],out_size=512, num_classes=10):
    partition_num_sites=[1,num_sites,num_sites,num_sites]
    return subWideResNet(num_blocks=blocks, widen_factors=widen_factors, partition_num_sites=partition_num_sites,out_size=512, num_classes=10, block=wide_basic)

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

        # self.base_model = subWideResNet18(num_sites=num_sites).to(device)
        self.base_model = subWideResNet34(num_sites=num_sites).to(device)
        # pytorch_total_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        # print(pytorch_total_params)
        if args.rank==0:
            # self.whole_model = WideResNet18().to(device)
            self.whole_model = WideResNet34().to(device)
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