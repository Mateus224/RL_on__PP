import time
from pcl_policy.pcl_rainbow.module import Embedding, NeighborEmbedding, OA, SA
from datetime import datetime

import torch
import torch.nn as nn


import torch.nn.functional as F
from torch_cluster import knn_graph

from torch.nn.utils import spectral_norm

import os
import math


import utils as util_functions

from torch.optim import lr_scheduler

#import dataset_loader_noise as cam_loader

from path import Path

import random
random.seed(0)
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.3):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)



class Tnet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self,k_transform):
        super().__init__()
        self.input_transform = Tnet(k=k_transform)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(k_transform,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1_seg = torch.nn.Conv1d(1088, 512, 1)
        self.conv2_seg = torch.nn.Conv1d(512, 256, 1)
        self.conv3_seg = torch.nn.Conv1d(256, 128, 1)
        self.bn1_seg = nn.BatchNorm1d(512)
        self.bn2_seg = nn.BatchNorm1d(256)
        self.bn3_seg = nn.BatchNorm1d(128)


    def forward(self, input):
        b, f, n = input.size() 
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)
        xb = F.relu(self.bn1(self.conv1(xb)))
        print(xb.shape)
        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)
        features=xb
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))

        
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        xb = xb.view(-1, 1024)

        xb = xb.view(-1, 1024, 1).repeat(1, 1, n)
        output_1= torch.cat([xb, features], 1)

        output_1 = F.relu(self.bn1_seg(self.conv1_seg(output_1)))
        output_1 = F.relu(self.bn2_seg(self.conv2_seg(output_1)))
        output_1 = F.relu(self.bn3_seg(self.conv3_seg(output_1)))

        output = nn.Flatten(1)(output_1)


        return output, matrix3x3, matrix64x64

class Transformers(nn.Module):
    def __init__(self,k_transform):
        super().__init__()
        self.input_transform = Tnet(k=k_transform)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(k_transform,256,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)


        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1_seg = torch.nn.Conv1d(2048, 512, 1)
        self.conv2_seg = torch.nn.Conv1d(512, 256, 1)
        self.conv3_seg = torch.nn.Conv1d(256, 128, 1)
        self.bn1_seg = nn.BatchNorm1d(512)
        self.bn2_seg = nn.BatchNorm1d(256)
        self.bn3_seg = nn.BatchNorm1d(128)

        self.oa1 = OA(256)
        self.oa2 = OA(256)
        self.oa3 = OA(256)
        self.oa4 = OA(256)

        self.linear = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )


    def forward(self, input):
        b, f, n = input.size() 
        matrix3x3 = self.input_transform(input)
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)
        xb = F.relu(self.bn1(self.conv1(xb)))
        x1 = self.oa1(xb)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)
        x = torch.cat([xb, x1, x2, x3, x4], dim=1)

        x = self.linear(x)
 
        xs = nn.MaxPool1d(x.size(-1))(x)

        xs = xs.view(-1, 1024, 1).repeat(1, 1, n)
        output_1= torch.cat([xs, x], 1)
        output_1 = F.relu(self.bn1_seg(self.conv1_seg(output_1)))
        output_1 = F.relu(self.bn2_seg(self.conv2_seg(output_1)))
        output_1 = F.relu(self.bn3_seg(self.conv3_seg(output_1)))

        output = nn.Flatten(1)(output_1)


        return output

class PointNet(nn.Module):
    def __init__(self, args, action_space ,nr_features=3):
        super().__init__()
        self.atoms = 51
        torch.backends.cudnn.enabled = False
        self.action_space = action_space
        self.transform = Transformers(k_transform=nr_features)

        self.fc1 = nn.Linear(65536, 512)
        self.fc2 = nn.Linear(519, 512)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc_h_v = spectral_norm(nn.Linear(512, 512))
        self.fc_h_a = spectral_norm(nn.Linear(512, 512))
        self.fc_z_v = NoisyLinear(512, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(512, action_space * self.atoms, std_init=args.noisy_std)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, pcl, pose, log=False):
        
        xb = self.transform(pcl)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = torch.flatten(xb, start_dim=1)
        pose = torch.flatten(pose, start_dim=1)
        xb = torch.cat([xb,pose], dim=1)
        xb = F.relu(self.fc2(xb))

        v=self.fc_h_v(xb)
        v_uuv = self.fc_z_v(F.relu(v))  # Value stream
        a=self.fc_h_a(xb)
        a_uuv = self.fc_z_a(F.relu(a))  # Advantage stream

        v_uuv, a_uuv = v_uuv.view(-1, 1, self.atoms), a_uuv.view(-1, self.action_space, self.atoms)
        
        q_uuv = v_uuv + a_uuv - a_uuv.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            q_uuv = F.log_softmax(q_uuv, dim=2)  # Log probabilities with action over second dimension
        else:
            
            q_uuv = F.softmax(q_uuv, dim=2)  # Probabilities with action over second dimension
        #return  q_uuv #q_uav,
        return q_uuv#, matrix3x3, matrix64x64

    
    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc_z' in name:
                module.reset_noise()


def pointnetloss(outputs, labels, m3x3, m64x64,k, alpha = 0.0001,):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    #id3x3 = torch.eye(6, requires_grad=True).repeat(bs,1,1)
    id3x3 = torch.eye(k, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)     
