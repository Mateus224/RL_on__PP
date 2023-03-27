import torch
import torch.nn as nn
import torch.nn.functional as F

from pcl_policy.pcl_rainbow.module import Embedding, NeighborEmbedding, NeighborEmbedding_own, NeighborEmbedding_origing, OA,OAO, MHeadOA, SA
from torch.nn.utils import spectral_norm
import math


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



class NaivePCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = SA(128)
        self.sa2 = SA(128)
        self.sa3 = SA(128)
        self.sa4 = SA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class SPCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 256)

        self.sa1 = OA(256)
        self.sa2 = OA(256)
        self.sa3 = OA(256)
        self.sa4 = OA(256)

        self.linear = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class PCT_M(nn.Module):
    def __init__(self, samples=[256, 256]):
        super().__init__()

        self.neighbor_embedding = NeighborEmbedding(samples)
        #self.neighbor_embedding = Embedding(3,128)
        self.oa11 = MHeadOA(256)
        #self.oa12 = MHeadOA(128)
        #self.oa13 = OA(128)
        #self.oa14 = OA(128)

        #self.oa1 = OA(256)
        self.oa2 = MHeadOA(256)
        self.oa3 = MHeadOA(256)
        self.oa4 = MHeadOA(256)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.linear1 = nn.Sequential(
            nn.Conv1d(1280, 1280, kernel_size=1, bias=False),
            nn.BatchNorm1d(1280),
            nn.LeakyReLU(negative_slope=0.2)
        )
    def forward(self, x):
        
        x1,x2,x3,x4 = self.neighbor_embedding(x)
        x_cat= torch.cat([ x1, x2,  x3, x4], dim=1)
        x10 = self.oa11(x_cat)
        x11 = self.oa2(x10)
        x12 = self.oa3(x11)
        x13 = self.oa4(x12)
        x = torch.cat([x_cat, x10, x11, x12, x13], dim=1)
        x = self.linear1(x)
        x_max = torch.max(x, dim=-1)[0]
        #x_mean = torch.mean(x, dim=-1)

        return x, x_max#, x_mean

class PCT_E(nn.Module):
    def __init__(self, samples=[256,128,512]):
        super().__init__()

        self.neighbor_embedding = NeighborEmbedding(samples)
        #self.neighbor_embedding = NeighborEmbedding_origing(samples)
        self.oa1 = OA(256)#MHeadOA(256)
        self.oa2 = OA(256)#MHeadOA(256)
        self.oa3 = OA(256)#MHeadOA(256)
        self.oa4 = OA(256)#MHeadOA(256)

        self.linear = nn.Sequential(
            nn.Conv1d(1280, 1280, kernel_size=1, bias=False),
            nn.BatchNorm1d(1280),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.neighbor_embedding(x)
        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.linear(x)
        #c = nn.MaxPool1d(x.size(-1))(x)
        #c = c.view(-1, 1024)
        #c = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        #x_mean = torch.mean(x, dim=-1)

        return x, x_max#, x_mean

class PCT(nn.Module):
    def __init__(self, samples=[128, 32]):
        super().__init__()

        self.neighbor_embedding = NeighborEmbedding(samples)
        #self.neighbor_embedding = NeighborEmbedding_origing(samples)
        self.oa1 = OA(512)#MHeadOA(256)
        self.oa2 = OA(512)#MHeadOA(256)
        self.oa3 = OA(512)#MHeadOA(256)
        self.oa4 = OA(512)#MHeadOA(256)

        self.linear = nn.Sequential(
            nn.Conv1d(2560, 2560, kernel_size=1, bias=False),
            nn.BatchNorm1d(2560),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.neighbor_embedding(x)
        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.linear(x)
        #c = nn.MaxPool1d(x.size(-1))(x)
        #c = c.view(-1, 1024)
        #c = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        #x_mean = torch.mean(x, dim=-1)

        return x, x_max#, x_mean


class Classification(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.linear1 = nn.Linear(2560, 512, bias=False)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, num_categories)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Segmentation(nn.Module):
    def __init__(self, part_num):
        super().__init__()

        self.part_num = part_num

        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean, cls_label):
        batch_size, _, N = x.size()

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature, cls_label_feature], dim=1)  # 1024 * 3 + 64

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x

class Policy(nn.Module):
    def __init__(self,args, actions):
        super().__init__()

        self.action_space = actions
        self.atoms =args.atoms

        self.convs1 = nn.Conv1d(512 + 512, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(65536, 512)
        self.fc2 = nn.Linear(519, 512)

        self.fc_h_v = spectral_norm(nn.Linear(65536, 512))
        self.fc_h_a = spectral_norm(nn.Linear(65536, 512))
        self.fc_z_v = NoisyLinear(512, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(512, self.action_space * self.atoms, std_init=args.noisy_std)

    
    def forward(self, x, c, a, p, log=False):
        batch_size, a, N = x.size()
        c = c.view(-1, 512, 1).repeat(1, 1, N)
        #a = c.view(-1, 1024, 1).repeat(1, 1, N)
        #p = p.view(-1, 7, 1).repeat(1, 1, N)
        x = torch.cat([x,c], dim=1)  # 1024 * 3 + 64
        x = F.relu(self.convs1(x))
        x = F.relu(self.convs2(x))
        x = F.relu(self.convs3(x))
        xb = torch.flatten(x, start_dim=1)
        #xb = F.relu(self.fc1(xb))
        #xb = torch.cat([xb,p], dim=1)
        #xb = F.relu(self.fc2(xb))
        
        v=self.fc_h_v(xb)
        v_uuv = self.fc_z_v(F.relu(v))  # Value stream
        a=self.fc_h_a(xb)
        a_uuv = self.fc_z_a(F.relu(a))  # Advantage stream

        v_uuv, a_uuv = v_uuv.view(-1, 1, self.atoms), a_uuv.view(-1, self.action_space, self.atoms)
        
        q_uuv = v_uuv + a_uuv - a_uuv.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            x = F.log_softmax(q_uuv, dim=2)  # Log probabilities with action over second dimension
        else:
            
            x = F.softmax(q_uuv, dim=2)  # Probabilities with action over second dimension
        #return  q_uuv #q_uav,
        return x



class NormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 3, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean):
        N = x.size(2)

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)
        
        x = torch.cat([x_max_feature, x_mean_feature, x], dim=1)

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x


"""
Classification networks.
"""

class NaivePCTCls(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = NaivePCT()
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


class SPCTCls(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = SPCT()
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


class PCTCls(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = PCT()
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


"""
Part Segmentation Networks.
"""

class NaivePCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
    
        self.encoder = NaivePCT()
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


class SPCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
    
        self.encoder = SPCT()
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


class PCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
    
        self.encoder = PCT(samples=[1024, 1024])
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x

class PCT_RL(nn.Module):
    def __init__(self, args, actions):
        super().__init__()
        print('sss')
        self.encoder = PCT_M()
        self.pol = Policy(args, actions)

    def forward(self, x, position, log=False):
        x,c, a = self.encoder(x)
        x = self.pol(x, c, a, position, log)
        return x

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc_z' in name:
                module.reset_noise()

class Conv_transformer(nn.Module):
    def __init__(self, samples=[512, 256]):
        super().__init__()

        #self.neighbor_embedding = NeighborEmbedding(samples)
        self.neighbor_embedding = Embedding(3,256)

        self.oa11 = OA(64)
        self.oa12 = OA(64)
        self.oa13 = OA(64)
        self.oa14 = OA(64)

        self.oa21 = OA(128)
        self.oa22 = OA(128)

        self.oa23 = OA(128)
        self.oa24 = OA(128)

        self.oa31 = OA(256)
        self.oa32 = OA(256)



        self.linearLayer_weights_11 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.linearLayer_weights_12 = nn.Conv1d(256, 256, kernel_size=1, bias=False)

        self.linearLayer_weights_2 = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        self.linearLayer_weights_3 = nn.Conv1d(1280, 1280, kernel_size=1, bias=False)
        self.linearLayer_weights_4 = nn.Conv1d(320, 64, kernel_size=1, bias=False)

        self.linearLayer_1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Hoa=OA(256)

        self.linearLayer_2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.neighbor_embedding(x)

        x01 = x.narrow(1,0,64)
        x02 = x.narrow(1,64,64)
        x03 = x.narrow(1,128,64)
        x04 = x.narrow(1,192,64)
 
        x11 = self.oa11(x01)
        x12 = self.oa12(x02)
        x13 = self.oa13(x03)
        x14 = self.oa14(x04)


        x_cat1 = torch.cat([x11, x12, x13, x14], dim=1)
        #print(x_cat1.shape)
        #x_cat2 = torch.cat([x03, x04, x13, x14], dim=1)

        x_11 =  F.relu(self.linearLayer_weights_11(x_cat1))
        #x_12 =  F.relu(self.linearLayer_weights_12(x_cat2))

        x01 = x_11.narrow(1,0,128)
        x02 = x_11.narrow(1,128,128)

        x1 = self.oa21(x01)
        x2 = self.oa23(x02)
        

        x_cat2 = torch.cat([ x1, x2], dim=1)
        x =  F.relu(self.linearLayer_weights_2(x_cat2))
        
        x01 = x.narrow(1,0,128)
        x02 = x_11.narrow(1,128,128)

        x4 = self.oa24(x3)        
        x2 = self.oa22(x1)
        
        x_cat3 = torch.cat([ x1, x2, x3, x4], dim=1)
        x =  F.relu(self.linearLayer_weights_2(x_cat2))

        x1 = self.oa31(x)
        x2 = self.oa32(x1)

        x_cat3 = torch.cat([x, x_cat1,x_cat2, x1, x2], dim=1)
        x =  F.relu(self.linearLayer_weights_3(x_cat3))
        """
        x1 = self.oa13(x03)
        x2 = self.oa23(x1)
        x3 = self.oa33(x2)
        x4 = self.oa43(x3)
        x_cat = torch.cat([x03, x1, x2, x3, x4], dim=1)
        x33 =  F.relu(self.linearLayer_weights_3(x_cat))

        x1 = self.oa14(x04)
        x2 = self.oa24(x1)
        x3 = self.oa34(x2)
        x4 = self.oa44(x3)
        x_cat = torch.cat([x04, x1, x2, x3, x4], dim=1)
        x44 =  F.relu(self.linearLayer_weights_4(x_cat))
        
        x_cat = torch.cat([x11, x22, x33, x44], dim=1)
        f_x=self.linearLayer_1(x_cat)
        f_x = self.Hoa(f_x)
        x=self.linearLayer_2(f_x)

        #c = nn.MaxPool1d(x.size(-1))(x)
        #c = c.view(-1, 1024)
        """
        #c = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        
        x_max = torch.max(x, dim=-1)[0]
        #x_mean = torch.mean(x, dim=-1)
        
        return x , x_max#, x_mean

class Multihead_PCT(nn.Module):
    def __init__(self, samples=[512, 256]):
        super().__init__()

        #self.neighbor_embedding = NeighborEmbedding(samples)
        self.neighbor_embedding = Embedding(3,256)

        self.oa11 = OA(64)
        self.oa21 = OA(64)
        self.oa31 = OA(64)
        self.oa41 = OA(64)

        self.oa12 = OA(64)
        self.oa22 = OA(64)
        self.oa32 = OA(64)
        self.oa42 = OA(64)

        self.oa13 = OA(64)
        self.oa23 = OA(64)
        self.oa33 = OA(64)
        self.oa43 = OA(64)

        self.oa14 = OA(64)
        self.oa24 = OA(64)
        self.oa34 = OA(64)
        self.oa44 = OA(64)

        self.linearLayer_weights_1 = nn.Conv1d(256, 64, kernel_size=1, bias=False)
        self.linearLayer_weights_2 = nn.Conv1d(256, 64, kernel_size=1, bias=False)
        self.linearLayer_weights_3 = nn.Conv1d(256, 64, kernel_size=1, bias=False)
        self.linearLayer_weights_4 = nn.Conv1d(256, 64, kernel_size=1, bias=False)

        #self.linearLayer_weights_f = nn.Linear(4*64, 64, bias=False)
        self.linear = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.neighbor_embedding(x)

        x01 = x.narrow(1,0,64)
        x02 = x.narrow(1,64,64)
        x03 = x.narrow(1,128,64)
        x04 = x.narrow(1,192,64)
 
        x1 = self.oa11(x01)
        x2 = self.oa21(x1)
        x3 = self.oa31(x2)
        x4 = self.oa41(x3)
        x_cat = torch.cat([x01, x1, x2, x3], dim=1)
        x11 =  F.relu(self.linearLayer_weights_1(x_cat))

        x1 = self.oa12(x02)
        x2 = self.oa22(x1)
        x3 = self.oa32(x2)
        x4 = self.oa42(x3)
        x_cat = torch.cat([x02, x1, x2, x3], dim=1)
        x22 =  F.relu(self.linearLayer_weights_2(x_cat))


        x1 = self.oa13(x03)
        x2 = self.oa23(x1)
        x3 = self.oa33(x2)
        x4 = self.oa43(x3)
        x_cat = torch.cat([x03, x1, x2, x3], dim=1)
        x33 =  F.relu(self.linearLayer_weights_3(x_cat))

        x1 = self.oa14(x04)
        x2 = self.oa24(x1)
        x3 = self.oa34(x2)
        x4 = self.oa44(x3)
        x_cat = torch.cat([x04, x1, x2, x3], dim=1)
        x44 =  F.relu(self.linearLayer_weights_4(x_cat))
        
        x_cat = torch.cat([x11, x22, x33, x44], dim=1)
        x = self.linear(x_cat)

        #c = nn.MaxPool1d(x.size(-1))(x)
        #c = c.view(-1, 1024)122448Mr!
        
        #c = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        #x_max = torch.max(x, dim=-1)[0]
        #x_mean = torch.mean(x, dim=-1)

        return x #, x_max, x_mean

class Policy2(nn.Module):
    def __init__(self,args, actions):
        super().__init__()

        self.action_space = actions
        self.atoms =args.atoms

        self.convs1 = nn.Conv1d(2560, 1024, 1)
        self.convs2 = nn.Conv1d(1024, 256, 1)
        self.convs3 = nn.Conv1d(256, 64, 1)
        #self.convs4 = nn.Conv1d(256, 128, 1)

        self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(64)
        #self.bns3 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(65536, 512)
        self.fc2 = nn.Linear(519, 512)

        self.fc_h_v = spectral_norm(nn.Linear(8192, 512))
        self.fc_h_a = spectral_norm(nn.Linear(8192, 512))
        self.fc_z_v = NoisyLinear(512, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(512, self.action_space * self.atoms, std_init=args.noisy_std)

    
    def forward(self, x, m, log=False):
        batch_size, a, N = x.size()
        c = m.view(-1, a, 1).repeat(1, 1, N)
        #a = c.view(-1, 1024, 1).repeat(1, 1, N)
        #p = p.view(-1, 7, 1).repeat(1, 1, N)
        x = torch.cat([x,c], dim=1)  # 1024 * 3 + 64
        x = F.relu(self.convs1(x))
        x = F.relu(self.convs2(x))
        x = F.relu(self.convs3(x))
        #x = F.relu(self.convs4(x))
        xb = torch.flatten(x, start_dim=1)
        #xb = F.relu(self.fc1(xb))
        #xb = torch.cat([xb,p], dim=1)
        #xb = F.relu(self.fc2(xb))
        
        v=F.relu(self.fc_h_v(xb))
        v_uuv = self.fc_z_v(F.relu(v))  # Value stream
        a=F.relu(self.fc_h_a(xb))
        a_uuv = self.fc_z_a(F.relu(a))  # Advantage stream

        v_uuv, a_uuv = v_uuv.view(-1, 1, self.atoms), a_uuv.view(-1, self.action_space, self.atoms)
        
        q_uuv = v_uuv + a_uuv - a_uuv.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            x = F.log_softmax(q_uuv, dim=2)  # Log probabilities with action over second dimension
        else:
            
            x = F.softmax(q_uuv, dim=2)  # Probabilities with action over second dimension
        #return  q_uuv #q_uav,
        return x

class Multihead_PCT_RL(nn.Module):
    def __init__(self, args, actions):
        super().__init__()
    
        self.encoder = PCT_E()
        self.policy2 = Policy2(args, actions)

    def forward(self, x, position, log=False):
        x, mean = self.encoder(x)
        x = self.policy2(x, mean, log)
        return x

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc_z' in name:
                module.reset_noise()

"""
Normal Estimation networks.
"""

class NaivePCTNormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.encoder = NaivePCT()
        self.ne = NormalEstimation()

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.ne(x, x_max, x_mean)
        return x


class SPCTNormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.encoder = SPCT()
        self.ne = NormalEstimation()

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.ne(x, x_max, x_mean)
        return x


class PCTNormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.encoder = PCT(samples=[1024, 1024])
        self.ne = NormalEstimation()

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.ne(x, x_max, x_mean)
        return x


if __name__ == '__main__':
    pc = torch.rand(4, 3, 1024).to('cuda')
    cls_label = torch.rand(4, 16).to('cuda')

    # testing for cls networks
    naive_pct_cls = NaivePCTCls().to('cuda')
    spct_cls = SPCTCls().to('cuda')
    pct_cls = PCTCls().to('cuda')

    print(naive_pct_cls(pc).size())
    print(spct_cls(pc).size())
    print(pct_cls(pc).size())

    # testing for segmentation networks
    naive_pct_seg = NaivePCTSeg().to('cuda')
    spct_seg = SPCTSeg().to('cuda')
    pct_seg = PCTSeg().to('cuda')

    print(naive_pct_seg(pc, cls_label).size())
    print(spct_seg(pc, cls_label).size())
    print(pct_seg(pc, cls_label).size())

    # testing for normal estimation networks
    naive_pct_ne = NaivePCTNormalEstimation().to('cuda')
    spct_ne = SPCTNormalEstimation().to('cuda')
    pct_ne = PCTNormalEstimation().to('cuda')

    print(naive_pct_ne(pc).size())
    print(spct_ne(pc).size())
    print(pct_ne(pc).size())