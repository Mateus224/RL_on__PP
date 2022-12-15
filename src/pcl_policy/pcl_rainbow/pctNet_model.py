import torch
import torch.nn as nn
import torch.nn.functional as F

from pcl_policy.pcl_rainbow.module import Embedding, NeighborEmbedding, OA, SA
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


class PCT(nn.Module):
    def __init__(self, samples=[512, 256]):
        super().__init__()

        #self.neighbor_embedding = NeighborEmbedding(samples)
        self.neighbor_embedding = Embedding(3,128)
        self.oa1 = OA(128)
        self.oa2 = OA(128)
        self.oa3 = OA(128)
        self.oa4 = OA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(640, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
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
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class Classification(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_categories)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

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
    
        self.encoder = PCT()
        self.pol = Policy(args, actions)

    def forward(self, x, position, log=False):
        x,c, a = self.encoder(x)
        x = self.pol(x, c, a, position, log)
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