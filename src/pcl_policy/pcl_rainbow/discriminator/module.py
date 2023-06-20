import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pcl_policy.pcl_rainbow.discriminator.util import sample_and_knn_group, knn_group


class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=256):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class EmbeddingV2(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=256):
        super(EmbeddingV2, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv20 = nn.Conv1d(out_channels, out_channels//8, kernel_size=1, bias=False)
        self.conv21 = nn.Conv1d(out_channels, out_channels//8, kernel_size=1, bias=False)
        self.conv22 = nn.Conv1d(out_channels, out_channels//8, kernel_size=1, bias=False)
        self.conv23 = nn.Conv1d(out_channels, out_channels//8, kernel_size=1, bias=False)
        self.conv24 = nn.Conv1d(out_channels, out_channels//8, kernel_size=1, bias=False)
        self.conv25 = nn.Conv1d(out_channels, out_channels//8, kernel_size=1, bias=False)
        self.conv26 = nn.Conv1d(out_channels, out_channels//8, kernel_size=1, bias=False)
        self.conv27 = nn.Conv1d(out_channels, out_channels//8, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn20 = nn.BatchNorm1d(out_channels//8)
        self.bn21 = nn.BatchNorm1d(out_channels//8)
        self.bn22 = nn.BatchNorm1d(out_channels//8)
        self.bn23 = nn.BatchNorm1d(out_channels//8)
        self.bn24 = nn.BatchNorm1d(out_channels//8)
        self.bn25 = nn.BatchNorm1d(out_channels//8)
        self.bn26 = nn.BatchNorm1d(out_channels//8)
        self.bn27 = nn.BatchNorm1d(out_channels//8)


    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x0 = F.relu(self.bn20(self.conv20(x)))
        x1 = F.relu(self.bn21(self.conv21(x)))
        x2 = F.relu(self.bn22(self.conv22(x)))
        x3 = F.relu(self.bn23(self.conv23(x)))
        x4 = F.relu(self.bn24(self.conv24(x)))
        x5 = F.relu(self.bn25(self.conv25(x)))
        x6 = F.relu(self.bn26(self.conv26(x)))
        x7 = F.relu(self.bn27(self.conv27(x)))
        return x0,x1,x2,x3,x4,x5,x6,x7


class SA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(SA, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        attention = self.softmax(energy)                      # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        
        # residual
        x = x + x_s

        return x

class SG_knn(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG_knn, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x, coords, k):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)           # (B, N, in_channels//2)
        new_feature = knn_group(s=self.s, k=32, coords=coords, features=x)  # [B, s, 3], [B, s, 32, in_channels]
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                 # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return coords, new_feature

class SG(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x, coords, k):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)           # (B, N, in_channels//2)
        new_xyz, new_feature = sample_and_knn_group(s=self.s, k=32, coords=coords, features=x)  # [B, s, 3], [B, s, 32, in_channels]
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                 # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return new_xyz, new_feature


class OwnPCT(nn.Module):
    def __init__(self, samples=[256, 128, 64]):
        super(OwnPCT, self).__init__()

        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.sg_knn = SG_knn(s=1620, in_channels=256, out_channels=128)
        self.sg0 = SG(s=512, in_channels=256, out_channels=128)
        self.sg1 = SG(s=128, in_channels=256, out_channels=128)#192

        #self.sg1 = SG(s=128, in_channels=256, out_channels=256)#192

        #self.sg2 = SG(s=32, in_channels=384, out_channels=384)#384
        #self.sg3 = SG(s=128, in_channels=256, out_channels=128)#384
        #self.sg4 = SG(s=64, in_channels=256, out_channels=128)#384
        #self.sg5 = SG(s=32, in_channels=256, out_channels=128)#384


        self.oa01 = OA(128)
        self.oa02 = OA(128) 

        self.oas0 =OA_2b(128, 128, 128)
        self.oas1 =OA_2b(128, 128, 128)

        self.oas_big =OA_2b(128, 128, 128)


        self.oa1 = OA(128)
        self.oa2 = OA(128)    
        self.oa3 = OA(128)
        self.oa4 = OA(128)
        self.oa5 = OA(128)
        self.oa6 = OA(128)

        self.linearL = nn.Sequential(
            nn.Conv1d(640, 640, kernel_size=1, bias=False), #786
            nn.BatchNorm1d(640),
            nn.LeakyReLU(negative_slope=0.2)
        ) 

        self.linear00 = nn.Sequential(
            nn.Conv1d(640, 640, kernel_size=1, bias=False), #786
            nn.BatchNorm1d(640),
            nn.LeakyReLU(negative_slope=0.2)
        ) 
        self.linear01 = nn.Sequential(
            nn.Conv1d(192, 192, kernel_size=1, bias=False), #786
            nn.BatchNorm1d(192),
            nn.LeakyReLU(negative_slope=0.2)
        ) 
        self.linear011 = nn.Sequential(
            nn.Conv1d(384, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.linear11 = nn.Sequential(
            nn.Conv1d(384, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Sequential(
            nn.Conv1d(192, 192, kernel_size=1, bias=False),#1536
            nn.BatchNorm1d(192),
            nn.LeakyReLU(negative_slope=0.2)
        ) 

        self.linear3= nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.linear4= nn.Sequential(
            nn.Conv1d(1890, 1890, kernel_size=1, bias=False),
            nn.BatchNorm1d(1890),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        if x.ndim==3:
            xyz = x.permute(0, 2, 1)  # [B, N ,3]
        elif x.ndim==2:
            xyz = x.permute(1,0)
        x = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features0 = F.relu(self.bn2(self.conv2(x))) # [B, 64, N]

        xyz1, features0 = self.sg_knn(features0, xyz, k=1620) 
        #xyz1, features1 = self.sg0(features0, xyz, k=32)         # [B, 128, 512]

        """
        x02 = self.oa01(features1)
        x03 = self.oa02(x02)
        x = torch.cat([ features1, x02, x03], dim=1)
        x = self.linear01(x)
        batch_size, _, N = x.size()
        x_max = torch.max(x, dim=-1)[0]
        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x = torch.cat([x,x_max_feature], dim=1)
        features1 = self.linear011(x)
        """

        #xyz1, features2 = self.sg1(features1, xyz1, k=32)  #features211       # [B, 128, 512]
        #x=self.oa00(features22) #features221
        #features222=self.oa001(features221)
        #--
        #features223=self.oa0011(features222)
        #features224=self.oa0012(features223)
        #--
        #x = torch.cat([ features22, x], dim=1)
        #x = self.linear0(x)
        #batch_size, _, N = x.size()
        #x_max = torch.max(x, dim=-1)[0]
        #_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        #x = torch.cat([x, x_max_feature], dim=1)
        #features22 = self.linear01(x)
        """
        x02 = self.oa01(features2)
        x03 = self.oa02(x02)
        x = torch.cat([ features2, x02, x03], dim=1)
        x = self.linear1(x)
        batch_size, _, N = x.size()
        x_max = torch.max(x, dim=-1)[0]
        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x = torch.cat([x,x_max_feature], dim=1)
        
        features2 = self.linear11(x)
        """
        #features335 = self.oas000(features33, features22) #x01
        #x=features335
        #batch_size, _, N = x.size()
        #x_max = torch.max(x, dim=-1)[0]
        #x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        #x = torch.cat([x,x_max_feature], dim=1)
        #features335 = self.linear00(x)


        #features334 = self.oas00(features22, features1) #x01
        #x=features334
        #batch_size, _, N = x.size()
        #x_max = torch.max(x, dim=-1)[0]
        #x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        #x = torch.cat([x,x_max_feature], dim=1)
        #features334 = self.linear0(x)

        #x001 = self.oas0(features2, features1) #x01
        #x=features444
        #batch_size, _, N = x.size()
        #x_max = torch.max(x, dim=-1)[0]
        #x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        #x = torch.cat([x,x_max_feature], dim=1)
        #x01 = self.linear1(x)
     


        #x002 = self.oas1(x001, features0)
        #x1 = torch.cat([features0, features1, features2], dim=2)
        #x01 = self.oas_big(x1, features0)
        x02 = self.oa3(features0)
        x03 = self.oa4(x02)
        x04 = self.oa5(x03)
        x05 = self.oa6(x04)
        
        x = torch.cat([features0, x05,x02, x03, x04], dim=1)
        x = self.linearL(x)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)
        return x, x_max, x_mean


class NeighborEmbedding(nn.Module):
    def __init__(self, samples=[512, 256]):
        super(NeighborEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.sg1 = SG(s=samples[0], in_channels=128, out_channels=128)
        self.sg2 = SG(s=samples[1], in_channels=256, out_channels=256)
    
    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]

        xyz1, features1 = self.sg1(features, xyz, 32)         # [B, 128, 512]
        xyz, features2 = self.sg2(features1, xyz1, 32)          # [B, 256, 256]

        return xyz, features2

class NeighborEmbeddingHigher(nn.Module):
    def __init__(self, samples=256):
        super(NeighborEmbeddingHigher, self).__init__()
        features=int(1024/samples)*8
        #fe= int(features/2)
        self.conv1 = nn.Conv1d(3, features, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(features, features, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(features)
        self.bn2 = nn.BatchNorm1d(features)

        self.sg1 = SG(s=samples, in_channels=features*2, out_channels=features*2-3)
        #self.sg2 = SG(s=samples, in_channels=288, out_channels=288)
        
    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]

        xyz, features = self.sg1(features, xyz)         # [B, 128, 512]
        #xyz, features2 = self.sg2(features1, xyz1)          # [B, 256, 256]
        #print(xyz.shape, features.shape)
        
        xyz = xyz.permute(0, 2, 1)
        features = torch.cat([xyz, features] ,dim=1)
        #print(features.shape)
        

        return xyz, features


class OA_3(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels, m):
        super(OA_3, self).__init__()
        
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels//m, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels , 1)
        self.trans_conv = nn.Conv1d(channels , channels , 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self,q, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(q).permute(0, 2, 1)
        
        x_k = self.k_conv(x)    
        x_v = self.v_conv(q)
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        #x = x_r+ q
        return x_r
        

class OA_2b(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels_m, q, x):
        super(OA_2b, self).__init__()

        #q= 32 256
        #k= 128   128

        self.q_conv = nn.Conv1d(x, channels_m//4  , 1, bias=False)
        self.k_conv = nn.Conv1d(q, channels_m//4 , 1, bias=False)
        self.v_conv = nn.Conv1d(x, channels_m , 1)

        self.x_old = nn.Conv1d(q, channels_m, 1)

        self.trans_conv = nn.Conv1d(channels_m, channels_m, 1)
        self.x_after_norm = nn.BatchNorm1d(channels_m)
        self.after_norm = nn.BatchNorm1d(channels_m)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self,q, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(q).permute(0, 2, 1)
        
        x_k = self.k_conv(x)    
        x_v = self.v_conv(q)
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here
        x_r = torch.bmm(x_v, attention)
        #old_x=self.x_after_norm(self.x_old(x))
        x_r = self.act(self.after_norm(self.trans_conv(x-x_r)))
        x = x_r+ x
        return x_r
       

class OA_2(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels_m,  d=2, o=1):
        super(OA_2, self).__init__()

        #q= 32 256
        #k= 128   128
        self.q_conv = nn.Conv1d(channels_m, channels_m//d , 1, bias=False)
        self.k_conv = nn.Conv1d(channels_m//d, channels_m//d, 1, bias=False)
        self.v_conv = nn.Conv1d(channels_m, channels_m//d, 1)
        self.trans_conv = nn.Conv1d(channels_m //d, channels_m//(d//o), 1)
        self.after_norm = nn.BatchNorm1d(channels_m//(d//o))
        
        self.x_old = nn.Conv1d(channels_m //d, channels_m//(d//o), 1)
        self.x_after_norm = nn.BatchNorm1d(channels_m//(d//o))
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self,q, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(q).permute(0, 2, 1)
        
        x_k = self.k_conv(x)    
        x_v = self.v_conv(q)
        
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x-x_r)))
        x = x_r+ x #self.x_after_norm(self.x_old(x))
        return x

 
class OA(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.dp1 = nn.Dropout(p=0.2)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        #x_r = self.dp1(x_r)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x +  x_r

        return x
        
class OAQ(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OAQ, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels , 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels , 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels , 1)
        #self.q_conv = nn.Conv1d(3, 64 , 1, bias=False)
        #self.k_conv = nn.Conv1d(3, 64 , 1, bias=False)
        #self.q_conv.weight = self.k_conv.weight
        #self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels , channels , 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, q,  x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(q).permute(0, 2, 1)
        
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(q - x_r)))
        x = q + x_r

        return x

class OAO(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OAO, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels // 2, 1)
        #self.q_conv = nn.Conv1d(3, 64 , 1, bias=False)
        #self.k_conv = nn.Conv1d(3, 64 , 1, bias=False)
        #self.q_conv.weight = self.k_conv.weight
        #self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels // 4, channels // 4, 1)
        self.after_norm = nn.BatchNorm1d(channels // 4)
        self.dp1 = nn.Dropout(p=0.2)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        #x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        #x = x + x_r
        #x_r = self.dp1(x_r) 
        x = self.act(self.after_norm(self.trans_conv(x_r)))

        return x

if __name__ == '__main__':
    """
    Please be careful to excute the testing code, because
    it may cause the GPU out of memory.
    """
    
    pc = torch.rand(32, 3, 1024).to('cuda')

    # testing for Embedding
    embedding = Embedding().to('cuda')
    out = embedding(pc)
    print("Embedding output size:", out.size())

    # testing for SA
    sa = SA(channels=out.size(1)).to('cuda')
    out = sa(out)
    print("SA output size:", out.size())

    # testing for SG
    coords = torch.rand(32, 1024, 3).to('cuda')
    features = torch.rand(32, 64, 1024).to('cuda')
    sg = SG(512, 128, 128).to('cuda')
    new_coords, out = sg(features, coords)
    print("SG output size:", new_coords.size(), out.size())

    # testing for NeighborEmbedding
    ne = NeighborEmbedding().to('cuda')
    out = ne(pc)
    print("NeighborEmbedding output size:", out.size())

    # testing for OA
    oa = OA(256).to('cuda')
    out = oa(out)
    print("OA output size:", out.size())
