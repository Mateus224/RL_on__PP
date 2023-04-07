import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcl_policy.pcl_rainbow.util import sample_and_knn_group, sample_and_knn_group1, sample_and_knn_group_


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

class SG(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels, k):
        super(SG, self).__init__()

        self.s = s
        self.k = k

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x, coords, k=32):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)           # (B, N, in_channels//2)
        xyz, new_feature, batch_index_arr01 = sample_and_knn_group_(s=self.s, k=k, coords=coords, features=x)  # [B, s, 3], [B, s, 32, in_channels]
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return xyz,new_feature, batch_index_arr01

class SG_1(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG_1, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)           # (B, N, in_channels//2)
        if self.s==600:
            new_feature = sample_and_knn_group(s=self.s, k=32, coords=coords, features=x)  # [B, s, 3], [B, s, 32, in_channels]
        else:
            new_feature = sample_and_knn_group1(s=self.s, k=16, coords=coords, features=x)
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return new_feature







class SG_4(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG_4, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.convE1 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.convE2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bnE1 = nn.BatchNorm1d(64)
        self.bnE2 = nn.BatchNorm1d(64)
    
    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        feature_k = x.narrow(1,0,128)
        feature_s1 = x.narrow(1,128,64)
        feature_s2 = x.narrow(1,192,64)
        feature_k = feature_k.permute(0, 2, 1)    
        #feature_k2 = feature_k2.permute(0, 2, 1)       
        new_feature1, new_feature2 = sample_and_knn_group1(s=self.s, k=22, coords=coords, features=feature_k)  # [B, s, 3], [B, s, 32, in_channels]
        #new_feature2 = sample_and_knn_group(s=self.s, k=22, coords=coords, features=feature_k2)
        #new_feature1 = F.relu(self.bnE1(self.convE1(new_feature1)))                   # [Bxs, in_channels, 32]
        #features1 = F.relu(self.bnE2(self.convE2(new_feature1)))  
        
        feature_k1=self.grouped_features(new_feature1)
        feature_k2=self.grouped_features0(new_feature2)
        #features2=self.grouped_features(new_feature2)
        #features3=self.grouped_features(new_feature3)
        #features4=self.grouped_features(new_feature4)
        return feature_k1, feature_k2, feature_s1, feature_s2
        
    def grouped_features(self, new_feature):
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return new_feature

    def grouped_features0(self, new_feature):
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bnE1(self.convE1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bnE2(self.convE2(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return new_feature
        
class NeighborEmbedding(nn.Module):
    def __init__(self, samples=[256, 128]):
        super(NeighborEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.sg0 = SG(s=samples[0], in_channels=128, out_channels=128)
        self.sg1 = SG(s=samples[1], in_channels=256, out_channels=256)
        #self.sg2 = SG(s=samples[2], in_channels=256, out_channels=256)
    
    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]

        xyz1,features1 = self.sg0(features, xyz,32)         # [B, 128, 512]
        _,features2 = self.sg1(features1, xyz1,32)          # [B, 256, 256]
        #features2 = self.sg2(features2, xyz,32)
        return features2

class NeighborEmbedding_own(nn.Module):
    def __init__(self, samples=[256, 128, 64]):
        super(NeighborEmbedding_own, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        self.sg0 = SG_1(s=600, in_channels=256, out_channels=256)
        self.oa01 = OA(256)
        self.oa02 = OA(256)
        self.sg1 = SG(s=64, in_channels=512, out_channels=256, k=32)
        self.oa11 = OA(256)
        self.oa12 = OA(256)
        self.sg2 = SG(s=8, in_channels=512, out_channels=256, k=4)
        self.oa21 = OA(256)
        self.oa22 = OA(256)   
        #self.sg3 = SG(s=samples[2], in_channels=512, out_channels=256)
        #self.oa31 = OA(256)
        #self.oa32 = OA(256)   
        #self.oao1 =OAS(256)
        #self.oao2 =OAS(256)
        #self.oao3 =OAS(256)
        #self.linear = nn.Sequential(
        #    nn.Conv1d(256, 256, kernel_size=1, bias=False),
        #    nn.BatchNorm1d(256),
        #    nn.LeakyReLU(negative_slope=0.2)
        #)
        #self.oa4 = OA(256)
    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]

        features0 = self.sg0(features, xyz) 
        features01=self.oa01(features0)
        features02=self.oa02(features01)
        xyz1, features1, batch_index_arr01 = self.sg1(features02, xyz, k=16)         # [B, 128, 512]
        features11=self.oa11(features1)
        features12=self.oa12(features11)
        xyz1, features2, batch_index_arr02 = self.sg2(features12, xyz1, k=4)         # [B, 128, 512]
        features21=self.oa11(features2)
        features22=self.oa12(features21)
        features02 = features02.permute(0, 2, 1)
        features12 = features12.permute(0, 2, 1)
        features22 = features22.permute(0, 2, 1)
        features12=self.index_points(features12, batch_index_arr02, features22) 
        features02=self.index_points(features02, batch_index_arr01, features12)
        x = features02.permute(0, 2, 1)
        
        
        
        #batch_index_arr02=batch_index_arr02.astype(int)
        #batch_index_arr01=batch_index_arr01.astype(int)
        #features02 = features02.permute(0, 2, 1)  # [B, N ,3]
        #print(features02[0,batch_index_arr01[:,:],:].shape)
        #features02[:,batch_index_arr01[:,:],:]=features12.permute(0, 2, 1)
        #features02[:,batch_index_arr01[:,batch_index_arr02[:,:]][:],:]=features22.permute(0, 2, 1)
        #features02 = features02.permute(0, 2, 1)

        return x


    def index_points(self, points, idx, new_points):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        
        Output:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        points[batch_indices, idx ,:] = new_points
        return points


class NeighborEmbedding_origing(nn.Module):
    def __init__(self, samples=[512, 256]):
        super(NeighborEmbedding_origing, self).__init__()

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

        xyz1, features1 = self.sg1(features, xyz)         # [B, 128, 512]
        _, features2 = self.sg2(features1, xyz1)          # [B, 256, 256]
        return features2

class MHeadOA(nn.Module):
    def __init__(self, features, heads=8):
        super(MHeadOA,self).__init__()
        self.heads=heads
        self.layers=nn.ModuleList()
        self.linear = nn.Sequential(
            nn.Conv1d(features*self.heads, features, kernel_size=1, bias=False),
            nn.BatchNorm1d(features),
            nn.LeakyReLU(negative_slope=0.2)
        )        
        for i in range(heads):
            self.layers.append(OA(features))


    
    def forward(self,x):
        #x1 = x.narrow(1,0,64)
        #x2 = x.narrow(1,64,64)
        #x3 = x.narrow(1,128,64)
        #x4 = x.narrow(1,192,64)
        outputs = [net(x) for net in self.layers]
        #x1=self.layers[0](x1)
        #x2=self.layers[1](x2)
        #x3=self.layers[2](x3)
        #x4=self.layers[3](x4)
        #x=torch.cat([x1,x2,x3,x3], dim=1)
        x=torch.cat(outputs, dim=1)
        x=self.linear(x)
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
        self.v_conv = nn.Conv1d(channels, channels , 1)
        #self.q_conv = nn.Conv1d(3, 64 , 1, bias=False)
        #self.k_conv = nn.Conv1d(3, 64 , 1, bias=False)
        #self.q_conv.weight = self.k_conv.weight
        #self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels , channels , 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
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
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        #x = self.act(self.after_norm(self.trans_conv(x_r)))

        return x

class OA_(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OA_, self).__init__()

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
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        #x = self.act(self.after_norm(self.trans_conv(x_r)))

        return x


class OAS(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OAS, self).__init__()

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

    def forward(self, q, x):
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
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        #x = self.act(self.after_norm(self.trans_conv(x_r)))

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

        self.trans_conv = nn.Conv1d(channels // 2, channels // 2, 1)
        self.after_norm = nn.BatchNorm1d(channels // 2)
        
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