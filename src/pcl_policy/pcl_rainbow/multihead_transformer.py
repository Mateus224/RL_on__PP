import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcl_policy.pcl_rainbow.module import Embedding as Emebed
from torch.nn.utils import spectral_norm



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



class SelfAttention(nn.Module):
    def __init__(self,embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size =embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim*heads == embed_size), 'Embed size needs to be div by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)


    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = values.reshape(N, key_len, self.heads, self.head_dim)
        queries = values.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        #print(queries.shape, 'qShape')
        
        #energy = torch.einsum("nqhd,nkhd-->nhqk", [queries, keys])
        energy = torch.einsum("nqhd, nkhd -> nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)  (32, 128, 4, 64)
        # key shape: (N, key_len, heads, heads_dim) (32, 3, 4, 64)
        # energy shape: (N, heads, query_len, key_dim) (32, 4, 128, 3)
        #energy2 shape: ()

        #if mask is not None:
        #    energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention=torch.softmax(energy/(self.embed_size**(1/2)),dim=3 )

        out =torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim )
        #attention shape: (N, heads, query_len, key_len?) (32, 4, 128, 3)
        #value shape: (N, value_len, heads, heads_dim) (32, 128, 4, 64)
        #after einsum (N,query_len, heads, head_dim) flat the last two dimesnsions
        out =self.fc_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention+query))
        #x = self.norm1(attention+query)
        forward =self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out
    
class Encoder(nn.Module):
    def __init__(
        self,
        points_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout
    ):
        super(Encoder, self).__init__()
        self.embed_size =embed_size
        self.device =device
        self.embedding = nn.Embedding(points_size, embed_size)
        self.position_embedding = nn.Embedding(1, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                    )
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.fc_out = nn.Linear(512,512)
        self.dropout = nn.Dropout(dropout)

    def forward (self, x, mask):
        N, features, point_size = x.shape
        #out = self.embedding(x)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        x = self.linear(x)
        x = nn.Flatten(1)(x)

        return x

class RL_PNet(nn.Module):
    def __init__(
        self,
        args,
        action_space,
        point_size=512,
        embed_size = 128,
        num_layers = 4,
        forward_expansion = 4,
        heads = 8,
        dropout =0,
        device = "cuda"
    ):
        super(RL_PNet, self).__init__()
        self.encoder =Encoder(
            point_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout
        )
        self.atoms = 51
        self.action_space = action_space
        self.device = device
        self.conv1 = nn.Conv1d(3, embed_size, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(embed_size, embed_size, kernel_size=1, bias=False)
        self.fc_h_v = spectral_norm(nn.Linear(65536, 512))
        self.fc_h_a = spectral_norm(nn.Linear(65536, 512))
        self.fc_z_v = NoisyLinear(512, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(512, action_space * self.atoms, std_init=args.noisy_std)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, pos, log=False):
        mask=None
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.swapaxes(1,2)
        x = self.encoder(x, mask)
        
        v=self.fc_h_v(x)
        v_uuv = self.fc_z_v(F.relu(v))  # Value stream
        a=self.fc_h_a(x)
        a_uuv = self.fc_z_a(F.relu(a))  # Advantage stream

        v_uuv, a_uuv = v_uuv.view(-1, 1, self.atoms), a_uuv.view(-1, self.action_space, self.atoms)
        
        q_uuv = v_uuv + a_uuv - a_uuv.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            q_uuv = F.log_softmax(q_uuv, dim=2)  # Log probabilities with action over second dimension
        else:
            
            q_uuv = F.softmax(q_uuv, dim=2)  # Probabilities with action over second dimension
        #return  q_uuv #q_uav,
        return q_uuv

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc_z' in name:
                module.reset_noise()
