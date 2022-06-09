import yaml
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def init_weigths(module):
    """ fanin initialization. """
    if type(module) == nn.Linear:
        size = module.weight.size()
        if len(size) == 2:
            fan_in = size[0]
        elif len(size) > 2:
            fan_in = np.prod(size[1:])
        else:
            raise Exception("Shape must be have dimension at least 2.")
        bound = 1. / np.sqrt(fan_in)
        module.weight.data.uniform_(-bound, bound)
        module.bias.data.uniform_(-bound, bound)


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()

        self.fc_f = nn.utils.spectral_norm(nn.Linear(in_channel, in_channel//8))
        self.fc_g = nn.utils.spectral_norm(nn.Linear(in_channel, in_channel//8))
        self.fc_h = nn.utils.spectral_norm(nn.Linear(in_channel, in_channel))

        self.softmax = nn.Softmax(-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: B, N
        f_projection = self.fc_f(x)  # BxN', N'=N//8
        g_projection = self.fc_g(x)  # BxN'
        h_projection = self.fc_h(x)  # BxN

        f_projection = f_projection.unsqueeze(-2)  # BxNxN'
        g_projection = g_projection.unsqueeze(-1)  # BxN'xN
        h_projection = h_projection.unsqueeze(-1)  # Bx1xN

        attention_map = f_projection @ g_projection  # BxNxN
        attention_map = self.softmax(attention_map)  # BxNxN

        out = torch.bmm(h_projection, attention_map)  # Bx1xN

        out = self.gamma*out + x
        return out


def adaIN(feature, mean_style, std_style, eps=1e-5):
    # feature: B, N, fdim
    # mean_style, std_style: B, 1, fdim
    std_feat = torch.std(feature, dim=-1, keepdim=True) + eps  # B, N, 1
    mean_feat = torch.mean(feature, dim=-1, keepdim=True)  # B, N, 1
    adain = std_style * (feature - mean_feat)/std_feat + mean_style
    return adain  # B, N, fdim


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU(inplace=False)

        self.fc1 = nn.utils.spectral_norm(nn.Linear(in_channel, in_channel))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(in_channel, in_channel))

    def forward(self, x, psi_slice):
        psi_chunks = psi_slice.chunk(4, dim=-1)

        res = x

        out = adaIN(x, psi_chunks[0], psi_chunks[1])
        out = self.leaky_relu(out)
        out = self.fc1(out)
        out = adaIN(out, psi_chunks[2], psi_chunks[0])
        out = self.leaky_relu(out)
        out = self.fc2(out)

        out = out + res
        return out


class Generator(nn.Module):
    slice_psi = (
        slice(0, 64*4),
        slice(64*4, 64*8),
    )

    def __init__(self):
        super().__init__()
        # in(=e): B, edim(=32), 1
        self.P = nn.Parameter(torch.rand(64*8, 32).normal_(0.0, 0.02))
        # out(=psi): B, psidim=(64*8), 1

        self.psi = nn.Parameter(torch.rand(64*8, 1))

        # in(=y): B, N, ydim(=14)
        self.fc1 = nn.Linear(14, 64)
        # self.fc2 = nn.Linear(64, 128)

        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)

        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5_1 = nn.Linear(16, 3)  # out1 = omega0 (B, 3)
        self.fc5_2 = nn.Linear(16, 4)  # out2 = u0 (B, 4)
        # out(=x) B, xdim(=7)

        self.apply(init_weigths)

        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.pooling = nn.AdaptiveMaxPool2d((16, 1))

    def forward(self, y, e):
        psi_hat = (self.P @ e[..., None]).transpose(1, 2)  # B, 1, psidim

        out = self.leaky_relu(self.fc1(y))  # B, N, 64
        # out = self.leaky_relu(self.fc2(out))

        out = self.res1(out, psi_hat[..., self.slice_psi[0]])  # B, N, 64
        out = self.res2(out, psi_hat[..., self.slice_psi[1]])  # B, N, 64

        # out = self.leaky_relu(self.fc3(out))
        out = self.leaky_relu(self.fc4(out))  # B, N, 16
        out = self.pooling(out).squeeze(-1)  # B, 16
        out_1 = self.fc5_1(out)  # B, 3
        out_2 = self.leaky_relu(self.fc5_2(out))  # B, 4
        out = torch.cat([out_1, out_2], dim=-1)  # B, 7
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # in(=x): B, xdim(=7)
        self.fc1_1 = nn.Linear(7, 16)
        # in(=y): B, N, ydim(=14)
        self.fc1_2 = nn.Linear(14, 16)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        # out(=r): B, K, 1

        self.W = nn.Parameter(
            torch.rand(32, config["num_tasks"]).normal_(0.0, 0.02))
        self.w_0 = nn.Parameter(torch.rand(32, 1).normal_(0.0, 0.02))
        self.b = nn.Parameter(torch.rand(1, 1).normal_(0.0, 0.02))

        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.pooling1 = nn.AdaptiveMaxPool2d((16, 1))
        self.pooling2 = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, y, i):
        out1 = self.leaky_relu(self.fc1_1(x))  # B, 16
        out2 = self.leaky_relu(self.fc1_2(y))  # B, N, 16
        out2 = self.pooling1(out2).squeeze(-1)  # B, 16
        out = torch.cat([out1, out2], dim=-1)  # B, 32
        out = self.leaky_relu(self.fc2(out))  # B, 32
        out = self.relu(self.fc3(out)).unsqueeze(1)  # B, 1, 32
        W_i = self.W[:, i][..., None].transpose(0, 1)  # 32, B, 1 -> B, 32, 1
        out = (out @ (W_i + self.w_0) + self.b).squeeze(-1)  # B, 1, 1
        return out, []


class Embedder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, c):
        out = self.model(c)
        return out


class LinearModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Linear(in_features, out_features, bias=False)

    def forward(self, input):
        out = self.model(input)
        return out


class Policy(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim):
        super().__init__()
        self.K = LinearModule(z_dim, x_dim * u_dim)
        self.x_dim = x_dim
        self.u_dim = u_dim

    def forward(self, z, x):
        K = self.K(z).reshape(-1, self.u_dim, self.x_dim).unsqueeze(1)
        u = - K @ x[..., None]
        return u.squeeze(-1)


class Critic(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim):
        super().__init__()
        self.H = LinearModule(z_dim, (x_dim + u_dim)*(x_dim + u_dim))
        self.xu_dim = x_dim + u_dim

    def forward(self, z, x, u):
        H = self.H(z).reshape(-1, self.xu_dim, self.xu_dim).unsqueeze(1)
        gradQ = H @ torch.cat([x, u], dim=-1)[..., None]
        return gradQ.squeeze(-1)
