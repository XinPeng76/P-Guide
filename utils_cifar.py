import copy
import os

import torch
from torch import distributed as dist
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )

import torch

def c_func(t, alpha=1.0, beta=1.0):
    return (t**alpha) * ((1 - t)**beta)*2

def dc_dt(t, alpha=1.0, beta=1.0):
    term1 = alpha * (t**(alpha-1)) * ((1 - t)**beta)
    term2 = beta * (t**alpha) * ((1 - t)**(beta-1))
    return (term1 - term2)*2

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x



import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        gamma = self.to_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.film = FiLM(out_ch, cond_dim)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = F.silu(self.conv1(x))
        h = self.conv2(h)
        h = self.film(h, cond)
        return F.silu(h + self.skip(x))


class LabelToImage_ResNet(nn.Module):
    def __init__(self, num_classes=10, img_shape=(3, 32, 32), base=64):
        super().__init__()

        self.img_shape = img_shape
        C, H, W = img_shape

        self.embed = nn.Embedding(num_classes + 1, 256)
        self.null_label = num_classes

        self.fc = nn.Linear(256, 256)

        # learned constant input (GAN-style)
        self.const = nn.Parameter(torch.randn(1, base, H, W))

        self.net = nn.Sequential(
            ResBlock(base, base, 256),
            ResBlock(base, base * 2, 256),
            ResBlock(base * 2, base * 2, 256),
            ResBlock(base * 2, base, 256),
        )

        self.out = nn.Conv2d(base, C * 2, 3, padding=1)  # mean + logvar

    def forward(self, y, force_uncond=False):
        if force_uncond:
            y = torch.full_like(y, self.null_label)

        emb = self.fc(self.embed(y))

        B = y.size(0)
        h = self.const.repeat(B, 1, 1, 1)

        for layer in self.net:
            h = layer(h, emb)

        out = self.out(h)
        mu, logvar = out.chunk(2, dim=1)

        logvar = torch.clamp(logvar, -10, 5)
        sigma = torch.exp(0.5 * logvar)

        return mu, sigma
