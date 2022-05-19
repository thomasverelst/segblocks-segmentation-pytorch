import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from torch.distributions import Bernoulli
import numpy as np
import random

BN_MOMENTUM = 0.02

def all_policies():
    return {
        'disabled': None,
        'random': PolicyRandom,
        'half': PolicyHalf,
        'reinforce': PolicyReinforce,
    }

class PolicyNet(nn.Module):
    def __init__(self, block_size: int, width_factor: float = 1):
        super().__init__()
        self.block_size = block_size

        features = int(64*width_factor)
        layers = []
        layers.append(nn.Conv2d(3, features, kernel_size=7, padding=3, dilation=1, stride=4, bias=False))
        layers.append(nn.BatchNorm2d(features,momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv2d(features, features, kernel_size=5, padding=2, dilation=1, stride=2, bias=False))
        layers.append(nn.BatchNorm2d(features,momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv2d(features, features, kernel_size=5, padding=2, dilation=1, stride=2, bias=False))
        layers.append(nn.BatchNorm2d(features,momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv2d(features, 1, kernel_size=5, padding=2, dilation=1, stride=2, bias=True))
        self.layers = nn.Sequential(*layers)

        self._ds_factor = 4*2*2*2 # downsampling factor of the net



    def forward(self, x):
        # rescale image to lower resolution
        SCALE = self._ds_factor / self.block_size
        x = F.interpolate(x, scale_factor=SCALE, mode='nearest')

        # run policy network
        x = self.layers(x)
        return x



class Policy(nn.Module, metaclass=ABCMeta):
    def __init__(self, block_size: int, quantize_percentage: float, percent_target: float, sparsity_weight: float, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.quantize_percentage = quantize_percentage
        self.percent_target = percent_target
        self.policy_net = PolicyNet(block_size, width_factor=1)
        self.sparsity_weight = sparsity_weight
    
    @abstractmethod
    def forward(self, x: torch.Tensor, meta: dict):
        raise NotImplementedError

    def loss(self, loss_per_pixel: torch.Tensor, meta: dict):
        return 0

    @staticmethod
    def _quantize_grid_percentage(grid, quantize_percentage):
        if quantize_percentage == 0:
            return grid
        # it is more efficient to quantize the number of executed blocks 
        # as cudnn.benchmark caches per tensor dimension
        grid_cpu = grid.cpu().bool()
        total = grid_cpu.numel()
        idx_not_exec = torch.nonzero(~grid_cpu.flatten()).squeeze(1).tolist()
        num_exec = total - len(idx_not_exec)
        multiple = int(total*quantize_percentage)
        num_exec_rounded = multiple * (1 + (num_exec - 1) // multiple)
        idx = random.sample(idx_not_exec, num_exec_rounded - num_exec)
        grid.flatten()[idx] = 1  
        # print('num_exec_rounded, num_exec, total, multiple', num_exec_rounded, num_exec, total, multiple)
        return grid

class PolicyHalf(Policy):
    def forward(self, x: torch.Tensor, meta: dict):
        N, C, H, W = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0
        grid = torch.zeros((N, H//self.block_size, W//self.block_size), device=x.device).bool()
        grid[:, int(grid.shape[1]/4):int(3*grid.shape[1]/4), :] = 1
        meta['grid'] = grid
        return grid, meta
        
class PolicyRandom(Policy):
    def forward(self, x: torch.Tensor, meta: dict):
        N, C, H, W = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0
        t = torch.randn((N, H//self.block_size, W//self.block_size), device=x.device)
        tflatsort = t.flatten().sort(descending=True)[0]
        thres = tflatsort[min(int(self.percent_target*len(tflatsort)), len(tflatsort)-1)]
        grid =  t > thres
        grid = self._quantize_grid_percentage(grid, self.quantize_percentage)
        meta['grid'] = grid
        return grid, meta

class PolicyReinforce(Policy):
    def forward(self, x: torch.Tensor, meta: dict):
        N, C, H, W = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0
        GRID_H, GRID_W = H//self.block_size, W//self.block_size
        logits = self.policy_net(x).squeeze(1)
        assert logits.shape == (N, GRID_H, GRID_W), f"{logits.shape} != {(N, GRID_H, GRID_W)}"

        if self.training:
            probs = torch.sigmoid(logits)
            m = Bernoulli(probs)
            grid = m.sample()
            grid = self._quantize_grid_percentage(grid, self.quantize_percentage)
            log_probs = m.log_prob(grid)
            meta['grid_probs'] = probs
            meta['grid_log_probs'] = log_probs
        else:
            grid = logits > 0
            grid = self._quantize_grid_percentage(grid, self.quantize_percentage)

        grid = grid.bool()
        meta['grid'] = grid
        return grid, meta
        
    def loss(self, loss_per_pixel: dict, meta: dict):
        grid = meta['grid']
        N, GH, GW = grid.shape

        # reward to get the target percentage
        block_use = float(grid.sum())/grid.numel() # percentage of high-res blocks
        reward_sparsity = float(self.percent_target - block_use) # lower reward if more blocks used than target

        # reward to optimize task
        loss_per_block = F.adaptive_avg_pool2d(loss_per_pixel, (GH, GW))
        loss_mean = loss_per_block.mean()
        reward_task = loss_per_block # higher loss = higher reward (better to run in high-res)
        advantage_task = reward_task - loss_mean # reduce variance by providing baseline
        
        # combine
        advantage = advantage_task + self.sparsity_weight*reward_sparsity # add sparsity reward (single float) to per-block advantage
        advantage[~grid] = -advantage[~grid] # for low-res blocks, flip advantage (i.e. low loss should provide better reward)
        meta['advantage'] = advantage
        meta['advantage_task'] = advantage_task
        meta['reward_sparsity'] = reward_sparsity
        
        log_probs = meta['grid_log_probs']

        # loss
        assert log_probs.dim() == 3
        assert advantage.shape == log_probs.shape, f"advantage {advantage.shape}, log_probs {log_probs.shape}"
        loss_policy = -log_probs * advantage.detach()
        loss_policy = loss_policy.mean()
        return loss_policy