import random
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

BN_MOMENTUM = 0.02


def all_policies() -> dict:
    return {
        "disabled": None,
        "random": PolicyRandom,
        "middle": PolicyMiddle,
        "heuristic": PolicyHeuristic,
        "reinforce": PolicyReinforce,
        "oracle": PolicyOracle,
    }


class PolicyNet(nn.Module):
    def __init__(self, block_size: int, width_factor: float = 1):
        super().__init__()
        self.block_size = block_size

        features = int(64 * width_factor)
        layers = []
        layers.append(nn.Conv2d(3, features, kernel_size=7, padding=3, dilation=1, stride=4, bias=False))
        layers.append(nn.BatchNorm2d(features, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv2d(features, features, kernel_size=5, padding=2, dilation=1, stride=2, bias=False))
        layers.append(nn.BatchNorm2d(features, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv2d(features, features, kernel_size=5, padding=2, dilation=1, stride=2, bias=False))
        layers.append(nn.BatchNorm2d(features, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv2d(features, 1, kernel_size=5, padding=2, dilation=1, stride=2, bias=True))
        self.layers = nn.Sequential(*layers)

        self._ds_factor = 4 * 2 * 2 * 2  # downsampling factor of the net

    def forward(self, x):
        # rescale image to lower resolution
        scale = self._ds_factor / self.block_size
        x = F.interpolate(x, scale_factor=scale, mode="nearest")

        # run policy network
        x = self.layers(x)
        return x

class Policy(nn.Module, metaclass=ABCMeta):
    """
    SegBlocks policy interface
    """

    def __init__(
        self,
        block_size: int,
        quantize_percentage: float,
        percent_target: float,
        sparsity_weight: float,
        **kwargs,
    ):
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
        multiple = int(total * quantize_percentage)
        num_exec_rounded = multiple * (1 + (num_exec - 1) // multiple)
        idx = random.sample(idx_not_exec, num_exec_rounded - num_exec)
        grid.flatten()[idx] = 1
        return grid


class PolicyMiddle(Policy):
    """
    Policy that executes half of the blocks in high-res,
    being complete rows vertically centered in the image.
    """

    def forward(self, x: torch.Tensor, meta: dict):
        N, C, H, W = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0
        grid = torch.zeros((N, H // self.block_size, W // self.block_size), device=x.device).bool()
        diff = self.percent_target
        lower = grid.shape[1] * (0.5 - diff / 2)
        higher = grid.shape[1] * (0.5 + diff / 2)
        # print('lower, higher', lower, higher)
        grid[:, int(lower) : int(higher), :] = 1
        meta["grid"] = grid
        return grid, meta


class PolicyRandom(Policy):
    """
    Policy that executes random blocks in high-res
    """

    def forward(self, x: torch.Tensor, meta: dict):
        N, C, H, W = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0
        t = torch.randn((N, H // self.block_size, W // self.block_size), device=x.device)
        tflatsort = t.flatten().sort(descending=True)[0]
        thres = tflatsort[min(int(self.percent_target * len(tflatsort)), len(tflatsort) - 1)]
        grid = t > thres
        grid = self._quantize_grid_percentage(grid, self.quantize_percentage)
        meta["grid"] = grid
        return grid, meta


class PolicyOracle(Policy):
    """
    Policy that executes random blocks in high-res
    """

    def forward(self, x: torch.Tensor, meta: dict):
        N, C, H, W = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0

        if meta.get("all_highres", False):
            grid = torch.ones((N, H // self.block_size, W // self.block_size), device=x.device).bool()
        elif meta.get("all_lowres", False):
            grid = torch.zeros((N, H // self.block_size, W // self.block_size), device=x.device).bool()
        else:
            loss_per_pixel_oracle_diff = meta["loss_per_pixel_oracle_lowres"] - meta["loss_per_pixel_oracle_highres"]
            loss_per_block_oracle = F.avg_pool2d(
                loss_per_pixel_oracle_diff, kernel_size=self.block_size, stride=self.block_size
            )
            grid = loss_per_block_oracle.gt(self.percent_target)
        meta["grid"] = grid
        return grid, meta


class PolicyHeuristic(Policy):
    """
    Policy that executes random blocks in high-res
    """

    def forward(self, x: torch.Tensor, meta: dict):
        N, C, H, W = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0

        scale_factor = 4
        x_down = F.avg_pool2d(x, kernel_size=scale_factor)
        x_up = F.interpolate(x_down, scale_factor=scale_factor, mode="nearest")
        diff = (torch.abs_(x_up - x)).sum(dim=1, keepdim=True)
        t = F.avg_pool2d(diff, kernel_size=self.block_size)
        t = t.squeeze(1)
        thres = self.percent_target
        grid = t > thres
        meta["grid"] = grid
        return grid, meta


class PolicyHeuristicSSIM(Policy):
    """
    Policy that executes random blocks in high-res
    """

    def forward(self, x: torch.Tensor, meta: dict):
        N, C, H, W = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0

        x_down = F.avg_pool2d(x, kernel_size=4)
        x_up = F.interpolate(x_down, scale_factor=4, mode="nearest")
        diff = (torch.abs_(x_up - x)).sum(dim=1, keepdim=True)
        t = F.avg_pool2d(diff, kernel_size=self.block_size)
        t = t.squeeze(1)
        # if self.thres:
        thres = self.percent_target
        grid = t > thres
        meta["grid"] = grid
        return grid, meta


class PolicyReinforce(Policy):
    """
    Policy that highres REINFORCE to train high-res policy
    """

    def forward(self, x: torch.Tensor, meta: dict):
        with torch.cuda.amp.autocast(enabled=False):
            N, C, H, W = x.shape
            assert H % self.block_size == 0
            assert W % self.block_size == 0
            GRID_H, GRID_W = H // self.block_size, W // self.block_size
            logits = self.policy_net(x).squeeze(1).float()
            assert logits.shape == (N, GRID_H, GRID_W), f"{logits.shape} != {(N, GRID_H, GRID_W)}"
            if self.training:
                probs = torch.sigmoid(logits)
                m = Bernoulli(probs)
                grid = m.sample()
                grid = self._quantize_grid_percentage(grid, self.quantize_percentage)
                log_probs = m.log_prob(grid)
                meta["grid_probs"] = probs
                meta["grid_log_probs"] = log_probs
            else:
                grid = logits > 0
                grid = self._quantize_grid_percentage(grid, self.quantize_percentage)

            grid = grid.bool()
            meta["grid"] = grid
            return grid, meta

    def loss(self, loss_per_pixel: dict, meta: dict):
        grid = meta["grid"]
        N, GH, GW = grid.shape

        # reward to get the target percentage
        block_use = float(grid.sum()) / grid.numel()  # percentage of high-res blocks
        # lower reward if more blocks used than target
        reward_sparsity = float(self.percent_target - block_use)

        # reward to optimize task
        loss_per_block = F.adaptive_avg_pool2d(loss_per_pixel, (GH, GW))
        loss_mean = loss_per_block.mean()
        reward_task = loss_per_block  # higher loss = higher reward (better to run in high-res)
        advantage_task = reward_task - loss_mean  # reduce variance by providing baseline

        # combine, by adding sparsity reward (single float) to per-block advantage
        advantage = advantage_task + self.sparsity_weight * reward_sparsity
        # for low-res blocks, flip advantage (i.e. low loss should provide better reward)
        advantage[~grid] = -advantage[~grid]
        meta["advantage"] = advantage
        meta["advantage_task"] = advantage_task
        meta["reward_sparsity"] = reward_sparsity

        log_probs = meta["grid_log_probs"]

        # loss
        assert log_probs.dim() == 3
        assert advantage.shape == log_probs.shape, f"advantage {advantage.shape}, log_probs {log_probs.shape}"

        loss_policy = -log_probs * advantage.detach()
        loss_policy = loss_policy.mean()
        return loss_policy
