from typing import Tuple
import torch
import torch.nn as nn

from .dualrestensor import DualResTensor
from .utils.profiler import timings
from .policy import Policy


def to_blocks(x: torch.Tensor, grid: torch.BoolTensor):
    """
    Split a tensor x[n,c,h,w] into blocks, according to the given boolean grid
    where grid[n,x,y] is True if the block for image n and position (x,y) 
    should be executed in high resolution
    """
    assert not isinstance(x, DualResTensor)
    out = DualResTensor.to_blocks(x, grid)
    return out


class SegBlocksModel(nn.Module):
    """
    Wrap a given net and policy into a single model
    """
    def __init__(self, net: torch.nn.Module, policy=None):
        super().__init__()
        self.net = net
        self.policy = policy

    def forward(self, x: torch.nn.Module, meta: dict):
        
        out, meta = execute_with_policy(self.net, self.policy, x, meta)
        return out, meta


def execute_with_policy(net: torch.nn.Module, policy: Policy, image: torch.Tensor, meta: dict) -> Tuple[torch.Tensor, dict]:
    """
    Execute the given network on the image, using the given boolean grid
    where grid[n,x,y] is True if the block for image n and position (x,y) 
    should be executed in high resolution
    """
    timings.add_count(len(image))
    if policy is not None and not meta.get('is_warmup', False):
        with timings.env("segblocks/policy"):
            # run the policy on the image
            grid, meta = policy(image, meta)
            out = execute_with_grid(net, image, grid)
    else:
        out = net(image)
    return out, meta


def execute_with_grid(net: torch.nn.Module, image: torch.Tensor, grid: torch.BoolTensor) -> torch.Tensor:
    """
    Execute the given network on the image, using the given boolean grid
    where grid[n,x,y] is True if the block for image n and position (x,y) 
    should be executed in high resolution.
    The returned result is a torch tensor (not blocks).
    """
    with timings.env("segblocks/to_blocks"):
        # split the image into blocks (DualResTensor)
        image = to_blocks(image, grid)

    with timings.env("segblocks/model"):
        # run the model
        out = net(image)

    with timings.env("segblocks/combine"):
        # combine
        out = out.combine()

    return out


def no_blocks(func):
    """
    decorator to run a function without blocks

    has a large performance cost due to required combine and split
    """

    def _no_blocks(x, *args):
        is_blocks = isinstance(x, DualResTensor)
        if is_blocks:
            blocks = x
            x = x.combine()

        x = func(x)

        if is_blocks:
            x = DualResTensor.to_blocks_like(x, blocks)
        return x

    return _no_blocks
