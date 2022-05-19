import torch
import torch.nn as nn

from .dualrestensor import DualResTensor
from .utils.profiler import timings


def to_blocks(x, grid):
    assert not isinstance(x, DualResTensor)
    out = DualResTensor.to_blocks(x, grid)
    return out


class SegBlocksModel(nn.Module):
    def __init__(self, net, policy=None):
        super().__init__()
        self.net = net
        self.policy = policy
        self._is_preallocated = False
    
    def forward(self, x, meta):
        out, meta = execute_with_policy(self.net, self.policy, x, meta, force_prealloc=not self._is_preallocated)
        self._is_preallocated = True
        return out, meta


def execute_with_policy(net, policy, image, meta, force_prealloc=False):
    timings.add_count(len(image))
    if policy is not None:
        with timings.env('segblocks/policy'):
            # run the policy on the image
            grid, meta = policy(image, meta)
        out = execute_with_grid(net, image, grid, force_prealloc=force_prealloc)
    else:
        out = net(image)
    return out, meta

def execute_with_grid(net, image, grid, force_prealloc=False):
    # if force_prealloc:
    #     # if force_prealloc, run the image without 0 highres blocks and will all highres blocks
    #     # to preallocate the memory and avoid memory fragmentation due to changing tensor sizes
    #     # only required for the first run of the model
    #     print('segblocks pre_alloc')
    #     grid2 = torch.ones_like(grid)
    #     image2 = to_blocks(image, grid2)
    #     _ = net(image2)
    #     grid2 = torch.zeros_like(grid)
    #     image2 = to_blocks(image, grid2)
    #     _ = net(image2)

    with timings.env('segblocks/to_blocks'):
        # split the image into blocks (DualResTensor)
        image = to_blocks(image, grid)

    with timings.env('segblocks/model'):
        # run the model
        out = net(image)

    with timings.env('segblocks/combine'):
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
