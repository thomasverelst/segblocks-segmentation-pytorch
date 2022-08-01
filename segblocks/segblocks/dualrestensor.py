from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from yaml import warnings

from segblocks.utils.profiler import timings

from .ops import batchnorm, blockcombine, blockpad, blocksplit

## flags
# verbose print
VERBOSE = False

# debug flag to disable blockpad
USE_BLOCKPAD = True

# bilinear interpolation is slow on small spatial dimensions, but trilinear is fast
USE_INTERPOLATION_SPEED_TRICK = True

# use custom batch norm with statistics over both high-res and low-res blocks
USE_DUALRES_BATCHNORM = True


## helper functions


def implements(torch_functions: Iterable):
    """
    Register a torch function override for DualResTensor
    in HANDLED_FUNCTIONS
    """

    @functools.wraps(torch_functions)
    def decorator(func):
        for torch_function in torch_functions:
            HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


# functions that can be handled by dualrestensors, will be handled by register
HANDLED_FUNCTIONS = {}


def is_dualrestensor(x: object) -> bool:
    """
    check if x is a dualres tensor
    """
    return isinstance(x, DualResTensor)


class DualResMetaData(object):
    """
    Class that keeps the metadata of a dualres tensor.

    Given a binary grid (N, GH, GW), the metadata consists of:
    - nhighres: number of highres blocks (grid[,,] = 1)
    - nlowres: number of lowres blocks (grid[,,] = 0)
    - map_hr (nhighres, GH, GW) of dtype int32, giving the mapping
        from highres index (0 to nhighres-1) to the original grid index(flattened)
    - map_lr (nlowres, GH, GW) of dtype int32, giving the mapping
        from lowres index (0 to nlowres-1) to the original grid index(flattened)
    - block_idx: int32 tensor of shape [N, GH, GW], giving the the mapping from
        the original grid index to the batch index in map_hr or map_lr
        for map_hr, the value is >= 0,
        for map_lr, the value is < 0 by subtracting N*GH*GW from the real index

    """

    def __init__(self, grid: torch.Tensor):
        assert grid.dtype == torch.bool
        assert grid.dim() == 3

        # create meta-data for bookkeeping in split/combine/pad operations

        self.grid = grid
        is_hr = grid  # True for blocks that should be executed in high resolution
        is_lr = ~grid  # True for blocks that should be executed in low resolution
        self.nhighres = int(grid.sum())  # number of highres blocks
        self.nlowres = grid.numel() - self.nhighres  # number of lowres blocks

        # block_idx is (N, 1, GRID_H, GRID_W) of dtype int32
        # each element is the batch index of that block in data_hr or data_lr
        # for lowres blocks, indexes are made negative by subtracting the total number of blocks (grid.numel())
        # this way, highres/lowres can be checked by the value in block_idx
        # i.e.
        # assert block_idx[grid == 1] >= 0
        # assert block_idx[grid == 0] < 0
        #
        # map_hr is a mapping from the batch index in data_hr to the original grid (flattened)
        # map_lr idem

        self.block_idx = torch.empty(grid.shape, dtype=torch.int32, device=grid.device)

        idx_hr = torch.arange(self.nhighres, dtype=torch.int32, device=grid.device)
        self.block_idx = self.block_idx.masked_scatter_(is_hr, idx_hr)
        self.map_hr = torch.nonzero(is_hr.flatten(), as_tuple=True)[0].int()

        idx_lr = torch.arange(self.nlowres, dtype=torch.int32, device=grid.device) - grid.numel()
        self.block_idx = self.block_idx.masked_scatter_(is_lr, idx_lr)
        self.map_lr = torch.nonzero(is_lr.flatten(), as_tuple=True)[0].int()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DualResMetaData):
            return False
        if not torch.equal(self.grid, other.grid):
            return False
        # the same grid always leads to the same block_idx, map_hr, map_lr
        # so no need to check
        return True

    def clone(self) -> DualResMetaData:
        return DualResMetaData(self.grid)


class DualResTensor(object):
    """
    Custom PyTorch Tensor
    https://pytorch.org/docs/stable/notes/extending.html
    """

    LOWRES_FACTOR = 2  # side length factor of lowres vs highres blocks

    def __init__(self, highres: torch.Tensor, lowres: torch.Tensor, metadata: DualResMetaData):
        self.highres = highres
        self.lowres = lowres
        self.metadata = metadata

    @property
    def block_size(self) -> int:
        """
        returns the size (length of a side in pixels) of a block
        """
        return self.highres.shape[-1]

    @property
    def grid_shape(self) -> torch.Size:
        """
        returns the shape of the grid
        4D tensor of shape (N, 1, GRID_H, GRID_W)
        with N the batch size
        """
        return self.metadata.grid.shape

    @property
    def represented_shape(self) -> Tuple[int, int, int, int]:
        """
        get the shape that the blocks represent
        as N, C, H, W
        """
        N = self.metadata.grid.size(0)
        C = self.highres.size(1)
        H = self.block_size * self.grid_shape[1]
        W = self.block_size * self.grid_shape[2]
        return N, C, H, W

    @property
    def shape(self) -> Tuple[torch.Size, torch.Size]:
        """
        returns the shape of both highres and lowres tensors
        """
        return (self.highres.shape, self.lowres.shape)

    def size(self, dim=None) -> Tuple[torch.Size, torch.Size]:
        """
        same as method shape, but with the optional dim argument
        to specify the dimension
        """
        if dim is not None:
            return (s[dim] for s in self.shape)
        else:
            return self.shape

    def dim(self) -> int:
        """
        returns the number of dimensions
        """
        return self.highres.dim()

    @property
    def dtype(self) -> torch.dtype:
        """
        return the data type
        """
        return self.highres.dtype

    @property
    def device(self) -> torch.device:
        """
        return the device
        """
        return self.highres.device

    @classmethod
    def to_blocks(cls, x: torch.Tensor, grid: torch.BoolTensor) -> DualResTensor:
        """
        Initialize DualResTensor from normal 4D tensor x with NCHW layout
        and grid tensor of dtype bool with shape (N, 1, GRID_H, GRID_W).
        The block size is determined as H//GRID_H, so H must be a multiple of GRID_H
        and W must be a multiple of GRID_W, with H//GRID_H == W//GRID_W.
        Only suports CUDA, and all tensors should be on a CUDA device
        """
        assert isinstance(x, torch.Tensor)
        assert isinstance(grid, torch.Tensor)
        assert grid.dtype == torch.bool
        assert grid.is_cuda
        assert x.is_cuda
        assert grid.shape[0] == x.shape[0]

        # create meta data from grid for bookkeeping in split / combine / pad operations
        metadata = DualResMetaData(grid)
        return cls._to_blocks_with_metadata(x, metadata)

    @classmethod
    def _to_blocks_with_metadata(cls, x: torch.Tensor, metadata: DualResMetaData) -> DualResTensor:
        with timings.env("dualrestensor/to_blocks"):
            N, C, H, W = x.shape  # shape of the original data
            _, GRID_H, GRID_W = metadata.grid.shape  # shape of the grid
            assert H % GRID_H == 0, f"{H} % {GRID_H} != 0"
            assert W % GRID_W == 0, f"{W} % {GRID_W} != 0"
            block_size = H // GRID_H
            assert block_size == W // GRID_W, f"{H} / {GRID_H} != {W} / {GRID_W}"

            # create tensors to store the data of high and low resolution blocks
            # blocks are stored as batches of blocks, with shape (number_of_blocks, C, block_size, block_size)
            # if no blocks for either high or low resolution, create tensor with 1 block
            size_hr = (max(metadata.nhighres, 1), C, block_size, block_size)
            data_hr = torch.empty(size_hr, device=metadata.grid.device, dtype=x.dtype)
            size_lr = (
                max(metadata.nlowres, 1),
                C,
                block_size // cls.LOWRES_FACTOR,
                block_size // cls.LOWRES_FACTOR,
            )
            data_lr = torch.empty(size_lr, device=metadata.grid.device, dtype=x.dtype)

            if metadata.nhighres == 0:
                data_hr.fill_(0)
            if metadata.nlowres == 0:
                data_lr.fill_(0)

            # copy data to data_hr and data_lr from x
            with timings.env("dualrestensor/split_func"):
                data_hr, data_lr = blocksplit.SplitFunction.apply(
                    x,
                    data_hr,
                    data_lr,
                    metadata.map_hr,
                    metadata.map_lr,
                    metadata.block_idx,
                    block_size,
                    cls.LOWRES_FACTOR,
                )

            if VERBOSE:
                print(f"DualResTensor.to_blocks: {x.shape} -> {data_hr.shape}, {data_lr.shape}")
            return DualResTensor(data_hr, data_lr, metadata)

    @classmethod
    def to_blocks_like(cls, x: torch.Tensor, other: DualResTensor) -> DualResTensor:
        """
        Initialize DualResTensor from another DualResTensor
        """
        return DualResTensor._to_blocks_with_metadata(x, other.metadata)

    def __repr__(self):
        return f"DualResTensor(highres={self.highres.shape}, lowres={self.lowres.shape}, grid={self.metadata.grid.shape}, blocksize={self.block_size}, represented_shape={self.represented_shape})"

    def combine(self) -> torch.Tensor:
        """
        Combine highres and lowres patches to a single NCHW tensor
        """
        with timings.env("dualrestensor/combine"):
            out = torch.empty(self.represented_shape, device=self.highres.device, dtype=self.highres.dtype)
            out = blockcombine.CombineFunction.apply(
                out,
                self.highres,
                self.lowres,
                self.metadata.block_idx,
                self.block_size,
                self.LOWRES_FACTOR,
            )

        if VERBOSE:
            print(f"DualResTensor.combine: {self.highres.shape} + {self.lowres.shape} -> {out.shape}")

        return out

    @classmethod
    def __torch_function__(cls, func: Callable, types: Tuple, args: Tuple = (), kwargs: Optional[Dict] = None) -> Any:
        """
        Handles pytorch functions
        """

        if kwargs is None:
            kwargs = {}
        if func.__name__ not in HANDLED_FUNCTIONS:
            print(f"{func.__name__} not handled")
            return NotImplemented
        out = HANDLED_FUNCTIONS[func.__name__](func, *args, **kwargs)
        if VERBOSE:
            print(f"{func.__name__}: {out.dtype} {out.shape}")
        return out

    def _dual_op(self, other: DualResTensor, op_name: str) -> DualResTensor:
        """
        General inplace operation
        """
        if not isinstance(other, DualResTensor):
            raise NotImplementedError(f"{type(other)} not supported")
        self.highres = getattr(self.highres, op_name)(other.highres)
        self.lowres = getattr(self.lowres, op_name)(other.lowres)
        return self

    def __add__(self, other: DualResTensor) -> DualResTensor:
        return self._dual_op(other, "__add__")

    def __sub__(self, other: DualResTensor) -> DualResTensor:
        return self._dual_op(other, "__sub__")

    def __mul__(self, other: DualResTensor) -> DualResTensor:
        return self._dual_op(other, "__mul__")


def apply_func_on_dualres(func: Callable, *args, **kwargs) -> DualResTensor:
    """
    Apply a function on both high-res and low-res tensors of DualResTensors
    """
    args_hr, args_lr = [], []
    kwargs_hr, kwargs_lr = {}, {}
    metadata = None

    def get_meta(block_meta):
        nonlocal metadata
        if metadata is None:
            metadata = block_meta
        else:
            assert metadata == block_meta, f"{metadata} != {block_meta}"

    for a in args:
        if isinstance(a, DualResTensor):
            get_meta(a.metadata)
            args_hr.append(a.highres)
            args_lr.append(a.lowres)
        else:
            args_hr.append(a)
            args_lr.append(a)
    for k, v in kwargs.items():
        if isinstance(v, DualResTensor):
            get_meta(a.metadata)
            kwargs_hr[k], kwargs_lr[k] = v.highres, v.lowres
        else:
            kwargs_hr[k], kwargs_lr[k] = v, v

    out_hr = func(*args_hr, **kwargs_hr)
    out_lr = func(*args_lr, **kwargs_lr)

    out = DualResTensor(highres=out_hr, lowres=out_lr, metadata=metadata)
    return out


@implements(["conv2d", "max_pool2d", "avg_pool2d", "lp_pool2d", "fractional_max_pool2d"])
def padded_functions(func: Callable, *args, **kwargs) -> DualResTensor:
    """
    Handles functions with sliding kernels that use padding to preserve
    the input dimensions

    Replaces zero-padding with BlockPadding, to preserve information flow
    between blocks
    """
    if not USE_BLOCKPAD:  # for debug purposes
        with timings.env("dualrestensor/blockpad_func"):
            return apply_func_on_dualres(func, *args, **kwargs)

    args = list(args)

    # get the padding amount from args/kwargs
    padding = kwargs.get("padding", None)
    if padding is None:
        padding = (
            args[4] if len(args) > 4 else 0
        )  # for some functions, padding is stored in args[4], but this is not a safe assumption

    # build zeros, to override the padding of the func
    zeros = 0
    if isinstance(padding, (tuple, list)):
        zeros = (0, 0)
        if padding[0] != padding[1]:
            raise NotImplementedError(f"Only support equal paddings, got {padding}")
        padding = padding[0]

    # override the padding of the func with zeros
    if "padding" in kwargs:
        kwargs["padding"] = zeros
    else:
        args[4] = zeros

    dualres = args[0]
    assert isinstance(dualres, DualResTensor)

    # if func has padding, replace with BlockPadding
    if padding > 0:
        with timings.env("dualrestensor/blockpad"):

            def run_func_padded(is_highres: bool) -> torch.Tensor:
                with timings.env("dualrestensor/blockpad"):
                    data_padded = blockpad.BlockPad.apply(
                        dualres.highres,
                        dualres.lowres,
                        dualres.metadata.map_hr,
                        dualres.metadata.map_lr,
                        dualres.metadata.block_idx,
                        padding,
                        is_highres,
                    )
                args[0] = data_padded
                with timings.env("dualrestensor/blockpad_func"):
                    out = func(*args, **kwargs)
                del data_padded
                return out

            out_hr = run_func_padded(is_highres=True)
            out_lr = run_func_padded(is_highres=False)

    else:
        with timings.env("dualrestensor/blockpad_func_nopadding"):
            highres, lowres = dualres.highres, dualres.lowres
            args[0] = highres
            out_hr = func(*args, **kwargs)
            args[0] = lowres
            out_lr = func(*args, **kwargs)

    out = DualResTensor(highres=out_hr, lowres=out_lr, metadata=dualres.metadata)
    return out


@implements(["batch_norm"])
def batch_norm_functions(func: Callable, *args, **kwargs) -> DualResTensor:
    """
    When training, we need to get batch norm statistics over both high-res and low-res tensors,
    instead of updating sequentially,
    During validation, batch statistics are saved batchnorm can be executed over both high-res
    and low-res tensors separately
    """
    if not USE_DUALRES_BATCHNORM:
        # debug mode that applies batchnorm on highres and lowres subsequently
        out = apply_func_on_dualres(func, *args, **kwargs)
    else:
        with timings.env("dualrestensor/custom_batch_norm"):
            dualres, running_mean, running_var = args
            weight = kwargs["weight"]
            bias = kwargs["bias"]
            training = kwargs["training"]
            momentum = kwargs["momentum"]
            eps = kwargs["eps"]

            out_hr, out_lr = batchnorm.block_batch_norm(
                dualres, running_mean, running_var, weight, bias, training, momentum, eps
            )
            out = DualResTensor(highres=out_hr, lowres=out_lr, metadata=dualres.metadata)
    return out


@implements(["relu", "hardtanh", "__add__", "__sub__"])
def per_resolution(func: Callable, *args, **kwargs) -> DualResTensor:
    """
    Operations that can be performed as-is on the high and low-resolution tensor
    """
    return apply_func_on_dualres(func, *args, **kwargs)


@implements(["interpolate", "upsample_bilinear"])
def interpolate(func: Callable, *args, **kwargs) -> DualResTensor:
    """
    bilinear interpolation is slow in blocks for some reason,
    but using trilinear as a trick works
    """
    with timings.env("dualrestensor/interpolate"):
        if not USE_INTERPOLATION_SPEED_TRICK or kwargs["mode"] != "bilinear":
            return apply_func_on_dualres(func, *args, **kwargs)
        else:
            args = list(args)
            dualres = args[0]
            assert isinstance(dualres, DualResTensor)
            kwargs["mode"] = "trilinear"
            if kwargs["scale_factor"] is not None:
                kwargs["scale_factor"] = (1, kwargs["scale_factor"], kwargs["scale_factor"])
            if kwargs["size"] is not None:
                kwargs["size"] = (1, kwargs["size"][0], kwargs["size"][1])
            args[0] = dualres.highres.unsqueeze(0)
            out_hr = func(*args, **kwargs).squeeze(0)
            args[0] = dualres.lowres.unsqueeze(0)
            out_lr = func(*args, **kwargs).squeeze(0)
            return DualResTensor(highres=out_hr, lowres=out_lr, metadata=dualres.metadata)


@implements(["mean", "sum", "max", "min,", "std", "var", "argmax", "count_nonzero", "nonzero"])
def channel_only(func: Callable, *args, **kwargs) -> DualResTensor:
    """
    Only compatible in batch dimension
    """
    if "dim" in kwargs and kwargs["dim"] == 1:
        pass
    else:
        warnings.warn(f"Functon {func.__name__} might behave differently with DualResTensor when dim != 1!")
    return apply_func_on_dualres(func, *args, **kwargs)


@implements(["adaptive_avg_pool2d", "adaptive_max_pool2d", "linear", "flip", "unsqueeze", "reshape", "view"])
def incompatible(func: Callable, *args, **kwargs):
    """
    Incompatible functions
    """
    raise NotImplementedError(f"{func.__name__} not supported for DualResTensor!")
