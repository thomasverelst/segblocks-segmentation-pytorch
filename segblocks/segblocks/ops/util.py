""" 
for cupy 
"""

import math
import os
import time
from collections import namedtuple
from string import Template
from typing import Iterable

import cupy
import torch

DTYPES_FLOAT = (torch.float16, torch.float32, torch.float64)

Stream = namedtuple("Stream", ["ptr"])

# CUDA_PATH = ('-I/usr/local/cuda/include','-I/usr/local/cuda-11.1/include','-I/usr/local/cuda-11.3/include','-I/usr/local/cuda-11.5/include','-I/usr/local/cuda-11.6/include')
CUDA_PATH = ["-I/usr/local/cuda-11.8/include"]
if "CUDA_HOME" in os.environ:
    CUDA_PATH.append(f"-I{os.path.join(os.environ['CUDA_HOME'], 'include')}")


def Dtype(t):
    if t.dtype == torch.float16:
        return "__half"
    elif t.dtype == torch.float32:
        return "float"
    elif t.dtype == torch.float64:
        return "double"
    else:
        raise NotImplementedError(t.dtype)


@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)

    options = list(CUDA_PATH[:])
    options.extend(("--restrict", "--use_fast_math"))
    kernel_coda = cupy.RawModule(code=code, options=tuple(options))
    kernel = kernel_coda.get_function(kernel_name)
    return kernel


def GET_BLOCKS(N, NTHREADS):
    return min((N + NTHREADS - 1) // (NTHREADS), 256 * 256 - 1)


def assertcuda(x, dtypes=None):
    if not isinstance(dtypes, Iterable):
        dtypes = (dtypes,)
    assert x.is_cuda
    assert x.is_contiguous()
    assert dtypes is None or x.dtype in dtypes, (x.dtype, dtypes)
    return True


CUDA_NUM_THREADS = 512
CUDA_NUM_BLOCKS = 2048


def get_threads_and_blocks(npixels, channels):
    threads_x = min(npixels, CUDA_NUM_THREADS)
    threads_y = min(channels, CUDA_NUM_THREADS // threads_x)

    blocks_x = min(GET_BLOCKS(npixels, threads_x), CUDA_NUM_BLOCKS)
    blocks_y = max(min(GET_BLOCKS(channels, threads_y) // blocks_x, CUDA_NUM_BLOCKS // blocks_x), 1)

    block = (threads_x, threads_y)
    grid = (blocks_x, blocks_y)
    return block, grid


_kernel_header_blocks = """
#include "cuda_fp16.h"

#define DTYPE ${dtype}
#define BLOCK_SIZE ${block_size}
#define BLOCK_SIZE_LOWRES (BLOCK_SIZE/LOWRES_FACTOR)
#define LOWRES_FACTOR ${lowres_factor}

#define BATCH_SIZE ${batch_size}
#define CHANNELS ${channels}
#define WIDTH ${width}
#define HEIGHT ${height}
#define GRID_W (WIDTH/BLOCK_SIZE)
#define GRID_H (HEIGHT/BLOCK_SIZE)

#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < n;                                       \
      i += blockDim.x * gridDim.x)

#define CUDA_CHANNEL_LOOP(c)                       \
  for (int c = blockIdx.y * blockDim.y + threadIdx.y; \
  c < CHANNELS; c += blockDim.y * gridDim.y) \

"""
