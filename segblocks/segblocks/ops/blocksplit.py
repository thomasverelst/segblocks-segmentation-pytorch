from typing import Tuple

import torch
from torch.autograd import Function

from .util import DTYPES_FLOAT, Dtype, Stream, _kernel_header_blocks, assertcuda, get_threads_and_blocks, load_kernel


class SplitFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        data_hr: torch.Tensor,
        data_lr: torch.Tensor,
        map_hr: torch.Tensor,
        map_lr: torch.Tensor,
        block_idx: torch.Tensor,
        block_size: int,
        lowres_factor: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert assertcuda(x, dtypes=x.dtype)
        assert assertcuda(data_hr, dtypes=x.dtype)
        assert assertcuda(data_lr, dtypes=x.dtype)
        assert assertcuda(map_hr, dtypes=torch.int32)
        assert assertcuda(map_lr, dtypes=torch.int32)
        assert assertcuda(block_idx, dtypes=torch.int32)

        assert data_hr.shape[2:] == (block_size, block_size)
        assert data_lr.shape[2:] == (block_size // lowres_factor, block_size // lowres_factor)
        assert len(data_hr) >= len(map_hr)
        assert len(data_lr) >= len(map_lr)
        assert len(map_hr) + len(map_lr) == block_idx.numel()

        N, C, H, W = x.shape
        npixels_high = len(map_hr) * block_size**2
        npixels_low = len(map_lr) * (block_size // lowres_factor) ** 2
        npixels = npixels_high + npixels_low

        block, grid = get_threads_and_blocks(npixels, C)

        f = load_kernel(
            "split_kernel",
            _split_kernel,
            dtype=Dtype(x),
            batch_size=N,
            channels=C,
            height=H,
            width=W,
            block_size=block_size,
            lowres_factor=lowres_factor,
            do_avg=int(True),
        )
        f(
            block=block,
            grid=grid,
            args=[
                x.data_ptr(),
                data_hr.data_ptr(),
                data_lr.data_ptr(),
                map_hr.data_ptr(),
                map_lr.data_ptr(),
                block_idx.data_ptr(),
                int(npixels_high),
                int(npixels_low),
            ],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream),
        )

        ctx.save_for_backward(block_idx)
        ctx.block_size = block_size
        ctx.lowres_factor = lowres_factor
        ctx.NCHW = (N, C, H, W)
        return data_hr, data_lr

    @staticmethod
    def backward(ctx, grad_hr, grad_lr):
        block_idx = ctx.saved_variables[0]
        grad_hr = grad_hr.contiguous()
        grad_lr = grad_lr.contiguous()
        assert assertcuda(grad_hr, dtypes=DTYPES_FLOAT)
        assert assertcuda(grad_lr, dtypes=DTYPES_FLOAT)

        assert len(grad_hr) == 0 or grad_hr.shape[2:] == (ctx.block_size,) * 2
        assert len(grad_lr) == 0 or grad_lr.shape[2:] == (ctx.block_size // ctx.lowres_factor,) * 2

        N, C, H, W = ctx.NCHW
        x = torch.zeros((N, C, H, W), device="cuda", dtype=grad_hr.dtype)
        npixels = N * H * W

        block, grid = get_threads_and_blocks(npixels, C)

        f = load_kernel(
            "split_kernel_backward",
            _split_kernel_backward,
            dtype=Dtype(grad_hr),
            batch_size=N,
            channels=C,
            height=H,
            width=W,
            block_size=ctx.block_size,
            lowres_factor=ctx.lowres_factor,
            do_avg=int(True),
        )
        f(
            block=block,
            grid=grid,
            args=[x.data_ptr(), grad_hr.data_ptr(), grad_lr.data_ptr(), block_idx.data_ptr(), int(npixels)],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream),
        )

        return x, None, None, None, None, None, None, None


_split_kernel = (
    _kernel_header_blocks
    + """
#define WIDTH ${width}
#define HEIGHT ${height}
#define DO_AVG ${do_avg}

extern "C"
__global__ void split_kernel(
    const DTYPE* __restrict__ const data_in, 
    DTYPE* __restrict__ const highres_out, DTYPE* __restrict__ const lowres_out, 
    const int* __restrict__ const highres_map, const int* __restrict__ const lowres_map, 
    const int* const block_idx, const int npixels_high, const int npixels_low){

CUDA_KERNEL_LOOP(i, npixels_high + npixels_low){
    bool is_highres = i < npixels_high;
    int j = i;
    int THIS_BLOCK_SIZE;
    DTYPE* data_out;
    const int*  data_map;
    if(is_highres){
        THIS_BLOCK_SIZE = BLOCK_SIZE;
        data_out = highres_out;
        data_map = highres_map;
    }else{
        j -= npixels_high;
        THIS_BLOCK_SIZE = BLOCK_SIZE_LOWRES;
        data_out = lowres_out;
        data_map = lowres_map;
    }

    int bn = j / (THIS_BLOCK_SIZE*THIS_BLOCK_SIZE);
    int bh = (j / THIS_BLOCK_SIZE) % THIS_BLOCK_SIZE;
    int bw = j % THIS_BLOCK_SIZE;
    int bi = bn*CHANNELS*THIS_BLOCK_SIZE*THIS_BLOCK_SIZE + bh*THIS_BLOCK_SIZE + bw;
    
    const int block_id = data_map[bn]; // linear patch id
    const int n_grid = block_id / (GRID_W*GRID_H);
    const int h_grid = (block_id / GRID_W) % GRID_H;
    const int w_grid = block_id % GRID_W;

    int n = n_grid;
    int h = h_grid*BLOCK_SIZE + bh*(1+!is_highres);
    int w = w_grid*BLOCK_SIZE + bw*(1+!is_highres);
    int o = n*CHANNELS*WIDTH*HEIGHT + h*WIDTH + w;

    //assert(n < BATCH_SIZE);
    //assert(h < HEIGHT);
    //assert(w < WIDTH);

    CUDA_CHANNEL_LOOP(c){ // loop over channels
        DTYPE val = 0;
        val += data_in[o + c*WIDTH*HEIGHT];
        if(DO_AVG && !is_highres){
            for(int ky=0; ky<LOWRES_FACTOR; ++ky){
                for(int kx=0; kx<LOWRES_FACTOR; ++kx){
                    if (kx==0 & ky==0) continue;
                    val += data_in[o + c*WIDTH*HEIGHT + ky*WIDTH + kx];
                }
            }
            val /= (DTYPE) (LOWRES_FACTOR*LOWRES_FACTOR);
        }
        data_out[bi + c*THIS_BLOCK_SIZE*THIS_BLOCK_SIZE] = val;
    }
} // close kernel loop
} // close kernel
"""
)

_split_kernel_backward = (
    _kernel_header_blocks
    + """
#define WIDTH ${width}
#define HEIGHT ${height}

extern "C"
__global__ void split_kernel_backward(
    DTYPE* __restrict__ const grad_data_out, 
    const DTYPE* __restrict__ const grad_highres_in, const DTYPE* __restrict__ const grad_lowres_in, 
    const int* const block_idx, const int npixels){

CUDA_KERNEL_LOOP(i, npixels){
    const int n = i / (HEIGHT*WIDTH);     // batch element
    const int h = (i / WIDTH) % HEIGHT;   // row
    const int w = i % WIDTH;              // column

    const int gh = h / BLOCK_SIZE;        // row in grid
    const int gw = w / BLOCK_SIZE;        // column in grid

    const int ph = h % BLOCK_SIZE;        // row in patch
    const int pw = w % BLOCK_SIZE;        // column in patch

    const int gidx = n*GRID_W*GRID_H+gh*GRID_W+gw;         // linearized grid index
    int pn = block_idx[gidx];
    const bool is_highres = pn >= 0;
    pn += (!is_highres)*BATCH_SIZE*GRID_H*GRID_W;
    const int i_x = n*CHANNELS*HEIGHT*WIDTH + h*WIDTH + w; // index of channel 0 in x
    
    CUDA_CHANNEL_LOOP(c){ // loop over channels
        DTYPE val;
        if(is_highres){
            // if highres
            int i_block = pn*CHANNELS*BLOCK_SIZE*BLOCK_SIZE + ph*BLOCK_SIZE + pw;
            val = grad_highres_in[i_block + c*BLOCK_SIZE*BLOCK_SIZE];
        }else{
            // if lowres
            int i_block = pn*CHANNELS*BLOCK_SIZE_LOWRES*BLOCK_SIZE_LOWRES + (ph/LOWRES_FACTOR)*BLOCK_SIZE_LOWRES + pw/LOWRES_FACTOR;
            val = grad_lowres_in[i_block + c*BLOCK_SIZE_LOWRES*BLOCK_SIZE_LOWRES]/((DTYPE) (LOWRES_FACTOR*LOWRES_FACTOR));
        }
        grad_data_out[i_x + c*WIDTH*HEIGHT] = val;
    }
} // close kernel loop
} // close kernel
"""
)
