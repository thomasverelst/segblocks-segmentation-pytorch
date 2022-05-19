import torch
from torch.autograd import Function

from .util import (Dtype, Stream, _kernel_header_blocks, assertcuda,
                   get_threads_and_blocks, load_kernel, DTYPES_FLOAT)


class CombineFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, data_hr: torch.Tensor, data_lr: torch.Tensor, 
                block_idx: torch.Tensor, block_size: int, lowres_factor: int) -> torch.Tensor:
        assert assertcuda(x, dtypes=DTYPES_FLOAT)
        assert assertcuda(data_hr, dtypes=DTYPES_FLOAT)
        assert assertcuda(data_lr, dtypes=DTYPES_FLOAT)
        assert assertcuda(block_idx, dtypes=torch.int32)

        ctx.save_for_backward(block_idx)
        ctx.block_size = block_size
        ctx.lowres_factor = lowres_factor
        ctx.highres_shape = data_hr.shape
        ctx.lowres_shape = data_lr.shape

        assert len(data_hr) == 0 or data_hr.shape[2:] == (block_size, block_size)
        assert len(data_lr)  == 0 or data_lr.shape[2:]  == (block_size//lowres_factor, block_size//lowres_factor)

        N,C,H,W = x.shape
        npixels = N*H*W
        
        block, grid = get_threads_and_blocks(npixels, C)

        f = load_kernel('combine_kernel', _combine_kernel,  dtype=Dtype(x),
                 batch_size=N, channels=C, height=H, width=W,
                block_size=block_size, lowres_factor=lowres_factor)
        f(block=block, grid=grid,
            args=[
                x.data_ptr(), data_hr.data_ptr(), data_lr.data_ptr(), 
                block_idx.data_ptr(), npixels
            ],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return x

    @staticmethod
    def backward(ctx, grad_combined):
        grad_combined = grad_combined.contiguous()
        
        block_idx = ctx.saved_variables[0]
        N,C,H,W = grad_combined.shape
        npixels = N*H*W
        
        block, grid = get_threads_and_blocks(npixels, C)

        highres_grad = torch.zeros(ctx.highres_shape, device=grad_combined.device, dtype=grad_combined.dtype)
        lowres_grad = torch.zeros(ctx.lowres_shape, device=grad_combined.device, dtype=grad_combined.dtype)

        f = load_kernel('combine_kernel_bw', _combine_kernel_bw,  dtype=Dtype(grad_combined),
                 batch_size=N, channels=C, height=H, width=W,
                block_size=ctx.block_size, lowres_factor=ctx.lowres_factor)
        f(block=block, grid=grid,
            args=[
                grad_combined.data_ptr(), highres_grad.data_ptr(), lowres_grad.data_ptr(), 
                block_idx.data_ptr(), npixels
            ], stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return None, highres_grad, lowres_grad, None, None, None, None


_combine_kernel = _kernel_header_blocks+'''
#define WIDTH ${width}
#define HEIGHT ${height}

extern "C"
__global__ void combine_kernel(
    DTYPE* __restrict__ const data_out, 
    const DTYPE* __restrict__ const highres_in, const DTYPE* __restrict__ const lowres_in, 
    const int* const block_idx, const int npixels){

CUDA_KERNEL_LOOP(i, npixels){
    const int n = i / (HEIGHT*WIDTH);     // batch element
    const int h = (i / WIDTH) % HEIGHT;   // row
    const int w = i % WIDTH;              // column
    const int i_x = n*CHANNELS*HEIGHT*WIDTH + h*WIDTH + w; // index of channel 0 in x

    const int gh = h / BLOCK_SIZE;        // row in grid
    const int gw = w / BLOCK_SIZE;        // column in grid

    const int ph = h % BLOCK_SIZE;        // row in patch
    const int pw = w % BLOCK_SIZE;        // column in patch


    const int gidx = n*GRID_W*GRID_H+gh*GRID_W+gw;         // linearized grid index
    int block_id = block_idx[gidx];
    const bool is_highres = block_id >= 0;

    int i_block, c_stride;
    const DTYPE* data_in;
    if(is_highres){
        i_block = block_id*CHANNELS*BLOCK_SIZE*BLOCK_SIZE + ph*BLOCK_SIZE + pw;
        c_stride = BLOCK_SIZE*BLOCK_SIZE;
        data_in = highres_in;
    }else{
        block_id += BATCH_SIZE*GRID_H*GRID_W;
        i_block = block_id*CHANNELS*BLOCK_SIZE_LOWRES*BLOCK_SIZE_LOWRES + (ph/LOWRES_FACTOR)*BLOCK_SIZE_LOWRES + pw/LOWRES_FACTOR;
        c_stride = BLOCK_SIZE_LOWRES*BLOCK_SIZE_LOWRES;
        data_in = lowres_in;
    }
    
    CUDA_CHANNEL_LOOP(c){ // loop over channels
        data_out[i_x + c*WIDTH*HEIGHT] = data_in[i_block + c*c_stride];
    }
} // close kernel loop
} // close kernel
'''

_combine_kernel_bw = _kernel_header_blocks+'''
#define WIDTH ${width}
#define HEIGHT ${height}

extern "C"
__global__ void combine_kernel_bw(
    const DTYPE* __restrict__ const grad_data_in, 
    DTYPE* __restrict__ const grad_highres_out, DTYPE* __restrict__ const grad_lowres_out, 
    const int* const block_idx, const int npixels){

CUDA_KERNEL_LOOP(i, npixels){
    const int n = i / (HEIGHT*WIDTH);     // batch element
    const int h = (i / WIDTH) % HEIGHT;   // row
    const int w = i % WIDTH;              // column
    const int i_x = n*CHANNELS*HEIGHT*WIDTH + h*WIDTH + w; // index of channel 0 in x

    const int gh = h / BLOCK_SIZE;        // row in grid
    const int gw = w / BLOCK_SIZE;        // column in grid

    const int ph = h % BLOCK_SIZE;        // row in patch
    const int pw = w % BLOCK_SIZE;        // column in patch

    const int gidx = n*GRID_W*GRID_H+gh*GRID_W+gw;         // linearized grid index
    int block_id = block_idx[gidx];
    const bool is_highres = block_id >= 0;
    if(!is_highres) block_id += BATCH_SIZE*GRID_H*GRID_W;

    CUDA_CHANNEL_LOOP(c){ // loop over channels
        DTYPE val = grad_data_in[i_x + c*WIDTH*HEIGHT];
        if(is_highres){
            // if highres
            int i_block = block_id*CHANNELS*BLOCK_SIZE*BLOCK_SIZE + ph*BLOCK_SIZE + pw;
            grad_highres_out[i_block + c*BLOCK_SIZE*BLOCK_SIZE] = val;
        }else{
            // if lowres
            int i_block = block_id*CHANNELS*BLOCK_SIZE_LOWRES*BLOCK_SIZE_LOWRES + (ph/LOWRES_FACTOR)*BLOCK_SIZE_LOWRES + pw/LOWRES_FACTOR;
            atomicAdd(grad_lowres_out + i_block + c*BLOCK_SIZE_LOWRES*BLOCK_SIZE_LOWRES, val);
        }
    }
} // close kernel loop
} // close kernel
'''
