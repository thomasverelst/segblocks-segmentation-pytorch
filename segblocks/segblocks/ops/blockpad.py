import torch
from torch.autograd import Function

from .util import DTYPES_FLOAT, Dtype, Stream, _kernel_header_blocks, assertcuda, get_threads_and_blocks, load_kernel


class BlockPad(Function):
    """
    Function to pad tensors using BlockPad
    """

    @staticmethod
    def forward(
        ctx,
        data_hr: torch.Tensor,
        data_lr: torch.Tensor,
        map_hr: torch.Tensor,
        map_lr: torch.Tensor,
        block_idx: torch.Tensor,
        padding: int,
        is_highres: bool,
    ) -> torch.Tensor:
        """
        Returns a block-padded version of the data tensor.
        When is_highres is True, the high-res data tensor (data_hr) is padded.
        When is_highres is False, the low-res data tensor (data_lr) is padded.

        data_hr: highres tensor
        data_lr: lowres tensor
        map_hr: highres mapping (from DualResMetadata)
        map_lr: lowres mapping (from DualResMetadata)
        block_idx: block index (from DualResMetadata)
        padding: padding size in pixels (equal padding on all sides)
        is_highres: whether to pad the highres data tensor (True) or the lowres data tensor (False)
        """
        do_avg = 1  # use average resampling
        assert assertcuda(data_hr, dtypes=DTYPES_FLOAT)
        assert assertcuda(data_lr, dtypes=DTYPES_FLOAT)
        assert assertcuda(block_idx, dtypes=torch.int32)
        assert assertcuda(map_hr, dtypes=torch.int32)
        assert assertcuda(map_lr, dtypes=torch.int32)
        assert len(map_hr) + len(map_lr) == block_idx.numel()
        assert data_hr.shape[2] == data_hr.shape[3]
        assert data_lr.shape[2] == data_lr.shape[3]
        assert data_hr.shape[2] % data_lr.shape[2] == 0
        ctx.save_for_backward(map_hr, map_lr, block_idx)
        ctx.padding = padding
        ctx.do_avg = do_avg
        ctx.is_highres = is_highres

        batch_size, grid_h, grid_w = block_idx.shape
        block_size = data_hr.shape[2]
        lowres_factor = data_hr.shape[2] // data_lr.shape[2]
        height, width = grid_h * block_size, grid_w * block_size

        ctx.block_size = block_size  # block size of highres data
        ctx.lowres_factor = lowres_factor  # size factor of highres compared to lowres

        data_in = data_hr if is_highres else data_lr
        data_map = map_hr if is_highres else map_lr
        num_blocks, C, side, _ = data_in.shape
        block_size_pad = side + 2 * padding
        size = (num_blocks, C, block_size_pad, block_size_pad)
        data_out = torch.empty(size, device=data_in.device, dtype=data_in.dtype)
        if len(data_map) == 0:
            data_out.fill_(0)
        else:
            npixels = len(data_map) * block_size_pad**2  # number of output pixels to process
            block, grid = get_threads_and_blocks(npixels, C)  # get CUDA block and grid

            # build a kernel
            f = load_kernel(
                "blockpad_kernel_forward",
                _repad_kernel_avg_sep,
                dtype=Dtype(data_hr),
                batch_size=batch_size,
                channels=C,
                height=height,
                width=width,
                block_size=block_size,
                lowres_factor=lowres_factor,
                padding=padding,
                is_highres=int(is_highres),
                do_avg=int(do_avg),
            )
            # execute kernel
            f(
                block=block,
                grid=grid,
                args=[
                    data_hr.data_ptr(),
                    data_lr.data_ptr(),
                    data_map.data_ptr(),
                    data_out.data_ptr(),
                    block_idx.data_ptr(),
                    int(npixels),
                ],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream),
            )
        return data_out

    @staticmethod
    def backward(ctx, grad_data_pad: torch.Tensor):
        grad_highres_in = grad_data_pad.contiguous()
        assert assertcuda(grad_highres_in, dtypes=DTYPES_FLOAT)

        map_hr, map_lr, block_idx = ctx.saved_variables
        block_size = ctx.block_size
        channels = grad_data_pad.size(1)
        lowres_factor = ctx.lowres_factor
        padding = ctx.padding
        do_avg = ctx.do_avg

        batch_size, grid_h, grid_w = block_idx.shape
        height, width = grid_h * ctx.block_size, grid_w * ctx.block_size

        def create_tensors(data_map, block_size):
            n = max(len(data_map), 1)
            size = (n, channels, block_size, block_size)
            return torch.zeros(size, device=grad_data_pad.device, dtype=grad_data_pad.dtype)

        grad_hr = create_tensors(map_hr, block_size)
        grad_lr = create_tensors(map_lr, block_size // lowres_factor)

        data_map = map_hr if ctx.is_highres else map_lr
        if len(data_map) > 0:
            npixels = len(data_map) * grad_data_pad.shape[2] ** 2
            block, grid = get_threads_and_blocks(npixels, channels)
            fac = load_kernel(
                "blockpad_kernel_backward",
                _repad_kernel_avg_sep_bw,
                dtype=Dtype(grad_data_pad),
                batch_size=batch_size,
                channels=channels,
                height=height,
                width=width,
                block_size=block_size,
                lowres_factor=lowres_factor,
                padding=padding,
                is_highres=int(ctx.is_highres),
                do_avg=int(do_avg),
            )
            fac(
                block=block,
                grid=grid,
                args=[
                    grad_hr.data_ptr(),
                    grad_lr.data_ptr(),
                    grad_data_pad.data_ptr(),
                    data_map.data_ptr(),
                    block_idx.data_ptr(),
                    int(npixels),
                ],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream),
            )
        return grad_hr, grad_lr, None, None, None, None, None


_repad_kernel_avg_sep = (
    _kernel_header_blocks
    + """
#define IS_HIGHRES ${is_highres}
#define DO_AVG ${do_avg}
#define PADDING ${padding}

extern "C"
__global__ void blockpad_kernel_forward(
    const DTYPE* __restrict__  const data_hr, const DTYPE* __restrict__ const data_lr,
    const int* __restrict__ const data_map, DTYPE* __restrict__ const data_out,
    const int* __restrict__ const block_idx, const int npixels) {

const int BS = IS_HIGHRES ? BLOCK_SIZE : BLOCK_SIZE_LOWRES; // block size
const int BS_OUT = BS + 2*PADDING; // block size with padding (block size of output)

CUDA_KERNEL_LOOP(i, npixels){ 
    // loop over every output pixel (padded blocks)
    const DTYPE* data_in = IS_HIGHRES ? data_hr : data_lr; // input data
    int BS_IN = BS; // block size of input

    const int n_out = i / (BS_OUT*BS_OUT);        // batch  
    const int h_out = (i / BS_OUT) % BS_OUT;      // height
    const int w_out = i % BS_OUT;                 // width

    int n_in = n_out;
    int h_in = h_out - PADDING;
    int w_in = w_out - PADDING;

    // check if this position is in a block's padding 
    const bool is_left = w_out < PADDING;
    const bool is_right = w_out >= BS + PADDING;
    const bool is_top = h_out < PADDING;
    const bool is_bottom = h_out >= BS + PADDING;

    const bool is_pad = is_left|is_right|is_top|is_bottom; // is in padding
    bool zero_pad = false; // if should be zero-padded (block at image edge)
    bool downsample = false; // if this block is low-res and neighbor is highres

    if(is_pad){
        // find the position of the block in the grid
        const int block_id = data_map[n_out]; // linear block id
        const int h_grid = (block_id / GRID_W) % GRID_H;
        const int w_grid = block_id % GRID_W;
        
        // check if it is in the zero-padding (image side)
        zero_pad = ((is_left & w_grid==0) || (is_right & w_grid==GRID_W-1) || \
                   (is_top & h_grid==0) || (is_bottom & h_grid==GRID_H-1));
        if(!zero_pad){
            // pad by copying values from neighbour
            
            // get the block_id of the neighbor
            // avoid conditional statements subtracting/adding boolean values
            int block_id_in = block_id;
            block_id_in -= is_left; // left neighbor
            block_id_in += is_right; // right neighbor
            block_id_in -= GRID_W*is_top; // top neighbor
            block_id_in += GRID_W*is_bottom; // bottom neighbor

            // get the pixel position
            n_in = block_idx[block_id_in];
            h_in = h_in + is_top*BS - is_bottom*BS;
            w_in = w_in + is_left*BS - is_right*BS;
            
            const bool is_highres_in = n_in >= 0; // check if neighbor is highres
            if(is_highres_in){ // if neighbor is highres
                if(!IS_HIGHRES){ // and if this block is lowres
                    h_in *= LOWRES_FACTOR; // adjust pixel position
                    w_in *= LOWRES_FACTOR; // adjust pixel position
                    data_in = data_hr; // set data input tensor
                    BS_IN = BLOCK_SIZE; // set block size
                    downsample = true; // should downsample data (combining pixels of highres input)
                }
            }else{ // neighbor is lowres
                n_in += BATCH_SIZE*GRID_H*GRID_W; // lowres blocks have negative indices
                if(IS_HIGHRES){ // and this block is highres
                    h_in /= LOWRES_FACTOR; // adjust pixel position
                    w_in /= LOWRES_FACTOR; // adjust pixel position
                    data_in = data_lr; // set data input tensor
                    BS_IN = BLOCK_SIZE_LOWRES; // set block size
                }
            }
            //assert(n_in >= 0);
            //assert(h_in >= 0);
            //assert(w_in >= 0);
            //assert(h_in < BS_IN);
            //assert(w_in < BS_IN);
        }
    }

    // channel 0 index of data tensors
    const int b_in = n_in*BS_IN*BS_IN*CHANNELS + h_in*BS_IN + w_in;
    const int b_out = n_out*BS_OUT*BS_OUT*CHANNELS + h_out*BS_OUT + w_out; 

    CUDA_CHANNEL_LOOP(c){
        DTYPE val = 0;
        if(!zero_pad){ // if current pixel was not in zero-padding (image side)
            val = data_in[b_in + c*BS_IN*BS_IN]; // get value from input
            if(DO_AVG && !IS_HIGHRES && downsample){ // average multiple values from highres neighbor if this is lowres
                #pragma unroll
                for(int ky=0; ky<LOWRES_FACTOR; ++ky){
                    for(int kx=0; kx<LOWRES_FACTOR; ++kx){
                        if (kx==0 & ky==0) continue; // was already included in val
                        val += data_in[b_in + c*BS_IN*BS_IN + BS_IN*ky + kx];
                    }
                }
                val /= (DTYPE) (LOWRES_FACTOR*LOWRES_FACTOR);
            }
        }
        data_out[b_out + c*BS_OUT*BS_OUT] = val; // copy data to output
    }
} // closes kernel_loop
} // closes kernel
"""
)


_repad_kernel_avg_sep_bw = (
    _kernel_header_blocks
    + """
#define IS_HIGHRES ${is_highres}
#define DO_AVG ${do_avg}
#define PADDING ${padding}

extern "C"
__global__ void blockpad_kernel_backward(
    DTYPE* __restrict__  const grad_hr_out, DTYPE* __restrict__ const grad_lr_out,
    const DTYPE* __restrict__ const data_in, const int* __restrict__ const data_map,
    const int* __restrict__ const block_idx, const int npixels) {

const int BS = IS_HIGHRES ? BLOCK_SIZE : BLOCK_SIZE_LOWRES; // block size
const int BS_IN = BS + 2*PADDING; // block size with padding (block size of input)

CUDA_KERNEL_LOOP(i, npixels){
    // loop over every input pixel (padded blocks)
    DTYPE* data_out = IS_HIGHRES ? grad_hr_out : grad_lr_out; // output data
    int BS_OUT = BS; // block size of output

    const int n_in = i / (BS_IN*BS_IN);        // batch  
    const int h_in = (i / BS_IN) % BS_IN;      // height
    const int w_in = i % BS_IN;                  // width

    int n_out = n_in;
    int h_out = h_in - PADDING;
    int w_out = w_in - PADDING;

    // check if this position is in block's padding 
    const bool is_left = w_in < PADDING;
    const bool is_right = w_in >= BS + PADDING;
    const bool is_top = h_in < PADDING;
    const bool is_bottom = h_in >= BS + PADDING;

    const bool is_pad = is_left|is_right|is_top|is_bottom;
    bool zero_pad = false;
    bool downscale = false;

    if(is_pad){
        // find position of patch it is in
        const int block_id = data_map[n_in]; // linear patch id
        const int h_grid = (block_id / GRID_W) % GRID_H;
        const int w_grid = block_id % GRID_W;
        
        // check if it is in the side zero-padding
        zero_pad = ((is_left & w_grid==0) | (is_right & w_grid==GRID_W-1) | \
                   (is_top & h_grid==0) | (is_bottom & h_grid==GRID_H-1));
        if(!zero_pad){
            // pad by copying from neighbour
            
            int block_id_out = block_id;
            block_id_out -= is_left; // left neighbor
            block_id_out += is_right; // right neighbor
            block_id_out -= GRID_W*is_top; // top neighbor
            block_id_out += GRID_W*is_bottom; // bottom neighbor

            n_out = block_idx[block_id_out];
            h_out = h_out + is_top*BS - is_bottom*BS;
            w_out = w_out + is_left*BS - is_right*BS;
            
            const bool is_highres_out = n_out >= 0;
            if(is_highres_out){
                if(!IS_HIGHRES){
                    h_out *= LOWRES_FACTOR;
                    w_out *= LOWRES_FACTOR;
                    data_out = grad_hr_out;
                    BS_OUT = BLOCK_SIZE;
                    downscale = true;
                }
            }else{
                n_out += BATCH_SIZE*GRID_H*GRID_W;
                if(IS_HIGHRES){
                    h_out /= LOWRES_FACTOR;
                    w_out /= LOWRES_FACTOR;
                    data_out = grad_lr_out;
                    BS_OUT = BLOCK_SIZE_LOWRES;
                }
            }
            //assert(n_out >= 0);
            //assert(h_out >= 0);
            //assert(w_out >= 0);
            //assert(h_out < BS_OUT);
            //assert(w_out < BS_OUT);
        }
    }

    // channel 0 index 
    const int b_in = n_in*BS_IN*BS_IN*CHANNELS + h_in*BS_IN + w_in;
    const int b_out = n_out*BS_OUT*BS_OUT*CHANNELS + h_out*BS_OUT + w_out; 

    CUDA_CHANNEL_LOOP(c){
        if(!zero_pad){
            DTYPE val = data_in[b_in + c*BS_IN*BS_IN];

            if(DO_AVG && !IS_HIGHRES && downscale){
                val /= (DTYPE) (LOWRES_FACTOR*LOWRES_FACTOR);
                for(int ky=0; ky<LOWRES_FACTOR; ++ky){
                    for(int kx=0; kx<LOWRES_FACTOR; ++kx){
                         atomicAdd(data_out + b_out + c*BS_OUT*BS_OUT + BS_OUT*ky + kx, val);
                    }
                }
            }else{
                atomicAdd(data_out + b_out + c*BS_OUT*BS_OUT, val);
            }
        }
    }
} // closes kernel_loop
} // closes kernel
"""
)
