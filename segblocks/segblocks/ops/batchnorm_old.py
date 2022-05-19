import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
try:
    from . import patchtensor 
except:
    pass

from torch.utils.checkpoint import checkpoint

def patch_bn(bn, p):
    assert isinstance(p, patchtensor.PatchTensor)
    assert bn.affine
    assert bn.training

    if bn.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = bn.momentum

    if bn.training and bn.track_running_stats:
        if bn.num_batches_tracked is not None:
            bn.num_batches_tracked = bn.num_batches_tracked + 1
            if bn.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = bn.momentum

    hr, lr = p.highres, p.lowres

    if bn.training:
        if p.has_hr:
            n_hr = hr.shape[0]*hr.shape[2]*hr.shape[3]
            var_hr, mean_hr = torch.var_mean(hr, [0, 2, 3], unbiased=False)
        else:
            n_hr, var_hr, mean_hr = 0, 0, 0

        if p.has_lr:
            n_lr = lr.shape[0]*lr.shape[2]*lr.shape[3]
            var_lr, mean_lr = torch.var_mean(lr, [0, 2, 3], unbiased=False)
        else:
            n_lr, var_lr, mean_lr = 0, 0, 0

        n = n_hr + n_lr
        var =  var_hr*(float(n_hr)/n) + var_lr*(float(n_lr)/n)
        mean = mean_hr*(float(n_hr)/n) + mean_lr*(float(n_lr)/n)

        with torch.no_grad():
            bn.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * bn.running_mean
            bn.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * bn.running_var
    else:
        mean = bn.running_mean
        var = bn.running_var

    do_efficient=False
    if p.has_hr:
        # hr = BN_func.apply(hr, mean, var, bn.weight, bn.bias, bn.affine, bn.eps)
        hr = do_checkpoint(apply_bn, (hr, mean, var, bn.weight, bn.bias, bn.affine, bn.eps), efficient=do_efficient)
        
    if p.has_lr:
        lr = do_checkpoint(apply_bn, (lr, mean, var, bn.weight, bn.bias, bn.affine, bn.eps), efficient=do_efficient)
        # lr = apply_bn(lr, mean, var, bn.weight, bn.bias, bn.affine, bn.eps)

    return p.new(hr, lr)

def do_checkpoint(func, args, efficient=True):
    if efficient:
        return checkpoint(func, *args)
    else:
        return func(*args)

# import torchvision
# @torch.jit.script
def apply_bn(x:torch.Tensor, mean:torch.Tensor, var:torch.Tensor, weight:torch.Tensor, bias:torch.Tensor, affine: bool, eps: float):
    sqrtvar = torch.sqrt(var + eps)
    out = x.sub(mean[None, :, None, None]).div(sqrtvar[None, :, None, None])
    # out = torch.addcmul(bias[None, :, None, None], out, weight[None, :, None, None])
    return out.mul(weight[None, :, None, None]).add(bias[None, :, None, None])
    # return out
    # out = (x-mean[None, :, None, None])/sqrtvar[None, :, None, None]
    # return (out * weight[None, :, None, None]) + (bias[None, :, None, None])
    # out = torchvision.transforms.functional.normalize(x, mean=mean, std=sqrtvar, inplace=True)
    # if affine:
    # return out
    # x_hat = x.sub(mean.detach()[None, :, None, None]).div_(sqrtvar.detach()[None, :, None, None])

    # out = x_hat




from torch.autograd import Function
class BatchNormFunction(Function):

    @classmethod
    def expand(cls, x):
        return x[None, :, None, None]

    @staticmethod
    def forward(ctx, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, 
                     weight: torch.Tensor, bias: torch.Tensor, affine: bool, eps: float):
        c = x.size(1)
        assert x.dim() == 4
        assert mean.dim() == 1
        assert var.dim() == 1
        assert weight.dim() == 1
        assert bias.dim() == 1
        assert c == mean.size(0)
        assert c == var.size(0)
        assert c == weight.size(0)
        assert c == bias.size(0)

        sqrtvar = torch.sqrt_(var + eps)
        mean = BatchNormFunction.expand(mean)
        sqrtvar = BatchNormFunction.expand(sqrtvar)
        weight = BatchNormFunction.expand(weight)
        bias = BatchNormFunction.expand(bias)

        x_hat, out = BatchNormFunction.func_forward(x, mean, sqrtvar, weight, bias, affine)
        ctx.save_for_backward(x_hat, weight, sqrtvar)
        return out

    @staticmethod
    # @torch.jit.script
    def func_forward(x, mean: torch.Tensor, sqrtvar: torch.Tensor,  weight: torch.Tensor, bias: torch.Tensor, affine: bool):
        x_hat = (x - mean).div_(sqrtvar)
        if affine:
            out = (weight * x_hat).add_(bias)
        else:
            out = x_hat
        return x_hat, out

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, weight, sqrtvar = ctx.saved_tensors
        N = grad_output.numel() // grad_output.size(1) #  batch_size * width * height        
        grad_input, grad_weight, grad_bias = BatchNormFunction.func_backward(grad_output, x_hat, weight, sqrtvar, N)
        return grad_input, None, None, grad_weight, grad_bias, None, None

    @staticmethod
    # @torch.jit.script
    def func_backward(grad_output: torch.Tensor, x_hat: torch.Tensor, weight: torch.Tensor, sqrtvar: torch.Tensor, N: int):
        grad_x_hat = grad_output * weight
        term_1 = N * grad_x_hat
        term_2 = torch.sum(grad_x_hat, dim=(0, 2, 3), keepdim=True)
        term_3 = x_hat * torch.sum(grad_x_hat * x_hat, dim=(0, 2, 3), keepdim=True)
        
        grad_input = (1/N)*(1/sqrtvar) * (term_1 - term_2 - term_3)
        grad_weight = torch.sum(torch.mul(grad_output, x_hat), dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias





# bnfunc = torch.jit.script(BN_func)

if __name__ == "__main__":
    pass
    from torch.autograd import gradcheck
    # set deterministic cuda 
    torch.backends.cudnn.enabled = False
    torch.manual_seed(0)

    dtype = torch.double

    C = 2
    x = torch.rand( (1, C, 2, 2), requires_grad=True, dtype=dtype)
    mean = x.mean(dim=(0, 2, 3), keepdim=False).detach()
    var = x.var(dim=(0, 2, 3), keepdim=False, unbiased=False).detach()
    # var = torch.zeros_like(mean)

    # mean = torch.randn((3,))
    # sqrtvar = torch.randn((3,))
    weight = torch.rand( (C, ), requires_grad=True, dtype=dtype)
    bias = torch.rand( (C, ), requires_grad=True, dtype=dtype)
    affine = True
    eps = 1e-8



    inputs = (x, mean, var, weight, bias, affine, eps)
    torch.autograd.gradcheck(apply_bn, inputs)

    #fw 
    if True:
        out = BatchNormFunction.apply(*inputs)
        print('out', out)
        loss = out.sum()
        loss.backward(retain_graph=True)
        xgrad  = x.grad.clone()
        print('xgrad', xgrad[0])

    # zero grad
    x.grad.zero_()



    if True:
        out2 = apply_bn(*inputs)
        print('out2', out2)
        loss2 = out2.sum()
        loss2.backward(retain_graph=True)
        xgrad2  = x.grad.clone()

        print('xgrad2', xgrad2[0])
    

    # print(gradcheck(BN_func.apply, inputs))

    # torch gradcheck

    

