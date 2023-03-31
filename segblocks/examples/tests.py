import torch
import torch.nn as nn
import segblocks

torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.use_deterministic_algorithms(False) # backward of upsample is not compatible
torch.set_anomaly_enabled(True)

N, C, H, W = 2, 3, 12, 12
block_size = 4

net_conv = nn.Sequential(nn.Conv2d(C, C, kernel_size=3, stride=1, padding=3 // 2)).cuda()
net_conv_bn_relu = nn.Sequential(
    nn.Conv2d(C, C, kernel_size=3, stride=1, padding=3 // 2), 
    nn.BatchNorm2d(C), 
    nn.ReLU()
).cuda()


if True:
    ### all high-res blocks should be identical to running without blocks
    x = torch.randn((N, C, H, W), dtype=torch.float, device="cuda", requires_grad=False)
    grid = torch.ones(N, H // block_size, W // block_size, device="cuda").bool()

    # run with blocks
    a = x.clone()
    a.requires_grad = True
    out_blocks = segblocks.execute_with_grid(net_conv_bn_relu, a, grid)

    # run without blocks
    b = x.clone()
    b.requires_grad = True
    out_noblocks = net_conv_bn_relu(b)

    assert torch.allclose(out_blocks, out_noblocks, rtol=1e-3, atol=1e-5)
    print("Check all-highres blocks forward ok")

    # check backward
    loss_a = a.sum()
    loss_b = b.sum()

    loss_a.backward(retain_graph=True)
    loss_b.backward(retain_graph=True)

    assert torch.allclose(a.grad, b.grad, rtol=1e-3, atol=1e-5)
    print("Check all-highres blocks backward ok")


if True:
    ## combined high/lowres is harder to check, use torch gradcheck
    # note that gradcheck requires fp64
    x = torch.randn((N, C, H, W), dtype=torch.double, device="cuda", requires_grad=True)
    # grid = torch.ones(N, H//block_size, W//block_size, device='cuda').bool()
    grid = torch.randn(N, H // block_size, W // block_size, device="cuda") > 0

    # split and combine
    def func_split_combine(x, grid):
        y = segblocks.to_blocks(x, grid)
        z = y.combine()
        return z

    assert torch.autograd.gradcheck(func_split_combine, (x, grid))
    print("Gradcheck split-combine ok")

    def func_split_net_combine(x, grid, net):
        y = segblocks.to_blocks(x, grid)
        y = net(y)
        z = y.combine()
        return z

    # split, conv and combine
    assert torch.autograd.gradcheck(func_split_net_combine, (x, grid, net_conv.double()))
    print("Gradcheck split-net_conv-combine ok")
    
    # split, conv_bn_relu and combine
    assert torch.autograd.gradcheck(func_split_net_combine, (x, grid, net_conv_bn_relu.double()), nondet_tol=1e-3)
    print("Gradcheck split-net_conv_bn_relu-combine ok")
