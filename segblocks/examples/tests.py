import torch
import torch.nn as nn
import segblocks

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

N, C, H, W = 2, 3, 12, 12
block_size = 4

net = nn.Sequential(nn.Conv2d(C, C, kernel_size=3, stride=1, padding=3 // 2)).cuda()


if True:
    ### all high-res blocks should be identical to running without blocks
    x = torch.randn((N, C, H, W), dtype=torch.float, device="cuda", requires_grad=False)
    grid = torch.ones(N, H // block_size, W // block_size, device="cuda").bool()

    # run with blocks
    a = x.clone()
    a.requires_grad = True
    out_blocks = segblocks.execute_with_grid(net, a, grid)

    # run without blocks
    b = x.clone()
    b.requires_grad = True
    out_noblocks = net(b)

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
    ## combind high/lowres is harder to check, use torch gradcheck
    x = torch.randn((N, C, H, W), dtype=torch.double, device="cuda", requires_grad=True)
    # grid = torch.ones(N, H//block_size, W//block_size, device='cuda').bool()
    grid = torch.randn(N, H // block_size, W // block_size, device="cuda") > 0

    net_double = net.double()

    # split and combine
    def func_split_combine(x, grid):
        y = segblocks.to_blocks(x, grid)
        z = y.combine()
        return z

    assert torch.autograd.gradcheck(func_split_combine, (x, grid))
    print("Gradcheck split-combine ok")

    # split and combine
    def func_split_net_combine(x, grid):
        y = segblocks.to_blocks(x, grid)
        y = net_double(y)
        z = y.combine()
        return z

    assert torch.autograd.gradcheck(func_split_net_combine, (x, grid))
    print("Gradcheck split-net-combine ok")
