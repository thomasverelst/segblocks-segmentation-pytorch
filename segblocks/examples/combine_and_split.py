import torch
import torch.nn as nn
import segblocks


#################### RUN NET WITH RANDOM DATA ##################

# define data shapes
N, C, H, W = 2, 1, 8, 8
block_size = 4

assert H % block_size == 0
assert W % block_size == 0
# define net
net = nn.Sequential(
    nn.Conv2d(C, C, kernel_size=3, stride=1, padding=3 // 2),
    nn.BatchNorm2d(C),
    nn.ReLU(),
).cuda()

# make random data
x = torch.randn(N, C, H, W, device="cuda")
grid = torch.randn(N, H // block_size, W // block_size, device="cuda") > 0

# run net
print("x:", str(x))
blocks = segblocks.to_blocks(x, grid)
print("blocks:", blocks)
blocks = net(blocks)
out = blocks.combine()
print("out:", out.shape)


################################## SMALL TESTS ##################################


def test_net(net):
    def run(x, grid):
        print("x", x)
        print("grid", grid)

        # split x into blocks
        blocks = segblocks.to_blocks(x, grid)
        print("before net: blocks.highres", blocks.highres)
        print("before net: blocks.lowres", blocks.lowres)

        # run net
        blocks = net(blocks)
        print("after net: blocks.highres", blocks.highres)
        print("after net: blocks.lowres", blocks.lowres)

        # combine blocks to full tensor
        out = blocks.combine()
        print("out", out)

    print("\n------------------------------ ALL HIGHRES ------------------------------")
    x = torch.arange(start=0, end=(N * C * H * W), step=1, dtype=torch.float, device="cuda").reshape(N, C, H, W).cuda()
    grid = torch.ones(N, H // block_size, W // block_size, device="cuda").bool()
    run(x, grid)

    print("\n------------------------------ ALL LOWRES ------------------------------")
    x = torch.arange(start=0, end=(N * C * H * W), step=1, dtype=torch.float, device="cuda").reshape(N, C, H, W).cuda()
    grid = torch.zeros(N, H // block_size, W // block_size, device="cuda").bool()
    run(x, grid)

    print("\n------------------------------ MIX HIGHRES/LOWRES ------------------------------")
    x = torch.arange(start=0, end=(N * C * H * W), step=1, dtype=torch.float, device="cuda").reshape(N, C, H, W).cuda()
    grid = torch.randn(N, H // block_size, W // block_size, device="cuda") > 0
    run(x, grid)


print("\n\n###################################### SPLIT AND COMBINE ######################################")
N, C, H, W = 1, 1, 4, 4
block_size = 2
test_net(nn.Identity().cuda())

print("\n\n###################################### SPLIT, CONV AND COMBINE ######################################")


N, C, H, W = 1, 1, 8, 8
block_size = 4

test_net(nn.Conv2d(C, C, kernel_size=3, stride=1, padding=3 // 2).cuda())
