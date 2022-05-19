'''
example to run resnet with dual-res segblocks
the convolutional backbone is ran in blocks 
however, the adaptive averaging pooling before the final linear layer cannot be executed in blocks
'''


import segblocks
import torchvision
import torch
import torch.nn as nn

def run_resnet(model, x, grid, verbose=False):
    ## run convolutional backbone (feature extractor) in blocks
    feature_extractor = nn.Sequential(
        model.conv1, 
        model.bn1, 
        model.relu, 
        model.maxpool, 
        model.layer1, 
        model.layer2, 
        model.layer3, 
        model.layer4
    )

    blocks = segblocks.to_blocks(x, grid) # convert x to blocks
    if verbose:
        print('>> Tensor: blocks:', blocks.shape)
    blocks_features = feature_extractor(blocks) # run feature extractor on blocks, with BlockPad module automatically inserted for convolutions
    if verbose:
        print('>> Tensor: blocks_features:', blocks_features.shape)
    features = blocks_features.combine() # combine blocks to full tensor
    if verbose:
        print('>> Tensor: features:', features.shape)

    # run head
    x = model.avgpool(features)
    x = torch.flatten(x, 1)
    out = model.fc(x)
    if verbose:
        print('>> Tensor: out', out.shape)
    return out
# data
N, C, H, W = 4, 3, 512, 512
block_size = 128

x = torch.randn(N, C, H, W, device='cuda') # image data

# define model
model = torchvision.models.resnet18(pretrained=True)
model = model.cuda().eval()

# execute examples

print('\n------------------------------ RANDOM HIGH/LOWRES ------------------------------')
# run with all blocks in high resolution
grid = torch.randn(N, H//block_size, W//block_size, device='cuda') > 0
out = run_resnet(model, x, grid, verbose=True)

print('\n------------------------------ ALL HIGHRES ------------------------------')
# run with all blocks in high resolution
grid = torch.ones(N, H//block_size, W//block_size, device='cuda').bool()
out = run_resnet(model, x, grid, verbose=True)
print('out with all blocks', out)
# should be equivalent to running without blocks
out_noblocks = model(x)
print('out without blocks', out_noblocks)
assert torch.allclose(out, out_noblocks, rtol=1e-3, atol=1e-5)


print('\n------------------------------ ALL LOWRES ------------------------------')
# run with all blocks in low resolution
grid = torch.zeros(N, H//block_size, W//block_size, device='cuda').bool()
out = run_resnet(model, x, grid, verbose=True)
print('out with all blocks', out)
# should be equivalent to running without blocks on a downsampled image
x_ds = torch.nn.functional.avg_pool2d(x, kernel_size=2)
out_noblocks = model(x_ds)
print('out without blocks', out_noblocks)
assert torch.allclose(out, out_noblocks, rtol=1e-3, atol=1e-5)



# example to show that average pooling is not supported
# try:
#     print('x:', x.shape)
#     blocks = segblocks.to_blocks(x, grid)
#     print('blocks:', blocks.shape)
#     blocks = model(blocks)
#     out = blocks.combine()
#     print('out:', out.shape)
# except Exception as e:
#     print(e)