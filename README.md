# SegBlocks - Block-wise adaptive resolution networks

PyTorch implementation of SegBlocks: Block-Based Adaptive Resolution Networks for Fast Segmentation (TPAMI)

SegBlocks accelerates networks by adjusting the processing resolution per image region. The block-based adaptive execution is achieved using CUDA modules to split and combine the image in blocks, along with a BlockPad module to preserve feature continuity between blocks. A policy trained with reinforcement learning decides which blocks to execute in high resolution. 

- [x] SegBlocks framework
- [x] Cityscapes dataset
- [x] SwiftNet with ResNet-18 backbone training scripts and backbone
- [ ] SwiftNet with EfficientNet-Lite
- [ ] CamVid and Mapillary datasets
- [ ] Better code documentation

## Installation

### Requirements

Create a new Anaconda env and activate it
    
    conda create -n segblocks python=3.9 -y
    conda activate segblocks

Install PyTorch 1.11.0 (note: newer versions might also work, but not tested)

    conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

Install CuPy (see installation instructions of CuPy in case of installation problems) (note: tested with CuPy 10.4 but newer might also work)
    
    pip install cupy-cuda113==10.4

Install other requirements

    pip install -r requirements.txt

Install segblocks package

    cd ./segblocks
    python setup.py develop
    cd ..

If needed, install other missing packages with `pip`.

### Dataset

Get the Cityscapes dataset from https://www.cityscapes-dataset.com/
Change the `CITYSCAPES_DATA_DIR` variable in `argparser.py` to point to the Cityscapes data root.

## Validation and FPS benchmark

### Checkpoints

Download checkpoints from [OneDrive](https://1drv.ms/u/s!ApImBF1PK3gnjpZ3KzdYj-PeozKnPQ?e=AGFCn9)
and unpack the zip in the code root so that the folder structure becomes

    ./exp/cityscapes/swiftnet_rn18/reinforce40/checkpoint_best.pth



| Model         | Block policy         | mIoU [val] | GMACs | speed [GTX 1080 Ti 11 GB] |
|---------------|----------------------|------------|-------|---------------------------|
| SwiftNet-RN18 | baseline (no blocks) | 76.3       | 103.6 | 39.1 FPS                  |
| SwiftNet-RN18 | RL policy (40%)      | 76.2       | 55.3  | 58.1 FPS                  |

### SwiftNet-RN18 

#### REINFORCE 40% policy

Validation (accuracy and FLOPS/GMACS):

    configs/cityscapes/swiftnet_rn18/reinforce40_val.sh

FPS 

    configs/cityscapes/swiftnet_rn18/reinforce40_speed.sh

#### baseline

Validation (accuracy and FLOPS/GMACS):

    configs/cityscapes/swiftnet_rn18/baseline_val.sh

FPS 

    configs/cityscapes/swiftnet_rn18/baseline_speed.sh

## Training, validation and FPS benchmark

A few configs are supplied:

Train a model with REINFORCE policy at 40% blocks: 

    configs/cityscapes/swiftnet_rn18/reinforce40_train.sh

Train a baseline model: 

    configs/cityscapes/swiftnet_rn18/baseline_train.sh

More configs (EfficientNet backbone) coming later.

## SegBlocks block-based networks code
The code to run networks in blocks can be found in `./segblocks/` with some examples on how to use this (e.g. on ResNet) in `./segblocks/examples/`


