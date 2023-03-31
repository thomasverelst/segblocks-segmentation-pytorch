import argparse
import segblocks.policy

CITYSCAPES_DATA_DIR = '/esat/visicsrodata/datasets/cityscapes'
CITYSCAPES_DATA_DIR = '/esat/tiger/tverelst/dataset/cityscapes'
CAMVID_DATA_DIR = '/esat/tiger/tverelst/dataset/camvid/CamVid/'

def parse_args():
    # initialize argparser
    parser = argparse.ArgumentParser(description='SegBlocks semantic segmentation')

    # mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'speed'], help='Run mode')
    parser.add_argument('--name', type=str, default='default', help='Experiment name for save dir')
    parser.add_argument('--resume', type=str, default='', help='')
    parser.add_argument('--resume-best', type=parse_bool, default=False, help='resume best model instead of latest')
    parser.add_argument('--viz', type=str, nargs='*', default=[], choices=['policy', 'pred'], help='vizualisations to make during validation')
    parser.add_argument('--num-viz-images', type=int, default=5, help='Number of visualisation images to save at validation (value is ceiled to batch size)')
    parser.add_argument('--profiler', type=parse_bool, default=False, help='run profiler, showing time cost of policy and segblocks modules (slows down execution), only when --mode==speed')

    # dataset and augmentations
    parser.add_argument('--dataset', type=str, default='cityscapes', help='dataset')
    parser.add_argument('--cityscapes-data-dir', type=str, default=CITYSCAPES_DATA_DIR, help='data directory')
    parser.add_argument('--camvid-data-dir', type=str, default=CAMVID_DATA_DIR, help='CamVid data directory')
    parser.add_argument('--num-workers', type=int, default=7, help='number of dataloader workers')
    parser.add_argument('--res', type=int, default=2048, help='image width')
    parser.add_argument('--crop-res', type=int, default=1024, help='crop size during training')
    parser.add_argument('--jitter', type=float, default=0.3, help='data augmentation color jitter')
    parser.add_argument('--rotate', type=float, default=0, help='data augmentation rotate degrees')
   
    # optimizer and schedule
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
    parser.add_argument('--lr-finetune-factor', type=float, default=0.25, help='learning rate factor for backbone finetune')
    parser.add_argument('--lr-end', type=float, default=1e-6, help='Learning rate at end of cosine annealing')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--loss', type=str, default='crossentropy', choices=['crossentropy', 'bootstrap_ce'], help='loss type')

    # model 
    parser.add_argument('--backbone', type=str, default='resnet18', help='SwiftNet backbone')
    parser.add_argument('--pretrain-backbone', type=parse_bool, default=True, help='pretrain backbone')
    parser.add_argument('--num-features', type=int, default=128, help='number of features in decoder')
    parser.add_argument('--fp16', type=parse_bool, default=True, help='use half precision for validation (training is always fp32)')

    # segblocks
    parser.add_argument('--segblocks-policy', type=str, default='disabled', choices=segblocks.policy.all_policies().keys(), help='Segblocks policy')
    parser.add_argument('--segblocks-block-size', type=int, default=128, help='input block size in pixels')
    parser.add_argument('--segblocks-percent-target', type=float, default=0.5, help='percentage of high-res blocks target, for REINFORCE policy')
    parser.add_argument('--segblocks-quantize-percentage', type=float, default=1/16, help='quantize the percentage of high-res blocks, by ceiling, for efficiency reasons')
    parser.add_argument('--segblocks-sparsity-weight', type=float, default=1, help='weight for sparsity reward')
    parser.add_argument('--segblocks-policy-lr-factor', type=float, default=1, help='learning rate factor relative to --lr for segblocks policy')
    parser.add_argument('--segblocks-policy-wd-factor', type=float, default=1, help='weight decay factor relative to --weight-decay for segblocks policy')

    args = parser.parse_args()
    return args


def parse_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



