import os
import os.path as osp
import time

import torch
import torch.nn.functional as F
import tqdm
from easydict import EasyDict as edict

import argparser
import segblocks
from segblocks.utils import flopscounter, profiler
from utils import misc, viz
from utils.logger import logger
from utils.losses import BootstrappedCrossEntropyLoss
from utils.metrics import StreamSegMetrics

torch.backends.cudnn.benchmark = True


def train_epochs(C, R):
    # train and validate over epochs
    
    logger.out()
    for epoch in range(R.progress.epoch+1, C.epochs):
        R.progress.epoch = epoch
        start = time.perf_counter()
        # with torch.autograd.detect_anomaly():
        R = train(C, R)
        if C.val_every == 0 or epoch % C.val_every == 0 or epoch == C.epochs-1:
            R = val(C, R)
            
            score = R.metrics['Mean IoU']

            R.progress.score = score
            is_best = score > R.progress.score_best
            if is_best:
                R.progress.score_best = score
                R.progress.score_best_epoch = R.progress.epoch

            R.logger.log_float('metrics/epoch', R.progress.epoch)
            R.logger.log_float('metrics/score', score)
            R.logger.log_float('metrics/score_best', R.progress.score_best)
            R.logger.log_float('metrics/score_best_epoch', R.progress.score_best_epoch)
            
            R = save_checkpoint(C, R)
            if is_best:
                R = save_checkpoint(C, R, postfix='_best')
        
        if R.scheduler:
            R.scheduler.step()
        logger.log_float('epoch_time', time.perf_counter()-start)
        logger.out()
    return R

def train(C, R):
    # train single epoch
    logger.subheader(f'TRAIN - {C.name} - Epoch {R.progress.epoch}/{C.epochs}')
    logger.info(f"Learning rates: { [p['lr'] for p in R.optimizer.param_groups]  }")
    R = set_fp32(C, R) # training always in FP32
    R.model.train()
    assert R.dtype == torch.float32, 'training only supports FP32'

    for image, target, meta in tqdm.tqdm(R.dataloaders.train, mininterval=10):
        image = image.to(R.device, dtype=R.dtype, non_blocking=True)
        target = target.to(R.device, dtype=torch.long, non_blocking=True)

        out, meta = R.model(image, meta)

        loss, loss_per_pixel = R.loss_function(out, target)
        logger.log_float_interval('train/loss_task', loss.item())

        if R.model.policy is not None:
            loss_policy = R.model.policy.loss(loss_per_pixel, meta)
            loss += loss_policy

            logger.log_float_interval('train/segblocks_block_percent', lambda: float(meta['grid'].sum())/meta['grid'].numel())
            logger.log_float_interval('train/policy_reward_sparsity', meta['reward_sparsity'])
            logger.log_float_interval('train/policy_advantage_task_abs', meta['advantage_task'].abs().mean())
            logger.log_float_interval('train/policy_advantage_abs', meta['advantage'].abs().mean())
            logger.log_float_interval('train/loss_policy', loss_policy.item())
            if 'grid_probs' in meta:
                logger.log_float_interval('train/policy_prob_mean', lambda: meta['grid_probs'].mean())
                logger.log_float_interval('train/policy_prob_std', lambda: meta['grid_probs'].std())

        R.logger.log_float_interval('train/loss_total', loss.item())

        R.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        R.optimizer.step()
        logger.tick()
   
    logger.out()
    return R

def val(C, R):
    # validate single epoch
    logger.subheader(f'VAL - {C.name} - Epoch {R.progress.epoch}/{C.epochs}')
    R = set_precision(C, R) # switch to FP16 if argparser 'fp16' argument is True
    R.model.eval()

    with torch.no_grad():
        metrics = StreamSegMetrics(R.dataset.num_classes, R.dataset.class_names)
        R.model = flopscounter.add_flops_counting_methods(R.model)
        R.model.start_flops_count(only_conv_and_linear=True)

        num_images = 0
        num_blocks_highres = 0
        num_blocks_total = 0
        
        for image, target, meta in tqdm.tqdm(R.dataloaders.val, mininterval=10):
            # if num_images > 10:
            #     break
            image = image.to(R.device, dtype=R.dtype, non_blocking=True)
            
            out, meta = R.model(image, meta)
            
            out = F.interpolate(out, size=target.shape[1:3], mode='bilinear')
            pred = out.detach().max(dim=1)[1]
            metrics.update(target.cpu().numpy(), pred.cpu().numpy())

            if num_images < C.num_viz_images:
                if 'policy' in C.viz:
                    viz.viz_policy(C, R, image, out, meta)
                if 'pred' in C.viz:
                    viz.viz_pred(C, R, image, out, meta)

            if 'grid' in meta:
                num_blocks_highres += meta['grid'].sum()
                num_blocks_total += meta['grid'].numel()

            logger.tick()
            num_images += len(image)
        R.metrics = metrics.get_results()
        if num_blocks_total > 0:
            logger.info(f'Percentage of high-res blocks: {float(num_blocks_highres)/num_blocks_total}')
        logger.info(f'Metrics: {R.metrics}')
        R.model.stop_flops_count()
        logger.info(R.model.total_flops_cost_repr(submodule_depth=2))
        
        logger.out()
    return R

def speed(C, R):
    from segblocks.utils.profiler import timings

    # speedtest
    logger.subheader(f'SPEED - {C.name} - Epoch {R.progress.epoch}/{C.epochs}')
    R = set_precision(C, R) # switch to FP16 if argparser 'fp16' argument is True
    R.model.eval()
    # R.model = bn_fusion.fuse_bn_recursively(R.model)
    R.model = flopscounter.add_flops_counting_methods(R.model)
    NUM_BATCHES = len(R.dataloaders.val)//2

    with torch.no_grad():
        num_batches = 0
        num_timed_images = 0
        time_start = 0

        # preload images to prevent IO bottleneck
        logger.info('Preloading images...')
        batches = []
        for batch, _, _ in R.dataloaders.val:
            batches.append(batch)
            if len(batches) >= NUM_BATCHES:
                break

        logger.info('Warmup...')
        torch.backends.cudnn.benchmark = True
        for batch in batches:
            batch = batch.to(R.device, dtype=R.dtype, non_blocking=True)
            if num_batches == NUM_BATCHES//2: 
                logger.info('Test speed...')
                # torch.backends.cudnn.benchmark = False
                R.model.start_flops_count(only_conv_and_linear=True)
                torch.cuda.synchronize()
                time_start = time.perf_counter()

            out, _ = R.model(batch, {})

            num_batches += 1
            if num_batches >= NUM_BATCHES//2:
                num_timed_images += len(batch)
        torch.cuda.synchronize()
        time_end = time.perf_counter()
        time_per_image = (time_end-time_start)/num_timed_images
        logger.info(f'Time per image: {time_per_image} (={1/time_per_image} FPS)')
        logger.info(str(timings))
        logger.info(R.model.total_flops_cost_repr(submodule_depth=1))

    logger.out()
    return R





def build_dataset(C, R):
    if C['dataset'] == 'cityscapes':
        R.logger.subheader('Initializing Cityscapes dataset')
        import albumentations as A
        import albumentations.pytorch.transforms as APT
        import cv2
        from torch.utils.data import DataLoader

        from dataloaders.cityscapes import Cityscapes
        
        R.dataset = {}
        R.dataset.mean = (73.1584/255, 82.9090/255, 72.3924/255)
        R.dataset.std = ((44.9149/255, 46.1529/255, 45.3192/255))
        R.dataset.num_classes = Cityscapes.num_classes
        R.dataset.class_names = Cityscapes.train_id_to_class_names
        R.dataset.ignore_id = Cityscapes.ignore_id

        h, w = C.res//2, C.res
        train_transform = A.Compose([
            A.RandomScale((-0.5,1.0), p=1), # note: albumentations scale factor differs from torchvision
            A.PadIfNeeded(min_height=C.crop_res, min_width=C.crop_res, border_mode=cv2.BORDER_CONSTANT, value=R.dataset.mean, mask_value=R.dataset.ignore_id),
            A.RandomCrop(C.crop_res, C.crop_res),
            A.ColorJitter(brightness=C.jitter, contrast=C.jitter, saturation=C.jitter, hue=0.05),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=R.dataset.mean, std=R.dataset.std),
            APT.ToTensorV2(),
        ])
        val_transform = A.Compose([
            A.Resize(h,w),
            A.Normalize(mean=R.dataset.mean, std=R.dataset.std),
            APT.ToTensorV2(),
        ])
        ds_train = Cityscapes(root=C.data_dir, split='train', transform=train_transform, transform_target=True)
        ds_val = Cityscapes(root=C.data_dir, split='val', transform=val_transform, transform_target=False)

        R.dataloaders = {}
        R.dataloaders.train = DataLoader(ds_train, batch_size=C.batch_size, shuffle=True, num_workers=C.num_workers)
        R.dataloaders.val = DataLoader(ds_val, batch_size=C.batch_size, shuffle=False, num_workers=C.num_workers, pin_memory=C.mode == 'speed')
    return R

def set_fp16(C, R):
    logger.info('Model: Using half precision (FP16)')
    R.model = R.model.half()
    R.dtype = torch.float16
    return R

def set_fp32(C, R):
    R.logger.info('Model: Using full precision (FP32)')
    R.model = R.model.float()
    R.dtype = torch.float32
    return R

def set_precision(C, R):
    if C.fp16:
        return set_fp16(C, R)
    else:
        return set_fp32(C, R)

def build_model(C, R):
    R.logger.subheader('Initializing SwiftNet model')
    import models.seg.swiftnet.backbones as backbones
    from models.seg.swiftnet.swiftnet import SwiftNet

    if 'efficientnet' in C.backbone:
        num_features = 48 # EfficientNet(-Lite) uses fewer features in decoder, to balance computational cost
    else:
        num_features = 128
        
    backbone = backbones.__dict__[C.backbone](pretrained=C.pretrain_backbone)
    swiftnet = SwiftNet(backbone=backbone, num_classes=R.dataset.num_classes, num_features=num_features)
    
    policy_class = segblocks.policy.all_policies()[C.segblocks_policy]
    if policy_class is not None:
        policy = policy_class(
            block_size = C.segblocks_block_size, 
            percent_target = C.segblocks_percent_target, 
            quantize_percentage = C.segblocks_quantize_percentage,
            sparsity_weight = C.segblocks_sparsity_weight)
    else:
        policy = None
    
    R.model = segblocks.SegBlocksModel(net=swiftnet, policy=policy)
    R.model.to(R.device)

    total_params = misc.get_n_params(R.model.parameters())
    ft_params = misc.get_n_params(R.model.net.fine_tune_params())
    ran_params = misc.get_n_params(R.model.net.random_init_params())
    spp_params = misc.get_n_params(R.model.net.spp.parameters())
    logger.info(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')
    logger.info(f'SPP params: {spp_params:,}')

    return R

def build_optimizer(C, R):
    R.logger.subheader('Initializing Adam optimizer')
    import torch.optim as optim

    net = R.model.net

    fine_tune_factor = 4 if C.pretrain_backbone else 1
    optim_params = [
        {'params': net.random_init_params(), 'lr': C.lr, 'weight_decay': C.weight_decay},
        {'params': net.fine_tune_params(), 'lr': C.lr / fine_tune_factor,
        'weight_decay': C.weight_decay / fine_tune_factor}
    ]

    policy = R.model.policy
    if policy is not None:
        optim_params.append({'params': R.model.policy.parameters(), 'lr': C.lr * C.segblocks_policy_lr_factor, 'weight_decay': C.weight_decay  * C.segblocks_policy_wd_factor})

    R.optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
    logger.info(f'Optimizer: Adam with lr, fine_tune_factor, weight_decay, epochs: {C.lr}, {fine_tune_factor}, {C.weight_decay}, {C.epochs}')
    return R

def build_scheduler(C, R):
    R.logger.subheader('Initializing CosineAnnealingLR scheduler')
    import torch.optim as optim
    R.scheduler = optim.lr_scheduler.CosineAnnealingLR(R.optimizer, C.epochs,
        C.lrmin, last_epoch=R.progress.epoch)
    return R

def build_loss(C, R):
    R.logger.subheader('Initializing loss function')
    from utils.losses import SemsegCrossEntropy
    if C.loss == 'crossentropy':
        R.loss_function = SemsegCrossEntropy(R.dataset.num_classes, R.dataset.ignore_id)
    elif C.loss == 'bootstrap_ce':
        R.loss_function = BootstrappedCrossEntropyLoss(ignore_id=R.dataset.ignore_id)
    else:
        raise NotImplementedError(f'Loss {C.loss} not implemented')
    return R


def save_checkpoint(C, R, postfix=''):
    logger.info(f'> Saving model to {R.save_path}')
    save_dict = {
        'state_dict': R.model.state_dict(),
        'optimizer': R.optimizer.state_dict(),
        'scheduler': R.scheduler.state_dict() if R.scheduler is not None else None,
        'progress': R.progress,
        'logger_step': logger.step,
        'config': C,
    }
    torch.save(save_dict, os.path.join(R.save_path, f'checkpoint{postfix}.pth'))
    logger.info(f'> Saved checkpoint to {os.path.join(R.save_path, f"checkpoint{postfix}.pth")}')
    return R

def load_checkpoint(C, R):
    logger.subheader('Loading from checkpoint')
    fn = 'checkpoint_best.pth' if C.resume_best else 'checkpoint.pth'
    ckp = os.path.join(R.save_path, fn)
    if os.path.exists(ckp):
        logger.info(f'> Checkpoint found, loading from {ckp}')
        save_dict = torch.load(ckp)
        R.model.load_state_dict(save_dict['state_dict'], strict=True)
        R.optimizer.load_state_dict(save_dict['optimizer'])
        R.scheduler.load_state_dict(save_dict['scheduler']) if save_dict['scheduler'] else None
        logger.step = save_dict['logger_step']
        R.progress = save_dict['progress']
        logger.info(f'> Loaded checkpoint with epoch {R.progress.epoch} and score {R.progress.score}')
    else:
        logger.info(f'> No checkpoint found in {ckp}')
    return R


def init():
    torch.manual_seed(0)
    
    args = argparser.parse_args()
    C = edict(vars(args))

    assert torch.cuda.is_available(), 'SegBlocks requires CUDA!'

    SAVE_DIR = 'exp'
    R = edict({
        'logger': logger,
        'device': torch.device('cuda'),
        'progress': {
            'epoch': -1,
            'score': 0,
            'score_best': 0,
            'score_best_epoch': -1,
        },
        'metrics': {},
        'save_path': osp.join(SAVE_DIR, C.name)
    })
    os.makedirs(R.save_path, exist_ok=True)
    
    logger.init(save_path=R.save_path, interval=100)
    logger.header('SegBlocks - semantic segmentation')
    logger.info(f'Arguments: C = {C}')

    profiler.timings.enabled = C.profiler

    return C, R

def main():
    C, R = init()
    R = build_dataset(C, R)
    R = build_model(C, R)
    R = build_optimizer(C, R)
    R = build_scheduler(C, R)
    R = build_loss(C, R)
    R = load_checkpoint(C, R)

    if C.mode == 'train':
        R = train_epochs(C, R)
    elif C.mode == 'val':
        R = val(C, R)
    elif C.mode == 'speed':
        R = speed(C, R)
    else:
        raise NotImplementedError(f'Mode {C.mode} not implemented')

    logger.header(f'Done -- {C.name}')
    return R.progress


if __name__ == '__main__':
    main()

