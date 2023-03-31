import numpy as np
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import os
import cv2

MAX_WIDTH = 1024


def resize(im):
    assert im.ndim == 3
    assert im.shape[2] == 3
    h, w = im.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        im = cv2.resize(im, fx=scale, fy=scale, dsize=None)
    return im


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def viz_policy(C: dict, R: dict, inputs: torch.Tensor, outputs: torch.Tensor, meta: dict):
    grid = meta["grid"]
    for b in range(inputs.shape[0]):  # loop over batch dimension
        filename = os.path.splitext(meta["file"][b])[0] + ".jpg"
        path = os.path.join(R.save_path, "viz", "policy", "grid_" + filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        image = inputs[b].cpu()
        denorm = Denormalize(mean=R.dataset.mean, std=R.dataset.std)
        image = (denorm(image) * 255).numpy().transpose(1, 2, 0).astype(np.uint8)  # H, W, C
        image = resize(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        g = (grid[b].byte() * 255).cpu().numpy()
        g = cv2.resize(g, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        g = cv2.applyColorMap(g, cv2.COLORMAP_VIRIDIS)

        alpha = 0.5
        out = cv2.addWeighted(g, alpha, image, 1 - alpha, 0)
        cv2.imwrite(path, out)
        R.logger.info("Saved viz of policy to {}".format(path))


def viz_pred(C: dict, R: dict, inputs: torch.Tensor, outputs: torch.Tensor, meta: dict):
    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
    for b in range(inputs.shape[0]):  # loop over batch dimension
        filename = os.path.splitext(meta["file"][b])[0] + ".jpg"
        path = os.path.join(R.save_path, "viz", "pred", "pred_" + filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        image = inputs[b].cpu()
        denorm = Denormalize(mean=R.dataset.mean, std=R.dataset.std)
        image = (denorm(image) * 255).numpy().transpose(1, 2, 0).astype(np.uint8)  # H, W, C

        pred = R.dataloaders.train.dataset.decode_target(preds[b]).astype(np.uint8)

        alpha = 0.7
        out = cv2.addWeighted(pred, alpha, image, 1 - alpha, 0)
        cv2.imwrite(path, resize(cv2.cvtColor(out, cv2.COLOR_RGB2BGR)))
        R.logger.info("Saved viz of pred to {}".format(path))
