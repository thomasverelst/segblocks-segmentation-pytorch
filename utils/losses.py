from torch import nn as nn
from torch.nn import functional as F
import torch
from typing import Optional


class SemsegCrossEntropy(nn.Module):
    def __init__(self, num_classes=19, ignore_id=19):
        super(SemsegCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        print("SemsegCrossEntropy loss with ignore_id {}".format(ignore_id))

    def _loss(self, y, t, reduction):
        y = F.interpolate(y, size=t.shape[1:3], mode="bilinear")
        return F.cross_entropy(y, target=t, ignore_index=self.ignore_id, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        loss = self._loss(logits, targets, "none")
        return loss.mean(), loss


# Adapted from: https://github.com/PingoLH/FCHarDNet/blob/master/ptsemseg/loss/loss.py
class BootstrappedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, k=4096, thresh=0.3, weight: Optional[torch.Tensor] = None, ignore_id: int = -100) -> None:
        super(BootstrappedCrossEntropyLoss, self).__init__(
            weight=weight, size_average=None, ignore_index=ignore_id, reduce=None, reduction="none"
        )
        self.k = k
        self.thresh = thresh
        print(f"BootstrappedCrossEntropyLoss with ignore_id {ignore_id}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = F.interpolate(logits, size=targets.shape[1:3], mode="bilinear")
        loss_pixelwise = super(BootstrappedCrossEntropyLoss, self).forward(logits, targets)

        loss_bootstrapped = 0.0
        loss_pixelwise_sorted, _ = torch.sort(loss_pixelwise.view(logits.shape[0], -1), descending=True)

        # Bootstrap from each image not entire batch
        for i in range(logits.shape[0]):
            sorted_loss = loss_pixelwise_sorted[i]
            if sorted_loss[self.k] > self.thresh:
                loss_image = sorted_loss[sorted_loss > self.thresh]
            else:
                loss_image = sorted_loss[: self.k]
            loss_bootstrapped += torch.mean(loss_image)
        loss_bootstrapped /= logits.shape[0]
        return loss_bootstrapped, loss_pixelwise
