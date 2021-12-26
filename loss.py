import torch
import torch.nn as nn
import torch.nn.functional as F


class L1RelativeLoss(nn.Module):
    def __init__(self, sampling_mean=1, weight=1e4, size_average=None, reduce=None, reduction='mean'):
        super().__init__()
        self.sampling_mean = sampling_mean
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, output, target):
        l1 = F.l1_loss(output, target, reduction=self.reduction)
        relative_loss = (output - target) / (target + self.sampling_mean)
        relative_loss = abs(torch.mean(relative_loss).item())
        loss = l1 + self.weight * relative_loss
        return loss
