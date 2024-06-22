
import torch.nn as nn
import torch.nn.functional as F
import torch

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, preds, target):
        return torch.sqrt(F.mse_loss(preds, target))
