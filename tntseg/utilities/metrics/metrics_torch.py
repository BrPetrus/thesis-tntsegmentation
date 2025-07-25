import torch
from torch import nn
from torch.types import Tensor
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred: Tensor, label: Tensor, eta: float = 1e-6) -> Tensor:
        TP = torch.sum(pred * label)
        FP = torch.sum((1-label)*pred)
        FN = torch.sum(label*(1-pred))

        return TP / (TP + self.alpha*FP + self.beta*FN + eta)

class FocalTverskyLoss(TverskyLoss):
    def __init__(self, alpha: float, beta: float, gamma: float, *args, **kwargs):
        super().__init__(alpha, beta, *args, **kwargs)
        self.gamma = gamma

    def forward(self, pred, label, eta = 1e-6) -> Tensor:
        return torch.pow(1 - super().forward(pred=pred, label=label, eta=eta), self.gamma)
        
class DiceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, pred, label) -> Tensor:
        pred = F.sigmoid(pred)
        numerator = 2 * torch.sum(pred, label)
        # TODO: why sometimes squared?
        denom = torch.sum(torch.pow(pred, 2)) + torch.sum(torch.pow(label, 2))
        return 1 - numerator / (denom)