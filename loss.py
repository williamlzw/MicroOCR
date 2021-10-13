from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import CTCLoss as TorchCTCLoss


class CTCLoss(nn.Module):
    def __init__(self, blank_idx, reduction='sum'):
        super().__init__()
        self.loss_func = TorchCTCLoss(
            blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(self, pred: Tensor, label: Tensor, label_length: Tensor) -> Dict[str, Tensor]:
        pred = pred.permute(1, 0, 2)
        batch_size = pred.size(1)
        pred = pred.log_softmax(2)
        preds_lengths = torch.tensor(
            [pred.size(0)] * batch_size, dtype=torch.long)
        loss = self.loss_func(pred, label, preds_lengths, label_length)
        return {'loss': loss}
