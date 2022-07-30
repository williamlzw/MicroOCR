from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import CTCLoss as TorchCTCLoss


class CTCLoss(nn.Module):
    def __init__(self, blank_idx: int, reduction: str = 'mean'):
        super().__init__()
        self.loss_func = TorchCTCLoss(
            blank=blank_idx, reduction=reduction, zero_infinity=True)

    def focal_ctc_loss(self, ctc_loss,alpha=0.25,gamma=0.5): # 0.99,1
        prob = torch.exp(-ctc_loss)
        focal_loss = alpha*(1-prob).pow(gamma)*ctc_loss
        return focal_loss.mean()

    def forward(self,
                pred: Tensor,
                label: Tensor,
                label_length: Tensor) -> Dict[str, Tensor]:
        pred = pred.permute(1, 0, 2)
        batch_size = pred.size(1)
        pred = pred.log_softmax(2)
        preds_lengths = torch.LongTensor([pred.size(0)] * batch_size)
        loss = self.loss_func(pred, label, preds_lengths, label_length)
        #loss = self.focal_ctc_loss(loss)
        return dict(loss=loss)