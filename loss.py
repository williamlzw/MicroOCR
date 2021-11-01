from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import CTCLoss as TorchCTCLoss


class CTCLoss(nn.Module):
    def __init__(self, blank_idx: int, reduction: str = 'sum'):
        super().__init__()
        self.loss_func = TorchCTCLoss(
            blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(self,
                pred: Tensor,
                label: Tensor,
                label_length: Tensor) -> Dict[str, Tensor]:
        pred = pred.permute(1, 0, 2)
        batch_size = pred.size(1)
        pred = pred.log_softmax(2)
        preds_lengths = torch.tensor(
            [pred.size(0)] * batch_size, dtype=torch.long)
        loss = self.loss_func(pred, label, preds_lengths, label_length)
        return dict(loss=loss)


class SARLoss(nn.Module):
    def __init__(self, ignore_index=0, reduction='mean', **kwargs):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)

    def format(self, outputs, targets):
        # targets[0, :], [start_idx, idx1, idx2, ..., end_idx, pad_idx...]
        # outputs[0, :, 0], [idx1, idx2, ..., end_idx, ...]

        # ignore first index of target in loss calculation
        targets = targets[:, 1:].contiguous()
        # ignore last index of outputs to be in same seq_len with targets
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()
        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        losses = dict(loss_ce=loss_ce)
        return losses