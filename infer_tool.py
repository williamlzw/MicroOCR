from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms

from train import build_rec_model
from keys import character
from ctc_label_converter import CTCLabelConverter
from collatefn import resize_with_specific_height
from train import load_rec_model


class RecInfer:
    def __init__(self, cfg):
        self.device = torch.device('cpu')
        self.model = build_rec_model(cfg, len(character)+1)
        load_rec_model(cfg.model_path, self.model)
        self.model.to(self.device)
        self.model.eval()
        self.converter = CTCLabelConverter(character)
        self.transforms = transforms.ToTensor()

    def predict(self, img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        img = resize_with_specific_height(30, img)
        tensor = self.transforms(img)
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out: Tensor = self.model(tensor)

        txt = self.converter.decode(out.softmax(
            dim=2).detach().cpu().numpy(), False)
        return txt
