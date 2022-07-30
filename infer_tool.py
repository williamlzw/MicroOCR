from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms

from train import build_model, load_model, build_conveter
from collatefn import resize_with_specific_height


class RecInfer:
    def __init__(self, cfg):
        self.device = torch.device('cpu')
        character = []
        with open(cfg.vocabulary_path, mode='r', encoding='utf-8') as fa:
            lines = fa.readlines()
            for line in lines:
                character.append(line.strip())
        self.converter = build_conveter(character)
        self.model = build_model(cfg.in_channels, cfg.nh, cfg.depth, self.converter.num_of_classes)
        load_model(cfg.model_path, self.model)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = transforms.ToTensor()

    def predict(self, img: np.ndarray) -> List[Tuple[str, List[np.ndarray]]]:
        img = resize_with_specific_height(32, img)
        tensor = self.transforms(img)
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out: Tensor = self.model(tensor)
        txt = self.converter.decode(out, False)
        return txt
