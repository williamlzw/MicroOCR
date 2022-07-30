from typing import Dict, List, Union, Any
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from img_aug import DataProcess, cv2pil, pil2cv


class TextLineDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 label_file_list: Union[str, List[str]],
                 character: str,
                 in_channels: int,
                 augmentation: bool = False):
        self.in_channels = in_channels
        self.aug = DataProcess()
        self.augmentation = augmentation
        self._get_image_info_list(label_file_list)
        self.str2idx = dict(zip(character, range(len(character))))
        self.str2idx[' '] = len(self.str2idx)
        self.data_dir = data_dir

    def _get_image_info_list(self,
                             file_list: Union[str, List[str]]) -> List[str]:
        if isinstance(file_list, str):
            file_list = [file_list]
        self.data_lines = []
        for file in file_list:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.data_lines.extend(lines)

    def __len__(self) -> int:
        return len(self.data_lines)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_name, label = self.data_lines[index].strip().split('\t')
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imdecode(np.fromfile(
            img_path, dtype=np.uint8), cv2.IMREAD_COLOR if self.in_channels == 3 else cv2.IMREAD_GRAYSCALE)
        if self.in_channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentation:
            image = pil2cv(self.aug.aug_img(cv2pil(image)))
        return dict(images=image, labels=label)
    
