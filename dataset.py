from typing import Dict, List, Union
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class RecTextLineDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 label_file_list: Union[str, List[str]],
                 character: str):
        text_lines = self._get_image_info_list(label_file_list)
        self.str2idx = dict(zip(character, range(len(character))))
        self.data_lines = []
        for line_index in tqdm(text_lines):
            img_name, label = line_index.strip('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            if True in [c not in self.str2idx for c in label]:
                continue
            if os.path.exists(img_path) is False:
                continue
            self.data_lines.append((img_path, label))

    def _get_image_info_list(self,
                             file_list: Union[str, List[str]]) -> List[str]:
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for file in file_list:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                data_lines.extend(lines)
        return data_lines

    def _find_max_length(self) -> int:
        return max({len(index[1]) for index in self.data_lines})

    def __len__(self) -> int:
        return len(self.data_lines)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        img_path, label = self.data_lines[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return dict(img_path=img_path, images=image, labels=label)
