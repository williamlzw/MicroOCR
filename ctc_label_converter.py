from typing import List, Tuple

import torch
from torch import Tensor
import numpy as np


class CTCLabelConverter(object):
    def __init__(self, character: str):
        dict_character = list(character)
        self.num_of_classes = len(character)+1
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        self.character = ['_'] + dict_character

    def encode(self, text: List[str]) -> List[Tensor]:
        length = [len(s) for s in text]
        t = [self.dict[char] for s in text for char in s]
        return torch.tensor(t, dtype=torch.long), \
            torch.tensor(length, dtype=torch.long)

    def decode(self,
               preds: np.ndarray,
               raw: bool = False) -> List[Tuple[str, np.ndarray]]:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        result_list = []
        for word, prob in zip(preds_idx, preds_prob):
            if raw:
                result_list.append(
                    (''.join([self.character[int(i)] for i in word]), prob))
            else:
                result = []
                conf = []
                for i, index in enumerate(word):
                    if word[i] != 0 and (not (i > 0 and word[i - 1] == word[i])):
                        result.append(self.character[int(index)])
                        conf.append(prob[i])
                result_list.append((''.join(result), conf))
        return result_list


if __name__ == "__main__":

    pass
