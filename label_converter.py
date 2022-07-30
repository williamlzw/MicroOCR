from typing import List, Tuple

import torch
from torch import Tensor
import numpy as np


class CTCLabelConverter(object):
    def __init__(self, character: str):
        list_character = list(character)
        self.num_of_classes = len(character)+2
        self.idx2char = []
        self.idx2char.append('_')
        for line in list_character:
            line = line.strip()
            if line != '':
                self.idx2char.append(line)
        self.idx2char.append(' ')
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def str2idx(self, strings):
        """Convert strings to indexes.
        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            indexes (list[list[int]]): [1,2,3,3,4,5,4,6,3,7].
        """
        indexes = []
        for string in strings:
            for char in string:
                char_idx = self.char2idx.get(char)
                if char_idx is None:
                    raise Exception(f'Chararcter: {char} not in dict')
                indexes.append(char_idx)
        return indexes

    def encode(self, strings: List[str]) -> Tuple[Tensor, Tensor]:
        targets_lengths = [len(s) for s in strings]
        targets = self.str2idx(strings)
        return torch.LongTensor(targets), torch.LongTensor(targets_lengths)

    def decode(self,
               preds: Tensor,
               raw: bool = False) -> List[Tuple[str, np.ndarray]]:
        preds = preds.softmax(dim=2)
        preds_score, preds_idx = preds.max(dim=2)
        preds_idx = preds_idx.detach().cpu().numpy().tolist()
        preds_score = preds_score.detach().cpu().numpy().tolist()
        result_list = []
        for word, score in zip(preds_idx, preds_score):
            if raw:
                result_list.append(
                    (''.join([self.idx2char[char_idx] for char_idx in word]), score))
            else:
                char_list = []
                score_list = []
                for i, char_idx in enumerate(word):
                    if char_idx != 0 and (not (i > 0 and word[i - 1] == char_idx)):
                        char_list.append(self.idx2char[char_idx])
                        score_list.append(score[i])
                result_list.append((''.join(char_list), score_list))
        return result_list