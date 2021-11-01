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


class AttentionLabelConverter(object):
    def __init__(self, character: str, start_end_token: str = '<BOS/EOS>', unknown_token: str = '<UKN>',
                 padding_token: str = '<PAD>', max_seq_len: int = 25):
        dict_character = list(character)
        self.max_seq_len = max_seq_len

        self.char2idx = dict()
        self.idx2char = list()
        self.idx2char.append(start_end_token)
        self.start_idx = 0
        self.end_idx = 1
        self.idx2char.append(padding_token)
        self.padding_idx = 2
        self.idx2char.append(unknown_token)
        self.unknown_idx = 3

        self.num_of_classes = len(character)
        for i, char in enumerate(dict_character):
            self.char2idx[char] = i + 3
            self.idx2char.append(i)

    def str2idx(self, strings):
        """Convert strings to indexes.
        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        """
        indexes = []
        for string in strings:
            if self.lower:
                string = string.lower()
            index = []
            for char in string:
                char_idx = self.char2idx.get(char, self.unknown_idx)
                if char_idx is None:
                    raise Exception(f'Chararcter: {char} not in dict,'
                                    f' please check gt_label and use'
                                    f' custom dict file,'
                                    f' or set "with_unknown=True"')
                index.append(char_idx)
            indexes.append(index)
        return indexes

    def idx2str(self, indexes):
        """Convert indexes to text strings.
        Args:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        Returns:
            strings (list[str]): ['hello', 'world'].
        """

        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            strings.append(''.join(string))

        return strings

    def encode(self, strings: List[str]) -> Tensor:
        """
        Convert text-string into tensor.
        Args:
            strings (list[str]): ['hello', 'world']
        Returns:
            targets (Tensor(bsz * max_seq_len))
        """
        tensors, padded_targets = [], []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.LongTensor(index)
            tensors.append(tensor)
            # target tensor for loss
            src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
            src_target[-1] = self.end_idx
            src_target[0] = self.start_idx
            src_target[1:-1] = tensor
            padded_target = (torch.ones(self.max_seq_len) *
                             self.padding_idx).long()
            char_num = src_target.size(0)
            if char_num > self.max_seq_len:
                padded_target = src_target[:self.max_seq_len]
            else:
                padded_target[:char_num] = src_target
            padded_targets.append(padded_target)
        targets = torch.stack(padded_targets, 0).long()

        return targets

    def decode(self, outputs: Tensor) -> List[Tuple[str, np.ndarray]]:
        """
        Convert output tensor to text-index
        Args:
            outputs (tensor): model outputs with size: N * T * C
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]]
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                                         [0.9,0.9,0.98,0.97,0.96]]
        """
        batch_size = outputs.size(0)
        ignore_indexes = [self.padding_idx]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = outputs[idx, :, :]
            max_value, max_idx = torch.max(seq, -1)
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == self.end_idx:
                    break
                str_index.append(char_index)
                str_score.append(char_score)

            indexes.append(str_index)
            scores.append(str_score)

        return indexes, scores


if __name__ == "__main__":

    pass
