from typing import Dict, List, Any

from torch import Tensor


class RecMetric(object):
    def __init__(self, converter):
        """
        文本识别相关指标计算类
        :param converter: 用于label转换的转换器
        """
        self.converter = converter

    def __call__(self,
                 predictions: Tensor,
                 labels: List[str]) -> Dict[str, Any]:
        preds_list = self.converter.decode(predictions)
        raws_list = self.converter.decode(predictions, True)
        word_correct = 0
        char_correct = 0
        show_str = []

        for (raw_str, raw_score), (pred_str, pred_score), target_str in zip(
                raws_list, preds_list, labels):
            show_str.append(f'{raw_str} -> {pred_str} -> {target_str}')
            if pred_str == target_str:
                word_correct += 1
            for idxa, idxb in zip(pred_str, target_str):
                if idxa == idxb:
                    char_correct += 1
        return dict(word_correct=word_correct,
                    char_correct=char_correct,
                    show_str=show_str)