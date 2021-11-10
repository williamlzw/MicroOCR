from typing import Dict

from torch import Tensor

from label_converter import CTCLabelConverter


class RecMetric(object):
    def __init__(self, converter: CTCLabelConverter):
        """
        文本识别相关指标计算类
        :param converter: 用于label转换的转换器
        """
        self.converter = converter

    def __call__(self,
                 predictions: Tensor,
                 labels: Tensor) -> Dict[str, Tensor]:
        word_correct, char_correct = 0, 0
        predictions = predictions.softmax(dim=2).detach().cpu().numpy()
        preds_str = self.converter.decode(predictions)
        raws_str = self.converter.decode(predictions, True)
        show_str = []
        for (raw, raw_conf), (pred, pred_conf), target in zip(
                raws_str, preds_str, labels):
            show_str.append(f'{raw} -> {pred} -> {target}')
            if pred == target:
                word_correct += 1
            for idxa, idxb in zip(pred, target):
                if idxa == idxb:
                    char_correct += 1
        return dict(word_correct=word_correct,
                    char_correct=char_correct,
                    show_str=show_str)
