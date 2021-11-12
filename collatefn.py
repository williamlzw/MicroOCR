from typing import List, Dict

import torch
from torch import Tensor
import numpy as np
import cv2
from torchvision import transforms


def resize_with_specific_height(input_h, img: np.ndarray) -> np.ndarray:
    """
    将图像resize到指定高度
    :param _img:    待resize的图像
    :return:    resize完成的图像
    """
    resize_ratio = input_h / img.shape[0]
    return cv2.resize(img, (0, 0), fx=resize_ratio,
                      fy=resize_ratio, interpolation=cv2.INTER_LINEAR)


def width_pad_img(img: np.ndarray,
                  target_width: int,
                  pad_value: int = 0) -> np.ndarray:
    """
    将图像进行高度不变，宽度的调整的pad
    :param _img:    待pad的图像
    :param _target_width:   目标宽度
    :param _pad_value:  pad的值
    :return:    pad完成后的图像
    """
    height, width, channels = img.shape
    if target_width > width:
        to_return_img = np.ones(
            [height, target_width, channels], dtype=img.dtype) * pad_value
        to_return_img[:height, :width, :] = img
    else:
        to_return_img = img
    return to_return_img


class RecCollateFn:
    """
    将图片缩放到固定高度,宽度取当前批次最长的RecCollateFn
    """

    def __init__(self, input_h: int=32):
        self.input_h = input_h
        self.transforms = transforms.ToTensor()

    def __call__(self,
                 batch: List[Dict[str, np.ndarray]]) \
            -> Dict[str, List[np.ndarray]]:
        resize_images: List[Tensor] = []
        # 统一缩放到指定高度
        all_same_height_images = [
            resize_with_specific_height(
                self.input_h, batch_index['images']) for batch_index in batch]
        # 取出最大宽度
        max_img_w = max({m_img.shape[1] for m_img in all_same_height_images})
        # 确保最大宽度是8的倍数
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        labels = []
        for i in range(len(batch)):
            labels.append(batch[i]['labels'])
            img = width_pad_img(
                all_same_height_images[i], max_img_w)
            img = self.transforms(img)
            resize_images.append(img)
        images = torch.stack(resize_images)
        return {'images': images, 'labels': labels}
