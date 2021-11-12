import os
import random

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 0), textSize=25):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    fontStyle = ImageFont.truetype(
        "simsunb.ttf", textSize, encoding="utf-8")

    for i, text_index in enumerate(text):
        draw.text((left+i*30, top), text_index, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def gen_img(max_length):
    """
    max_length:验证码位数
    """
    code_str = ''
    for i in range(max_length):
        code_str += alphabet[random.randint(0, len(alphabet)-1)]
    left = random.randint(10, 12)
    top = random.randint(1, 3)
    img = np.zeros((30, max_length*30+left, 3), np.uint8)
    img[:] = [random.randint(50, 250), random.randint(
        50, 250), random.randint(50, 250)]
    text_color = (random.randint(150, 250), random.randint(
        150, 250), random.randint(150, 250))
    img = cv2ImgAddText(img, code_str, left, top, text_color)
    return code_str, img


def gen_dataset(img_root, count):
    if not os.path.exists(img_root):
        os.makedirs(img_root)
    label_name = img_root + '.txt'
    with open(label_name, mode='w+', encoding="utf-8") as fs:
        for i in range(count):
            length = random.randint(3,6)
            code, img = gen_img(length)
            img_name = img_root + '/' + str(i).zfill(5)+'.jpg'
            fs.write(img_name+'\t'+code+"\n")
            cv2.imwrite(img_name, img)


if __name__ == '__main__':
    gen_dataset('train', 5000)
    gen_dataset('test', 500)
