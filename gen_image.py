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
        "font/simsun.ttc", textSize, encoding="utf-8")

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

    img = np.zeros((30, 120, 3), np.uint8)
    img[:] = [50, 100, 150]
    left = random.randint(10, 12)
    top = random.randint(1, 3)
    img = cv2ImgAddText(img, code_str, left, top)
    return code_str, img


def gen_test():
    os.makedirs('test1')
    codes = []
    for i in range(500):
        code, img = gen_img(4)
        codes.append(code)
        cv2.imwrite('test1/'+str(i).zfill(5)+'.jpg', img)

    i = 0
    with open('test1.txt', mode='w+') as fs:
        for code in codes:
            fs.write('test1/'+str(i).zfill(5)+'.jpg'+'\t'+code+"\n")
            i += 1
        fs.close()


def gen_train():
    os.makedirs('train1')
    codes = []
    for i in range(5000):
        code, img = gen_img(4)
        codes.append(code)
        cv2.imwrite('train1/'+str(i).zfill(5)+'.jpg', img)

    i = 0
    with open('train1.txt', mode='w+') as fs:
        for code in codes:
            fs.write('train1/'+str(i).zfill(5)+'.jpg'+'\t'+code+"\n")
            i += 1
        fs.close()


if __name__ == '__main__':
    gen_train()
    gen_test()
