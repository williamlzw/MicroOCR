import os
import random

from PIL import Image
from captcha.image import ImageCaptcha

alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
gen = ImageCaptcha(160, 60)


def gen_img(max_length):
    """
    max_length:验证码位数
    """
    content = [random.randrange(0, len(alphabet)) for _ in range(max_length)]
    s = ''.join([alphabet[i] for i in content])
    d = gen.generate(s)
    img = Image.open(d)
    return s, img


def gen_dataset(img_root, count):
    if not os.path.exists(img_root):
        os.makedirs(img_root)
    label_name = img_root + '.txt'
    with open(label_name, mode='w+', encoding="utf-8") as fs:
        for i in range(count):
            length = random.randint(3, 8)
            length = 4
            code, img = gen_img(length)
            img_name = str(i).zfill(5)+'.jpg'
            img_path = img_root + '/' + img_name
            fs.write(img_name+'\t'+code+"\n")
            img.save(img_path)


if __name__ == '__main__':
    gen_dataset('D:/dataset/gen/train', 10000)
    gen_dataset('D:/dataset/gen/test', 500)