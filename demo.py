import argparse
import time

import cv2

from infer_tool import RecInfer


def main():
    parser = argparse.ArgumentParser(description='MicroOCR')
    parser.add_argument('--model_path',
                        default='save_model/micro_nh16_depth2_epoch17_word_acc1.000000_char_acc1.000000.pth',
                        help='model path')
    parser.add_argument('--nh', default=16, type=int, help='nh')
    parser.add_argument('--depth', default=2, type=int, help='depth')
    parser.add_argument('--use_lstm', default=False, help='use lstm', type=bool)
    cfg = parser.parse_args()
    infer = RecInfer(cfg)
    img = cv2.imread('00000.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    out = infer.predict(img)
    t1 = time.time()
    print(out, '{:.2f}ms'.format((t1-t0)*1000))


if __name__ == "__main__":
    main()
