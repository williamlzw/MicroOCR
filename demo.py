import argparse
import time

import cv2

from infer_tool import RecInfer


def main():
    parser = argparse.ArgumentParser(description='MicroOCR')
    parser.add_argument('--model_path',
                        default='save_model/micro_nh32_depth2_epoch100_word_acc0.842000_char_acc0.935000.pth',
                        help='model path')
    parser.add_argument('--nh', default=32, type=int, help='nh')
    parser.add_argument('--depth', default=2, type=int, help='depth')
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
