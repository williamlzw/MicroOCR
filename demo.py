import argparse
import time

import cv2

from infer_tool import RecInfer


def main():
    parser = argparse.ArgumentParser(description='MicroOCR')
    parser.add_argument('--vocabulary_path', default='english.txt',
                        help='vocabulary path')
    parser.add_argument('--model_path',
                        default='./save_model/micromlp_nh64_depth2_best_rec.pth',
                        help='model path')
    parser.add_argument('--nh', default=64, type=int, help='nh')
    parser.add_argument('--depth', default=2, type=int, help='depth')
    parser.add_argument(
        '--in_channels', default=3, help='in channels', type=int)
    cfg = parser.parse_args()
    infer = RecInfer(cfg)
    img = cv2.imread('D:/dataset/gen/test/00000.jpg',
                     cv2.IMREAD_COLOR if cfg.in_channels == 3 else cv2.IMREAD_GRAYSCALE)
    if cfg.in_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    out = infer.predict(img)
    t1 = time.time()
    print(out, '{:.2f}ms'.format((t1-t0)*1000))


if __name__ == "__main__":
    main()
