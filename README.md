## MicroOCR
a micro OCR network.

The project has 5000 training task training sets and 500 test sets. 

Only 35 epochs are used to verify the accuracy of the set to reach 1.0. 

The training of GTX1060 3GB graphics card takes 3 minutes.

## Task Parameter Reference:
Simple recognition task:nh=16,depth=2

Chinese recognition task:nh=64,depth=2

Complex background recognition task:nh=128 or more,depth=2 or more

## Model parameters
Nh    | Depth | nclass  | Params size(MB) |Model size(KB)| Total train(epoch) | Word acc
:----:|:-----:|:-------:|:---------------:|:------------:|:------------------:|:--------:
2     |  2    |    62   |      0.005      |      17      |        9000        |    0.996
4     |  2    |    62   |      0.01       |      22      |        338         |    1.0
8     |  2    |    62   |      0.02       |      31      |        362         |    1.0
16    |  2    |    62   |      0.04       |      53      |        35          |    1.0
32    |  2    |    62   |      0.08       |      95      |        39          |    1.0
64    |  2    |    62   |      0.17       |      193     |        79          |    1.0
128   |  2    |    62   |      0.41       |      438     |        79          |    1.0
256   |  2    |    62   |      1.06       |      1119    |        14          |    1.0
512   |  2    |    62   |      3.13       |      3250    |        8           |    1.0
1024  |  2    |    62   |      10.25      |      10584   |        7           |    1.0


## Script Description

```shell
MicroOCR
├── README.md                                   # Descriptions about MicroNet
├── collatefn.py                                # collatefn
├── ctc_label_converter.py                      # label converter
├── dataset.py                                  # Data preprocessing for training and evaluation
├── demo.py                                     # demo
├── gen_image.py                                # generate image for train and eval
├── infer_tool.py                               # inference tool
├── keys.py                                     # character
├── loss.py                                     # Ctcloss definition
├── metric.py                                   # accuracy metric for MicroNet
├── model.py                                    # MicroNet
├── train.py                                    # train the model
```

## Generate data for train and eval
```shell
python gen_image.py
```

## Training
```shell
python train.py
```

## Inference
```shell
python demo.py
```