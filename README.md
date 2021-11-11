## MicroOCR
a micro OCR network.

The project has 5000 training task training sets and 500 test sets. 

The training of GTX1060 3GB graphics card takes 3 minutes.

## Task Parameter Reference:
Simple recognition task:nh=32 or 64,depth=2

Chinese recognition task:nh=128 or 256,depth=2

Complex background recognition task:nh=512 or more,depth=2

## Model parameters
Nh    | Depth | Nclass  |    Params size  |Model size(KB)| Total train(epoch) | Word acc
:----:|:-----:|:-------:|:---------------:|:------------:|:------------------:|:--------:
16    |  2    |    62   |      9.726k     |      50      |        99          |    0.782
32    |  2    |    62   |      20.414k    |      93      |        100         |    0.842
64    |  2    |    62   |      44.862k    |      190     |        49          |    0.810
128   |  2    |    62   |      106.046k   |      434     |        45          |    0.882
256   |  2    |    62   |      277.566k   |      1113    |        50          |    0.872
512   |  2    |    62   |      817.214k   |      3239    |        45          |    0.884
1024  |  2    |    62   |      2.682942M  |      10563   |        49          |    0.894


## Script Description

```shell
MicroOCR
├── README.md                                   # Descriptions about MicroNet
├── collatefn.py                                # collatefn
├── label_converter.py                          # label converter
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