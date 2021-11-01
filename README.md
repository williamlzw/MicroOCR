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
Nh    | Depth | Nclass  | Use lstm|    Params size  |Model size(KB)| Total train(epoch) | Word acc
:----:|:-----:|:-------:|:-------:|:---------------:|:------------:|:------------------:|:--------:
2     |  2    |    62   |    no   |      1.224k     |      17      |        9000        |    0.996
4     |  2    |    62   |    no   |      2.402k     |      22      |        338         |    1.0
4     |  2    |    62   |    yes  |      23.202k    |      104     |        229         |    1.0
8     |  2    |    62   |    no   |      4.806k     |      31      |        362         |    1.0
8     |  2    |    62   |    yes  |      29.766k    |      129     |        97          |    1.0
16    |  2    |    62   |    no   |      9.806k     |      51      |        17          |    1.0
16    |  2    |    62   |    yes  |      43.086k    |      182     |        27          |    1.0
32    |  2    |    62   |    no   |      20.574k    |      95      |        13          |    1.0
32    |  2    |    62   |    yes  |      70.494k    |      289     |        31          |    1.0
64    |  2    |    62   |    no   |      45.182k    |      193     |        10          |    1.0
64    |  2    |    62   |    yes  |      128.382k   |      517     |        21          |    1.0
128   |  2    |    62   |    no   |      106.686k   |      438     |        10          |    1.0
128   |  2    |    62   |    yes  |      256.446k   |      1020    |        17          |    1.0
256   |  2    |    62   |    no   |      278.846k   |      1119    |        8           |    1.0
256   |  2    |    62   |    yes  |      561.726k   |      2217    |        12          |    1.0
512   |  2    |    62   |    no   |      819.774k   |      3250    |        9           |    1.0
512   |  2    |    62   |    yes  |      2.196542M  |      8613    |        9           |    1.0
1024  |  2    |    62   |    no   |      2.688062M  |      10584   |        10          |    1.0
1024  |  2    |    62   |    yes  |      8.68563M   |      33982   |        9           |    1.0


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