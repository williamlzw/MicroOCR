## MicroOCR
a micro OCR network with 0.07mb params.

        Layer (type)               Output Shape         Param #

            Conv2d-1            [-1, 64, 8, 32]           3,136
       BatchNorm2d-2            [-1, 64, 8, 32]             128
              GELU-3            [-1, 64, 8, 32]               0
         ConvBNACT-4            [-1, 64, 8, 32]               0
            Conv2d-5            [-1, 64, 8, 32]             640
       BatchNorm2d-6            [-1, 64, 8, 32]             128
              GELU-7            [-1, 64, 8, 32]               0
         ConvBNACT-8            [-1, 64, 8, 32]               0
            Conv2d-9            [-1, 64, 8, 32]           4,160
      BatchNorm2d-10            [-1, 64, 8, 32]             128
             GELU-11            [-1, 64, 8, 32]               0
        ConvBNACT-12            [-1, 64, 8, 32]               0
       MicroBlock-13            [-1, 64, 8, 32]               0
           Conv2d-14            [-1, 64, 8, 32]             640
      BatchNorm2d-15            [-1, 64, 8, 32]             128
             GELU-16            [-1, 64, 8, 32]               0
        ConvBNACT-17            [-1, 64, 8, 32]               0
           Conv2d-18            [-1, 64, 8, 32]           4,160
      BatchNorm2d-19            [-1, 64, 8, 32]             128
             GELU-20            [-1, 64, 8, 32]               0
        ConvBNACT-21            [-1, 64, 8, 32]               0
       MicroBlock-22            [-1, 64, 8, 32]               0
          Flatten-23              [-1, 64, 256]               0
    AdaptiveAvgPool1d-24           [-1, 64, 30]               0
           Linear-25               [-1, 30, 60]           3,900

    Total params: 17,276
    Trainable params: 17,276
    Non-trainable params: 0
    Input size (MB): 0.05
    Forward/backward pass size (MB): 2.90
    Params size (MB): 0.07
    Estimated Total Size (MB): 3.02

## Script Description

```shell
MicroOCR
├── README.md                                   # Descriptions about MicroNet
├── collatefn.py                                # collatefn
├── ctc_label_converter.py                      # accuracy metric for MicroNet
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