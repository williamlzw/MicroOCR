## MicroOCR
a micro OCR network with 0.07mb params.

The project has 5000 training task training sets and 500 test sets. 

Only 35 epochs are used to verify the accuracy of the set to reach 1.0. 

The training of GTX1060 3GB graphics card takes 3 minutes.

## Task Parameter Reference:
Simple recognition task:nh=16,depth=2

Chinese recognition task:nh=64,depth=2

Complex background recognition task:nh=128 or more,depth=2 or more

## Model parameters
     nh=16,depth=2,nclass=60

        Layer (type)               Output Shape         Param # 
            Conv2d-1            [-1, 16, 8, 32]             784 
       BatchNorm2d-2            [-1, 16, 8, 32]              32 
              GELU-3            [-1, 16, 8, 32]               0 
         ConvBNACT-4            [-1, 16, 8, 32]               0 
            Conv2d-5            [-1, 16, 8, 32]             160 
       BatchNorm2d-6            [-1, 16, 8, 32]              32 
              GELU-7            [-1, 16, 8, 32]               0 
         ConvBNACT-8            [-1, 16, 8, 32]               0 
            Conv2d-9            [-1, 16, 8, 32]             272 
      BatchNorm2d-10            [-1, 16, 8, 32]              32 
             GELU-11            [-1, 16, 8, 32]               0 
        ConvBNACT-12            [-1, 16, 8, 32]               0 
       MicroBlock-13            [-1, 16, 8, 32]               0 
           Conv2d-14            [-1, 16, 8, 32]             160
      BatchNorm2d-15            [-1, 16, 8, 32]              32
             GELU-16            [-1, 16, 8, 32]               0
        ConvBNACT-17            [-1, 16, 8, 32]               0
           Conv2d-18            [-1, 16, 8, 32]             272
      BatchNorm2d-19            [-1, 16, 8, 32]              32
             GELU-20            [-1, 16, 8, 32]               0
        ConvBNACT-21            [-1, 16, 8, 32]               0
       MicroBlock-22            [-1, 16, 8, 32]               0
          Flatten-23              [-1, 128, 32]               0
           Linear-24               [-1, 32, 60]           7,740

     Total params: 9,548
     Trainable params: 9,548
     Non-trainable params: 0

     Input size (MB): 0.05
     Forward/backward pass size (MB): 0.73
     Params size (MB): 0.04
     Estimated Total Size (MB): 0.82

     ===========================================================
     nh=64,depth=2,nclass=60

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
          Flatten-23              [-1, 512, 32]               0
           Linear-24               [-1, 32, 60]          30,780

     Total params: 44,156
     Trainable params: 44,156
     Non-trainable params: 0

     Input size (MB): 0.05
     Forward/backward pass size (MB): 2.89
     Params size (MB): 0.17
     Estimated Total Size (MB): 3.10
     ===========================================================
     nh=128,depth=2,nclass=60

        Layer (type)               Output Shape         Param #

            Conv2d-1           [-1, 128, 8, 32]           6,272
       BatchNorm2d-2           [-1, 128, 8, 32]             256
              GELU-3           [-1, 128, 8, 32]               0
         ConvBNACT-4           [-1, 128, 8, 32]               0
            Conv2d-5           [-1, 128, 8, 32]           1,280
       BatchNorm2d-6           [-1, 128, 8, 32]             256
              GELU-7           [-1, 128, 8, 32]               0
         ConvBNACT-8           [-1, 128, 8, 32]               0
            Conv2d-9           [-1, 128, 8, 32]          16,512
      BatchNorm2d-10           [-1, 128, 8, 32]             256
             GELU-11           [-1, 128, 8, 32]               0
        ConvBNACT-12           [-1, 128, 8, 32]               0
       MicroBlock-13           [-1, 128, 8, 32]               0
           Conv2d-14           [-1, 128, 8, 32]           1,280
      BatchNorm2d-15           [-1, 128, 8, 32]             256
             GELU-16           [-1, 128, 8, 32]               0
        ConvBNACT-17           [-1, 128, 8, 32]               0
           Conv2d-18           [-1, 128, 8, 32]          16,512
      BatchNorm2d-19           [-1, 128, 8, 32]             256
             GELU-20           [-1, 128, 8, 32]               0
        ConvBNACT-21           [-1, 128, 8, 32]               0
       MicroBlock-22           [-1, 128, 8, 32]               0
          Flatten-23             [-1, 1024, 32]               0
           Linear-24               [-1, 32, 60]          61,500

     Total params: 104,636
     Trainable params: 104,636
     Non-trainable params: 0

     Input size (MB): 0.05
     Forward/backward pass size (MB): 5.76
     Params size (MB): 0.40
     Estimated Total Size (MB): 6.21

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