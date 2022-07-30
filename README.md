## MicroOCR
a micro OCR network.

This model can handle complex tasks without lstm, and its accuracy and speed are better than resnet and crnn models.

## Script Description

```shell
MicroOCR
├── README.md                                   # descriptions about MicroNet
├── average_meter.py                            # average meter
├── collatefn.py                                # batch data processing
├── label_converter.py                          # label converter
├── dataset.py                                  # data preprocessing for training and evaluation
├── demo.py                                     # inference
├── gen_image.py                                # generate image for train and eval
├── img_aug.py                                  # img augmentation
├── infer_tool.py                               # inference tool
├── logger.py                                   # logger
├── loss.py                                     # ctcloss definition
├── model.py                                    # MicroMLPNet
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