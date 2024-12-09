# MNIST Fashion - Optuna Hyperparameter Tuning

This project demonstrates the use of **Optuna** for automatic hyperparameter optimization on the MNIST Fashion dataset
![alt text](https://github.com/lyf1436/MnistFashion-Optuna/blob/main/newplot%20(2).png)
![alt text](https://github.com/lyf1436/MnistFashion-Optuna/blob/main/newplot%20(3).png)
![alt text](https://github.com/lyf1436/MnistFashion-Optuna/blob/main/newplot%20(4).png)
![alt text](https://github.com/lyf1436/MnistFashion-Optuna/blob/main/newplot%20(5).png)
---

## Task

Perform **segmentation** and **classification** on the MNIST Fashion dataset. Each image is segmented into different fashion categories, including a background class.

---
## Hyperparameters Studied:
learning rate
learning rate decay
batch size
class weight

---
## Model Architecture

The model is a minimal implementation of 4-layer CNN with a skip connection

```plaintext
SegmentationClassificationNet(
  (block1_conv): Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (block1_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (block1_bn): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (block2_conv): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (block2_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (block2_bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (block3_upsample): ConvTranspose2d(8, 16, kernel_size=(2, 2), stride=(2, 2))
  (block3_conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (block3_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (block4_upsample): ConvTranspose2d(16, 11, kernel_size=(2, 2), stride=(2, 2))
  (block4_conv): Conv2d(11, 11, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (block4_bn): BatchNorm2d(11, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
