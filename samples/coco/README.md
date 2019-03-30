# Guide

To perform a fine-tuning/continue the training:
```shellscript
python -W ignore coco.py train --logs=logs/ --model=imagenet --dataset=/data/coco --edge-loss-filters sobel-x sobel-y laplace --edge-loss-smoothing --run-name=sobel_xy_laplace_gauss1
```

To perform a full training/train from scratch:
```shellscript
python -W ignore coco.py train --logs=logs/ --model=last --dataset=/data/coco --edge-loss-filters sobel-x sobel-y laplace --edge-loss-smoothing --run-name=sobel_xy_laplace_gauss1
```