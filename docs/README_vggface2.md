# Fine-tuning of Squeeze and Excitation Network Pre-trained on VGG Face2 Dataset


## Run fine-tune
```
python finetune.py
```

* `--image_size`, type=int, default=256, help='training data size == (x.size[0])'
* `--out_features`, type=int, default=136, help='number of classes == y'
* `--batch_size`, type=int, default=128, help='batch size'
* `--num_epochs`, type=int, default=30, help='number of epochs to run'
* `--lr`, type=float, default=0.001, help='learning rate'
* `--feature_extract`, type=bool, default=True, help='finetune fc layer only (True) / all layers (False)'

* `--root`, type=str, default='./datasets/generated', help='training data dir'
* `--model_path`, type=str, default='./pretrained/senet50_scratch_weight.pkl', help='pretrained SENet model path'

* `--save_epoch`, type=int, default=1, help='epoch to save images for validation'
* `--val_save_dir`, type=str, default='./assets/out', help='save directory for validation file'
* `--log_dir`, type=str, default='log', help='save directory for log file'
* `--ckpt_dir`, type=str, default='ckpt', help='save directory for checkpoint file'

* `--n_identity`, type=int, default=8631, help='number of classes pretrained in SENet'
* `--pretrained_size`, type=int, default=224, help='image size pretrained in SENet'


## Dataset

VGGFace2 dataset, see [authors' site](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).

- [Ref1](https://github.com/ox-vgg/vgg_face2/issues/2)
- [Ref2](https://github.com/ox-vgg/vgg_face2/issues/50)

## Pretrained models

The followings are PyTorch models converted from [Caffe models](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) authors of [2] provide.

|arch_type|download link|
| :--- | :---: |
|`resnet50_ft`|[link](https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU)|
|`senet50_ft`|[link](https://drive.google.com/open?id=1YtAtL7Amsm-fZoPQGF4hJBC9ijjjwiMk)|
|`resnet50_scratch`|[link](https://drive.google.com/open?id=1gy9OJlVfBulWkIEnZhGpOLu084RgHw39)|
|`senet50_scratch`|[link](https://drive.google.com/open?id=11Xo4tKir1KF8GdaTCMSbEQ9N4LhshJNP)|


## References

1. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141). [arXiv](https://arxiv.org/abs/1709.01507?spm=a2c41.13233144.0.0)

2. ZQ. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman, VGGFace2: A dataset for recognising faces across pose and age, 2018.   
    [site](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/), [arXiv](https://arxiv.org/abs/1710.08092)