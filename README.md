# Surrogate Gradient Field (SGF) for Latent Space Manipulation in Pytorch

This is an unofficial implementation of the paper ["Surrogate Gradient Field for Latent Space Manipulation (CVPR 2021)"](https://arxiv.org/abs/2104.09065) in Pytorch. Please notice that this implementation may differ in details compared to the original paper due to the empricial reasons.

![sgf_result](./docs/sgf_result.jpg)

### (Jul. 16, 2021) Current issues in the result (TODO, working on)
- the ID of face changes
    - how to fix? Will add more supervision (binary attributes)


## Requirements
Please install the environment by running:
```
bash install.sh
```
- which will install libraries such as:
```
pip install tensorflow-gpu==1.15
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
pip install scipy>=0.17.0
pip install requests==2.22.0
pip install Pillow==6.2.1
pip install h5py==2.9.0
pip install imageio==2.9.0
pip install imageio-ffmpeg==0.4.2
pip install tqdm==4.49.0
pip install click pyspng ninja
pip install opencv-python
pip install scikit-image
pip install numba
```


## Run SGF
To see the manipulation result:
```
python sgf.py --G_path 'path/to/generator.pkl' --SE_path 'path/to/se.pth' --AUX_path 'path/to/aux.pth' --save_result 1
```

---

## SGF step by step
SGF consists of multiple steps to follow. I will briefly introduce the concept of each step and the code to run respectively.

### Step 1: Sample image generation using StyleGAN2 [`x`]
- Generate 100K samples images using StyleGAN2 to train SENet
```
python generate.py --outdir=data/train/images --seeds=0,100000 --resize 256
python generate.py --outdir=data/val/images --seeds=100000,100500 --resize 256
python generate.py --outdir=data/test/images --seeds=100500,101000 --resize 256
```

### Step 2: Label images [`c`]
- Label images using Azure Face API / open source Face landmark detection algorithm
```
python face_align.py --indir train
python face_align.py --indir val
python face_align.py --indir test
```

- If you want to see the landmark result
```
python face_align.py --indir test --plot 1
```


### Step 3: Fine-tune Squeeze and Excitation Network using images [`x`] and labels [`c`]
- Used is SE ResNet 50 pretrained on VGG Face2 dataset
```
python finetune.py --pretrained_path 'path/to/model.pkl'
python finetune.py --mode test --model_path 'path/to/model.pth'
```

### Step 4: Train Auxiliary (FC-layer) Network [`mapping: (z, c) -> z`]
- 6 FC layers for Z space, and 15 layers for W space
- AdaIN is used to mix features (`z` and `c`) in the same way as StyleGAN v1
- Refer to Appendix B in the paper

```
python fc_layer.py --ckpt_dir 'path/to/save_dir'
python fc_layer.py --mode test --ckpt_dir 'path/to/save_dir' --ckpt_fname 'filename.pth'
```

### Step 5: Calculate gradient in the surrogate gradient field and update [`z`]
- Refer to Algo 1 in the original paper
- Manipulate C to suit your purpose
```
python sgf.py --G_path 'path/to/generator.pkl' --SE_path 'path/to/se.pth' --AUX_path 'path/to/aux.pth' --save_result 1
```


## Acknowledgement
Many thanks to the first author of the original paper, [Minjun Li](https://minjun.li/). The reproducing was not possible without Minjun's help.

## References
- [Li, M., Jin, Y., & Zhu, H. (2021). Surrogate Gradient Field for Latent Space Manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision ](https://arxiv.org/abs/2104.09065)

Also, the implementaion is based on many works:
- [Face Alignment](https://arxiv.org/abs/1703.07332)
    - [Official code](https://github.com/1adrianb/face-alignment)
- [StyleGAN2](https://arxiv.org/abs/1912.04958)
    - [Official code](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [SENet](https://arxiv.org/abs/1709.01507?spm=a2c41.13233144.0.0) & [VGG Face2 dataset](https://arxiv.org/abs/1710.08092)
    - [Official code](https://github.com/ox-vgg/vgg_face2)
    - [Pytorch code](https://github.com/cydonia999/VGGFace2-pytorch)
- [AdaIN](https://arxiv.org/abs/1703.06868)