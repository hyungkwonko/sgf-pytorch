# To train on landmark labels


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