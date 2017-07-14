# [MultiTextureSynthesis](https://sites.google.com/site/yijunlimaverick/texturesynthesis)
Torch implementation of our CVPR17 [paper](https://arxiv.org/abs/1703.01664) on multi-texture synthesis. For academic use only.

## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- Torch 
- Pretrained [VGG](https://drive.google.com/open?id=0B8_MZ8a8aoSeSG84N3pqcGpYT3M) model (download and put it under data/pretrained/)

## Task 1: Diverse synthesis

We first realize the diverse synthesis on single-texture. Given one texture example, the generator should be powerful enough to combine elements in various way.

-- Training

```
th single_texture_diverse_synthesis_train.lua -texture YourTextureExample.jpg -image_size 256 -diversity_weight -1.0
```

-- Testing

```
th single_texture_diverse_synthesis_test.lua 
```
After obtaining all diverse results, run gif.m (data/test_out/) in Matlab to convert them to an .avi video for view.

## Task 2: Multi-texture synthesis

- Training

Collect your texture image set (e.g., data/texture60/) before the training.

```
th multi_texture_synthesis_train.lua
```

- Testing

We release a 60-texture synthesis [model](https://drive.google.com/open?id=0B8_MZ8a8aoSeS0FncWpzTUNoblk) that synthesize the provided 60-texture set (ind_texture =1,2,...,60) in data/texture60/ folder.

```
th multi_texture_synthesis_test.lua -ind_texture 24
```


## Task 3: Multi-style transfer

In the synthesis, each bit in the selection unit represents a texture example. In the transferring, we employ a set of selection maps where each map represents one style image when initalized as a noise map (e.g., from the uniform distribution).

Collect your style image set (e.g., data/style1000/) before the training. For large number of style images (e.g., 1000), it is suggested to convert all images (e.g., ,jpg) to a HDF5 file for fast reading.

```
th convertHDF5.lua -images_path YourImageSetPath -save_to XXX.hdf5 -resize_to 512
```

- Training

```
th multi_style_transfer_train.lua -image_size 512
```

- Testing

We release a 1000-style transfer [model](https://drive.google.com/open?id=0B8_MZ8a8aoSeZnRESGg5Z0RpVzQ) that transfer this 1000-style [set](https://drive.google.com/open?id=0B8_MZ8a8aoSeajRLcEtIUjBjR3c) (ind_texture =1,2,...,1000).

```
th multi_style_transfer_test.lua 
```


## Citation
```
@inproceedings{DTS-CVPR-2017,
    author = {Li, Yijun and Fang, Chen and Yang, Jimei and Wang, Zhaowen and Lu, Xin and Yang, Ming-Hsuan},
    title = {Diversified Texture Synthesis with Feed-forward Networks},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2017}
}
```

## Acknowledgement
- This work was performed during a summer internship at [Adobe Research](https://research.adobe.com/).
- Codes are heavily borrowed from popular implementations of several great work, including [NeuralArt](https://github.com/jcjohnson/neural-style), [TextureNet](https://github.com/DmitryUlyanov/texture_nets), and [FastNeuralArt](https://github.com/jcjohnson/fast-neural-style).
