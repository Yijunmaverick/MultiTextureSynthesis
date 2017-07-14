# [MultiTextureSynthesis](https://sites.google.com/site/yijunlimaverick/texturesynthesis)
Torch implementation of our CVPR17 [paper](https://arxiv.org/abs/1703.01664) on multi-texture synthesis. For academic use only.

## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- Torch 
- Pretrained [VGG](https://drive.google.com/open?id=0B8_MZ8a8aoSeSG84N3pqcGpYT3M) model (download and put it under data/pretrained/)

## Task 1: Diverse synthesis

We first realize the diverse synthesis on single-texture. Given one texture example, the generator should be powerful enough to combine elements in various way.

### Training

```
th single_texture_diverse_synthesis_train.lua -texture YourTextureExample.jpg -image_size 256 -diversity_weight -1.0
```

# Testing

```
th single_texture_diverse_synthesis_test.lua 
```

## Task 2: Multi-texture synthesis

## Task 3: Multi-style transfer

We incorporate our idea of introducing a selection unit to handle the transferring of different styles. For each bit in
the selection unit, we generate a corresponding noise map (e.g., from the uniform distribution) and concatenate these
maps with the encoded features from the content, which are then decoded to the transferred result.

## Training
- Follow the [DCGAN](https://github.com/soumith/dcgan.torch) to prepare the data (CelebA). The only differece is that the face we cropped is of size 128x128. Please modify Line 10 in their [crop_celebA.lua](https://github.com/soumith/dcgan.torch/blob/master/data/crop_celebA.lua) file. We use the standard train&test split of the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.

- Modify the training data path in ./matlab/FaceCompletion_training/GFC_caffeinit.m file.

- Download our face parsing model [Model_parsing](https://drive.google.com/open?id=0B8_MZ8a8aoSeaXlUR296TzM2NW8) and put it under ./matlab/FaceCompletion_training/model/ folder.

- We provide an initial [model](https://drive.google.com/open?id=0B8_MZ8a8aoSeWWtldlhXSjdydVk) that is only trained with the reconstruction loss, as a good start point for the subsequent GAN training. Please download it and put it under ./matlab/FaceCompletion_training/model/ folder.

- Run ./matlab/FaceCompletion_training/demo_GFC_training.m for training.

## Testing
- Download our face completion model [Model_G](https://drive.google.com/open?id=0B8_MZ8a8aoSeQlNwY2pkRkVIVmM) and put it under ./matlab/FaceCompletion_testing/model/ folder. 
- Run ./matlab/FaceCompletion_testing/demo_face128.m for completion. TestImages are from the CelebA test dataset.

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
