# [ParCNetV2: Oversized Kernel with Enhanced Attention](https://arxiv.org/abs/2211.07157)

This repo is the official PyTorch implementation of **ParCNetV2** proposed by our paper "[ParCNetV2: Oversized Kernel with Enhanced Attention](https://arxiv.org/abs/2211.07157)".


![Figure1](/assets/accuracy_latency.png)
Figure 1: **Comparison between ParCNetV2 with the prevailing transformer (Swin), CNN (ConvNeXt), and large kernel CNNs (RepLKNet \& SLaK) when trained from scratch on ImageNet-1K at 224x224 resolution.**  Left: performance curve of model size vs. top-1 accuracy. Right: performance curve of inference latency vs. top-1 accuracy. **IG** represents using the *implicit gemm* acceleration algorithm.

## Introduction

Transformers have shown great potential in various computer vision tasks. By borrowing design concepts from transformers, many studies revolutionized CNNs and showed remarkable results. This paper falls in this line of studies. Specifically, we propose a new convolutional neural network, **ParCNetV2**, that extends the research line of ParCNetV1 by bridging the gap between CNN and ViT. It introduces two key designs: 1) **O**versized **C**onvolution (**OC**) with twice the size of the input, and 2) **B**ifurcate **G**ate **U**nit (**BGU**) to ensure that the model is input adaptive. Fusing OC and BGU in a unified CNN, ParCNetV2 is capable of flexibly extracting global features like ViT, while maintaining lower latency and better accuracy. Extensive experiments demonstrate the superiority of our method over other convolutional neural networks and hybrid models that combine CNNs and transformers.

## Overview
![Figure2](/assets/parc_evolution.png)
Figure 2: **The transitions from the original ParC V1 to ParC V2 block.** Compared with ParCNetV1, we first introduce oversized convolutions to further enhance capacity while simplifying architecture; then we design a bifurcate gate unit to improve efficiency and strengthen attention; finally, we propose a uniform local-global block and construct the whole network with this uniform block.


## Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; [timm](https://github.com/rwightman/pytorch-image-models) (`pip install timm==0.6.12`)

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train

We use batch size of 4096 by default and we show how to train models with 8 GPUs. For multi-node training, adjust `--grad-accum-steps` according to your situations.


```bash

DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/metaformer # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS
MASTER_PORT=29501

cd $CODE_PATH && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=$NUM_GPU \
--master_port=$MASTER_PORT \
train.py $DATA_PATH \
--model parcnetv2_tiny --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.2 --head-dropout 0.0 \
> log/parcnetv2_tiny.log 2>&1
```
Training scripts of other models are shown in [scripts](/scripts/).


<!-- ## Experiments

### ImageNet-1K classification

| Model        | Param(M) | Macs(G) | Top-1(%) |
|:-------------|:--------:|:-------:|:--------:|
| ParCNetV2-XT | 7.4      | 1.6     | 79.4     |
| ParCNetV2-T  | 25       | 4.3     | 83.5     |
| ParCNetV2-S  | 39       | 7.8     | 84.3     |
| ParCNetV2-B  | 56       | 12.5    | 84.6     | -->


## License

This project is released under the MIT license. Please see the [LICENSE](/LICENSE) file for more information.

## Citation

If you find this repository helpful, please consider citing:

```
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## Acknowledgement

This work is supported in part by Natural Science Foundation of China under Grant No. U20B2052, 61936011, in part by The National Key Research and Development Program of China under Grant No. 2018YFE0118400.

This repository is built using the following libraries and repositories.

1. [Timm](https://github.com/rwightman/pytorch-image-models)
2. [DeiT](https://github.com/facebookresearch/deit)
3. [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
4. [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
5. [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
6. [MetaFormer](https://github.com/sail-sg/metaformer)
