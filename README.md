# Towards Better Multi-head Attention via Channel-wise Sample Permutation

## Introduction

Transformer plays a central role in many fundamental deep learning models, e.g., the ViT in computer vision and the BERT and GPT in natural language processing, whose effectiveness is mainly attributed to its multi-head attention (MHA) mechanism. 
In this study, we propose a simple and novel channel-wise sample permutation (CSP) operator, achieving a new structured MHA with fewer parameters and lower complexity. 
Given an input matrix, CSP sorts grouped samples of each channel and then circularly shifts the sorted samples of different channels with various steps. 
This operator is equivalent to implicitly implementing cross-channel attention maps as permutation matrices, which achieves linear complexity and suppresses the risk of rank collapse when representing data. 
We replace the MHA of some representative models with CSP and test the CSP-based models in several discriminative tasks, including image classification and long sequence analysis. 
Experiments show that the CSP-based models achieve comparable or better performance with fewer parameters and lower computational costs than the classic Transformer and its state-of-the-art variants. 


## ViT

### Prepare Dataset

The `CIFAR-10` and `CIFAR-100` are downloaded from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).
The `ImageNet-1k` is downloaded from [https://www.image-net.org/](https://www.image-net.org/).

### Image Classification Based on ViT

To run image classification experiments, modify config.py and execute one_expe.py.

```bash
cd vit
python one_expe.py
```

## LRA

### Environment Setup 

This repository requires Python 3.8+ and Pytorch 1.11+.

```bash
cd lra/mega_csp
pip install -e .
```

### Prepare Dataset

Download the [processed data](https://dl.fbaipublicfiles.com/mega/data/lra.zip). The original data is from the [LRA repo](https://github.com/google-research/long-range-arena).

### Train MEGA using CSP on LRA 

To train MEGA using CSP on LRA, modify and run train_lra.sh.

```bash
bash train_lra.sh
```

- `model_name`: The model to be trained, can be one of `transformer`, `mega`, `lstm` and `flash`.
- `dataset_name`: The six tasks of the LRA, can be one of `listops`, `imdb-4000`, `aan`, `cifar10`, `pathfinder` and `path-x`.
