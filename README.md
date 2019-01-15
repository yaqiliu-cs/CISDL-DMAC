# CISDL-DMAC

Constrained image splicing detection and localization (CISDL) is a newly proposed challenging task for image forensics, which investigates two input suspected images and identifies whether one image has suspected regions pasted from the other. We propose a novel adversarial learning framework to train a deep matching network for CISDL. Our framework mainly consists of three building blocks: 1) a deep matching network based on atrous convolution (DMAC) aims to generate two high-quality candidate masks which indicate suspected regions of the two input images, 2) a detection network is designed to rectify inconsistencies between the two corresponding candidate masks, 3) a discriminative network drives the DMAC network to produce masks that are hard to distinguish from ground-truth ones. In DMAC, atrous convolution is adopted to extract features with rich spatial information, a correlation layer based on a skip architecture is proposed to capture hierarchical features, and atrous spatial pyramid pooling is constructed to localize tampered regions at multiple scales. The detection network and the discriminative network act as the losses with auxiliary parameters to supervise the training of DMAC in an adversarial way. Besides, a sliding window based matching strategy is investigated for high-resolution images matching. Extensive experiments, conducted on five groups of datasets, demonstrate the effectiveness of the proposed framework and the superior performance of DMAC.

## Citation

If you use this code for your research, please cite our paper.

```
@article{liu2018adversarial,
  title={Adversarial Learning for Image Forensics Deep Matching with Atrous Convolution},
  author={Liu, Yaqi and Zhao, Xianfeng and Zhu, Xiaobin and Cao, Yun},
  journal={arXiv preprint arXiv:1809.02791},
  year={2018}
}
```

## Getting Started

We provide PyTorch implementations for both DMAC and [DMVN](https://gitlab.com/rex-yue-wu/Deep-Matching-Validation-Network).

The code was written and supported by [Yaqi Liu](https://github.com/yaqiliu-cs).

### Prerequisites

- Linux
- Python 2
- CPU or NVIDIA GPU + CUDA

### Installation

- Install PyTorch 0.4+ and torchvision from http://pytorch.org and OpenCV 3.

- Clone this repo:
```bash
git clone https://github.com/yaqiliu-cs/CISDL-DMAC
cd CISDL-DMAC
```
