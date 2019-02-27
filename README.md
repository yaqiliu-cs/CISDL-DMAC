# CISDL-DMAC

Constrained image splicing detection and localization (CISDL) is a newly proposed challenging task for image forensics, which investigates two input suspected images and identifies whether one image has suspected regions pasted from the other. We propose a novel adversarial learning framework to train a deep matching network for CISDL. Our framework mainly consists of three building blocks: 1) a deep matching network based on atrous convolution (DMAC) aims to generate two high-quality candidate masks which indicate suspected regions of the two input images, 2) a detection network is designed to rectify inconsistencies between the two corresponding candidate masks, 3) a discriminative network drives the DMAC network to produce masks that are hard to distinguish from ground-truth ones. In DMAC, atrous convolution is adopted to extract features with rich spatial information, a correlation layer based on a skip architecture is proposed to capture hierarchical features, and atrous spatial pyramid pooling is constructed to localize tampered regions at multiple scales. The detection network and the discriminative network act as the losses with auxiliary parameters to supervise the training of DMAC in an adversarial way. Besides, a sliding window based matching strategy is investigated for high-resolution images matching.

## Citation

The code is for **Research Use Only**. If you use this code for your research, please cite our paper (The formal citation will be updated in the future).

```
@article{liuyaqi2019tifs,
  title={Adversarial Learning for Constrained Image Splicing Detection and Localization based on Atrous Convolution},
  author={Liu, Yaqi and Zhu, Xiaobin and Zhao, Xianfeng and Cao, Yun},
  journal={{IEEE} Trans. Inf. Forensics Security},
  year={2019}
}
```

## Getting Started

We provide PyTorch implementations for both DMAC and [DMVN](https://gitlab.com/rex-yue-wu/Deep-Matching-Validation-Network).

The code was written and supported by [Yaqi Liu](https://github.com/yaqiliu-cs).

### Prerequisites

- Linux
- Python 2
- NVIDIA GPU + CUDA

### Installation

- Install PyTorch 0.4+ and torchvision from http://pytorch.org and OpenCV 3.

- Clone this repo:
```bash
git clone https://github.com/yaqiliu-cs/CISDL-DMAC
cd CISDL-DMAC
```

### Pre-trained models
Download our pre-trained [DMAC-adv](https://drive.google.com/open?id=1NDNTrFbrJV0MgV7f780lGuya79_l4I8f) and [DMVN-BN](https://drive.google.com/open?id=1hu9tO-NbCMRmuFEKi0aFHYGDjPs4hGxO) models.

### DMAC demo

- You can run a simple demo:
```bash
python demo.py im1_1.jpg im1_2.jpg
```

### DMAC train
- Training Data Preparation

You can generate sufficient synthetic image pairs for training by running "combine_generation_train.m" based on the [MatlabAPI](https://github.com/cocodataset/cocoapi) of [the MS COCO dataset](http://cocodataset.org/#download).

Changing training paths and lists to your generated training sets, you can train your own DMAC and DMAC-adv models as follows:

- Training based on the single spatial cross-entropy loss
```bash
python train_ce_script.py
```

- Optimizing based on the multi-task loss
```bash
python train_adversary_script.py
```

### Generated testing sets
- [Combination Sets](https://drive.google.com/open?id=1LcoojC4T9oED6r-MuynUWKn11jihllVm).
- [Generated sets for invariance analyses](https://drive.google.com/open?id=1rk81lJ3McO9rO6Ktchqly7fJ3U0p8uuC).


### DMAC test
Download our generated [Combination Sets](https://drive.google.com/open?id=1LcoojC4T9oED6r-MuynUWKn11jihllVm).

- Test DMAC-adv model
```bash
python test_coco.py
```

- Test DMAC-adv model with a sliding window based matching strategy
```bash
python test_coco_slide.py
```

### DMVN test
Download our generated [Combination Sets](https://drive.google.com/open?id=1LcoojC4T9oED6r-MuynUWKn11jihllVm).

- Test DMVN-BN model
```bash
python test_coco_dmvn.py
```

## Authorship

**Yaqi Liu**
1. State Key Laboratory of Information Security, Institute of Information Engineering, Chinese Academy of Sciences, Beijing 100093, China.
2. School of Cyber Security, University of Chinese Academy of Sciences, Beijing 100093, China.

E-mail: liuyaqi@iie.ac.cn

