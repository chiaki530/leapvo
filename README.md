[![arXiv](https://img.shields.io/badge/arXiv-LEAPVO-red)](https://arxiv.org/abs/2403.18913)

# LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry
**[CVPR 2024]** The repository contains the official implementation of [LEAP-VO](https://github.com/chiaki530/leapvo). We aim to leverage **temporal context with long-term point tracking** to achieve motion estimation, occlusion handling, and track probability modeling.


> **LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry**<br> 
> [Weirong Chen](https://chiaki530.github.io/), [Le Chen](https://clthegoat.github.io/), [Rui Wang](https://rui2016.github.io/), [Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/)<br> 
> CVPR 2024

**[[Paper](https://arxiv.org/abs/2401.01887)] [[Project Page](https://chiaki530.github.io/projects/leapvo/)]**

<!-- ## Todo
- [ ] Improve LEAPVO visualization.
- [ ] Release LEAPVO training code.
- [ ] Release LEAPVO inference code and checkpoints. -->

## Installation
### Requirements
The code was tested on Ubuntu 20.04, PyTorch 1.12.0, CUDA 11.3 with 1 NVIDIA GPU (RTX A4000).

### Clone the repo
```
git clone https://github.com/chiaki530/leapvo.git
cd leapvo 
```

### Create a conda environment
```
conda env create -f environment.yml
conda activate leapvo
```
<!-- conda create -n leapvo python==3.10 -->

<!-- Install PyTorch and other dependencies
```
conda install pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
``` -->

<!-- Install LieTorch
```
pip install git+https://github.com/princeton-vl/lietorch.git
``` -->

### Install LEAPVO package
```
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

pip install .
```


## Demos


## Evaluation
We provide evaluation scripts for MPI-Sinel, AirDOS-Shibuya, and Replica.
### MPI-Sintel
Follow [MPI-Sintel](http://sintel.is.tue.mpg.de/) and download it to the data folder. For evaluation, we also need to download the [groundtruth camera pose data](http://sintel.is.tue.mpg.de/depth). 

```
bash scripts/eval_shibuya.sh
```

### TartanAir-Shibuya
Follow [TartanAir-Shibuya](https://github.com/haleqiu/tartanair-shibuya) and download it to the data folder.

```
bash scripts/eval/run_cotrackerslam_tartanair_shibuya_cvpr.sh
```

### Replica 
Follow [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf/) and download the Replica dataset into data folder.
```
bash scripts/eval_replica.sh
```

## Citations
If you find our repository useful, please consider citing our paper in your work:
```
@article{chen2024leap,
  title={LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry},
  author={Chen, Weirong and Chen, Le and Wang, Rui and Pollefeys, Marc},
  journal={arXiv preprint arXiv:2401.01887},
  year={2024}
}
```
## Acknowledgement
LEAP-VO is built on [CoTracker](https://github.com/facebookresearch/co-tracker) and [DPVO](https://github.com/princeton-vl/DPVO). 
We sincerely thank the authors for publicly releasing their great work.

<!-- : , [CoTracker](https://github.com/facebookresearch/co-tracker), [TAP-Vid](https://github.com/google-deepmind/tapnet), [DPVO](https://github.com/princeton-vl/DPVO), [ParticleSfM](https://github.com/bytedance/particle-sfm).  -->


<!-- ## License -->