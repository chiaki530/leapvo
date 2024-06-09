[![arXiv](https://img.shields.io/badge/arXiv-LEAPVO-red)](https://arxiv.org/abs/2403.18913)

# LEAP-VO 
**[CVPR 2024]** The repository contains the official implementation of our paper:

> **LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry**<br> 
> by [Weirong Chen](https://chiaki530.github.io/), [Le Chen](https://clthegoat.github.io/), [Rui Wang](https://rui2016.github.io/), and [Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/)

**[[Paper](https://arxiv.org/abs/2401.01887)] [[Project Page](https://chiaki530.github.io/projects/leapvo/)]**

## Todo
- [ ] Improve LEAPVO visualization.
- [ ] Release LEAPVO training code.
- [ ] Release LEAPVO inference code and checkpoints.

## Setup and Installation 
To get started with the code, clone this repository and install the required dependencies:
```
git clone https://github.com/felix-ch/f3loc.git
cd f3loc
conda env create -f environment.yml
conda activate leap (use leap environment)
```

## Models and Data 

### MPI-Sintel
Follow [mpi-sintel](http://sintel.is.tue.mpg.de/) and download it to the data folder. You also need to download the [groundtruth camera pose data](http://sintel.is.tue.mpg.de/depth) for evaluation. 


### AirDOS-Shibuya
Follow [tartanair-shibuya](https://github.com/haleqiu/tartanair-shibuya) and download it to the data folder.


## Evaluation

### MPI-Sintel
```
bash scripts/eval/run_cotrackerslam_sintel_cvpr_fixed_intrinsics.sh
```

### AirDOS-Shibuya
```
bash scripts/eval/run_cotrackerslam_tartanair_shibuya_cvpr.sh
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
We sincerely thank the authors of the following repositories for publicly releasing their code and data: [PIPs](https://github.com/aharley/pips), [CoTracker](https://github.com/facebookresearch/co-tracker), [TAP-Vid](https://github.com/google-deepmind/tapnet), [DPVO](https://github.com/princeton-vl/DPVO), [ParticleSfM](https://github.com/bytedance/particle-sfm). 


## License