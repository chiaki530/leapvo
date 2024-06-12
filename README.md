# LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry

**[CVPR 2024]** The repository contains the official implementation of [LEAP-VO](https://github.com/chiaki530/leapvo). We aim to leverage **temporal context with long-term point tracking** to achieve motion estimation, occlusion handling, and track probability modeling.


> **LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry**<br> 
> [Weirong Chen](https://chiaki530.github.io/), [Le Chen](https://clthegoat.github.io/), [Rui Wang](https://rui2016.github.io/), [Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/)<br> 
> CVPR 2024

**[[Paper](https://arxiv.org/abs/2401.01887)] [[Project Page](https://chiaki530.github.io/projects/leapvo/)]**

<div align="center">
  <p align="center">
  <a href="">
    <img src="./assets/demo.gif" alt="Logo" width="70%">
  </a>
</p>
</div>


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

### Install LEAP-VO package
```
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

pip install .
```

## Demos
Our method requires an RGB video and camera intrinsics as input. We provide the model checkpoint and example data on [Google Drive](https://drive.google.com/drive/folders/1muTSIpAvm61YrSZJhOrybcvd34BZ3wK7?usp=sharing). Please download `leap_kernel.pth` and place it in the `weights` folder, and download `samples` and place them in the `data` folder.

The demo can be run using:
```
python main/eval.py \
  --config-path=../configs \
  --config-name=demo \                                  # config file
  data.imagedir=data/samples/sintel_market_5/frames \   # path to image directory or video
  data.calib=data/samples/sintel_market_5/calib.txt \   # calibration file
  data.savedir=logs/sintel_market_5 \                   # save directory
  data.name=sintel_market_5 \                           # scene name
  save_trajectory=true \                                # save trajectory in TUM format
  save_video=true \                                     # save video visualization
  save_plot=true                                        # save trajectory plot
```

## Evaluations
We provide evaluation scripts for MPI-Sinel, TartanAir-Shibuya, and Replica.

### MPI-Sintel
Follow [MPI-Sintel](http://sintel.is.tue.mpg.de/) and download it to the `data` folder. For evaluation, we also need to download the [groundtruth camera pose data](http://sintel.is.tue.mpg.de/depth). The folder structure should look like
```
MPI-Sintel-complete
└── training
    ├── final
    └── camdata_left
```

Then run the evaluation script after setting the `DATASET` variable to custom location. 
```
bash scripts/eval_sintel.sh
```

### TartanAir-Shibuya
Follow [TartanAir-Shibuya](https://github.com/haleqiu/tartanair-shibuya) and download it to the `data` folder. Then run the evaluation script after setting the `DATASET` variable to custom location. 

```
bash scripts/eval_shibuya.sh
```

### Replica 
Follow [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf/) and download the Replica dataset into data folder. Then run the evaluation script after setting the `DATASET` variable to custom location. 
```
bash scripts/eval_replica.sh
```

## Citations
If you find our repository useful, please consider citing our paper in your work:
```
@InProceedings{chen2024leap,
  title={LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry},
  author={Chen, Weirong and Chen, Le and Wang, Rui and Pollefeys, Marc},
  journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
## Acknowledgement
We adapted some codes from some awesome repositories including [CoTracker](https://github.com/facebookresearch/co-tracker), [DPVO](https://github.com/princeton-vl/DPVO), and [ParticleSfM](https://github.com/bytedance/particle-sfm).
We sincerely thank the authors for open-sourcing their work and follow the License of CoTracker, DPVO and ParticleSfM.

