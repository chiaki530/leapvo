[![arXiv](https://img.shields.io/badge/arXiv-LEAPVO-red)](https://arxiv.org/abs/2403.18913)

# LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry [CVPR 2024]

![](assets/butterfly-3d-tracks.gif)

> [**LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry**](https://chiaki530.github.io/projects/leapvo/),  
> Weirong Chen, Le Chen, Rui Wang, Marc Pollefeys,
> CVPR 2024
> *Paper at [arxiv](https://arxiv.org/abs/2401.01887)*  


## News and ToDo
- [ ] Improve LEAPVO visualization.
- [ ] Release LEAPVO training code.
- [ ] Release LEAPVO inference code and checkpoints.

## Requirements 
To get started with the code, clone this repository and install the required dependencies:
```
git clone https://github.com/felix-ch/f3loc.git
cd f3loc
conda env create -f environment.yml
conda activate f3loc
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


## Citation
If you use this project or ideas from the paper for your research, please cite our paper:
```
@inproceedings{chen2024leap,
  title={LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry},
  author={Chen, Weirong and Chen, Le and Wang, Rui and Pollefeys, Marc},
  journal={arXiv preprint arXiv:2401.01887},
  year={2024}
}
```
## Acknowledgement
We sincerely thank the authors of the following repositories for publicly releasing their work:


## License