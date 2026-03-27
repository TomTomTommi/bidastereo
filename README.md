<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
Match-Stereo-Videos: Bidirectional Alignment <br> for Consistent Dynamic Stereo Matching<h1>      

<div align="center">
  <a href="https://arxiv.org/abs/2403.10755" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
  <a href="https://tomtomtommi.github.io/BiDAStereo/" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/Project-Page-9cf" alt="Project Page"></a>
</div>
</p>

![Reading](./assets/Reading.gif)

## Updated
The extension of this work is [[`BiDAVideo`](https://github.com/TomTomTommi/bidavideo)]

## Dataset

Download the following datasets and put in `./data/datasets`:
 - [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
 - [Sintel](http://sintel.is.tue.mpg.de/stereo)

Download the following dataset and link to the project `ln -s ./dynamic_replica ./bidastereo/`:
 - [Dynamic_Replica](https://dynamic-stereo.github.io/)


## Installation

Installation of BiDAStereo with PyTorch3D, PyTorch 1.12.1 & cuda 11.3

### Setup the root for all source files:
```
git clone https://github.com/TomTomTommi/BiDAStereo
cd bidastereo
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```
### Create a conda env:
```
conda create -n bidastereo python=3.8
conda activate bidastereo
```
### Install requirements
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
```

## Evaluation
To download the checkpoints, click the links below. Copy the checkpoints to `./bidastereo/checkpoints/`.

- [BiDAStereo](https://github.com/TomTomTommi/BiDAStereo/releases/tag/v0.0) trained on SceneFlow
- [BiDAStereo](https://github.com/TomTomTommi/BiDAStereo/releases/tag/v0.0) trained on SceneFlow and Dynamic Replica

To evaluate BiDAStereo:
```
sh evaluate_bidastereo.sh
sh evaluate_real.sh
```
The results are evaluated on an A6000 48GB GPU.
Evaluation on *Dynamic Replica* requires a 32GB GPU. If you don't have enough GPU memory, you can modify `kernel_size` from 20 to 10.

## Training
Training requires 8 V100 32GB GPUs or 4 A100 80GB GPUs. You can decrease `image_size` and / or `sample_len` if you don't have enough GPU memory.
```
sh train_bidastereo.sh
```

## Citing BiDAStereo
If you use BiDAStereo in your research, please use the following BibTeX entry.
```
@inproceedings{jing2024match,
  title={Match-stereo-videos: Bidirectional alignment for consistent dynamic stereo matching},
  author={Jing, Junpeng and Mao, Ye and Mikolajczyk, Krystian},
  booktitle={European Conference on Computer Vision},
  pages={415--432},
  year={2024},
  organization={Springer}
}
```
## Acknowledgement

In this project, we use parts of public codes and thank the authors for their contribution in:
- [DynamicStereo](https://github.com/facebookresearch/dynamic_stereo)
- [RAFT](https://github.com/princeton-vl/RAFT)
