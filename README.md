# Keeping an Eye on Things: Deep Learned Features for Long-Term Visual Localization

![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/overview.png?raw=true)

We learn visual features for localization across a full range of lighting change. We showed in our paper that the learned features could be used for successful localization in Visual Teach and repeat (VT&R) in closed-loop on a robot outdoors. In VT&R, we manually drive the robot to teach a path (and build a map) that the robot subsequently repeats autonoously.  

We train a neural network to predict sparse keypoints with associated descriptors and scores that can be used together with a classical pose estimator for localization. Our training pipeline includes a differentiable pose estimator such that training can be supervised with ground truth poses from data collected earlier. 

We insert the learned features into the existing VT&R pipeline to perform closed-loop path-following in unstructured outdoor environments. We show successful path following across all lighting conditions despite the robot's map being constructed using daylight conditions. In all, we validated the approach with 35 km of autonomous path-following experiments in challenging conditions.

The details of our method is detailed in the [paper](https://arxiv.org/abs/2109.04041):
```
@ARTICLE{gridseth_ral2022,
  author={Gridseth, Mona and Barfoot, Timothy D.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Keeping an Eye on Things: Deep Learned Features for Long-Term Visual Localization}, 
  year={2022},
  volume={7},
  number={2},
  pages={1016-1023},
  doi={10.1109/LRA.2021.3136867}}
```

## Learning Pipeline

Our method is inpired by Barnes et al. and their [paper](https://arxiv.org/abs/2001.10789) Under the Radar: Learning to Predict Robust Keypoints for Odometry Estimation and Metric Localisation in Radar. We have adapted a similar differentiable learning pipeline and network architecture to work for a stereo camera. The training pipeline takes a pair of stereo images and uses the neural network to extract keypoints, descriptors, and scores. We match keypoint using the descriptors and compute 3D points for each of the matched 2D keypoints. The 3D points are passed to a differentiable pose estimator that uses SVD. The pipeline is illustrated below:

![Test](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/pipeline.png?raw=true)

The neural network consists of an encoder and two decoders. The descriptors are extracted from the encode, while the two decoders provide the keypoint coordinates and scores, see the figure below.

![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/network.png?raw=true)

## Training Data

## Getting Started

We provide the docker 

### Docker

### Generate Datset

### Train

### Test

