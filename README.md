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

We train the network with data collected during autonomous path-following with a Grizzly robot using Multi-Experience VT&R. VT&R stores images and relative poses between in a spatio-temporal pose graph. Each time the robot repeats a path, new data is added to the pose graph. We can sample images and relative ground truth poses from a large range of appearance conditions from this pose graph. We train using data from the UTIAS Long-Term Localization and Mapping Dataset. The data is desribed in detail and can be download from this [website](http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/). 

## Getting Started

We explain how to run code for generating datasets, training the model, or testing the model. We provide a docker container that can be used to run the code and the python commands if running without docker is preferred.

### Docker

In the `docker` folder, we provide a Dockerfile that can be used to buid an image that will run the code. To build the docker image, run the following command:

```
cd docker
docker image build --shm-size=64g -t <docker_image_name> .
```

### Generate Datset

As mentioned above, we use data from the UTIAS Long-Term Localization and Mapping Dataset. We have written a script that will generate training, validation, and test datasets from the localization data. We sample image pairs and poses from the VT&R pose graph. The code for dataset generation is found in the `data` folder. Our code assumes that the data is structured the same way as described on the dataset [website](http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/). 

With our code, we have generated one set of trainging, testing, and validation datasets using the UTIAS Multiseason data and another set using the UTIAS In-The-Dark data. These can be downloaded using the `download_dataset.sh` script. This stores the data sample ids and ground truth poses needed in Pytorch. The images from the UTIAS Long-Term Localization and Mapping Dataset still need to be stored one the computer to be available to load.

Alternatively, if wanting to generate new datasets from scratch, this can be done by running a docker container as shown below. The `docker_data.sh` has to be updated with the correct file paths etc., where place holders are present.

```
cd docker
./docker_data.sh
```

Finally, the code can be run without the use of docker by executing the following command:

```
python -m data.build_train_test_dataset_loc --config <path_to_code>/deep_learned_visual_features/config/data.json
```

### Train

The model can be trained by running the following docker container, provided that updates are made to file path place holder in `docker_test.sh`:

```
cd docker
./docker_test.sh
```

The command for running the training code is the following:

```
python -m src.train --config <path_to_code>/deep_learned_visual_features/config/train.json
```

### Test

We provide model weights from training the network with the UTIAS Multiseason and UTIAS In-The-Dark datasets. They can be downloaded using the `download_networks.sh` script. We have the following model weight files:

`network_inthedark_layer32.pth`: the network was trained on data from the UTIAS In-The-Dark dataset and used in the lighting-change experiment described in the paper. The first layer of the network encoder is of size 32.

`network_multiseason_layer16.pth`: the network was trained on data from the UTIAS Multiseason dataset. The first layer of the network encoder is of size 16.

`network_mutiseason_inthedark_layer16.pth`: the network was trained on data from the UTIAS Multiseason dataset and then subsequently fine-tuned with that from the UTIAS In-The-Dark dataset. It was used in the generalization experiment described in the paper. The first layer of the network encoder is of size 16.

The testing can be done using the docker container provided that place holder file paths in `docker_test.sh` are updated:

```
cd docker
./docker_test.sh
```
The command for running the python test script is the following:

```
python -m src.test --config <path_to_code>/deep_learned_visual_features/config/test.json
```

## Experimental Results

We use the learned features in the VT&R system such that we can test them in closed-loop operation on a robot outdoors. The code for VT&R is available open-source [here](https://utiasasrl.github.io/). The experiments are described in detail in our paper. To summarize, we conducted two experiments. In the lighting-change experiment we trained the network using the UTIAS In-The-Dark dataset from 2016 and taught a new path in the same location around noon in August 2021. We repeated the path across a full range of lighting change from 3 a.m. until 10.30 p.m. with a 100% autonomy rate. In the figures below, we show example images from different conditons and a box plot of the inlier feature matches for each run. 

![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/conditions_lighting_change.png?raw=true)
![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/box_plot_lighting_change.png?raw=true)

In the generalization experiment we trained the network using the UTIAS In-The-Dark dataset from 2016 and the UTIAS Multiseason datasets from 2017 and taught a new path in August 2021. In this path we include new areas that have not been seen in he training data to test whether the features can generalize outside of the training paths. We teach the path around 1:30 p.m. and repeat it across a full range of lighting change from 4 a.m. until 9 p.m. with a 100% autonomy rate. In the figures below, we show example images from different conditons and a box plot of the inlier feature matches for each run. 

![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/conditions_generalization.png?raw=true)
![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/box_plot_generalization.png?raw=true)



