# Keeping an Eye on Things: Deep Learned Features for Long-Term Visual Localization

![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/overview.png?raw=true)

We learn visual features for localization across a large appearace change and used them in Visual Teach and repeat (VT&R) for closed-loop path-following on a robot outdoors. In VT&R, we manually drive the robot to teach a path (and build a map) that the robot subsequently repeats autonoously. This [video](https://www.youtube.com/watch?v=KkG6TQOVXak) shows a live demonstration of localization between daytime and after dark.  

We train a neural network to predict sparse keypoints with associated descriptors and scores that can be used together with a classical pose estimator for localization. Our training pipeline includes a differentiable pose estimator such that training can be supervised with ground truth poses from data collected earlier. 

We insert the learned features into the existing VT&R pipeline to perform closed-loop path-following in unstructured outdoor environments. We show successful path following across all lighting conditions despite the robot's map being constructed using daylight conditions. In all, we validated the approach with 35.5 km of autonomous path-following experiments in challenging conditions.

The details of our method can be found in the [paper](https://arxiv.org/abs/2109.04041):
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

Our method is inpired by Barnes et al. and their [paper](https://arxiv.org/abs/2001.10789) Under the Radar: Learning to Predict Robust Keypoints for Odometry Estimation and Metric Localisation in Radar. We have adapted a similar differentiable learning pipeline and network architecture to work for a stereo camera. The training pipeline takes a pair of stereo images and uses the neural network to extract keypoints, descriptors, and scores. We match keypoints using the descriptors and compute 3D points for each of the matched 2D keypoints. The 3D points are passed to a differentiable pose estimator that uses SVD. The pipeline is illustrated below:

![Test](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/pipeline.png?raw=true)

The neural network consists of an encoder and two decoders. The descriptors are extracted from the encoder, while the two decoders provide the keypoint coordinates and scores, see the figure below.

![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/network.png?raw=true)

## Training Data

We train the network with data collected during autonomous path-following with a Grizzly robot using Multi-Experience VT&R. VT&R stores images and relative poses between them in a spatio-temporal pose graph. Each time the robot repeats a path, new data is added to the pose graph. We can sample images and relative ground truth poses from a large range of appearance conditions from this pose graph. We train using data from the UTIAS Long-Term Localization and Mapping Dataset. The data is desribed in detail and can be download from this [website](http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/). 

## Getting Started

We explain how to run code for generating datasets, training the model, and testing the model. We provide a docker container that can be used to run the code. If running without docker is preferred, we provide a `requirements.txt` to list the neccessary dependencies and provide the the commands neeed to run the python scripts. 

### Docker

In the `docker` folder, we provide a Dockerfile that can be used to buid an image that will run the code. To build the docker image, run the following command:

```
cd docker
docker image build --shm-size=64g -t <docker_image_name> .
```

### Generate Datset

As mentioned above, we use data from the UTIAS Long-Term Localization and Mapping Dataset. We have written a script that will generate training, validation, and test datasets from the localization data. We sample image pairs and relative poses from the VT&R pose graph. The code for dataset generation is found under the `data` folder. Our code assumes that the data is structured the same way as described on the dataset [website](http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/). 

For the traiing and validation data we sample randomly from the pose graph for a chosen set of runs. For the test data we sample image pairs and relative poses sequentialy for the whole run to emulate localization for each run. We localize each of the chosen test runs to each other.

With our code, we have generated one set of training, testing, and validation datasets using the UTIAS Multiseason data and another set using the UTIAS In-The-Dark data. These can be downloaded using the `download_dataset.sh` script. This stores the data sample ids and ground truth labels (poses) needed in Pytorch. The images from the UTIAS Long-Term Localization and Mapping Dataset still need to be stored one the computer to be available to load.

Alternatively, if wanting to generate new datasets from scratch, this can be done by running a docker container as shown below. The `docker_data.sh` has to be updated with the correct file paths, where place holders are present.

```
cd docker
./docker_data.sh
```

Finally, the code can be run without the use of docker by executing the following command:

```
python -m data.build_train_test_dataset_loc --config <path_to_code>/deep_learned_visual_features/config/data.json
```

### Train

The model can be trained by running the following docker container, provided that updates are made to the file path place holders in `docker_test.sh`:

```
cd docker
./docker_test.sh
```

The command for running the training code without docker is the following:

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
### Configurations

In the `config` folder we provide json files that set the configuration parameters for the different tasks. 

#### Important note
The UTIAS Multiseason and UTIAS In-The-Dark datasets were collected with two different cameras and hence their calibration parameters are different. 

```
UTIAS Multiseason camera parameters:
width: 512
height: 384
focal length: 388.425
cx: 253.502
cy: 196.822
baseline: 0.239946
```
```
UTIAS In-The-Dark camera parameters:
width: 512
height: 384
focal length: 387.777
cx: 257.446
cy: 197.718
baseline: 0.239965
```

#### Generate Dataset

In `config/data.json`, we find the following configurations:

`dataset_name`: give the daaset a name to reference when loading it for training/testing.\
`num_train_samples`: the number of training samples we want to generate.\
`num_validation_samples`: the number of validation samples we want to generate.\
`max_temporal_len`: when localizing live vertices to the map vertices, allow up to this topoligical distance between them.\
`dataset`: image height and width.\
`train_paths`: list of name of paths we want to include in he training data ('multiseason', 'inthedark'). Use the name you have stored the dataset under on your computer, i.e. `/path_to_data/dataset_name/run_000001/...`. \
`test_paths`: list of name of paths we want to include in the test data.   
`sampling_ratios_train`: for each path in the training data, which portion of the total samples should it contribute. If we use two paths ('multiseason', 'inthedark') we could for example sample 0.7 from 'multiseason' and 0.3 from 'intthedark'.\
`sampling_ratios_valid`: same as above for validation data.\
`train_runs`: the runs from the paths to include in the training data.\
`validation_runs`: the runs from the paths to include in the validation data.\
`test_runs`:the runs from the paths to include in the test data.\
`temporal_len`: same topoligical distance as max_temporal_len above. Here we set one fixed distance for each test run (when the specific run is used as the teach/map run).

#### Train

In `config/train.json`, we find the following configurations:

`dataset_name`: name of the dataset to use. \
`checkpoint_name`: name for the checkpoint to store the model, alternatively load and resume training an exsiting model. \
`training`: \
&nbsp;&nbsp;&nbsp;&nbsp;`start_pose_estimation`: we can train initially using only the keypoint loss. This says which epoch to start pose estimation and using the pose loss. \
&nbsp;&nbsp;&nbsp;&nbsp;`max_epochs`: stop training after this many epochs. \
&nbsp;&nbsp;&nbsp;&nbsp;`patience`: how many epochs to run with validation loss not improving before stopping. \
`network`: \
&nbsp;&nbsp;&nbsp;&nbsp;`num_channels`: number of input channels, 3 for one RGB image. \
&nbsp;&nbsp;&nbsp;&nbsp;`num_classes`: number of classes for the output of the decoders, should be 1. \
&nbsp;&nbsp;&nbsp;&nbsp;`layer_size`: the size of the first layer of the encoder. Subsequent layer sizes are determined automatically. \
`pipeline`: height and with of the windows in which we detect a keypoint and a parameter for whether to do dense or sparse matching of descriptors. \
`outlier_rejection`: \
 &nbsp;&nbsp;&nbsp;&nbsp;`type`: outlier rejection is done with ground truth poses during training. \
 &nbsp;&nbsp;&nbsp;&nbsp;`dim`: doing outlier rejection for 2D points, 3D points, or in the plane (only considering x, y, heading). \
 &nbsp;&nbsp;&nbsp;&nbsp;`error_tolerance`: error threshold to be considered an inlier. \
`data_loader`: condifgurations for the data loader. \
`stereo`: camera calibration parameters. \
`dataset`: parameters for the dataset such as image heigh, width and whether to normalize the images and include disparity. \
`optimizer`: configurations for the optimizer. \
`scheduler`: configurations for the scheduler. \
`loss`: \
&nbsp;&nbsp;&nbsp;&nbsp;`types`: the loss types to use (`pose`, `pose_plane` (only SE(2)), `keypoint_2D` (error between 2D image coordinates), `keypoint_3D` error between 3D points), `keypoint_plane` (only SE(2)). \
&nbsp;&nbsp;&nbsp;&nbsp;`weights`: weights for the different types of losses.

#### Test

In `config/test.json`, we find the following configurations:

`dataset_name`: name of the dataset to use. \
`checkpoint_name`: name for the model checkpoint to load. \
`network`: \
&nbsp;&nbsp;&nbsp;&nbsp;`num_channels`: number of input channels, 3 for one RGB image. \
&nbsp;&nbsp;&nbsp;&nbsp;`num_classes`: number of classes for the output of the decoders, should be 1. \
&nbsp;&nbsp;&nbsp;&nbsp;`layer_size`: the size of the first layer of the encoder. Subsequent layer sizes are determined automatically. \
`pipeline`: height and with of the windows in which we detect a keypoint and a parameter for whether to do dense or sparse matching of descriptors. \
`outlier_rejection`: \
 &nbsp;&nbsp;&nbsp;&nbsp;`type`: outlier rejection is done with RANSAC during testing. \
 &nbsp;&nbsp;&nbsp;&nbsp;`dim`: doing outlier rejection for 2D points, 3D points, or in the plane (only considering x, y, heading). \
 &nbsp;&nbsp;&nbsp;&nbsp;`inlier_threshold`: minimum ratio of inliers rewuired to stop RANSAC early. \
 &nbsp;&nbsp;&nbsp;&nbsp;`error_tolerance`: error threshold to be considered an inlier. \
 &nbsp;&nbsp;&nbsp;&nbsp;`error_tolerance`: number of iterations to run RANSAC. \
`data_loader`: condifgurations for the data loader. \
`stereo`: camera calibration parameters. \
`dataset`: parameters for the dataset such as image heigh, width and whether to normalize the images and include disparity. \
`optimizer`: configurations for the optimizer. \
`scheduler`: configurations for the scheduler. \
`loss`: \
&nbsp;&nbsp;&nbsp;&nbsp;`types`: we include the pose loss type during testing so we know if we are estimation the full SE(3) pose (`pose`) or only SE(2) (`pose_plane`).

## Experimental Results

We use the learned features in the VT&R system such that we can test them in closed-loop operation on a robot outdoors. The code for VT&R is available open-source [here](https://utiasasrl.github.io/). The experiments are described in detail in our paper. To summarize, we conducted two experiments. In the lighting-change experiment we trained the network using the UTIAS In-The-Dark dataset from 2016 and taught a new path in the same location around noon in August 2021. We repeated the path across a full range of lighting change from 3 a.m. until 10.30 p.m. with a 100% autonomy rate. In the figures below, we show example images from different conditons and a box plot of the inlier feature matches for each run. 

![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/conditions_lighting_change.png?raw=true)
![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/box_plot_lighting_change.png?raw=true)

In the generalization experiment we trained the network using the UTIAS In-The-Dark dataset from 2016 and the UTIAS Multiseason datasets from 2017 and taught a new path in August 2021. In this path we included new areas that have not been seen in he training data to test whether the features can generalize outside of the training paths. We taught the path around 1:30 p.m. and repeated it across a full range of lighting change from 4 a.m. until 9 p.m. with a 100% autonomy rate. In the figures below, we show example images from different conditons and a box plot of the inlier feature matches for each run. 

![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/conditions_generalization.png?raw=true)
![](https://github.com/utiasASRL/deep_learned_visual_features/blob/main/figures/box_plot_generalization.png?raw=true)



