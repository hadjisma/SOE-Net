#ReadMe
This ReadMe file explains how to use the code for trainable SOE-Net
work by Isma Hadji (hadjisma@cse.yorku.ca) 
========================================================================================================================

* This folder contain all necessary functions to build SOE-Net

##Requirements:

* This code has been tested under the following environment settings:
	- cuda 9.0
	- cudnn 7.1
	- tensorflow 1.8
	- opencv 3.4

## Relevant scripts:

- **SOE_Net_model_full.py**
	- All necessary functions to build a trainable SOE_Net are contained in this script.
- **init_SOE_Net.py**
	- All functions to initialize SOE_Net's building blocks are here.
- **input_data.py**
	- All functions used in the input pipeline.
- **util.py**
	- functions to visualize videos and results.
- **SOE_MSOE_SO_TEST.py**
	- full fledged main code with examples to build and extract SOE, MSOE, SO, SOE-NET, MSOE-NET and SO-NET features. 
	It needs as input:
	1) a path to a datasets folder
	2) a dataset folder
	3) a sample video to be used for testing

* This code can be used to extract features to be used with the video segmentation code.

## Helper script:

The **shell_script_helpers** folder contains helper bash scripts that automate the entire process:
- **Video2Clips.sh** is a helper bash script to prepare input data pipeline. (i.e. convert videos to clips that are fed to SOE_Net)
- **SOE_MSOE_demo.sh** calls all necessary functions (including the data preparation scripts **Video2Clips.sh**) to extract the desired features.
	It needs as input:
	1) root_path:  a path to a datasets folder
	2) dataset: a dataset folder
	3) vid_name: a sample video to be used for testing. However, if no video name is provided the script will run over the entire dataset.
* For example, to extract all different features for the birdfall video call the script as: `SOE_MSOE_demo.sh /path/to/data dataset birdfall`



