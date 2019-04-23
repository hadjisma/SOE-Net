This ReadMe file explains where to find relevant code to use SOE-Net
Work done by Isma Hadji (hadjisma@cse.yorku.ca) 
If you use any part of this code please cite our paper:

@inproceedings{Hadji2017,
author = {I. Hadji and R. P. Wildes},
title = {A Spatiotemporal Oriented Energy Network for Dynamic Texture Recognition},
booktitle = {ICCV},
year = {2017},
}
========================================================================================================================

* This folder contain all necessary functions to build SOE-Net

Document:
=========
Please refer to our ICCV 2017 paper for detailed description of the proposed building blocks.

Requirements:
============
* This code has been tested under the following environment settings:
	- cuda 9.0
	- cudnn 7.1
	- tensorflow 1.8
	- opencv 3.4

Relevant scripts:
=================

* SOE_Net_model_full.py
	- All necessary functions to build the SOE_Net architecture are contained in this script.
* init_SOE_Net.py
	- All functions to initialize SOE_Net's building blocks are here.
* util.py
	- functions to visualize videos and results.
* configure.py
	- just a place to hold all global variables.


