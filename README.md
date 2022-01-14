# Implementation of online CGMM-MVDR beamforming
This repository implements online CGMM clustering and MVDR beamforming based on [Frame-by-Frame Closed-Form Update for Mask-Based Adaptive MVDR Beamforming](https://ieeexplore.ieee.org/document/8461850) and [Online MVDR Beamformer Based on Complex Gaussian Mixture Model With Spatial Prior for Noise Robust ASR](https://ieeexplore.ieee.org/document/7845594). Please cite these papers if you use this code. Note that this is not an official implementation of the above papers. 
The code was used in the [BUT system for Clarity challenge](https://claritychallenge.github.io/clarity2021-workshop/papers/Clarity_2021_CEC1_paper_final_zmolikova.pdf).

## Requirements
To use the CGMM-MVDR module itself, only `numpy` is needed. For running the provided notebook with example, these libraries are additionally required:
```
soundfile
scipy
matplotlib
```

## Content
The algorithm is implemented in the class `OnlineCGMMMVDR` in `cgmm_mvdr.py`. For an example how to use it, please see the Jupyter notebook `Example.ipynb`.

