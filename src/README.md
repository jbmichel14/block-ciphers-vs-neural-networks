# Src folder

## Source
The code in this folder was implemented for the semester project _Attacking Pseudo-randomness with Deep Learning_ by Jean-Baptiste Michel.

It contains the implementation of the framework (data generation and deep learning pipeline), as well as experiments and saved models.


## Overview

| File/Directory&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Explanation                                                                                                                                                  |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset | Datasets for Round-Reduced Ciphers |
| dataset_rr | This folder contains the code for round-reduced ciphers, Speck especially. Source: <https://github.com/differential-neural/An-Assessment-of-Differential-Neural-Distinguishers> |
| attacks-key-recovery | Folder with notebooks with experiments for key recovery attacks. |
| attacks-aes | Folder with notebooks with models for AES plaintext recovery attacks. |
| attacks-s-aes | Folder with notebooks with models for S-AES plaintext recovery attacks. |
| attacks-speck | Folder with notebooks with models for Speck plaintext recovery attacks. |
| saved_models | Folder with saved models in .h5 format. |
| attack_template.ipynb | Notebook that serves as a template to train neural networks for any type of attack. |
| pipeline.py | Training/testing framework implementation and useful methods. |


