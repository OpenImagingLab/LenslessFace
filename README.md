# [LenslessFace : An End-to-End Optimized Lensless System for Privacy-Preserving Face Verification](https://openimaginglab.github.io/LenslessFace/)

Code for the paper "An End-to-End Optimized Lensless System for Privacy-Preserving Face Verification".

# Get Started

    git clone https://github.com/OpenImagingLab/LenslessFace.git
    cd LenslessFace
    conda create -n lenslessface python=3.9
    conda activate lenslessface
    pip install -r requirements.txt



# Data
For training, we use the [Asian-Celeb](https://drive.google.com/file/d/1xTVBwoeNWiPS-KbH_6OFV32zME45_Z6q/view?usp=sharing) dataset. 
Tests are conducted on the [LFW](http://vis-www.cs.umass.edu/lfw/) dataset and [FCFD](https://computationalimaging.rice.edu/databases/flatcam-face-dataset/) dataset
, which should be downloaded and extracted to the data directory.

You can modify the arguments in `config_file` to change the dataset path.

# Training
For RGB-based teacher model training, run the following command:
```
./scripts/dist_train_teacher.sh config_file
```
An example of `config_file` is `configs/face_no_optical/rgb_teacher.py`.

For lensless-based student model training, run the following command:
```
./scripts/dist_train.sh config_file
```
An example of `config_file` is `configs/distill/face/base.py`.

For lensless-based face center detection model training, run the following command:
```
./scripts/dist_train_pose.sh config_file
```
An example of `config_file` is `configs/face_center_detection/base.py`.

# Testing
For aligned face verification, run the following command:
```
./scripts/test.sh config_file check_point_path
```
An example of `config_file` is `configs/distil/face/base.py`.

For random face verification, run the following command:
```
./scripts/test_random.sh config_file
```
An example of `config_file` is `configs/hybrid/optical/base_test.py`.
you need to modify the `cls_checkpoint` and `face_center_detection_checkpoint` in the `config_file`.

Acknowledgments
We thank the authors and maintainers of the following repositories for providing the frameworks and datasets that significantly facilitated our research:

* [OpenMMLab](https://github.com/open-mmlab)

* [LenslessClassification](https://github.com/ebezzam/LenslessClassification)
* [LenslessPiCam
](https://github.com/LCAV/LenslessPiCam)

Special thanks also go to the authors of the datasets we used for training and evaluation.

Citation
Please cite our paper if you find this repository useful for your research:

```
@misc{cai2024lenslessface,
      title={LenslessFace: An End-to-End Optimized Lensless System for Privacy-Preserving Face Verification}, 
      author={Xin Cai and Hailong Zhang and Chenchen Wang and Wentao Liu and Jinwei Gu and Tianfan Xue},
      year={2024},
      eprint={2406.04129},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
# License
This project is licensed under the terms of the MIT license.

