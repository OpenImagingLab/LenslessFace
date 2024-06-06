# LenslessFace

Code for the paper "An End-to-End Optimized Lensless System for Privacy-Preserving Face Verification".

# Get Started

    git clone https://github.com/
    cd LenslessFace
    conda create -n lenslessface python=3.9
    conda activate lenslessface
    pip install -r requirements.txt

# Data
Training dataset: [Asian-Celeb](https://drive.google.com/file/d/1xTVBwoeNWiPS-KbH_6OFV32zME45_Z6q/view?usp=sharing) dataset. 

Test dataset: [LFW](http://vis-www.cs.umass.edu/lfw/) dataset and [FCFD](https://computationalimaging.rice.edu/databases/flatcam-face-dataset/) dataset.

All datasets are downloaded and extracted to the `data` directory.
You can modify the arguments in `config_file` to change the dataset path.

# Training
For RGB-based teacher model training, run the following command:
```
./scripts/dist_train_teacher.sh config_file
```
An example of `config_file` is `configs/face_no_optical/rgb_teacher.py`.

For lenseless-based student model training, run the following command:
```
./scripts/dist_train.sh config_file
```
An example of `config_file` is `configs/distill/face/base.py`.

For lensless-based face center detection model training, run the following command:
```
./scripts/dist_train_pose.sh config_file
```
An example of `config_file` is `configs/pose/base.py`.

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
you need to modify the `cls_checkpoint` and `pose_checkpoint` in the `config_file`.


