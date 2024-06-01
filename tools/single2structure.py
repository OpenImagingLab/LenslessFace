

import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm

def restore_original_structure(path_dict_file, target_folder, restore_folder):
    with open(path_dict_file, 'r') as f:
        path_dict = json.load(f)
    file_list = os.listdir(target_folder)
    #filter all file not start with MyPic
    # file_list = [x for x in file_list if x.startswith('MyPic')]
    # the name is like 
    # MyPicFast_exp18000_index0-23286490-0
    # MyPic_exp40000_index13231-23286490-0, 
    # file_list.sort(key = lambda x: int(x.split('-')[0].split('index')[1]))
    file_list.sort(key = lambda x: int(x.split('_')[0]))
    for original_path, (current_path, unique_number) in path_dict.items():
        original_path = os.path.join(restore_folder, original_path)
        current_path = os.path.join(target_folder, file_list[unique_number])
        print(original_path, current_path)
        # break
        if not os.path.exists(original_path):
            os.makedirs(os.path.dirname(original_path), exist_ok=True)
        shutil.copy2(current_path, original_path)

if __name__ == '__main__':
    path_dict_file = "/root/caixin/data/lfw/lfw-112X96-single/path_dict.json"
    target_image_folder = '/root/caixin/data/lfw/lfw-deepfunneled-recon-single'
    restore_folder = "/mnt/caixin/RawSense/data/lfw/recon"  # The folder to restore the original structure
    os.makedirs(restore_folder, exist_ok=True)
    # path_dict_file = os.path.join(target_folder, path_dict_file)
    restore_original_structure(path_dict_file, target_image_folder, restore_folder)

 