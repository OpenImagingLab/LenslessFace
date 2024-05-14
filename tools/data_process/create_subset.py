#create a blance subset of the dataset for training
import os
import random
import shutil

source_dir = 'data/OrderAffectNet/train'
target_dir = 'data/OrderAffectNet/train_subset'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
for label in os.listdir(source_dir):
    img_list = os.listdir(os.path.join(source_dir, label))
    random.shuffle(img_list)
    if len(img_list) > 3800:    
        img_list = img_list[:3800]
    else:
        print(f'{label} has less than 3800 images.')

    if not os.path.exists(os.path.join(target_dir, label)):
        os.makedirs(os.path.join(target_dir, label))
    for img in img_list:
        shutil.copy(os.path.join(source_dir, label, img), os.path.join(target_dir, label, img))
# The subset is created in 'data/OrderAffectNet/train_subset'. You can use the following code to check the number of images in the subset.