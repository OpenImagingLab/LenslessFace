# given a image dataset do the center crop for the image
import os
import cv2
from tqdm import tqdm



source_imgs_path = "/root/caixin/data/lfw/lfw-deepfunneled-single"
target_dir = "/root/caixin/data/lfw/lfw-deepfunneled-172x172-single"
os.makedirs(target_dir, exist_ok=True)
for img_path in tqdm(os.listdir(source_imgs_path)):
    if img_path.endswith('.png') or img_path.endswith('.jpg'):
        source_img_path = os.path.join(source_imgs_path, img_path)
        print(source_img_path)
        img = cv2.imread(source_img_path)
        # the original size of the image is 250x250
        # the size of the center crop is 172x172
        img = img[39:211, 39:211]
        cv2.imwrite(os.path.join(target_dir, img_path), img)

