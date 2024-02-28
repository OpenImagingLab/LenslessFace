# add background for face images (torch tensor)
import os.path as osp
import cv2
from ..builder import PIPELINES
import torch.nn.functional as F
import numpy as np
import os
import random
from torchvision import transforms
import torch
# add background on the center of img 
@PIPELINES.register_module()
class AddBackground(object):
    def __init__(self, img_dir, size = (200,200), prob=1.0, ratio=(0.8, 1.2), scale=(0.3, 2),is_tensor=True):
        self.img_dir = img_dir
        self.prob = prob
        self.size = size
        self.ratio = ratio
        self.scale = scale
        self.is_tensor = is_tensor
        self.background_files = os.listdir(img_dir)
        # load 3000 background images
        if len(self.background_files) > 1500:
            self.background_files = self.background_files[:1500]
        self.background_files = [cv2.imread(osp.join(img_dir, f)) for f in self.background_files]
    # RandomResizedCrop for images read from opencv
    def _transform(self, img):
        cropped_size = self.size
        ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])
        img_size = img.shape[:2]
        # random crop
        w = int(scale * cropped_size[0])
        h = int(scale * cropped_size[1])
        tw = int(w * ratio)
        th = int(h * ratio)
        i = np.random.randint(0, img_size[0] - th)
        j = np.random.randint(0, img_size[1] - tw)
        img = img[i:i + th, j:j + tw, :]
        # resize
        img = cv2.resize(img, cropped_size)
        return img
  
    def __call__(self, results):
     
        for key in results.get('img_fields', ['img']):
            #random choose a background
            background = random.choice(self.background_files)
            background = self._transform(background)
            if self.is_tensor:
                background = transforms.ToTensor()(background)
        # if random.random() < self.prob:
            img = results[key]
            results[key + "_wobg"] = img
            # img = results['img']
            # img_size = img.shape[:2]
            # assert img_size greater than background size
            # assert img_size[0] >= self.size[0] and img_size[1] >= self.size[1]
            # img center crop with self.size
            if self.is_tensor:
                img_center_crop = img[:,(img.shape[1] - self.size[1])//2:(img.shape[1] - self.size[1])//2 + self.size[1], (img.shape[2] - self.size[0])//2:(img.shape[2] - self.size[0])//2 + self.size[0]]
                img_center_crop = torch.where(img_center_crop == 0, background, img_center_crop)
                img[:,(img.shape[1] - self.size[1])//2:(img.shape[1] - self.size[1])//2 + self.size[1], (img.shape[2] - self.size[0])//2:(img.shape[2] - self.size[0])//2 + self.size[0]] = img_center_crop

            else:
                img_center_crop = img[(img.shape[1] - self.size[1])//2:(img.shape[1] - self.size[1])//2 + self.size[1], (img.shape[2] - self.size[0])//2:(img.shape[2] - self.size[0])//2 + self.size[0],:]
                img_center_crop = np.where(img_center_crop == 0, background, img_center_crop)
                img[(img.shape[1] - self.size[1])//2:(img.shape[1] - self.size[1])//2 + self.size[1], (img.shape[2] - self.size[0])//2:(img.shape[2] - self.size[0])//2 + self.size[0],:] = img_center_crop

        return results