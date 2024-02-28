# transfer affine matrix to label for face center detection (shift and scale)

import os.path as osp
import cv2
from ..builder import PIPELINES
import torch.nn.functional as F
import numpy as np
import os
import random
from torchvision import transforms
import torch
@PIPELINES.register_module()
class Affine2label(object):
    def __init__(self):
        pass
    def __call__(self, results):
        results["target_weight"] = np.array([[1.0]])
        if "affine_matrix" not in results:
            results["target"] = np.array([0.5,0.5]).reshape(1,2)
            return results
        affine_matrix = results["affine_matrix"]
        # print(affine_matrix.shape)
        translate = affine_matrix[:,2]
        # print("translate", translate)

        # translate_0 = translate[0] * results["img"].shape[2] / 2 / 128
        # translate_1 = translate[1] * results["img"].shape[1] / 2 / 128
        translate_0 = translate[0] 
        translate_1 = translate[1]
        # print("translate_0, translate_1",translate_0, translate_1) 

        results["target"] = np.array([0.5 + translate_0 ,0.5 + translate_1]).reshape(1,2)
        # trans target and target_weight to tensor
        # results["target_np"] = results["target"]
        results["target"] = torch.from_numpy(results["target"]).float()
        results["target_weight"] = torch.from_numpy(results["target_weight"]).float()
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'img_norm_cfg={self.img_norm_cfg})'
        return repr_str