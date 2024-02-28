# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any

import mmcv
import numpy as np

from ..builder import PIPELINES
import torch.distributed as dist


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

        # try:
        #     img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        # except:
        #     with open('debug_{}.txt'.format(dist.get_rank()), 'a') as f:
        #         f.write(filename+'\n')
        #     print(filename)
        #     img = np.zeros([400, 400, 3]).astype(np.float32)

        if self.to_float32:
            img = img.astype(np.float32)

        results['image_file'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()    
class AddFaceCenter(object):
    #add face center info for face center estimation
    def __init__(self, ann_info = None, point=[0.5,0.5], scale_std = 200):
        if ann_info is None:
           self.ann_info =  {'image_size': np.array([128, 128]), 'heatmap_size': np.array([64, 64]), 'num_joints': 1, 'inference_channel': [0], 'num_output_channels': 1, 'dataset_channel': [[0]], 'use_different_joint_weights': False, 'flip_pairs': [], 'flip_index': [0], 'upper_body_ids': [], 'lower_body_ids': [], 'joint_weights': np.array([[1.]], dtype=np.float32), 'skeleton': []}
        else:
            self.ann_info = ann_info
        self.point = point
        self.scale_std = scale_std
    def __call__(self, results):
        results['ann_info'] = self.ann_info
        results['center'] = [self.point * results['img_shape'][1], self.point * results['img_shape'][0]]
        results['scale'] = [results['img_shape'][1] / self.scale_std, results['img_shape'][0] / self.scale_std]
        results['rotation'] = 0
        results['joints_3d'] = np.array([self.point[0] * results['img_shape'][1], self.point[1] * results['img_shape'][0], 0], dtype=np.float32).reshape(1, 3)
        results['joints_3d_visible'] = np.array([1, 1, 0], dtype=np.float32).reshape(1, 3)
        results['dataset'] = 'face_center'
        results['bbox'] = [0, 0, results['img_shape'][1], results['img_shape'][0]]
        results['bbox_score'] = 1
        results['box_size'] = results['img_shape'][1]
        return results

@PIPELINES.register_module()
class LoadImagePair(object):

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        imgfile1 = results['imgfile1']
        imgfile2 = results['imgfile2']

        img_bytes = self.file_client.get(imgfile1)
        img1 = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img1 = img1.astype(np.float32)

        img_bytes = self.file_client.get(imgfile2)
        img2 = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img2 = img2.astype(np.float32)

        results['img1'] = img1
        results['img2'] = img2
        results['img_fields'] = ['img1', 'img2']
        return results
