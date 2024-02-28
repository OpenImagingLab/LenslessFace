# amplitude mask for training

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Function
from ..builder import OPTICALS
from mmcls.models.opticals import SoftPsfConv
import cv2
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
from waveprop.pytorch_util import fftconvolve
import warnings
@OPTICALS.register_module()   
class LoadPsf(SoftPsfConv):
    def __init__(self, load_psf_path = None, **kwargs):
        super().__init__(**kwargs)
        super().compute_intensity_psf()
        if load_psf_path is not None:
            if load_psf_path.endswith("png") or load_psf_path.endswith("jpg"):
                _psf = cv2.imread(load_psf_path)
                # psf_vals = cv2.cvtColor(psf_vals, cv2.COLOR_BGR2RGB)
                # if psf_vals.shape[-1] == 3:
                #     psf_vals = cv2.cvtColor(psf_vals, cv2.COLOR_BGR2GRAY)
                _psf = _psf.astype(np.float32)
                if _psf.max() > 1:
                    _psf /= 255
                # print(_psf.shape)
                
                if _psf.shape[0] < _psf.shape[1]:
                    # rotate 90
                    _psf = np.rot90(_psf)
                    # flip _psf use opencv
                    _psf = cv2.flip(_psf, 1)
                # print(_psf.shape, self._psf.shape)
                # psf_vals /= 255
                # resize to the size of the self.psf_vals
                _psf = cv2.resize(_psf, (self._psf.shape[2], self._psf.shape[1]))
                _psf = torch.from_numpy(_psf).cuda()
                # from [H, W, C] to [C, H, W]
                _psf = _psf.permute(2,0,1)
                # print(_psf.shape, self._psf.shape)

                self._psf = _psf
                self._psf_saved = _psf
        
    def compute_intensity_psf(self, psf_vals=None):
        pass
    

    def forward(self, x, affine_matrix = None):
        #print("psf input shape ", x.shape)
        if x.min() < 0:
            warnings.warn("Got negative data. Shift to non-negative.")
            x -= x.min()
        self.before_optical = x.detach()
        # print(" self.psf_val.grad is not None ", self.psf_vals.grad)
        # compute intensity PSF from psf values
        self._psf = self._psf_saved
        self.psf_val_to_save = self.get_psf_val_to_make_mask(self.psf_vals)
        #print("psf shape", self._psf.shape)
        if self.n_psf_mask > 1:
            # add dimension for time-multiplexed measurements
            x = x.unsqueeze(1)
            
    
        # convolve with PSF
        if self.do_optical:
            x = fftconvolve(x, self._psf, axes=(-2, -1))
        #print("output shape: ", x.shape)
        if self.n_psf_mask > 1:
            # consider time-multiplexed as more channels
            x = x.flatten(1, 2)

        if self.add_noise is not None:
            x = self.add_noise(x)

     
        # if self.resize is not None:
        #     x = self.resize(x)

        # if self.resize is not None:
        #     x = self.resize(x)
        # make sure non-negative
        x = torch.clip(x, min=0)
    
        self.after_optical = x.detach()
        if self.use_stn:
            x = self.stn(x)
        # if affine_matrix is not None and use_stn is false, use affine_matrix to warp the image
        # elif affine_matrix is not None and not self.do_optical:
        #     grid = F.affine_grid(affine_matrix, x.size(),align_corners=False)
        #     x = F.grid_sample(x, grid,align_corners=False)
        elif self.do_affine:
            # apply translate and its inverse transform
            # only do the translation of affine matrix
            if affine_matrix is not None:
                affine_matrix_ = affine_matrix.clone()
                affine_matrix_[:,0,0] = 1
                affine_matrix_[:,0,1] = 0
                affine_matrix_[:,1,0] = 0
                affine_matrix_[:,1,1] = 1
                grid = F.affine_grid(affine_matrix_, x.size(), align_corners=False)
                x = F.grid_sample(x, grid, align_corners=False)
            # print("hello!")

        if self.center_crop is not None:
            x = self.center_crop(x)
        if self.downsample is not None:
            x = self.downsample(x)
        self.after_affine = x.detach()
 
        if self.return_measurement:
            return x
        # normalize after PSF 
        x = self.normalize(x)
        # normalize after PSF
        # x = self.conv_bn(x)
        if self.sensor_activation is not None:
            x = self.sensor_activation(x)
        # x = x.reshape(x.shape[0], x.shape[1] * self.split_psf[0] * self.split_psf[1], x.shape[2] // self.split_psf[0], x.shape[3] // self.split_psf[1])

        return x