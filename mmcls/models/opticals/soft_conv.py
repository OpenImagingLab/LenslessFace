# amplitude mask for training

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Function
from ..builder import OPTICALS
from mmcls.models.opticals import PsfConv
from mmcls.devices import SensorParam
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale

class BinarizeFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # x = (x + 1.0) / 2.0
        cliped_weights = x.clamp(min=0, max=1)
        binary_weights_no_grad = torch.round(x)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # x = x.clamp(min=0, max=1)
        return binary_weights

    @staticmethod
    def backward(ctx, grad_x):
        return grad_x
    
@OPTICALS.register_module()   
class SoftPsfConv(PsfConv):
    def __init__(self, feature_size = 3e-5, mask_kernel_shape = None, **kwargs):
        self.binarize = BinarizeFunction.apply
        self.feature_size = feature_size
        self.mask_kernel_shape = mask_kernel_shape
        self.use_binary = kwargs.get("binary", False)
        super().__init__(**kwargs)

    def init_psf_vals(self):
        rand_func = torch.rand
        self.feature_size = np.array(self.feature_size)
        if self.feature_size.shape == 1:
            self.feature_size = np.array([self.feature_size, self.feature_size])

        val_shape = self.sensor_config[SensorParam.SHAPE] * self.sensor_config[SensorParam.PIXEL_SIZE] / self.feature_size 
        val_shape = np.array(val_shape).astype(np.int)

        if self.mask_kernel_shape is not None:
            if isinstance(self.mask_kernel_shape, int):
                self.mask_kernel_shape = [self.mask_kernel_shape]
            padding_masks = [] 
            for mask_kernel_shape in self.mask_kernel_shape:
                mask_kernel_shapes = [mask_kernel_shape,mask_kernel_shape]
                # split padding_mask with self.split
           
                val_shape_split =  val_shape // self.split_psf
                # val_shape_split = np.array(val_shape_split).astype(np.int)
                padding_mask = np.zeros(val_shape)
                for i in range(self.split_psf[0]):
                    for j in range(self.split_psf[1]):
                        if val_shape_split[0] < mask_kernel_shapes[0] or val_shape_split[1] < mask_kernel_shapes[1]:
                            left = np.array([i,j]) * val_shape_split
                            right = left + val_shape_split
                        else:
                            left = (val_shape_split - mask_kernel_shape) // 2 + np.array([i,j]) * val_shape_split
                            right = left + mask_kernel_shapes
                        padding_mask[left[0]:right[0], left[1]:right[1]] = 1
                padding_masks.append(padding_mask)
        else:
            padding_masks = [np.ones(val_shape)]
        padding_masks = torch.concat([torch.from_numpy(padding_mask).float().cuda().unsqueeze(0) for padding_mask in padding_masks], dim=0)

        num_mask = padding_masks.shape[0]
          

        if self.n_psf_mask == 1:
            self.psf_vals = rand_func(num_mask, val_shape[0], val_shape[1], dtype=self.dtype, 
                requires_grad=self.requires_grad, device="cuda")
            # self.psf_vals = psf_vals
            if self.requires_grad:
                self.psf_vals = nn.Parameter(self.psf_vals)

                # self.psf_vals.retain_grad()
        else:
            self.psf_vals = [
                rand_func(num_mask, val_shape[0], val_shape[1], dtype=self.dtype,
                requires_grad=self.requires_grad,device="cuda")
            for _ in range(self.n_psf_mask)]
            # for psf_vals in self.psf_vals:
            #     psf_vals = psf_vals * padding_masks
            #     self.psf_vals.append(psf_vals)
            if self.requires_grad:
                self.psf_vals = [nn.Parameter(psf_vals) for psf_vals in self.psf_vals]
            
                # for psf_vals in self.psf_vals:
                #     psf_vals.retain_grad()
        self.padding_masks = padding_masks
        return self.psf_vals


    def get_mask_size(self):
        # the size of sensor
        return self.sensor_config[SensorParam.SHAPE] * self.sensor_config[SensorParam.PIXEL_SIZE] 
                
    def get_psf_mask(self, psf_vals,
    mask_dim = None):
        # for n_psf_mask == 1, psf_vals is a tensor of shape [num_mask, h, w]
        psf_vals = psf_vals * self.padding_masks
        psf_vals = psf_vals.unsqueeze(0)
        psf_vals = psf_vals - psf_vals.min()
        psf_vals = psf_vals  / (psf_vals.max() + 1e-6)
        if mask_dim is None:
            mask_dim = self.input_shape[1:]
        psf_vals_ = F.interpolate(psf_vals,
            size=tuple(mask_dim), mode='nearest-exact')
        # psf_vals_detach = psf_vals.clone().detach() 
        psf_vals_ = psf_vals_ - psf_vals_.min()
        psf_vals_ = psf_vals_  / (psf_vals_.max() + 1e-6)
        # psf_vals = F.sigmoid(psf_vals)
        psf_vals_ = psf_vals_.squeeze(0)
    
        psf_vals_ = torch.sum(psf_vals_, dim=0)
        mask = psf_vals_.repeat(3, 1).reshape(3,psf_vals_.shape[0],psf_vals_.shape[1])

        if self.use_binary:
            mask = self.binarize(mask)       
        # mask = mask.cuda()
        return mask
    def get_psf_val_to_make_mask(self, psf_vals):
        psf_vals = psf_vals * self.padding_masks
        psf_vals = psf_vals - psf_vals.min()
        psf_vals = psf_vals  / (psf_vals.max() + 1e-6)
        if self.use_binary:
            psf_vals = self.binarize(psf_vals)  
        return psf_vals
   
        
    def compute_intensity_psf(self, psf_vals=None):
        # print("!!!!!!")
        if psf_vals is None:
            psf_vals = self.psf_vals
       
        mask = self.get_psf_mask(psf_vals=psf_vals)

       
        self.mask = mask
        # if self.use_binary:
        #     self._psf = self.binarize(mask)
        # else:
        self._psf = mask

        # psfs = mask / torch.norm(mask.flatten())
        # if self.grayscale:
        #     self._psf = rgb_to_grayscale(torch.square(torch.abs(psfs)))
        # else:
        #     self._psf = torch.square(torch.abs(psfs))