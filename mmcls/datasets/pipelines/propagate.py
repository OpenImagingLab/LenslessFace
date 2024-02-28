import os.path as osp
import torch
import numpy as np

from ..builder import PIPELINES
from mmcls.devices import *
from lensless.io import load_psf
from lensless.util import resize, rgb2gray
import cv2
from torchvision import transforms
from lenslessutils import RealFFTConvolve2D,AddPoissonNoise
from skimage.util.noise import random_noise
from torch import nn
from scipy import ndimage
import copy

class AddNoise:
    def __init__(self, snr, noise_type, noise_mean, dtype):
        self.snr = snr
        self.noise_type = noise_type
        self.noise_mean = noise_mean
        self.dtype = dtype

    def __call__(self, measurement):
        # sig_var = np.linalg.norm(measurement)

        # if measurement.max() > 1:
        # normalize to 1 as mean is also normalized like so
        measurement /= measurement.max()

        measurement_np = measurement.cpu().numpy()

        sig_var = ndimage.variance(measurement_np)
        noise_var = sig_var / (10 ** (self.snr / 10))

        noisy = random_noise(
            measurement_np,
            mode=self.noise_type,
            clip=False,
            mean=self.noise_mean,
            var=noise_var,
        )
        return torch.tensor(np.array(noisy).astype(self.dtype)).to(measurement)

@PIPELINES.register_module()
class Propagated(object):

    def apply(self, img, do_pad=True):
        # if self.use_cuda:
        #     img = img.cuda()
        # print("img.shape", img.shape)
        #from GBR to RGB
        # 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if do_pad:
            img = self._transform(img)
        else:
            img = self._transform_no_pad(img)
            
        if self._transform_post:
            img = self._transform_post(img)
        # print(self._transform, self._transform_post)
        # img = transforms.ToTensor()(np.array(img))
        # cast to uint8 as on sensor
        if self.use_max_range or img.max() > 1:
            img /= img.max()
        img *= self.max_val
        img = img.to(dtype=self.dtype_out)
        return img
    
    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            results[key+ "_nopad"] = self.apply(img, do_pad=False)
            
            results[key] = self.apply(img)
      


            results['%s_shape'%key] = results[key].shape
            # results['ori_shape'] = results['img'].shape
        
        return results

    def __init__(
        self,
        object_height,
        scene2mask,
        mask2sensor,
        sensor,
        input_dim,
        psf_fp=None,
        single_psf=False,
        downsample_psf=None,
        output_dim=None,
        crop_psf=False,
        vflip=False,
        grayscale=False,
        use_cuda=True,
        scale=(1, 1),
        max_val=1,
        use_max_range=True,
        dtype=np.float32,
        # dtype_out=torch.uint8,  # simulate quantization of sensor
        dtype_out=torch.float32,
        noise_type=None,
        snr=40,
        down="resize",
        random_shift=False,
        random_height=None,
        rotate=False,
        perspective=False,
        keys = ['img'],
        **kwargs,
    ):

        self.dtype = dtype
        self.dtype_out = dtype_out
        self.max_val = max_val
        self.use_max_range = use_max_range
        self.use_cuda = use_cuda
        self.keys = keys    
        self.object_height = object_height
        self.input_dim = np.array(input_dim[:2])
        sensor_param = sensor_dict[sensor]
        sensor_size = sensor_param[SensorParam.SIZE]
     
        # -- load PSF
        if psf_fp is not None:
            psf = load_psf(fp=psf_fp, single_psf=single_psf, dtype=dtype)

            if crop_psf:
                # for compact support PSF like lens
                # -- keep full convolution
                self.conv_dim = np.array(psf.shape)

                # -- crop PSF around peak
                center = np.unravel_index(np.argmax(psf, axis=None), psf.shape)
                top = int(center[0] - crop_psf / 2)
                left = int(center[1] - crop_psf / 2)
                psf = psf[top : top + crop_psf, left : left + crop_psf]

            else:
                # for PSFs with large support, e.g. lensless
                if downsample_psf:
                    psf = resize(psf, 1 / downsample_psf, interpolation=cv2.INTER_CUBIC).astype(
                        dtype
                    )
                    if single_psf:
                        # cv2 drops the last dimension when it's 1..
                        psf = psf[:, :, np.newaxis]
                self.conv_dim = np.array(psf.shape)

            # reorder axis to [channels, width, height]
            if grayscale and not single_psf:
                # convert PSF to grayscale
                psf = rgb2gray(psf).astype(dtype)
                psf = psf[np.newaxis, :, :]
                self.conv_dim[2] = 1
            else:
                psf = np.transpose(psf, (2, 0, 1))

            # cast as torch array
            if self.use_cuda:
                psf = torch.tensor(psf).cuda()
            else:
                psf = torch.tensor(psf)
        else:

            # No PSF, output dimensions is same as dimension (before) convolution
            psf = None
            assert output_dim is not None
            self.conv_dim = np.array(output_dim)

        self.psf = psf

        # processing steps
        # -- convert to tensor and flip image if need be
        transform_list = [np.array, transforms.ToTensor()]
        transform_list_no_pad = [np.array, transforms.ToTensor()]
        # transform_list_no_pad.append(transforms.CenterCrop((112,96)))
        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=1.0))
            transform_list_no_pad.append(transforms.RandomVerticalFlip(p=1.0))

        # -- resize to convolution dimension and scale to desired height at object plane
        magnification = mask2sensor / scene2mask
        self.scene_dim = sensor_size / magnification

        if random_height is not None:
            # TODO combine with shifting which needs to know padding
            assert len(random_height) == 2
            assert random_height[0] <= random_height[1]

            def random_scale(image):

                object_height = np.random.uniform(low=random_height[0], high=random_height[1])
                object_height_pix = int(
                    np.round(object_height / self.scene_dim[1] * self.conv_dim[1])
                )
                scaling = object_height_pix / self.input_dim[1]
                object_dim = (np.round(self.input_dim * scaling)).astype(int).tolist()
                image = transforms.Resize(size=object_dim)(image)

                # -- pad rest with zeros
                padding = self.conv_dim[:2] - object_dim
                left = padding[1] // 2
                right = padding[1] - left
                top = padding[0] // 2
                bottom = padding[0] - top
                image = transforms.Pad(padding=(left, top, right, bottom))(image)
                return image

            transform_list.append(random_scale)
        else:
            assert isinstance(object_height, float)
            object_height_pix = int(np.round(object_height / self.scene_dim[1] * self.conv_dim[1]))
            # print(self.scene_dim,self.input_dim)
            scaling = object_height_pix / self.input_dim[1]
            object_dim = (np.round(self.input_dim * scaling)).astype(int).tolist()
            # print("object_dim", object_dim)
            # print("self.input_dim", self.input_dim)
            # print("object_height", object_height)
            # print("scene_dim", self.scene_dim)
            # print("object_height_pix", object_height_pix)
            # print()
            transform_list.append(
                transforms.Resize(size=tuple(object_dim))
            )
            transform_list_no_pad.append(
                transforms.Resize(size=tuple(self.conv_dim[:2]))
            )
            # -- pad rest with zeros
            padding = self.conv_dim[:2] - object_dim
            # print(self.conv_dim)
            left = padding[1] // 2
            right = padding[1] - left
            top = padding[0] // 2
            bottom = padding[0] - top
            transform_list.append(transforms.Pad(padding=(left, top, right, bottom)))

        if rotate:
            # rotate around center
            transform_list.append(transforms.RandomRotation(degrees=rotate))

        if perspective:
            transform_list.append(transforms.RandomPerspective(distortion_scale=perspective, p=1.0))

        # -- random shift
        if random_shift:

            assert (
                not random_height
            ), "Random height not supported with random shift, need padding info"

            def shift_within_sensor(image):
                hshift = int(np.random.uniform(low=-left, high=right))
                vshift = int(np.random.uniform(low=-bottom, high=top))
                return torch.roll(image, shifts=(vshift, hshift), dims=(1, 2))

            transform_list.append(shift_within_sensor)

        if grayscale:
            if input_dim[2] != 1:
                # convert to grayscale
                transform_list.append(transforms.Grayscale(num_output_channels=1))
        else:
            if input_dim[2] == 1:
                # 2D image so repeat on all channels
                transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1))) 

        self._transform = transforms.Compose(transform_list)
        self._transform_no_pad = transforms.Compose(transform_list_no_pad)
        # -- to do convolution on GPU (must faster)
        self._transform_post = None
        if self.psf is not None:
            self._transform_post = []

            conv_op = RealFFTConvolve2D(self.psf, img_shape=np.roll(self.conv_dim, shift=1))
            self._transform_post.append(conv_op)

            # -- resize to output dimension
            if output_dim is not None:
                if down == "resize":
                    self._transform_post.append(transforms.Resize(size=output_dim))
                elif down == "max" or down == "avg":
                    hidden = np.prod(output_dim)

                    # determine filter size, stride, and padding: https://androidkt.com/calculate-output-size-convolutional-pooling-layers-cnn/
                    k = int(np.ceil(np.sqrt(np.prod(self.conv_dim) / hidden)))
                    p = np.roots(
                        [4, 2 * np.sum(self.conv_dim), np.prod(self.conv_dim) - k**2 * hidden]
                    )
                    p = max(int(np.max(p)), 0) + 1
                    if down == "max":
                        pooler = nn.MaxPool2d(kernel_size=k, stride=k, padding=p)
                    else:
                        pooler = nn.AvgPool2d(kernel_size=k, stride=k, padding=p)
                    self._transform_post.append(pooler)
                else:
                    raise ValueError("Invalid downsampling approach.")

            if noise_type:
                if noise_type == "poisson":
                    transform_list.append(AddPoissonNoise(snr))
                else:
                    if sensor == SensorOptions.RPI_HQ.value:
                        bit_depth = 12
                        noise_mean = RPI_HQ_CAMERA_BLACK_LEVEL / (2**bit_depth - 1)
                    else:
                        noise_mean = 0
                    transform_list.append(AddNoise(snr, noise_type, noise_mean, dtype))

            self._transform_post = transforms.Compose(self._transform_post)
