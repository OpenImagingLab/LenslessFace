from torch import nn
import torch.nn.functional as F
import torch
import warnings
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale, resize
import numpy as np
from waveprop.pytorch_util import fftconvolve
from waveprop.spherical import spherical_prop
from waveprop.color import ColorSystem
from waveprop.rs import angular_spectrum
from waveprop.fresnel import fresnel_conv
from scipy import ndimage
from mmcls.devices import SensorParam, SensorOptions, sensor_dict
import cv2
from ..builder import OPTICALS
from ..utils import STN

class AddPoissonGaussianNoise:
  
    # expected_light_intensity is the max value of the light intensity, which is corresponding to the face image shape
    def __init__(self, tol=1e-4, expected_light_intensity = 6400, noise_ratio = 1.0):
        self.tol = tol
        self.expected_light_intensity = expected_light_intensity
        self.noise_ratio = noise_ratio

    def __call__(self, measurement):
            # for batches
        # ensure non-negative
        if measurement.min() < 0:
            warnings.warn("Got negative data. Shift to non-negative.")
            measurement = torch.clip(measurement, min=0)
        # G = 1 / light_level #
        # F = 9800
        signification_factor = 9800 * 64 / self.expected_light_intensity   # F / G
        measurement_after_poisson_noise = torch.poisson(measurement * signification_factor) / (signification_factor)
        measurement_after_poisson_noise = self.noise_ratio * (measurement_after_poisson_noise - measurement) + measurement
        # ratio = torch.std(measurement_after_poisson_noise) / torch.std(measurement_after_poisson_noise - measurement)
        # print("--------",torch.std(measurement_after_poisson_noise), torch.std(measurement_after_poisson_noise - measurement),ratio,"--------")
        gaussian_noise_std = (2.52 / 9800 * self.expected_light_intensity)

        gaussian_noise = torch.randn_like(measurement) * gaussian_noise_std
   
        measurement_after_noise = measurement_after_poisson_noise + self.noise_ratio * gaussian_noise

        # ratio = torch.std(measurement) / torch.std(measurement_after_noise - measurement)
        # print("--------",torch.std(measurement), torch.std(measurement_after_noise - measurement),ratio,"--------")

        return measurement_after_noise

class AddGaussianNoise:
    # if one grid can generate 2e-, then expected_light_intensity is 5000(grid)
    def __init__(self, tol=1e-4, expected_light_intensity = 2500, noise_ratio = 1.0):
        self.tol = tol
        self.expected_light_intensity = expected_light_intensity
        self.noise_ratio = noise_ratio

    def __call__(self, measurement):
            # for batches

        # ensure non-negative
        # if measurement.min() < 0:
        #     warnings.warn("Got negative data. Shift to non-negative.")
        #     measurement = torch.clip(measurement, min=0)
        #print max, mean of measurement
        # print("max, mean of measurement", torch.max(measurement), torch.mean(measurement))
        signification_factor =  9800 / self.expected_light_intensity 
        gaussian_noise_var = (2.52 / signification_factor) ** 2
        light_noise_var = measurement / signification_factor / 64
        gaussian_noise_var = gaussian_noise_var + light_noise_var
        gaussian_noise = torch.randn_like(measurement) * torch.sqrt(gaussian_noise_var)

        measurement_after_noise = measurement + self.noise_ratio * gaussian_noise

        # ratio = torch.std(measurement) / torch.std(measurement_after_noise - measurement)
        # print("--------",torch.std(measurement), torch.std(measurement_after_noise - measurement),ratio,"--------")

        return measurement_after_noise
#add gaussian noise with noise level
class AddGaussianNoise2:
    def __init__(self, noise_level = 20):
        self.snr = noise_level
    def __call__(self, measurement):
            # for batches
        # ensure non-negative
        if measurement.min() < 0:
            warnings.warn("Got negative data. Shift to non-negative.")
            measurement = torch.clip(measurement, min=0)
        # avg_intensity = torch.mean(measurement)
        gaussian_noise = torch.randn_like(measurement) * torch.sqrt((10 ** (-self.snr / 10)) * torch.var(measurement))
        measurement_after_noise = measurement + gaussian_noise
        return measurement_after_noise


@OPTICALS.register_module()
class PsfConv(nn.Module):
    """ 
    given sensor_config, crop_fact, scene2mask, mask2sensor,
    do the convolution with  psf
    and training the parameters of psf
    """

    def __init__(
        self,
        input_shape,
        sensor,
        scene2mask,
        mask2sensor,
        crop_fact=1,
        target_dim=None,  # try to get this while keeping aspect ratio of sensor
        down="crop", # "resize", "crop", "max", "avg"
        dtype=torch.float32,
        grayscale=False,
        sensor_activation=None,
        dropout=None,
        noise_type="gaussian",
        noise_ratio = 1.0,
        expected_light_intensity = 6400,
        return_measurement=False,
        requires_grad = True,
        output_dim=None,
        n_psf_mask=1,  # number of psf masks to optimize, would be time-multiplexed
        waveprop = "angular_spectrum", # the method to compute the propagation
        split_psf = [1,1], # split the psf and the after optical image into [w,h] parts
        use_stn = False,
        do_optical = True,
        load_weight_path = None,
        requires_grad_psf = True,
        noise_level = 20,
        **kwargs,
    ):
        """
        grayscale : whether input is grayscale, can then simplify to just grayscale PSF

        """
        super(PsfConv, self).__init__()
        if dtype == torch.float32:
            self.ctype = torch.complex64
        elif dtype == torch.float64:
            self.ctype = torch.complex128
        else:
            raise ValueError(f"Unsupported data type : {dtype}")

        if len(input_shape) == 2:
            input_shape = [1] + list(input_shape)
        assert len(input_shape) == 3

        # store configuration
        self.input_shape = np.array(input_shape)
        
        self.sensor_config = sensor_dict[sensor]
        self.crop_fact = crop_fact
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.grayscale = grayscale
        self.noise_ratio = 1
    
        self.dtype = dtype
        self.target_dim = target_dim
        self.return_measurement = return_measurement
        self.requires_grad = requires_grad  # for psf vals
        self.n_psf_mask = n_psf_mask
        self.last_psf_vals = None
        self.waveprop = waveprop
        self.split_psf = split_psf
        self.use_stn = use_stn
        self.do_optical = do_optical
        self.do_affine  = kwargs.get("do_affine", False)
        self.center_crop_size = kwargs.get("center_crop_size", None)
        # adding noise
        if noise_type:
            if noise_type == "poisson":
                add_noise = AddPoissonGaussianNoise(expected_light_intensity = expected_light_intensity, noise_ratio = noise_ratio)
            elif noise_type == "gaussian":
                add_noise = AddGaussianNoise(expected_light_intensity = expected_light_intensity, noise_ratio = noise_ratio)
            elif noise_type == "gaussian2":
                add_noise = AddGaussianNoise2(noise_level = noise_level)
                # TODO : hardcoded for Raspberry Pi HQ sensor
                # bit_depth = 12
                # noise_mean = RPI_HQ_CAMERA_BLACK_LEVEL / (2**bit_depth - 1) * 250

                # def add_noise(measurement):

                #     # normalize as mean is normalized to max value 1
                #     with torch.no_grad():
                #         max_vals = torch.max(torch.flatten(measurement, start_dim=1), dim=1)[0]
                #         max_vals = max_vals.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                #     # measurement /= max_vals

                #     # compute noise for each image
                #     measurement_np = measurement.clone().cpu().detach().numpy()
                #     noise = []
                #     for _measurement in measurement_np:

                #         # sig_var = ndimage.variance(_measurement)
                #         # noise_var = sig_var / (10 ** (snr / 10)) 
                #         noise_var = noise_mean
                #         noise.append(
                #             random_noise(
                #                 _measurement,
                #                 mode=noise_type,
                #                 clip=False,
                #                 mean=noise_mean,
                #                 var=noise_var,
                #             )
                #             - _measurement
                #         )

                #     noise = torch.tensor(np.array(noise).astype(np.float32)).cuda()

                #     return measurement + noise

            self.add_noise = add_noise
        else:
            self.add_noise = None

        # -- downsampling
        self.downsample = None
        if self.center_crop_size is not None :
            self.center_crop = transforms.CenterCrop(size=self.center_crop_size)
        else:
            self.center_crop = None

        if self.target_dim is not None or output_dim is not None:
            if output_dim is None:
                sensor_size = self.sensor_config[SensorParam.SHAPE]
                w = np.sqrt(np.prod(self.target_dim) * sensor_size[1] / sensor_size[0])
                h = sensor_size[0] / sensor_size[1] * w
                self.output_dim = np.array([int(h), int(w)])
            else:
                self.output_dim = np.array(output_dim)

            if not grayscale:
                self.output_dim = np.r_[self.output_dim, 3]

            if down == "resize":
                self.downsample = transforms.Resize(size=self.output_dim[:2].tolist())
            elif down == "crop":
                self.downsample = transforms.CenterCrop(size=self.output_dim[:2].tolist())
            elif down == "random_crop":
                self.downsample = transforms.RandomCrop(size=self.output_dim[:2].tolist())

            elif down == "max" or down == "avg":
                n_embedding = np.prod(self.output_dim)

                # determine filter size, stride, and padding: https://androidkt.com/calculate-output-size-convolutional-pooling-layers-cnn/
                k = int(np.ceil(np.sqrt(np.prod(self.input_shape[1:]) / n_embedding)))
                p = np.roots(
                    [
                        4,
                        2 * np.sum(self.input_shape),
                        np.prod(self.input_shape[1:]) - k**2 * n_embedding,
                    ]
                )
                p = max(int(np.max(p)), 0) + 1
                if down == "max":
                    self.downsample = nn.MaxPool2d(kernel_size=k, stride=k, padding=p)
                else:
                    self.downsample = nn.AvgPool2d(kernel_size=k, stride=k, padding=p)
                pooling_outdim = ((self.input_shape[1:] - k + 2 * p) / k + 1).astype(int)
                assert np.array_equal(self.output_dim[:2], pooling_outdim)
            else:
                raise ValueError("Invalid downsampling approach.")
        else:
            self.output_dim = self.input_shape

        self.resize = transforms.Resize(size=[self.output_dim[0] // self.split_psf[0] * self.split_psf[0], self.output_dim[1] // self.split_psf[1] * self.split_psf[1]])

        # -- decision network after sensor
        self.sensor_activation = sensor_activation
        self.flatten = nn.Flatten()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.psf_vals = self.init_psf_vals()
        # -- normalize after PSF
        if self.grayscale:
            # self.conv_bn = nn.BatchNorm2d(self.n_psf_mask)
            self.normalize = nn.InstanceNorm2d(self.n_psf_mask)
        else:
            # self.conv_bn = nn.BatchNorm2d(self.n_psf_mask * 3)
            self.normalize = nn.InstanceNorm2d(self.n_psf_mask)

        # -- initialize PSF from psf values and pre-compute constants
        # object to mask (modeled with spherical propagation which can be pre-computed)
        self.color_system = ColorSystem.rgb()
        if grayscale:
            # self.color_system = ColorSystem.gray()
            self.color_system.wv = np.array([550]) * 1e-9
            self.color_system.n_wavelength = 1
        self.d1 = np.array(self.get_mask_size()) / self.input_shape[1:]
        self.spherical_wavefront = spherical_prop(
            in_shape=self.input_shape[1:],
            d1=self.d1,
            wv=self.color_system.wv,
            dz=self.scene2mask,
            return_psf=True,
            is_torch=True,
            dtype=self.dtype,
        )
        self.spherical_wavefront = self.spherical_wavefront.cuda()

        if self.use_stn:
            self.stn = STN()

        #print("self.spherical_wavefront.shape",self.spherical_wavefront.shape)
        self._psf = None
        self._H = None  # pre-compute free space propagation kernel
        self._H_exp = None

        if load_weight_path is not None:
            if load_weight_path.endswith(".pth"):
                checkpoint = torch.load(load_weight_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                sub_model_name = ["architecture.model",'architecture.model.backbone']
                sub_model_state_dict = {}
                for sub_model in sub_model_name:
                    sub_model = sub_model + ".optical."
                    sub_model_state_dict = {k.replace(sub_model, ''): v for k, v in state_dict.items() if k.startswith(sub_model)}
                    if sub_model_state_dict:
                        continue
                if not sub_model_state_dict:
                    print(state_dict.keys())
                assert sub_model_state_dict
                self.load_state_dict(sub_model_state_dict, strict=False)


        if not requires_grad_psf:
            self.psf_vals.requires_grad = False

        psf_vals = self.psf_vals
        self.compute_intensity_psf(psf_vals=psf_vals)



        # for name,parameters in self.named_parameters():
        #     print(name,':',parameters.size())

    def init_psf_vals(self):
        """
        init psf variables for the shape of the sensor and the parameters of the mask
        return PSF variables for computing psf mask
        psf_vals -> psf_mask -> intensity_psf
        """
        return NotImplementedError
   


    def set_psf_vals(self, psf_vals):
        """
        only works if requires_grad = False
        """

        # np.testing.assert_array_equal(psf_vals.shape, self.psf_vals.shape)
        self.psf_vals = psf_vals

        # recompute intensity PSF
        self.compute_intensity_psf()

    def get_psf(self, numpy=False):
        psf_vals = self.psf_vals      
        self.compute_intensity_psf(psf_vals=psf_vals)

        if numpy:
            if self.n_psf_mask == 1:
                return self._psf.cpu().detach().numpy().squeeze()
            else:
                return [_psf.cpu().detach().numpy().squeeze() for _psf in self._psf]
        else:
            return self._psf

    def get_mask_size(self):
        """
        get mask physical size for computing d1
        """
        return NotImplementedError
    
    def set_mask2sensor(self, mask2sensor):
        self.mask2sensor = mask2sensor

        # recompute intensity PSF
        self.compute_intensity_psf()

    def save_psf(self, fp, bit_depth=8):
        psf = self.get_psf(numpy=True)
        if not self.grayscale:
            psf = np.transpose(psf, (1, 2, 0))
        psf /= psf.max()

        # save as int
        psf *= 2**bit_depth - 1
        if bit_depth <= 8:
            psf = psf.astype(dtype=np.uint8)
        else:
            psf = psf.astype(dtype=np.uint16)
        if not self.grayscale:    
            cv2.imwrite(fp, cv2.cvtColor(psf, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(fp, psf)
    def forward(self, x, affine_matrix = None):
        #print("psf input shape ", x.shape)
        if x.min() < 0:
            warnings.warn("Got negative data. Shift to non-negative.")
            x -= x.min()
        self.before_optical = x.detach()
        # print(" self.psf_val.grad is not None ", self.psf_vals.grad)
        # compute intensity PSF from psf values
        psf_vals = self.psf_vals  # apply any (physical) pre-processing
        #print("psf vals shape ", psf_vals.shape)
        self.compute_intensity_psf(psf_vals=psf_vals)

        self.psf_val_to_save = self.get_psf_val_to_make_mask(psf_vals)
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
        self.after_optical = x.detach()
        # normalize after PSF
        # x = self.conv_bn(x)
        if self.sensor_activation is not None:
            x = self.sensor_activation(x)
        # x = x.reshape(x.shape[0], x.shape[1] * self.split_psf[0] * self.split_psf[1], x.shape[2] // self.split_psf[0], x.shape[3] // self.split_psf[1])

        return x

    def get_psf_mask(self, psf_vals,target_dim = None):
        """
        return PSF mask for computing intensity PSF
        """
        return NotImplementedError

    def get_psf_val_to_make_mask(self, psf_vals):
        """
        return PSF val to make mask
        """
        return NotImplementedError
    
    def compute_intensity_psf(self, psf_vals=None):

        if psf_vals is None:
            psf_vals = self.psf_vals

        # assert psf_vals.max() <= 1
        # assert psf_vals.min() >= 0

        if self.n_psf_mask == 1:
            # TODO : backward compatability but can make consistent with multiple masks

            # -- get psf mask, i.e. deadspace modeling, quantization (todo), non-linearities (todo), etc
            mask = self.get_psf_mask(psf_vals=psf_vals)
            self.mask = mask
            # apply mask
            u_in = mask * self.spherical_wavefront
         
            # mask to sensor
            if self._H is None:
                # precompute at very start, not dependent on input pattern, just its shape
                # TODO : benchmark how much it actually saves
                self._H = torch.zeros(
                    [self.color_system.n_wavelength] + list(self.input_shape[1:] * 2),
                    dtype=self.ctype,
                    device=u_in.device,
                )
                for i in range(self.color_system.n_wavelength):
                    self._H[i] = angular_spectrum(
                        u_in=u_in[i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=u_in.device,
                        return_H=True,
                    )

            psfs = torch.zeros(u_in.shape, dtype=self.ctype, device=u_in.device)
            for i in range(self.color_system.n_wavelength):
                if self.waveprop == "angular_spectrum":
                    psfs[i], _, _ = angular_spectrum(
                        u_in=u_in[i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=u_in.device,
                        H=self._H[i],
                        # H_exp=self._H_exp[i],
                    )
                elif self.waveprop == "fresnel":
                    psfs[i], _, _ = fresnel_conv(
                        u_in=u_in[i],
                        wv=self.color_system.wv[i],
                        d1=self.d1[0],
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=u_in.device,
                    )
                else:
                    raise ValueError("Invalid waveprop method.")

            # psfs = psfs / torch.norm(psfs.flatten())
     

        else:

            assert len(psf_vals) == self.n_psf_mask

            # compute masks
            masks_dim = [self.n_psf_mask, self.color_system.n_wavelength] + list(
                self.input_shape[1:]
            )
            masks = torch.zeros(masks_dim, dtype=self.dtype, device=self.spherical_wavefront.device)
            for i in range(self.n_psf_mask):
                masks[i] = self.get_psf_mask(
                    psf_vals=psf_vals[i]
                )

            # apply mask
            u_in = masks * self.spherical_wavefront

            # mask to sensor
            if self._H is None:
                # precompute at very start, not dependent on input pattern, just its shape
                # TODO : benchmark how much it actually saves
                self._H = torch.zeros(
                    [self.color_system.n_wavelength] + list(self.input_shape[1:] * 2),
                    dtype=self.ctype
                ).cuda()
                for i in range(self.color_system.n_wavelength):
                    self._H[i] = angular_spectrum(
                        u_in=u_in[0][i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=u_in.device,
                        return_H=True,
                    )

            psfs = torch.zeros(u_in.shape, dtype=self.ctype, device=u_in.device)
            for n in range(self.n_psf_mask):
                for i in range(self.color_system.n_wavelength):
                    psfs[n][i], _, _ = angular_spectrum(
                        u_in=u_in[n][i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=u_in.device,
                        H=self._H[i],
                    )

            norm_fact = (
                torch.norm(psfs.flatten(1, 3), dim=1, keepdim=True).unsqueeze(2).unsqueeze(2)
            )
            psfs = psfs / norm_fact

        # intensity psf
        if self.grayscale:
            self._psf = rgb_to_grayscale(torch.square(torch.abs(psfs)))
        else:
            self._psf = torch.square(torch.abs(psfs))