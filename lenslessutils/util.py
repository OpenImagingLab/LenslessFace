import torch
from torchvision.transforms.functional import crop
import numpy as np
from scipy import ndimage
import warnings

import cv2
from lensless.util import resize
import time


ctypes = [torch.complex64, torch.complex128]
# @ https://github.com/ebezzam/LenslessClassification/blob/main/lenslessclass/util.py

class RealFFTConvolve2D:
    def __init__(self, filter, mode=None, axes=(-2, -1), img_shape=None, device=None):
        """
        Operator that performs convolution in Fourier domain, and assumes
        real-valued signals. Useful if convolving with the same filter, i.e.
        avoid computing FFT of same filter.

        Parameters
        ----------
        filter array_like
            2D filter to use. Must be of shape (channels, height, width) even if
            only one channel.
        img_shape : tuple
            If image different shape than filter, specify here.
        dtype : float32 or float64
            Data type to use for optimization.
        """
        assert torch.is_tensor(filter)
        if device is not None:
            filter = filter.to(device)
        self.device = device

        self.filter_shape = filter.shape
        if img_shape is None:
            self.img_shape = filter.shape
        else:
            assert len(img_shape) == 3
            self.img_shape = img_shape
        if axes is None:
            self.shape = [
                self.filter_shape[i] + self.img_shape[i] - 1 for i in range(len(self.filter_shape))
            ]
        else:
            self.shape = [self.filter_shape[i] + self.img_shape[i] - 1 for i in axes]
        self.axes = axes
        if mode is not None:
            if mode != "same":
                raise ValueError(f"{mode} mode not supported ")

        self.filter_freq = torch.fft.rfftn(filter, self.shape, dim=axes)

    def __call__(self, x):
        orig_device = x.device
        if self.device is not None:
            x = x.to(self.device)
        x_freq = torch.fft.rfftn(x, self.shape, dim=self.axes)
        ret = torch.fft.irfftn(self.filter_freq * x_freq, self.shape, dim=self.axes)

        y_pad_edge = int((self.shape[0] - self.img_shape[self.axes[0]]) / 2)
        x_pad_edge = int((self.shape[1] - self.img_shape[self.axes[1]]) / 2)
        return crop(
            ret,
            top=y_pad_edge,
            left=x_pad_edge,
            height=self.img_shape[self.axes[0]],
            width=self.img_shape[self.axes[1]],
        ).to(orig_device)


def fftconvolve(in1, in2, mode=None, axes=None):
    """
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/signaltools.py#L554-L668

    TODO : add support for mode (padding) and axes

    """

    s1 = in1.shape
    s2 = in2.shape
    if axes is None:
        shape = [s1[i] + s2[i] - 1 for i in range(len(s1))]
    else:
        shape = [s1[i] + s2[i] - 1 for i in axes]
    if mode is not None:
        if mode != "same":
            raise ValueError(f"{mode} mode not supported ")

    is_complex = False
    if in1.dtype in ctypes or in2.dtype in ctypes:
        is_complex = True
        sp1 = torch.fft.fftn(in1, shape, dim=axes)
        sp2 = torch.fft.fftn(in2, shape, dim=axes)
        ret = torch.fft.ifftn(sp1 * sp2, shape, dim=axes)
    else:
        sp1 = torch.fft.rfftn(in1, shape, dim=axes)
        sp2 = torch.fft.rfftn(in2, shape, dim=axes)
        ret = torch.fft.irfftn(sp1 * sp2, shape, dim=axes)

    # same shape, mode="same"
    # TODO : assuming 2D here
    if axes is None:
        y_pad_edge = int((shape[0] - s1[0]) / 2)
        x_pad_edge = int((shape[1] - s1[1]) / 2)
        if is_complex:
            _real = crop(ret.real, top=y_pad_edge, left=x_pad_edge, height=s1[0], width=s1[1])
            _imag = crop(ret.imag, top=y_pad_edge, left=x_pad_edge, height=s1[0], width=s1[1])
            return torch.complex(_real, _imag)
        else:
            return crop(ret, top=y_pad_edge, left=x_pad_edge, height=s1[0], width=s1[1])
    else:
        y_pad_edge = int((shape[0] - s1[axes[0]]) / 2)
        x_pad_edge = int((shape[1] - s1[axes[1]]) / 2)
        if is_complex:
            _real = crop(
                ret.real, top=y_pad_edge, left=x_pad_edge, height=s1[axes[0]], width=s1[axes[1]]
            )
            _imag = crop(
                ret.imag, top=y_pad_edge, left=x_pad_edge, height=s1[axes[0]], width=s1[axes[1]]
            )
            return torch.complex(_real, _imag)
        else:
            return crop(ret, top=y_pad_edge, left=x_pad_edge, height=s1[axes[0]], width=s1[axes[1]])


def device_checks(device=None, single_gpu=False):

    use_cuda = torch.cuda.is_available()
    multi_gpu = False
    device_ids = None
    if device is None:
        if use_cuda:
            print("CUDA available, using GPU.")
            device = "cuda:0"
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1 and not single_gpu:
                multi_gpu = True
                print(f"-- using {n_gpus} GPUs")
                device_ids = np.arange(n_gpus).tolist()
        else:
            device = "cpu"
            print("CUDA not available, using CPU.")
    else:
        if device == "cpu":
            use_cuda = False
        else:
            try:
                gpu_id = int(device.split(":")[1])
            except:
                raise ValueError(
                    "Bad device specification. Should be 'cpu' or something like 'cuda:1' to set the GPU ID."
                )
            assert use_cuda, f"No GPU availlable but device set to {device}."
            n_gpus = torch.cuda.device_count()
            assert gpu_id < n_gpus, f"GPU {device} not available"
            if n_gpus > 1 and not single_gpu:
                multi_gpu = True
                print(f"-- using {n_gpus} GPUs")
                device_ids = np.arange(n_gpus)
                device_ids[[0, gpu_id]] = device_ids[[gpu_id, 0]]
                device_ids = device_ids.tolist()
            elif device is not None:
                device_ids = [int(device.split(":")[1])]

    return device, use_cuda, multi_gpu, device_ids


class AddPoissonNoise:
    def __init__(self, snr, tol=1e-4):
        self.snr = snr
        self.tol = tol

    def __call__(self, measurement):

        if len(measurement.shape) == 4:
            # for batches

            # ensure non-negative
            with torch.no_grad():
                measurement_np = measurement.detach().cpu().numpy()
            min_vals = measurement_np.reshape(len(measurement), -1).min(axis=1)[
                :, np.newaxis, np.newaxis, np.newaxis
            ]
            measurement -= torch.tensor(min_vals).to(measurement)

            # generate poisson noise
            noise = torch.poisson(measurement)

            # scale noise for target SNR
            with torch.no_grad():
                measurement_np = measurement.detach().cpu().numpy()
                noise_np = noise.detach().cpu().numpy()
            sig_var = np.array([ndimage.variance(_meas) for _meas in measurement_np])
            noise_var = np.array(
                [np.maximum(ndimage.variance(_noise), self.tol) for _noise in noise_np]
            )
            fact = np.sqrt(sig_var / noise_var / (10 ** (self.snr / 10)))
            fact = np.array(fact)[:, np.newaxis, np.newaxis, np.newaxis]
            fact = torch.tensor(fact).to(measurement)

        else:

            measurement -= measurement.min()
            noise = torch.poisson(measurement)

            # rescale noise to target SNR
            sig_var = ndimage.variance(measurement.numpy())
            noise_var = np.maximum(ndimage.variance(noise.numpy()), self.tol)
            fact = np.sqrt(sig_var / noise_var / (10 ** (self.snr / 10)))

        return measurement + fact * noise


