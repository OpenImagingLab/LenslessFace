from math import cos, pi
import numpy as np
from mmcv.runner.hooks import HOOKS,Hook
from math import cos, pi
import os
@HOOKS.register_module()
class CropUpdaterHook(Hook):
    def __init__(self,
                min_size = (112,96),
                max_size = (172,172),
                by_epoch = True,
                max_progress = None,
                **kwargs):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.by_epoch = by_epoch
        self.max_progress = max_progress
    def get_bg_size(self,runner):
        if self.by_epoch:
            if self.max_progress is None:
                max_progress = runner.max_epochs
            else:
                max_progress = int(self.max_progress * runner.max_epochs)
            progress = runner.epoch if runner.epoch < max_progress else max_progress
            
        else:
            if self.max_progress is None:
                max_progress = runner.max_iters
            else:
                max_progress = int(self.max_progress * runner.max_iters)
            progress = runner.iter if runner.iter < max_progress else max_progress
            

        # annealing_cos
        size0 = self.min_size[0] + (self.max_size[0] - self.min_size[0]) * (1 - cos(pi * progress / max_progress)) / 2
        size1 = self.min_size[1] + (self.max_size[1] - self.min_size[1]) * (1 - cos(pi * progress / max_progress)) / 2
        size0 = int(size0)
        size1 = int(size1)
        return (size0,size1)

   
    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        size = self.get_bg_size(runner)
        for transform in runner.data_loader.dataset.pipeline.transforms:
            if transform.__class__.__name__ == 'CenterCrop':
                transform.crop_size = size
                if runner.model.device_ids[0] == 0:
                    print("CenterCrop.size: ",size)
            if transform.__class__.__name__ == 'Propagated':
                input_dim = [size[0],size[1],3]
                object_height = transform.object_height * size[0] / transform.input_dim[0]
                propagated_args = dict(
                    mask2sensor=0.002,
                    scene2mask=0.4,
                    object_height=object_height,
                    sensor='IMX250',
                    single_psf=False,
                    grayscale=False,
                    input_dim=input_dim,
                    output_dim=[308, 257, 3])
                transform.__init__(**propagated_args)
               