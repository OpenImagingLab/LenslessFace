from math import cos, pi
import numpy as np
from mmcv.runner.hooks import HOOKS,Hook
from math import cos, pi
@HOOKS.register_module()
class NoiseUpdaterHook(Hook):
    def __init__(self,
                min_ratio = 0.,
                by_epoch = True,
                max_progress = None,
                **kwargs):
        super().__init__()
        self.min_ratio = min_ratio
        self.by_epoch = by_epoch
        self.max_progress = max_progress
    def get_noise_ratio(self,runner):
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
        ratio = self.min_ratio + (1 - self.min_ratio) * (1 - cos(pi * progress / max_progress)) / 2
        return ratio
    def before_train_iter(self, runner):
        if self.by_epoch:
            return
        ratio = self.get_noise_ratio(runner)
        if runner.model.module.backbone.optical.add_noise is not None:
            runner.model.module.backbone.optical.add_noise.noise_ratio = ratio
    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        ratio = self.get_noise_ratio(runner)
        if runner.model.module.backbone.optical.add_noise is not None:
            runner.model.module.backbone.optical.add_noise.noise_ratio = ratio