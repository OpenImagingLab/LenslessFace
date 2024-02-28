from math import cos, pi
import numpy as np
from mmcv.runner.hooks import HOOKS,Hook
from math import cos, pi
from mmcls.datasets.pipelines.background import AddBackground
import os
@HOOKS.register_module()
class BGUpdaterHook(Hook):
    def __init__(self,
                min_size = 50,
                max_size = 100,
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
        size = self.min_size + (self.max_size - self.min_size) * (1 - cos(pi * progress / max_progress)) / 2
        size = int(size)
        return (size,size)

    def update_bg_size(self, dataset, runner):
        for transform in dataset.pipeline.transforms:
                if transform.__class__.__name__ == 'AddBackground':
                    transform.size = self.get_bg_size(runner)
                    # if runner.model.device_ids[0] == 0:
                    #     print("AddBackground.size: ",transform.size)

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        if runner.data_loader.dataset.__class__.__name__ == 'HybridDataset':
            hybrid_dataset = runner.data_loader.dataset
            if hasattr(hybrid_dataset, 'cls_dataset'):
                self.update_bg_size(hybrid_dataset.cls_dataset, runner)
            if hasattr(hybrid_dataset, 'det_dataset'):
                self.update_bg_size(hybrid_dataset.det_dataset, runner)
            if hasattr(hybrid_dataset, 'pose_dataset'):
                self.update_bg_size(hybrid_dataset.pose_dataset, runner)
            
        else:
            dataset = runner.data_loader.dataset
            self.update_bg_size(dataset, runner)
    
    def before_train_iter(self, runner):
        if self.by_epoch:
            return
        if runner.data_loader.__class__.__name__ == 'HybridIterLoader':
            hybrid_loader = runner.data_loader._iter_loaders
            for key in ['cls','det','pose']:
                if key in hybrid_loader.keys():
                    hybrid_dataset = hybrid_loader[key]._dataloader.dataset
                    self.update_bg_size(hybrid_dataset, runner)
        else:
            dataset = runner.data_loader.dataset
            self.update_bg_size(dataset, runner)