from math import cos, pi
import numpy as np
from mmcv.runner.hooks import HOOKS,Hook
# from mmcls.datasets.pipelines.auto_augment import TorchAffineRTS 
import os
@HOOKS.register_module()
class AffineUpdaterHook(Hook):
    def __init__(self,
                min_translate = 0.0,
                max_translate = 0.2,
                min_scale = 0.0,
                max_scale = 0.2,
                by_epoch = True,
                max_progress = None,
                **kwargs):
        super().__init__()
        self.min_translate = min_translate
        self.max_translate = max_translate
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.by_epoch = by_epoch
        self.max_progress = max_progress
        self.apply_translate = kwargs.get('apply_translate',True)
        self.apply_scale = kwargs.get('apply_scale',True)
    def get_progress(self,runner):
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
        return progress,max_progress
    def get_translate(self,runner):
        progress,max_progress = self.get_progress(runner)
        # annealing_cos
        translate = self.min_translate + (self.max_translate - self.min_translate) * (1 - cos(pi * progress / max_progress)) / 2
        return (translate,translate)
    def get_scale(self,runner):
        progress,max_progress = self.get_progress(runner)
        # annealing_cos
        scale = self.min_scale + (self.max_scale - self.min_scale) * (1 - cos(pi * progress / max_progress)) / 2
        return scale

    def update_translate(self, dataset, runner):
        for transform in dataset.pipeline.transforms:
            if transform.__class__.__name__ == 'TorchAffineRTS':
                transform.translate = self.get_translate(runner)
                if runner.model.device_ids[0] == 0:
                    print("TorchAffineRTS.translate: ",transform.translate)
    def update_scale(self, dataset, runner):
        for transform in dataset.pipeline.transforms:
            if transform.__class__.__name__ == 'TorchAffineRTS':
                transform.scale = self.get_scale(runner)
                if runner.model.device_ids[0] == 0:
                    print("TorchAffineRTS.scale: ",transform.scale)
               
    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        if runner.data_loader.dataset.__class__.__name__ == 'HybridDataset':
            hybrid_dataset = runner.data_loader.dataset
            if hasattr(hybrid_dataset, 'cls_dataset'):
                if self.apply_translate:
                    self.update_translate(hybrid_dataset.cls_dataset, runner)
                if self.apply_scale:
                    self.update_scale(hybrid_dataset.cls_dataset, runner)
             

            if hasattr(hybrid_dataset, 'det_dataset'):
                if self.apply_translate:
                    self.update_translate(hybrid_dataset.det_dataset, runner)
                if self.apply_scale:
                    self.update_scale(hybrid_dataset.det_dataset, runner)
        

            if hasattr(hybrid_dataset, 'pose_dataset'):
                if self.apply_translate:
                    self.update_translate(hybrid_dataset.pose_dataset, runner)
                if self.apply_scale:
                    self.update_scale(hybrid_dataset.pose_dataset, runner)
            
            
        else:
            dataset = runner.data_loader.dataset
            if self.apply_translate:
                self.update_translate(dataset, runner)
            if self.apply_scale:
                self.update_scale(dataset, runner)
         

    
    def before_train_iter(self, runner):
        if self.by_epoch:
            return
        if runner.data_loader.__class__.__name__ == 'HybridIterLoader':
            hybrid_loader = runner.data_loader._iter_loaders
            for key in ['cls','det','pose']:
                if key in hybrid_loader.keys():
                    hybrid_dataset = hybrid_loader[key]._dataloader.dataset
                    if self.apply_translate:
                        self.update_translate(hybrid_dataset, runner)
                    if self.apply_scale:
                        self.update_scale(hybrid_dataset, runner)
                   

        else:
            dataset = runner.data_loader.dataset
            if self.apply_translate:
                self.update_translate(dataset, runner)
            if self.apply_scale:
                self.update_scale(dataset, runner)
