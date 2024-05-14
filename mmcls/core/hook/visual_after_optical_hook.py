from math import cos, pi
import numpy as np
from mmcv.runner.hooks import HOOKS,Hook
import os 
import cv2 as cv
from torchvision.utils import save_image
@HOOKS.register_module()
class VisualAfterOpticalHook(Hook):
    def __init__(self,  do_distill = False, visual_freq = 100, visual_num = 16) -> None:
        super().__init__()
        self.do_distill = do_distill
        self.visual_freq = visual_freq
        self.visual_images = visual_num
    def binarization(self,x):
        y = (x + 1.0) / 2.0
        y = np.clip(y, 0, 1) 
        y = np.round(y) * 255
        return y
    def normalize_01(self,x):
        mmin = x.min()
        mmax = x.max()
        print(mmin, mmax)
        y = (x-mmin)/(mmax-mmin)
        return y

    def after_train_iter(self, runner):
        if runner.iter % self.visual_freq == 0 and runner.model.device_ids[0] == 0:
            os.makedirs(os.path.join(runner.work_dir,"visualizations"),exist_ok=True)
            if hasattr(runner.model.module, 'optical'):
                model = runner.model.module
            elif hasattr(runner.model.module, 'backbone'):
                if hasattr(runner.model.module.backbone, 'optical'):
                    model = runner.model.module.backbone
            elif hasattr(runner.model.module, 'architecture'):
                if hasattr(runner.model.module.architecture.model, 'optical'):
                    model = runner.model.module.architecture.model
                else:
                    model = runner.model.module.architecture.model.backbone
            else:
                RuntimeError("No optical in model")
        
            after_optical = model.optical.after_optical
            before_optical = model.optical.before_optical
            after_affine = model.optical.after_affine
            # for i in range(after_optical.shape[0]):
            after_save_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_after_optical"  + ".png")
            save_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_before_optical"  + ".png")
            after_affine_save_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_after_affine"  + ".png")
            # print(after_optical.shape,before_optical.shape)
            if len(after_optical) >=self.visual_images:
                after_optical = save_image(after_optical[:self.visual_images,:,:,:],after_save_path, nrow = 4, normalize = True)
                before_optical = save_image(before_optical[:self.visual_images,:,:,:],save_path, nrow = 4, normalize = True)
                after_affine = save_image(after_affine[:self.visual_images,:,:,:],after_affine_save_path, nrow = 4, normalize = True)
            else:
                after_optical = save_image(after_optical,after_save_path, nrow = 4, normalize = True)
                before_optical = save_image(before_optical,save_path, nrow = 4, normalize = True)
            
        
               