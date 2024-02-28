from math import cos, pi
import numpy as np
from mmcv.runner.hooks import HOOKS,Hook
import os 
import cv2 as cv
import copy
@HOOKS.register_module()
class VisualConvHook(Hook):
    def __init__(self, do_distill = False, visual_freq = 100) -> None:
        super().__init__()
        self.visual_freq = visual_freq
        self.mask = None
        self.do_distill = do_distill
        
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


    def after_train_iter(self, runner, split_channel=False): # weather split RGB channel for mask and psf 
        # print("finished! %d\n"%runner.iter)
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

        if runner.iter % self.visual_freq == 0 and runner.model.device_ids[0] == 0:
            os.makedirs(os.path.join(runner.work_dir,"visualizations"),exist_ok=True)

            # conv_weight = runner.model.module.backbone.conv0.weight.cpu().detach().numpy()
            conv_weight = model.optical._psf.cpu().detach().numpy()
            mask = model.optical.psf_val_to_save.cpu().detach().numpy()
            if split_channel:
                for i in range(conv_weight.shape[0]):
                    save_psf_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_psf_channel_" + str(i) + ".png")
                    save_mask_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_mask_channel_" + str(i) + ".png")
                    # visual_conv = conv_weight[i,:,:,:].transpose(1,2,0)
                    # visual_conv = conv_weight[i,:,:].transpose(1,0)
                    # visual_mask = mask[i,:,:].transpose(1,0)
                    visual_conv = conv_weight[i,:,:]
                    visual_mask = mask[i,:,:]
                    visual_conv = self.normalize_01(visual_conv)
                    visual_mask = self.binarization(visual_mask)
                    cv.imwrite(save_psf_path, visual_conv)
                    cv.imwrite(save_mask_path,visual_mask)
            else:
                save_psf_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_psf" + ".png")
                save_mask_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_mask"  + ".png")
                mask_change_path = os.path.join(runner.work_dir,"visualizations", str(runner.iter) + "_mask_change"  + ".png")
                visual_mask = mask.transpose(1,2,0)
                model.optical.save_psf(save_psf_path)
                visual_mask = visual_mask / (visual_mask.max() + 1e-6)
                visual_mask = np.clip(visual_mask, 0, 1)
                visual_mask = (visual_mask * 255).astype(np.int)
                # print(visual_mask)
                if self.mask is None:
                    self.mask = copy.deepcopy(visual_mask)
                mask_changed = np.abs(self.mask - visual_mask) 
                mask_changed = mask_changed / (mask_changed.max() + 1e-6)
                mask_changed = mask_changed.astype(np.int) * 255
                print(np.nonzero(mask_changed))
                self.mask = copy.deepcopy(visual_mask)
                cv.imwrite(mask_change_path,mask_changed)
                cv.imwrite(save_mask_path,visual_mask)