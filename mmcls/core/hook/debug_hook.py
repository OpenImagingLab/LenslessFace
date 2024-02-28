from math import cos, pi
import numpy as np
from mmcv.runner.hooks import HOOKS,Hook
import cv2 as cv
from mmcv.runner.dist_utils import master_only
@HOOKS.register_module()
class DebugHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    @master_only
    def before_train_iter(self, runner):
        print("begin at iter %d\n"%runner.iter)
    @master_only    
    def after_iter(self, runner):
        print("finished! %d\n"%runner.iter)