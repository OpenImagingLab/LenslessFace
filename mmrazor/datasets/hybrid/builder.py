from mmcv.utils import Registry, build_from_cfg, digit_version
from mmcls.datasets import build_dataset as build_cls_dataset
from mmpose.datasets import build_dataset as build_pose_dataset

DATASETS = Registry('dataset')

def build_hybrid_dataset(cfg):
    return build_from_cfg(cfg, DATASETS)


def build_hybrid_dataset_train(cfg):
    func_dict = dict(cls=build_cls_dataset,
                     pose=build_pose_dataset)
    dataset_dict = {}
    for key, item in cfg.items():
        if item is not None:
            dataset_dict[key] = func_dict[key](item)
    return dataset_dict