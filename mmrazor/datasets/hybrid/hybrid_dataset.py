from .builder import DATASETS
from torch.utils.data import Dataset
from mmcls.datasets import build_dataset as build_cls_dataset
from mmpose.datasets import build_dataset as build_pose_dataset
import numpy as np
from mmcv.utils import print_log


@DATASETS.register_module()
class HybridDataset(Dataset):
    def __init__(self,
                 cls_dataset=None,
                 pose_dataset=None,
                 test_mode=False
                 ):
        length = 0
        num_datasets = 0
        if cls_dataset is not None:
            self.cls_dataset = build_cls_dataset(cls_dataset)
            length = max(length, self.cls_dataset.__len__())
            num_datasets += 1
        if pose_dataset is not None:
            self.pose_dataset = build_pose_dataset(pose_dataset)
            length = max(length, self.pose_dataset.__len__())
            num_datasets += 1
        assert num_datasets > 0
        self.test_mode = test_mode
        self.length = length
        # self.length = 1000
        # print('DEBUG!')
        # self.cls_dataset.data_infos = self.cls_dataset.data_infos[:10]
        # self.det_dataset.data_infos = self.det_dataset.data_infos[:8]
        # self.pose_dataset.db = self.pose_dataset.db[:4]
        # self.length = 10

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not self.test_mode:
            input_dict = {}
            if hasattr(self, 'cls_dataset'):
                index_cls = np.random.randint(low=0, high=self.cls_dataset.__len__())
                input_dict['cls'] = self.cls_dataset.__getitem__(index_cls)
            if hasattr(self, 'pose_dataset'):
                index_pose = np.random.randint(low=0, high=self.pose_dataset.__len__())
                input_dict['pose'] = self.pose_dataset.__getitem__(index_pose)
        else:
            input_dict = {}
            if hasattr(self, 'cls_dataset'):
                if index < self.cls_dataset.__len__():
                    input_dict['cls'] = self.cls_dataset.__getitem__(index)
            if hasattr(self, 'pose_dataset'):
                if index < self.pose_dataset.__len__():
                    input_dict['pose'] = self.pose_dataset.__getitem__(index)

        return dict(input_dict=input_dict)

    def evaluate(self,
                 results,
                 logger=None,
                 cls_args={},
                 pose_args={}):
        eval_results = {}

        cls_results = []
        pose_results = []
        for res in results:
            if 'cls_result' in res.keys():
                cls_results.append(res['cls_result'])
            if 'pose_result' in res.keys():
                pose_results.append(res['pose_result'])

        if hasattr(self, 'cls_dataset'):
            #print(cls_results)
            eval_results['cls_result'] = self.cls_dataset.evaluate(cls_results, logger=logger, **cls_args)
 
        if hasattr(self, 'pose_dataset'):
            eval_results['pose_result'] = self.pose_dataset.evaluate(pose_results, logger=logger, **pose_args)
            print_log('pose result:', logger=logger)
            for k, v in eval_results['pose_result'].items():
                msg = "{}:{}".format(k, v)
                print_log(msg, logger=logger)

        return eval_results
























