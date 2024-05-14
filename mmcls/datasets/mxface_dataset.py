# import mxnet as mx
from torch.utils.data import Dataset
import os
import numpy as np
import numbers
import torch
from .pipelines import Compose

from .builder import DATASETS
@DATASETS.register_module()
class MXFaceDataset(Dataset):
    CLASSES = 93431
    def __init__(self, data_root, pipeline):
        super(MXFaceDataset, self).__init__()

        self.pipeline = Compose(pipeline)
   
        self.data_root = data_root
        # self.local_rank = local_rank
        path_imgrec = os.path.join(data_root, 'train.rec')
        path_imgidx = os.path.join(data_root, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        #bgr to rgb
        sample = sample[:, :, ::-1]
        out = dict(img=sample, gt_label = label)
        out = self.pipeline(out)
        return out

    def __len__(self):
        return len(self.imgidx)