import torch
from torch.utils.data import Dataset
import os
from .pipelines import Compose
from .builder import DATASETS
import numpy as np

@DATASETS.register_module()
class MS1MDataset(Dataset):
	CLASSES = 93431

	def __init__(self, data_root, pipeline, test_mode=False):
		super(MS1MDataset, self).__init__()
		self.pipeline = Compose(pipeline)
		self.root_dir = data_root
		self.img_prefix = data_root
		self.filenames = []
		self.labels = []
		print("start!")
		with open(os.path.join(data_root, "file_label.txt")) as f:
			for line in f.readlines():
				filename, label = line.strip().split(" ")
				self.filenames.append(filename)
				self.labels.append(int(label))
				

	def __getitem__(self, index):
		data_info = dict(
			img_prefix=self.img_prefix,
			img_info=dict(filename=self.filenames[index]),
			gt_label=np.array(self.labels[index], dtype=np.int64))
		try:
			out = self.pipeline(data_info)
		except:
			index_new = np.random.randint(0, self.__len__()-1)
			out = self.__getitem__(index_new)
		# print("hello!")
		# print("out[img].min() >= 0", out["img"].min() >= 0)
		return out

	def __len__(self):
		return len(self.labels)
		# return 1000   # for debug
