import torch
from torch.utils.data import Dataset
import os
from .pipelines import Compose
from .builder import DATASETS
import numpy as np

@DATASETS.register_module()
class Celeb(Dataset):
	CLASSES = 93955

	def __init__(self, img_prefix, imglist_root, label_root, pipeline, test_mode=False):
		super(Celeb, self).__init__()
		self.pipeline = Compose(pipeline)
		#self.root = root
		self.img_prefix = img_prefix
		self.imglist_root = imglist_root
		self.label_root = label_root
		with open(self.label_root) as f:
			label = [line.strip() for line in f.readlines()]
		labels = [int(l) for l in label]

		#一定要用strip,因为原txt文件每行后面会带‘\n‘字符；
		with open(self.imglist_root) as f:
			data = [line.strip() for line in f.readlines()]
		self.filenames = data
		self.labels = labels

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
