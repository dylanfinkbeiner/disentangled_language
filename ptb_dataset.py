import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PTB_Dataset(Dataset):
	def __init__(self):
		#Fetch data
		self.data = None

		self.len = None
		self.x_data = None
		self.y_data = None
	
	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len


