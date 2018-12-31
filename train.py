import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from parser import BiAffineParser
from ptb_dataset import PTB_Dataset

import os

NUM_EPOCHS = 100
BATCH_SIZE = 20

WEIGHTS_PATH = '../weights'
MODEL_NAME = ''

def main():


	parser = BiAffineParser(params)

    if os.path.isdir(WEIGHTS_PATH):
        parser.load_state_dict(torch.load(WEIGHT_PATH))

	#TODO get model params to feed to optimizer

	if train:
	train_dataset = PTB_Dataset(train_file)
	loader = DataLoader(dataset=dataset,
			batch_size=BATCH_SIZE,
			shuffle=True)

	num_batches = num_samples / BATCH_SIZE

		#Optimizer
		optim = Adam(params, lr=1e-1)

		parser.train()
		for e in range(NUM_EPOCHS):

			for b in range(num_batches):
				optim.zero_grad()

				'''Get batch of data (since batches are tensors with shape
				   [batchsize, ...], does tuple unpacking unpack these components
				   into [batchsize, words], [batchsize, pos],... tensors?)
				'''
				words, pos, sent_lens, arcs, deprels = next(loader)

				#Forward pass
				S, L = parser(words, pos)

				#Calculate losses
				loss =  loss_heads(S, arcs)
				loss += loss_labels(L, deprels)

				loss.backward()
				optim.step()
		

		#Save weights
		if not os.path.isdir(WEIGHTS_PATH):
			os.mkdir(WEIGHTS_PATH)
		torch.save(parser.state_dict(), '%s-%s' % (SAVE_PATH, MODEL_NAME))
	
	elif test:
		pass
			

def make_dicts():

	return x2i, i2x

'''map sentences to tensors of word-indices'''
def sent_to_i(sentences, x2i):
	s2i = []
	for j in enumerate(sentences):
		s2i.append([x2i(k) for k in sentences[j]]) #XXX definitely not right
	
	return s2i

def train_test_split(data):
	pass

#Eventually, probably should pull loss functions out into a utilities file
def loss_heads(S, heads):
	'''
	S - should be something like a tensor w/ shape
	    (batch_size, sent_len, sent_len); also, these are
	    head scores BEFORE softmax applied

	heads - should be a list of integers (the indices)
	'''
	Y_heads = Variable(torch.LongTensor(heads), autograd=False)

	#Cross-entropy between S and 
	return F.cross_entropy(S, Y_heads)

def loss_rels(L, deprels):
	'''
	L - should be tensor w/ shape (batch_size, sent_len, d_rel)

	deprels - should be a list of dependency relations as they are indexed in the dict
	'''

	Y_labels = Variable(torch.LongTensor(deprels), autograd=False)

	return F.cross_entropy(L, Y_labels)

if name == '__main__':
	main()
	
