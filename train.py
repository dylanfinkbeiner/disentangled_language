import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from parser import BiAffineParser
from ptb_dataset import PTB_Dataset

import os

NUM_EPOCHS = 100
BATCH_SIZE = 10

def main():
	dataset = PTB_Dataset()
	loader = DataLoader(dataset=dataset,
			batch_size=BATCH_SIZE,
			shuffle=True)


	num_batches = num_samples / BATCH_SIZE

	parser = BiAffineParser()

	#TODO get model params to feed to parser

	#Optimizer
	optim = Adam(params, lr=1e-1)

	parser.train()
	for e in range(NUM_EPOCHS):

		for b in range(num_batches):
			optim.zero_grad()

			#Get batch data
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

def loss_heads(S, heads):
	#Get softmax of outputs from parser forward pass

	#Cross-entropy between S and 
	return F.cross_entropy(S, heads)

def loss_rels(L, deprels):

	return F.cross_entropy(L, labels)

if name == '__main__':
	main()
	
