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

CONLLU_FILE = './data/sentences.conllu'

def main():
	parser = BiAffineParser(params)

    if os.path.isdir(WEIGHTS_PATH):
        parser.load_state_dict(torch.load(WEIGHT_PATH))

	#TODO get model params to feed to optimizer

    if train:
        train_dataset = get_dataset(CONLLU_FILE)
        loader = custom_data_loader(train_dataset, BATCH_SIZE) 

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
                words, pos, sent_lens, heads, rels = next(loader)

                #Forward pass
                S, L = parser(words, pos)

                #Calculate losses
                loss =  loss_heads(S, heads)
                loss += loss_labels(L, rels)

                loss.backward()
                optim.step()


        #Save weights
        if not os.path.isdir(WEIGHTS_PATH):
            os.mkdir(WEIGHTS_PATH)
        torch.save(parser.state_dict(), '%s-%s' % (SAVE_PATH, MODEL_NAME))

    elif test:
        pass

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

def loss_rels(L, rels):
    '''
    L - should be tensor w/ shape (batch_size, sent_len, d_rel)

    deprels - should be a list of dependency relations as they are indexed in the dict
    '''

    Y_labels = Variable(torch.LongTensor(rels), autograd=False)

    return F.cross_entropy(L, Y_labels)

if name == '__main__':
    main()

