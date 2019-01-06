import numpy as np
import torch
from torch import unsqueeze
import torch.nn as nn
import torch.nn.functional as F
import sys

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

'''
    An implementation of Jabberwocky dropout-heavy
    BiAffine Attention Dependency Parser
'''

alpha = 40 #For calculating word dropout rates

class BiaffineParser(nn.Module):
    def __init__(self,
            word_e_size=100,
            pos_e_size=25, #original Dozat/Manning paper uses 100
            word_vocab_size=None,
            pos_vocab_size=None,
            hidden_size=400,
            lstm_layers=3,
            d_arc=500,
            d_rel=100,
            num_relations=46,
            embedding_dropout=0.33,
            lstm_dropout=0.33,
            arc_dropout=0.33,
            rel_dropout=0.33):

        super(BiaffineParser, self).__init__()

        #Embeddings
        self.word_emb = nn.Embedding(word_vocab_size, word_e_size, padding_idx=0)
        self.pos_emb = nn.Embedding(pos_vocab_size, pos_e_size, padding_idx=0)

        #Dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        #LSTM
        self.lstm = nn.LSTM(input_size=word_e_size + pos_e_size,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
                dropout=lstm_dropout)

        #Arc-scoring MLPs
        self.h_arc_dep   = nn.Sequential(
                nn.Linear(hidden_size*2, d_arc),
                nn.ReLU(),
                nn.Dropout(p=arc_dropout))
        self.h_arc_head  = nn.Sequential(
                nn.Linear(hidden_size*2, d_arc),
                nn.ReLU(),
                nn.Dropout(p=arc_dropout))

        #Label-scoring MLPs
        self.h_rel_dep  = nn.Sequential(
                nn.Linear(hidden_size*2, d_rel),
                nn.ReLU(),
                nn.Dropout(p=rel_dropout))
        self.h_rel_head = nn.Sequential(
                nn.Linear(hidden_size*2, d_rel),
                nn.ReLU(),
                nn.Dropout(p=rel_dropout))

        '''
            Amsterdam initializes weights/biases uniformly using
            a std deviation
        '''
        #Biaffine Attention Parameters
        self.W_arc = nn.Parameter(torch.randn(d_arc, d_arc))
        self.b_arc = nn.Parameter(torch.randn(d_arc))

        #In paper, shape is (d, d, r) but in implementation Kasai uses (d,r,d)
        self.U_rel = nn.Parameter(torch.randn(d_rel, num_relations, d_rel))
        #self.U_rel = torch.randn(d_rel, d_rel, num_relations)
        self.W_rel = nn.Parameter(torch.randn(d_rel, num_relations))
        self.b_rel = nn.Parameter(torch.randn(num_relations))


    def forward(self, words, pos, sent_lens, train=True):
        '''
        x_words - list/LongTensor of mappings to integers from x2nums dict
        x_pos - list/LongTensor ...
        sent_lens - a LIST of ints
        '''

        #Embeddings
        w_embs = self.word_emb(words) # (b, l, w_e)
        p_embs = self.pos_emb(pos) # (b, l, p_e)

        # cat(_, -1) means concatenate along final axis
        lstm_input = self.embedding_dropout(torch.cat([w_embs, p_embs], -1)) # (b, l, w_e + p_e)

        #Packing
        packed_input = pack_padded_sequence(lstm_input, sent_lens, batch_first=True)

        #Feed to LSTM
        H, _ = self.lstm(packed_input) # H, presumably, is the tensor of h_k's described in Kasai

        #Unpack (second return is just sent_lens again)
        unpacked, _ = pad_packed_sequence(H, batch_first=True) # (b, l, 2*hidden_size)

        #Recap so far: H is a (b,l,800) tensor; first axis: sentence | second: word | third: lstm output

        H_arc_head = self.h_arc_head(unpacked) # (b, l, d_arc)
        H_arc_dep  = self.h_arc_dep(unpacked) # (b, l, d_arc)
        H_rel_head = self.h_rel_head(unpacked) # (b, l, d_rel)
        H_rel_dep  = self.h_rel_dep(unpacked) # (b, l, d_rel)


        #XXX MAY WANT TO CONSIDER USING TORCH'S BILINEAR LAYER INSTEAD

        #This chunk is basically a torch paraphrase of Kasai et al's tensorflow imp.
        b, l, d_arc = H_arc_head.size() #l for "longest_sentence"
        S = torch.mm(H_arc_dep.view(b*l, d_arc), self.W_arc).view(b, l, d_arc) # Implementing W_arc * h_i_dep
        S = torch.bmm(S, H_arc_head.permute(0, 2, 1)) # (b, l, l), implementing 
        bias = torch.mm(H_arc_head.view(b*l, d_arc), self.b_arc.unsqueeze(1)) # (b*l, 1)
        #bias : Unsqueezing here allows intuitive matrix-vector multiply
        bias = bias.view(b, l, 1).permute(0, 2, 1) # (b, 1, l)
        S += bias # (b, l, l) where pre-softmax s_i vectors are on 3rd axis of S

        if train: #Predict heads by greedy maximum on scores in S
                #XXX Axis 2 might be wrong, should double check which axis represents 'classes'
                head_preds = torch.argmax(S, 2) # (b, l)

        else: #Call MST algorithm to predict heads
            #predictions = predict_heads(S)
            head_preds = torch.argmax(S, 2) #XXX Wrong (temporary, until MST algorithm implemented)

        d_rel, num_rel, _ = self.U_rel.size()
        #Again, basically copypasted from Kasai
        one_hot_pred = torch.zeros(b, l, l) 
        for i in range(b): # ith sentence
            for j in range(l): # jth word
                k = head_preds[i,j] # k is index of predicted head
                one_hot_pred[i,j,k] = 1
        H_rel_head = torch.bmm(one_hot_pred, H_rel_head) # (b, l, d_rel)
        #H_rel_head: Now the i-th row of this matrix is h_p_i^(rel_head), the MLP output for predicted head of ith word
        U_rel = self.U_rel.view(-1, num_rel * d_rel) # (d_rel, num_rel * d_rel)
        interactions = torch.mm(H_rel_head.view(b*l, d_rel), U_rel).view(b*l, num_rel, d_rel) # (b*l, num_rel, d_rel)
        interactions = torch.bmm(interactions, H_rel_dep.view(b*l, d_rel, 1)).view(b, l, num_rel) # (b*l, l, num_rel)
        sums = (torch.mm((H_rel_head+H_rel_dep).view(b*l, d_rel), self.W_rel) + self.b_rel).view(b, l, num_rel)
        L = interactions + sums # (b, l, num_rel) where pre-softmax l_i vectors are on 3rd axis of L

        #Pre-softmax scores for heads, labels
        return S, L, head_preds
