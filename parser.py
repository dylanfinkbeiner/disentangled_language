import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

'''
    An implementation of Jabberwocky dropout-heavy
    BiAffine Attention Dependency Parser
'''

alpha = 40

class BiaffineParser(nn.Module):
    def __init__(self, 
            word_e_size=100,
            pos_e_size=25,
            word_vocab_size=None,
            pos_vocab_size=None,
            hidden_size=400,
            num_layers=3,
            d_arc=500,
            d_rel=100,
            num_relations=46,
            input_dropout=0.33):

        super(BiaffineParser, self).__init__()

        #Embeddings
        self.word_emb = nn.Embedding(word_vocab_size, word_e_size)
        self.pos_emb = nn.Embedding(pos_vocab_size, pos_e_size)

        #Dropout
        self.input_dropout = nn.Dropout(p=input_dropout)

        #LSTM 
        self.lstm = nn.LSTM(input_size=word_e_size+pos_e_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=0.33)

        #Arc-scoring MLPs
        self.h_arc_dep   = nn.Sequential(
                nn.Linear(hidden_size*2, d_arc),
                nn.ReLU())
        self.h_arc_head  = nn.Sequential(
                nn.Linear(hidden_size*2, d_arc),
                nn.ReLU())

        #Label-scoring MLPs
        self.h_rel_dep  = nn.Sequential(
                nn.Linear(hidden_size*2, d_rel),
                nn.ReLU())
        self.h_rel_head = nn.Sequential(
                nn.Linear(hidden_size*2, d_rel),
                nn.ReLU())

        #Biaffine Attention Parameters
        self.W_arc = torch.randn(d_arc, d_arc)
        self.b_arc = torch.randn(d_arc, 1)

        #self.U_rel = torch.randn(d_rel, d_rel, num_relations)
        self.U_rel = torch.randn(d_rel, num_relations, d_rel)
        self.W_rel = torch.randn(num_relations, d_rel)
        self.b_rel = torch.randn(num_relations, 1)


    def forward(self, words, pos, sent_lens, train=True):
        '''
        x_words - list/LongTensor of mappings to integers from x2nums dict
        x_pos - list/LongTensor ...
        sent_len - 
        '''

        #Embeddings
        w_embs = self.word_emb(words)
        p_embs = self.pos_emb(pos)

        print( w_embs.shape) #(b_size, longest_sent, 100)
        print( p_embs.shape) #(b_size, longest_sent, 25)


        lstm_input = self.input_dropout(torch.cat([w_embs, p_embs], -1))

        print( lstm_input.shape) #(b_size, longest_sent, 125)

        #Packing
        packed_input = pack_padded_sequence(lstm_input, sent_lens, batch_first=True)

        #Feed to LSTM
        H, _ = self.lstm(packed_input)

        #Unpack
        unpacked, _ = pad_packed_sequence(H, batch_first=True) 

        print(unpacked.shape) #(b_size, longest_sent, 800)

        H_arc_head = self.h_arc_head(unpacked)
        H_arc_dep  = self.h_arc_dep(unpacked)
        H_rel_head = self.h_rel_head(unpacked)
        H_rel_dep  = self.h_rel_dep(unpacked)

        print(H_rel_dep.shape)#(b_size, longest_sent, 100)
        print(H_arc_head.shape) #(b_size, longest_sent, 500)

        #This chunk is basically a torch paraphrase of Kasai et al's tensorflow imp.
        b, l, d_arc = H_arc_head.size() #l for "longest_sentence"
        S = torch.mm(H_arc_dep.view(b*l, d_arc), self.W_arc).view(b, l, d_arc)
        S = torch.mm(S, H_arc_head.permute(0, 2, 1))
        bias = torch.mm(H_arc_head.view(b*l, d_arc), self.b_arc.unsqueeze(1))
        bias = bias.view(b, l, 1).permute(0, 2, 1) #(B x 1 x L), for broadcasting
        S += bias
        #XXX I don't actually understand what is going on, though

        if train: #Predict heads by greedy maximum on scores in S
            pass

        else: #Call MST algorithm to predict heads
            pass


        _, _, num_rel = self.U_rel.size()
        #Again, basically copypasted from Kasai

        one_hot_pred = predictions
        H_rel_head = #Filtering
        U_rel = self.U_rel.view(-1, num_rel * d_rel)
        interactions = torch.mm(H_rel_head.view(b*l, d_rel), U_rel).view(b*l, num_rel, d_rel)
        interactions = torch.mm(interactions, H_rel_dep.view(b*l, d_rel, 1)).view(b, l, num_rel)
        sums = (torch.mm((H_rel_head+H_rel_dep).view(b*l, d_rel), self.W_rel) + self.b_rel).view(b, l, num_rel)
        L = interactions + sums

        #Pre-softmax scores for heads, labels
        return S, L
