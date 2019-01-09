import numpy as np
import torch
from torch import unsqueeze
import torch.nn as nn
import torch.nn.functional as F
import sys

import mst

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

'''
    An implementation of Jabberwocky dropout-heavy
    BiAffine Attention Dependency Parser
'''

alpha = 40 #For calculating word dropout rates...

class BiaffineParser(nn.Module):
    def __init__(self,
            word_e_size=100,
            pos_e_size=25, #Original Dozat/Manning paper uses 100
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
            rel_dropout=0.33,
            padding_idx=None):

        super(BiaffineParser, self).__init__()

        #Embeddings
        self.word_emb = nn.Embedding(word_vocab_size, word_e_size, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(pos_vocab_size, pos_e_size, padding_idx=padding_idx)

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
        self.h_arc_dep = nn.Sequential(
                nn.Linear(hidden_size*2, d_arc),
                nn.ReLU(),
                nn.Dropout(p=arc_dropout))
        self.h_arc_head = nn.Sequential(
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

        #Biaffine Attention Parameters
        self.W_arc = nn.Parameter(torch.randn(d_arc, d_arc))
        self.b_arc = nn.Parameter(torch.randn(d_arc))

        #Note to self: in paper, U shape is (d, d, r) but in implementation Kasai uses (d,r,d)
        self.U_rel = nn.Parameter(torch.randn(d_rel, num_relations, d_rel))
        self.W_rel = nn.Parameter(torch.randn(d_rel, num_relations))
        self.b_rel = nn.Parameter(torch.randn(num_relations))


    def forward(self, words, pos, sent_lens):
        '''
        ins:
            words::Tensor
            pos::Tensor
            sent_lens::List

        outs:
            S::Tensor - Shape(b, l, l); [i,j] is pre-softmax vector s_j, i.e.
                        logits for dist. over heads of jth word in ith sentence
            L::Tensor - Shape(b, l, num_rel); [i,j] is pre-softmax vector
                        l_i as described in paper
            head_preds::Tensor - Shape(b, l); [i,j] entry is
                                 prediction of jth word in ith sentence
        '''

        #if not self.train:
        #    self.word_emb.weight.data[,:] = 0.0 # Zero-out "unk" word at test time

        w_embs = self.word_emb(words) # (b, l, w_e)
        p_embs = self.pos_emb(pos) # (b, l, p_e)

        lstm_input = self.embedding_dropout(torch.cat([w_embs, p_embs], -1)) # (b, l, w_e + p_e)

        packed_input = pack_padded_sequence(lstm_input, sent_lens, batch_first=True)

        H, _ = self.lstm(packed_input) # H, presumably, is the tensor of h_k's described in Kasai
        H, _ = pad_packed_sequence(H, batch_first=True) # (b, l, 2*hidden_size)

        #Recap so far: H is a (b,l,800) tensor; first axis: sentence | second: word | third: lstm output

        H_arc_head = self.h_arc_head(H) # (b, l, d_arc)
        H_arc_dep  = self.h_arc_dep(H) # (b, l, d_arc)
        H_rel_head = self.h_rel_head(H) # (b, l, d_rel)
        H_rel_dep  = self.h_rel_dep(H) # (b, l, d_rel)

        b, l, d_arc = H_arc_head.size() # l for "longest_sentence"
        S = torch.mm(H_arc_dep.view(b*l, d_arc), self.W_arc).view(b, l, d_arc) # Implementing W_arc * h_i_dep
        S = torch.bmm(S, H_arc_head.permute(0, 2, 1)) # (b, l, l)
        bias = torch.mm(H_arc_head.view(b*l, d_arc), self.b_arc.unsqueeze(1)) # (b*l, 1)
        #bias : Unsqueezing here allows intuitive matrix-vector multiply
        bias = bias.view(b, l, 1).permute(0, 2, 1) # (b, 1, l)
        S += bias # (b, l, l) where logits vectors s_i are 3rd axis of S

        if self.training: # Greedy
            head_preds = torch.argmax(S, 2) # (b, l, l) -> (b, l)

        else: # Single-rooted, acyclic heads
            head_preds = mst_preds(S, sent_lens)
            #head_preds = torch.argmax(S, 2) #XXX Wrong (temporary, until MST algorithm implemented)

        d_rel, num_rel, _ = self.U_rel.size()
        #Again, basically copypasted from Kasai
        one_hot_pred = torch.zeros(b, l, l)
        #NOTE  This one hot code could probably be replaced by clever use of index_select
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
        L = interactions + sums # (b, l, num_rel) where logits vectors l_i are 3rd axis of L

        return S, L, head_preds


## From https://github.com/chantera/biaffineparser/blob/master/pytorch_model.py#L86
'''THE POINT OF THIS FUNCTION IS...

'''
def mst_preds(S, sent_lens):
       #ROOT_TOKEN = 

    heads_batch = []
    
    S = S.data.cpu().numpy() # Take to numpy arrays

    for arcs_logits, length in zip(S, sent_lens):
        arcs_probs = softmax2d(arcs_logits[:length, :length]) # Select out THE ACTUAL SENTENCE
        head_preds = mst.mst(arcs_probs) # NOTE Input to mst is softmax of arc scores
        
        #label_probs = softmax2d(label_logit[np.arange(length), arcs])
        #labels = np.argmax(label_probs, axis=1) # NOTE Simple argmax to get label predictions
        #labels[0] = ROOT
        #tokens = np.arange(1, length)
        #roots = np.where(labels[tokens] == ROOT)[0] + 1
        #if len(roots) < 1:
        #    root_arc = np.where(arcs[tokens] == 0)[0] + 1
        #    labels[root_arc] = ROOT
        #elif len(roots) > 1:
        #    label_probs[roots, ROOT] = 0
        #    new_labels = \
        #        np.argmax(label_probs[roots], axis=1)
        #    root_arc = np.where(arcs[tokens] == 0)[0] + 1
        #    labels[roots] = new_labels
        #    labels[root_arc] = ROOT
        heads_batch.append(head_preds)
        #labels_batch.append(labels)

    print(len(heads_batch))
    print(heads_batch[0].shape)
    print(heads_batch[4].shape)
    sys.exit()
    return heads_batch # (b, l, l)

def softmax2d(x): #Just doing softmax of row vectors
    y = x - np.max(x, axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y
