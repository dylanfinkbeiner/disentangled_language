import sys
import time
import logging
from memory_profiler import profile

import numpy as np
import torch
from torch import unsqueeze
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, \
        pack_padded_sequence, pad_sequence

import mst

log = logging.getLogger(__name__)
file_handler = logging.FileHandler('../log/parser.log')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
log.addHandler(file_handler)
log.addHandler(stream_handler)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
log.setLevel(logging.DEBUG)

'''
    An implementation of Jabberwocky dropout-heavy
    BiAffine Attention Dependency Parser
'''

alpha = 40  # For calculating word dropout rates...

'''
Hypothesis: We can break up parser into to submodules:
    1) BiLSTM
        in: (words, pos, sent_len)
        out:  (lstm_out, (h_n, c_n)) [Should this be unpacked?]

    2) BiAffine Attention
        in: (lstm_out, sent_len)
        out: (arc_scores, rel_scores, head_preds)
'''


class BiLSTM(nn.Module):
    def __init__(
            self,
            word_e_size=100,
            pos_e_size=25,  # Original Dozat/Manning paper uses 100
            word_vocab_size=None,
            pos_vocab_size=None,
            hidden_size=400,
            lstm_layers=3,
            lstm_dropout=0.33,
            embedding_dropout=0.33,
            padding_idx=None):
        super(BiLSTM, self).__init__()

        # Embeddings (words initialized to zero) 
        self.word_emb = nn.Embedding(
                word_vocab_size,
                word_e_size,
                padding_idx=padding_idx)
        self.word_emb.weight.data.copy_(
                torch.zeros(word_vocab_size, word_e_size))

        self.pos_emb = nn.Embedding(
            pos_vocab_size,
            pos_e_size,
            padding_idx=padding_idx)

        # Dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # LSTM
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
                input_size=(word_e_size + pos_e_size),
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
                dropout=lstm_dropout)

    def forward(self, words, pos, sent_lens):
        '''
            input will be words, pos, sent_lens (NOT IN ORDER)
            except that words, pos will be tensors whose size depends on the
            maximum length
        '''

        h_size = self.hidden_size

        #if not self.train:
        #    self.word_emb.weight.data[,:] = 0.0 # Zero-out "unk" word at test time

        # Sort the words, pos, sent_lens
        lens_sorted, indices = torch.sort(torch.LongTensor(sent_lens), descending=True)
        words = words.index_select(0, indices) # NOTE Keep in mind, this is consuming additional memory!
        pos = pos.index_select(0, indices)

        w_embs = self.word_emb(words) # (b, l, w_e)
        p_embs = self.pos_emb(pos) # (b, l, p_e)

        lstm_input = self.embedding_dropout(
                torch.cat([w_embs, p_embs], -1)) # (b, l, w_e + p_e)

        packed_input = pack_padded_sequence(
                lstm_input, lens_sorted, batch_first=True)

        outputs, (h_n, c_n) = self.lstm(packed_input)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True) # (b, l, 2*hidden_size)

        # Un-sort
        indices_inverted = torch.argsort(indices)
        outputs = outputs.index_select(0, indices_inverted)

        return outputs, (h_n, c_n)


class BiAffineAttention(nn.Module):
    def __init__(
            self,
            hidden_size=400,
            d_arc=500,
            d_rel=100,
            num_relations=46,
            arc_dropout=0.33,
            rel_dropout=0.33):
        super(BiAffineAttention, self).__init__()

        # Arc-scoring MLPs
        self.h_arc_dep = nn.Sequential(
                nn.Linear(hidden_size*2, d_arc),
                nn.ReLU(),
                nn.Dropout(p=arc_dropout))
        self.h_arc_head = nn.Sequential(
                nn.Linear(hidden_size*2, d_arc),
                nn.ReLU(),
                nn.Dropout(p=arc_dropout))

        # Label-scoring MLPs
        self.h_rel_dep = nn.Sequential(
                nn.Linear(hidden_size*2, d_rel),
                nn.ReLU(),
                nn.Dropout(p=rel_dropout))
        self.h_rel_head = nn.Sequential(
                nn.Linear(hidden_size*2, d_rel),
                nn.ReLU(),
                nn.Dropout(p=rel_dropout))

        # Biaffine Attention parameters
        self.W_arc = nn.Parameter(torch.randn(d_arc, d_arc))
        self.b_arc = nn.Parameter(torch.randn(d_arc))

        self.U_rel = nn.Parameter(torch.randn(d_rel, num_relations, d_rel))
        self.W_rel = nn.Parameter(torch.randn(d_rel, num_relations))
        self.b_rel = nn.Parameter(torch.randn(num_relations))


    def forward(self, H, sent_lens):
        #Recap so far: H is a (b,l,800) tensor; first axis: sentence | second: word | third: lstm output
        H_arc_head = self.h_arc_head(H) # (b, l, d_arc)
        H_arc_dep  = self.h_arc_dep(H) # (b, l, d_arc)
        H_rel_head = self.h_rel_head(H) # (b, l, d_rel)
        H_rel_dep  = self.h_rel_dep(H) # (b, l, d_rel)

        b, l, d_arc = H_arc_head.size() # l for "longest_sentence"
        S_arc = torch.mm(H_arc_dep.view(b * l, d_arc), self.W_arc).view(b, l, d_arc) # Implementing W_arc * h_i_dep
        S_arc = torch.bmm(S_arc, H_arc_head.permute(0, 2, 1)) # (b, l, l)
        bias = torch.mm(H_arc_head.view(b*l, d_arc), self.b_arc.unsqueeze(1)) # (b*l, 1)
        #bias : Unsqueezing here allows intuitive matrix-vector multiply
        bias = bias.view(b, l, 1).permute(0, 2, 1) # (b, 1, l)
        S_arc += bias # (b, l, l) where logits vectors s_i are 3rd axis of S

        if self.training:  # Greedy
            head_preds = torch.argmax(S_arc, 2) # (b, l, l) -> (b, l)

        else:  # Single-rooted, acyclic graph of head-dependency relations
            head_preds = mst_preds(S_arc, sent_lens) # S:(b, l, l) -> [length-b list of np arrays]
            head_preds = pad_sequence([torch.Tensor(s).long() for s in head_preds],
                    batch_first=True, padding_value=0) # (b, l)

        d_rel, num_rel, _ = self.U_rel.size()

        for i in range(b):
            H_rel_head[i] = H_rel_head[i].index_select(0, head_preds[i].view(-1))

        # H_rel_head: Now the i-th row of this matrix is h_p_i^(rel_head), the MLP output for predicted head of ith word
        U_rel = self.U_rel.view(-1, num_rel * d_rel) # (d_rel, num_rel * d_rel)
        interactions = torch.mm(H_rel_head.view(b * l, d_rel), U_rel).view(b * l, num_rel, d_rel) # (b*l, num_rel, d_rel)
        interactions = torch.bmm(interactions, H_rel_dep.view(b * l, d_rel, 1)).view(b, l, num_rel) # (b*l, l, num_rel)
        sums = (torch.mm((H_rel_head + H_rel_dep).view(b*l, d_rel), self.W_rel) + self.b_rel).view(b, l, num_rel)
        S_rel = interactions + sums # (b, l, num_rel) where logits vectors l_i are 3rd axis of L

        return S_arc, S_rel, head_preds


class BiaffineParser(nn.Module):
    def __init__(
            self,
            word_e_size=100,
            pos_e_size=25,  # Original Dozat/Manning paper uses 100
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

        self.BiLSTM = BiLSTM(
                word_e_size=word_e_size,
                pos_e_size=pos_e_size,
                word_vocab_size=word_vocab_size,
                pos_vocab_size=pos_vocab_size,
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                embedding_dropout=embedding_dropout,
                lstm_dropout=lstm_dropout)

        self.BiAffineAttention = BiAffineAttention(
            hidden_size=hidden_size,
            d_arc=d_arc,
            d_rel=d_rel,
            num_relations=num_relations,
            arc_dropout=arc_dropout,
            rel_dropout=rel_dropout)
        
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

        outputs, (h_n, c_n) = self.BiLSTM(words, pos, sent_lens)
        return self.BiAffineAttention(outputs, sent_lens)


## From https://github.com/chantera/biaffineparser/blob/master/pytorch_model.py#L86
def mst_preds(S, sent_lens):
    heads_batch = []
    
    batch_logits = S.data.cpu().numpy() # Take to numpy arrays

    for sent_logits, true_length in zip(batch_logits, sent_lens):
        sent_probs = softmax2d(sent_logits[:true_length, :true_length]) # Select out THE ACTUAL SENTENCE
        head_preds = mst.mst(sent_probs) # NOTE Input to mst is softmax of arc scores

        heads_batch.append(head_preds)
        
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
        #labels_batch.append(labels)

    return heads_batch # (b, l, l)

def softmax2d(x): #Just doing softmax of row vectors
    y = x - np.max(x, axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y


def average_hiddens(H1, H2, sent_lens):
    H1 = H1.sum(axis=1)
    H2 = H2.sum(axis=1)

    #sent_lens[0] = torch.Tensor(sent_lens[0]).view(-1, 1)
    sent_lens = torch.Tensor(sent_lens).view(2, -1, 1)  # Column vector

    H1 / sent_lens[0]
    H2 / sent_lens[1]

