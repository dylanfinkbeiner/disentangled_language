import os
import logging

import numpy as np
import torch
from torch import unsqueeze
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, \
        pack_padded_sequence, pad_sequence

import mst

LOG_DIR = '../log/'
LOG_PATH = os.path.join(LOG_DIR, 'parser.log')

log = logging.getLogger('__parser__')
file_handler = logging.FileHandler(LOG_PATH)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
log.addHandler(file_handler)
log.addHandler(stream_handler)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
log.setLevel(logging.DEBUG)


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
            padding_idx=None,
            unk_idx=None):
        super(BiLSTM, self).__init__()

        self.unk_idx = unk_idx

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
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        h_size = self.hidden_size

        # Zero-out "unk" word at test time
        if not self.training:
            self.word_emb.weight.data[self.unk_idx,:] = 0.0 

        # Sort the words, pos, sent_lens (necessary for pack_padded_sequence)
        lens_sorted = sent_lens
        words_sorted = words
        pos_sorted = pos
        if(sent_lens.shape[0] > 1):
            lens_sorted, indices = torch.sort(lens_sorted, descending=True)
            indices = indices.to(device)
            words_sorted = words_sorted.index_select(0, indices)
            pos_sorted = pos_sorted.index_select(0, indices)
            del words
            del pos

        w_embs = self.word_emb(words_sorted) # (b, l, w_e)
        p_embs = self.pos_emb(pos_sorted) # (b, l, p_e)

        lstm_input = self.embedding_dropout(torch.cat([w_embs, p_embs], -1))

        packed_input = pack_padded_sequence(
                lstm_input, lens_sorted, batch_first=True)

        outputs, (h_n, c_n) = self.lstm(packed_input)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True) # (b, l, 2*hidden_size)

        # Un-sort
        if(sent_lens.shape[0] > 1):
            indices_inverted = torch.argsort(indices)
            outputs = outputs.index_select(0, indices_inverted)

        return outputs, (h_n, c_n)


class BiAffineAttention(nn.Module):
    def __init__(
            self,
            hidden_size=400,
            d_arc=500,
            d_rel=100,
            num_relations=None,
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
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
            arc_preds = torch.argmax(S_arc, 2) # (b, l, l) -> (b, l)

        else:  # Single-rooted, acyclic graph of head-dependency relations
            arc_preds = mst_preds(S_arc, sent_lens) # S:(b, l, l) -> [length-b list of np arrays]
            arc_preds = pad_sequence([torch.Tensor(s).long() for s in arc_preds],
                    batch_first=True, padding_value=0) # (b, l)

        # head_preds should be (b,l), not (b,l-1), as head of a word might be root
        arc_preds = arc_preds.to(device)

        d_rel, num_rel, _ = self.U_rel.size()

        for i in range(b):
            H_rel_head[i] = H_rel_head[i].index_select(0, head_preds[i].view(-1))

        # H_rel_head: Now the i-th row of this matrix is h_p_i^(rel_head), the MLP output for predicted head of ith word
        U_rel = self.U_rel.view(-1, num_rel * d_rel) # (d_rel, num_rel * d_rel)
        interactions = torch.mm(H_rel_head.view(b * l, d_rel), U_rel).view(b * l, num_rel, d_rel) # (b*l, num_rel, d_rel)
        interactions = torch.bmm(interactions, H_rel_dep.view(b * l, d_rel, 1)).view(b, l, num_rel) # (b*l, l, num_rel)
        # Multiply W_rel on the left by the sum of the the H's
        sums = (torch.mm((H_rel_head + H_rel_dep).view(b*l, d_rel), self.W_rel) + self.b_rel).view(b, l, num_rel)
        S_rel = interactions + sums # (b, l, num_rel) where logits vectors l_i are 3rd axis of L

        return S_arc, S_rel, arc_preds


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
            num_relations=None,
            embedding_dropout=0.33,
            lstm_dropout=0.33,
            arc_dropout=0.33,
            rel_dropout=0.33,
            padding_idx=None,
            unk_idx=None):
        super(BiaffineParser, self).__init__()

        self.BiLSTM = BiLSTM(
                word_e_size=word_e_size,
                pos_e_size=pos_e_size,
                word_vocab_size=word_vocab_size,
                pos_vocab_size=pos_vocab_size,
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                embedding_dropout=embedding_dropout,
                lstm_dropout=lstm_dropout,
                padding_idx=padding_idx,
                unk_idx=unk_idx)

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
            words - LongTensor
            pos - LongTensor
            sent_lens - list of integers

        outs:
            S_arc - Tensor containing scores for arcs
            S_rel - Tensor containing scores for
            arc_preds - Tensor of predicted arcs
        '''

        outputs, (h_n, c_n) = self.BiLSTM(words, pos, sent_lens)
        return self.BiAffineAttention(outputs, sent_lens)


## From https://github.com/chantera/biaffineparser/blob/master/pytorch_model.py#L86
def mst_preds(S_arc, sent_lens):
    arcs_batch = []
    
    batch_logits = S_arc.data.cpu().numpy() # Take to numpy arrays

    for sent_logits, true_length in zip(batch_logits, sent_lens):
        sent_probs = softmax2d(sent_logits[:true_length, :true_length]) # Select out THE ACTUAL SENTENCE (including ROOT token)
        arc_preds = mst.mst(sent_probs) # NOTE Input to mst is softmax of arc scores
        arcs_batch.append(arc_preds)
        
    return arcs_batch # (b, l, l)


def softmax2d(x): #Just doing softmax of row vectors
    y = x - np.max(x, axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y
