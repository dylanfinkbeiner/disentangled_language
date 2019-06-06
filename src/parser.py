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


class Embeddings(nn.Module):
    def __init__(
            self,
            word_e_size=None,
            pos_e_size=None,
            pretrained_e=None,
            word_vocab_size=None,
            pos_vocab_size=None,
            hidden_size=None,
            embedding_dropout=None,
            padding_idx=None,
            unk_idx=None,
            device=None):
        super(Embeddings, self).__init__()

        self.unk_idx = unk_idx

        # Embeddings (words initialized to zero, per Jabberwocky paper) 
        self.word_emb = nn.Embedding(
                word_vocab_size,
                word_e_size,
                padding_idx=padding_idx)
        #self.word_emb.weight.data.copy_(
        #        torch.zeros(word_vocab_size, word_e_size))
        if pretrained_e is not None:
            print('Using pretrained word embeddings!')
            self.word_emb.weight.data.copy_(
                    torch.Tensor(pretrained_e))

        if pos_e_size is not None and pos_vocab_size is not None:
            self.pos_emb = nn.Embedding(
                pos_vocab_size,
                pos_e_size,
                padding_idx=padding_idx)

        # Dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    def forward(self, words, sent_lens, pos=None):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pos_flag = True if pos is not None else False

        # Zero-out "unk" word at test time
        #if not self.training:
        #    self.word_emb.weight.data[self.unk_idx,:] = 0.0 

        # Sort the words, pos, sent_lens (necessary for pack_padded_sequence)
        lens_sorted = sent_lens
        words_sorted = words
        if pos_flag:
            pos_sorted = pos
        indices = None
        if(sent_lens.shape[0] > 1):
            lens_sorted, indices = torch.sort(lens_sorted, descending=True)
            indices = indices.to(device)
            words_sorted = words_sorted.index_select(0, indices)
            del words
            if pos_flag:
                pos_sorted = pos_sorted.index_select(0, indices)
                del pos

        w_embs = self.word_emb(words_sorted) # (b, l, w_e)
        if pos_flag:
            p_embs = self.pos_emb(pos_sorted) # (b, l, p_e)

        dropout_input = w_embs if not pos_flag else torch.cat([w_embs, p_embs], -1)

        lstm_input = self.embedding_dropout(dropout_input) # (b, l, w_e + p_e)
        packed_lstm_input = pack_padded_sequence(
                lstm_input, lens_sorted, batch_first=True)

        return packed_lstm_input, indices, lens_sorted


class SemanticRNN(nn.Module):
    def __init__(
            self,
            input_size=None,
            hidden_size=None,
            num_layers=None,
            dropout=None,
            device=None):
        super(SemanticRNN, self).__init__()

        # LSTM
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
                bias=True)


    def forward(self, packed_lstm_input):
        outputs, _ = self.lstm(packed_lstm_input)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        return outputs


class SyntacticRNN(nn.Module):
    def __init__(
            self,
            input_size=None,
            hidden_size=None,
            num_layers=None,
            dropout=None,
            device=None):
        super(SyntacticRNN, self).__init__()

        # LSTM
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
                bias=True)

        # Must now manually do what was once automatic in Torch's 3-layer BiLSTM
        #self.output_dropout = nn.Dropout(p=dropout)

    def forward(self, packed_lstm_input):
        outputs, _ = self.lstm(packed_lstm_input)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True) 

        #outputs = self.output_dropout(outputs)
        return outputs


class FinalRNN(nn.Module):
    def __init__(
            self,
            input_size=None,
            hidden_size=None,
            num_layers=None,
            dropout=None,
            device=None):
        super(FinalRNN, self).__init__()

        # LSTM
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
                bias=True)

    def forward(self, packed_lstm_input):
        outputs, _ = self.lstm(packed_lstm_input)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True) 

        return outputs


class BiAffineAttention(nn.Module):
    def __init__(
            self,
            hidden_size=None,
            d_arc=None,
            d_rel=None,
            num_relations=None,
            arc_dropout=None,
            rel_dropout=None):
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
            arc_preds = pad_sequence([torch.LongTensor(s) for s in arc_preds],
                    batch_first=True, padding_value=0) # (b, l)

        # head_preds should be (b,l), not (b,l-1), as head of a word might be root
        arc_preds = arc_preds.to(device)

        d_rel, num_rel, _ = self.U_rel.size()

        for i in range(b):
            H_rel_head[i] = H_rel_head[i].index_select(0, arc_preds[i].view(-1))

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
            word_e_size=None,
            pos_e_size=None,  # Original Dozat/Manning paper uses 100
            pretrained_e=None,
            word_vocab_size=None,
            pos_vocab_size=None,
            syn_h=None, sem_h=None, final_h=None,
            syn_nlayers=None, sem_nlayers=None, final_nlayers=None,
            embedding_dropout=None,
            lstm_dropout=None,
            d_arc=None,
            d_rel=None,
            num_relations=None,
            arc_dropout=None,
            rel_dropout=None,
            padding_idx=None,
            unk_idx=None,
            device=None):
        super(BiaffineParser, self).__init__()

        self.Embeddings = Embeddings(
                word_e_size=word_e_size,
                pos_e_size=pos_e_size,
                word_vocab_size=word_vocab_size,
                pos_vocab_size=pos_vocab_size,
                pretrained_e=pretrained_e,
                embedding_dropout=embedding_dropout,
                padding_idx=padding_idx,
                unk_idx=unk_idx,
                device=device
                ).to(device)

        token_e_size = (word_e_size + pos_e_size) if pos_e_size is not None else word_e_size

        self.SyntacticRNN = SyntacticRNN(
                input_size=token_e_size,
                hidden_size=syn_h,
                num_layers=syn_nlayers,
                dropout=lstm_dropout,
                ).to(device)
        self.final_dropout = nn.Dropout(p=lstm_dropout).to(device)

        self.POSMLP = nn.Sequential(nn.Linear(2*syn_h, pos_vocab_size)).to(device)
                #nn.ReLU()
                #nn.Dropout(p=arc_dropout)

        self.SemanticRNN = SemanticRNN(
                input_size=token_e_size,
                hidden_size=sem_h,
                num_layers=sem_nlayers,
                dropout=lstm_dropout,
                ).to(device)
        
        self.FinalRNN = FinalRNN(
                input_size=(2*sem_h + 2*syn_h),
                hidden_size=final_h,
                num_layers=final_nlayers,
                dropout=lstm_dropout,
                ).to(device)

        self.BiAffineAttention = BiAffineAttention(
            hidden_size=final_h,
            d_arc=d_arc,
            d_rel=d_rel,
            num_relations=num_relations,
            arc_dropout=arc_dropout,
            rel_dropout=rel_dropout
            ).to(device)
        
    def forward(self, words, sent_lens, pos=None):
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

        packed_lstm_input, indices, lens_sorted = self.Embeddings(words, sent_lens, pos=pos)

        #Packed outputs
        syntactic_outputs = self.SyntacticRNN(packed_lstm_input)
        semantic_outputs = self.SemanticRNN(packed_lstm_input)
        syntactic_outputs = self.final_dropout(syntactic_outputs)
        semantic_outputs = self.final_dropout(semantic_outputs)

        #This might be unnecessary, should double check
        #syn_h = syntactic_outputs.shape[-1] // 2
        #sem_h = semantic_outputs.shape[-1] // 2
        #forward = torch.cat([syntactic_outputs[:,:,:syn_h], semantic_outputs[:,:,:sem_h]], dim=-1)
        #backward = torch.cat([syntactic_outputs[:,:,syn_h:], semantic_outputs[:,:,sem_h:]], dim=-1)

        final_inputs = torch.cat([syntactic_outputs, semantic_outputs], dim=-1)
        #final_inputs = torch.cat([syntactic_outputs, torch.zeros(semantic_outputs.shape).to(semantic_outputs.device)], dim=-1)
        final_inputs = pack_padded_sequence(final_inputs, lens_sorted, batch_first=True)
        
        final_outputs = self.FinalRNN(final_inputs)

        if(final_outputs.shape[0] > 1):
            final_outputs = unsort(final_outputs, indices)

        S_arc, S_rel, arc_preds = self.BiAffineAttention(final_outputs, sent_lens)

        return S_arc, S_rel, arc_preds


def unsort(batch, indices):
    indices_inverted = torch.argsort(indices)
    batch = batch.index_select(0, indices_inverted)
    return batch


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
