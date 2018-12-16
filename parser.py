import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
	Should be implementation of Jabberwocky dropout-heavy
	BiAffine Dependency Parser, specifically using frequency based
	dropout with alpha=40
'''

alpha = 40

class BiaffineParser(nn.Module):
	def __init__(   word_e_size=100,
			pos_e_size=25,
			vocab_size,
			pos_vocab_size,
			hidden_size=400,
			num_layers=1,
			d_arc=100,
			d_rel=500,
			num_relations=46,
			input_dropout=0.33):
		super(BiaffineParser, self).__init__()

		#Embeddings
		self.word_em = nn.Embedding(vocab_size, word_e_size)
		self.pos_em  = nn.Embedding(pos_vocab_size, pos_e_size)

		#Dropout
		self.in_dropout = nn.Dropout(p=input_dropout)

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
				nn.Linear(hidden_sizei*2, d_rel),
				nn.ReLU())
		self.h_rel_head = nn.Sequential(
				nn.Linear(hidden_size*2, d_rel),
				nn.ReLU())

		#Biaffine Attention
		self.W_arc = torch.randn(d_arc, d_arc)
		self.b_arc = torch.randn(d_arc, 1)

		self.U_rel = torch.randn(d_rel, d_rel, num_relations)
		self.W_rel = torch.randn(num_relations, d_rel)
		self.b_rel = torch.randn(num_relations, 1)


	def forward(self, x_words, x_pos, sent_len):
		#Embeddings
		w_embs = self.word_em(x_words)
		p_embs = self.pos_em(x_pos)
		embs   = torch.cat([w_embs, p_embs], -1)
		lstm_input = self.in_dropout(embs)

		#Feed to LSTM
		H, (h_n,c_n) = self.lstm(lstm_input) #XXX currently assuming lstm outputs a list of output states for each word...
		# h_n and c_n will have shape (2, #batches, hidden_size) if LSTM 1-layer deep

		#MLPs XXX if you feed a list of h_k's in, does the MLP output a list of outputs?
		H_arc_head = self.h_arc_head(H)
		H_arc_dep  = self.h_arc_dep(H)
		H_rel_head = self.h_rel_head(H)
		H_rel_dep  = self.h_rel_dep(H)

		#Arc-score ('s' is input to softmax)
		#s = torch.zeros(sent_len, self.d_arc, 1)
		HW = torch.mm(H_arc_head, self.W_arc) 
		Hb = torch.mm(H_arc_head, self.b_arc)

		S = torch.tensor([ nn.Softmax(torch.mm(HW, H_arc_dep[i]) for i in range(sent_len)) ])

		#Collect predicted heads of words by greedy maximum
		p_heads = torch.zeros(sent_len, 1)
		for i in range(sent_len):
			_, p_heads[i] = torch.max(s[i], 1) #Probably not the max along correct dim

		#Label-score
		U
		l = torch.zeros(sent_len, self_d_rel, 1)
		
		#Pre-softmax scores for heads, labels
		return S, L
