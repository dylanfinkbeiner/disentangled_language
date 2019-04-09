import torch
import torch.functional as F

def loss_heads(S_arc, head_targets, pad_idx=-1):
    '''
    S - should be something like a tensor w/ shape
        (batch_size, sent_len, sent_len); also, these are
        head scores BEFORE softmax applied

    heads - should be a list of integers (the indices)
    '''
    # For input to cross_entropy, shape must be (b, C, ...) where C is number of classes
    return F.cross_entropy(S_arc.permute(0,2,1).cpu(), head_targets, ignore_index=pad_idx)


def loss_rels(S_rel, rel_targets, pad_idx=-1):
    '''
    L - should be tensor w/ shape (batch_size, sent_len, d_rel)

    rels - should be a list of dependency relations as they are indexed in the dict
    '''

    return F.cross_entropy(S_rel.permute(0,2,1).cpu(), rel_targets, ignore_index=pad_idx)


def loss_sem_rep(h1, h2, hn, margin=0.4, syn_size=None, h_size=None):
    '''
        Based on the loss function from Wieting et al (2018), where the
        BiLSTM hidden state is treated as a sentence embedding and the goal
        is to maximize cosine similarity of embeddings of paraphrases
        and minimize similarity of embeddings of "negative samples".
    '''

    #f_para_attract = F.cosine_similarity(h1[:,syn_size:h_size], h2[:,syn_size:h_size]) # (b,sem_size), (b,sem_size) -> (b)
    #b_para_attract = F.cosine_similarity(h1[:,h_size+syn_size:], h2[:,h_size+syn_size:]) # (b,sem_size), (b,sem_size) -> (b)

    sem_h1 = torch.cat((h1[:,syn_size:h_size], h1[:,h_size+syn_size:]), dim=-1)
    sem_h2 = torch.cat((h2[:,syn_size:h_size], h2[:,h_size+syn_size:]), dim=-1)
    sem_hn = torch.cat((hn[:,syn_size:h_size], hn[:,h_size+syn_size:]), dim=-1)

    para_attract = F.cosine_similarity(sem_h1, sem_h2) # (b,sem_size), (b,sem_size) -> (b)

    #f_neg_repel = F.cosine_similarity(h1[:,syn_size:h_size], hn[:,syn_size:h_size]) # (b,2*d), (b,2*d) -> (b)
    #b_neg_repel = F.cosine_similarity(h1[:,h_size+syn_size:], hn[:,h_size+syn_size:]) # (b,2*d), (b,2*d) -> (b)

    neg_repel = F.cosine_similarity(sem_h1, sem_hn) # (b,sem_size), (b,sem_size) -> (b)

    losses = F.relu(margin - para_attract + neg_repel) # (b)

    return losses.sum()


def loss_syn_rep(outputs_batch, outputs_paired, scores, syn_size=None, h_size=None):
    '''
        inputs:
            outputs_batch - (b, 2*h_size) tensor
            outputs_paired - (b, 2*h_size) tensor
            scores - weights per sentence pair of batch, their LAS "similarity"
            syn_size - size, in units, of syntactic representation component of hidden state

        outputs:
            losses, where loss for sentence pair (x,y) is 1-cos(x,y) * score(x,y)
    '''
    # All should be (b, l , syn_size) tensors
    #f_batch = outputs_batch[:,0:syn_size]
    #b_batch = outputs_batch[:,h_size:h_size+syn_size]
    #f_paired = outputs_paired[:,0:syn_size]
    #b_paired = outputs_paired[:,h_size:h_size+syn_size]
    syn_batch = torch.cat((outputs_batch[:,0:syn_size], outputs_batch[:,h_size:h_size+syn_size]), dim=-1)
    syn_paired = torch.cat((outputs_paired[:,0:syn_size], outputs_paired[:,h_size:h_size+syn_size]), dim=-1)

    # (b)
    #f_loss = 1 - F.cosine_similarity(f_batch, f_paired, dim=-1)
    #b_loss = 1 - F.cosine_similarity(b_batch, b_paired, dim=-1)

    losses = 1 - F.cosine_similarity(syn_batch, syn_paired, dim=-1)

    #losses = f_loss + b_loss

    losses *= scores.view(-1)

    return losses.sum()
    

