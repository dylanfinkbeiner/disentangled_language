import torch
import torch.nn.functional as F


def loss_arcs(S_arc, arc_targets, pad_idx=-1):
    '''
    S - should be something like a tensor w/ shape
        (batch_size, sent_len, sent_len); also, these are
        arcs scores BEFORE softmax applied

    arcs - should be a list of integers (the indices)
    '''
    # For input to cross_entropy, shape must be (b, C, ...) where C is number of classes
    return F.cross_entropy(S_arc.permute(0,2,1).cpu(), arc_targets, ignore_index=pad_idx)


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

    sem_h1 = torch.cat((h1[:,syn_size:h_size], h1[:,h_size+syn_size:]), dim=-1)
    sem_h2 = torch.cat((h2[:,syn_size:h_size], h2[:,h_size+syn_size:]), dim=-1)
    sem_hn = torch.cat((hn[:,syn_size:h_size], hn[:,h_size+syn_size:]), dim=-1)

    para_attract = F.cosine_similarity(sem_h1, sem_h2) # (b,sem_size), (b,sem_size) -> (b)

    neg_repel = F.cosine_similarity(sem_h1, sem_hn) # (b,sem_size), (b,sem_size) -> (b)

    losses = F.relu(margin - para_attract + neg_repel) # (b)

    return losses.sum()


def loss_syn_rep(h_batch, h_paired, scores, syn_size=None, h_size=None):
    '''
        inputs:
            h_batch - (b, 2*h_size) tensor
            h_paired - (b, 2*h_size) tensor
            scores - weights per sentence pair of batch, their LAS "similarity"
            syn_size - size, in units, of syntactic representation component of hidden state

        outputs:
            losses, where loss for sentence pair (x,y) is 1-cos(x,y) * score(x,y)
    '''
    # All should be (b, l , syn_size) tensors
    syn_batch = torch.cat((h_batch[:,0:syn_size], h_batch[:,h_size:h_size+syn_size]), dim=-1)
    syn_paired = torch.cat((h_paired[:,0:syn_size], h_paired[:,h_size:h_size+syn_size]), dim=-1)

    losses = 1 - F.cosine_similarity(syn_batch, syn_paired, dim=-1)

    losses *= scores.view(-1)

    return losses.sum()
    

