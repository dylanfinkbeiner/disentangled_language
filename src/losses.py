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


def loss_sem_rep(sem_h1, sem_h2, sem_hn1, sem_hn2=None, margin=None):
    '''
        Based on the loss function from Wieting et al (2018), where the
        BiLSTM hidden state is treated as a sentence embedding and the goal
        is to maximize cosine similarity of embeddings of paraphrases
        and minimize similarity of embeddings of "negative samples".

        Last checked for correctness on May 9
    '''

    #sem_h1 = torch.cat((h1[:,syn_size:h_size], h1[:,h_size+syn_size:]), dim=-1)
    #sem_h2 = torch.cat((h2[:,syn_size:h_size], h2[:,h_size+syn_size:]), dim=-1)
    #sem_hn1 = torch.cat((hn1[:,syn_size:h_size], hn1[:,h_size+syn_size:]), dim=-1)

    para_attract = F.cosine_similarity(sem_h1, sem_h2) # (b,sem_size), (b,sem_size) -> (b)

    neg1_repel = F.cosine_similarity(sem_h1, sem_hn1) # (b,sem_size), (b,sem_size) -> (b)

    losses = F.relu(margin - para_attract + neg1_repel) # (b)

    if sem_hn2 is not None:
        #sem_hn2 = torch.cat((hn2[:,syn_size:h_size], hn2[:,h_size+syn_size:]), dim=-1)
        neg2_repel = F.cosine_similarity(sem_h2, sem_hn2) # (b,sem_size), (b,sem_size) -> (b)
        losses += F.relu(margin - para_attract + neg2_repel)

    return losses.mean()


def loss_pos(logits, target_pos, pad_idx=-1):
    #Cross entropy
    loss_nn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    losses = loss_nn(logits.view(-1, logits.shape[-1]), target_pos.flatten())

    return losses.mean()


def loss_stag(logits, stag_targets, pad_idx=-1):
    loss_nn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    # logits are (b, l, len(pos_dict)), view takes it to (b*l, len(pos_dict))
    losses = loss_nn(logits.view(-1, logits.shape[-1]), stag_targets.flatten())

    return losses.mean()


