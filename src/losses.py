import torch
import torch.nn.functional as F


def loss_arcs(S_arc, arc_targets, pad_idx=-1):
    # For input to cross_entropy, shape must be (b, C, ...) where C is number of classes
    return F.cross_entropy(S_arc.permute(0,2,1), arc_targets, ignore_index=pad_idx)


def loss_rels(S_rel, rel_targets, pad_idx=-1):
    return F.cross_entropy(S_rel.permute(0,2,1), rel_targets, ignore_index=pad_idx)


def loss_sem_rep(sem_h1, sem_h2, sem_hn1, sem_hn2=None, margin=None):
    para_attract = F.cosine_similarity(sem_h1, sem_h2) # (b,sem_size), (b,sem_size) -> (b)
    neg1_repel = F.cosine_similarity(sem_h1, sem_hn1) # (b,sem_size), (b,sem_size) -> (b)
    losses = F.relu(margin - para_attract + neg1_repel) # (b)

    if sem_hn2 is not None:
        neg2_repel = F.cosine_similarity(sem_h2, sem_hn2) # (b,sem_size), (b,sem_size) -> (b)
        losses += F.relu(margin - para_attract + neg2_repel)

    return losses.mean()


def loss_stag(logits, stag_targets, pad_idx=-1):
    loss_nn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    losses = loss_nn(logits.view(-1, logits.shape[-1]), stag_targets.flatten())

    return losses.mean()


def loss_ptag(logits, pos_targets, pad_idx=-1):
    loss_nn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    losses = loss_nn(logits.view(-1, logits.shape[-1]), pos_targets.flatten())

    return losses.mean()
