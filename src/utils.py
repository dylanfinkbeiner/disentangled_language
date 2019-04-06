import torch

def attachment_scoring(head_preds=None, 
        rel_preds=None, 
        head_targets=None, 
        rel_targets=None, 
        sent_lens=None, 
        root_included=False, 
        keep_dim=False):
    '''
        input:
            head_preds::Tensor - Has shape (b, l), -1 padded
            rel_preds::Tensor -  Has shape (b, l), -1 padded
            head_targets::Tensor - (-1)-padded (b, l) tensor of ints
            rel_targets::Tensor - (-1)-padded (b, l) tensor of ints


        returns:
            UAS - average number of correct head predictions
            LAS - average number of correct relation predictions
    '''

    sent_lens = torch.Tensor(sent_lens).view(-1, 1)
    b, l = head_preds.shape

    # This way we can be sure padding values do not contribute to score when we do .eq() calls
    head_preds = torch.where(
            head_targets != -1,
            head_preds,
            torch.zeros(head_preds.shape).long())
    rel_preds = torch.where(
            rel_targets != -1,
            rel_preds,
            torch.zeros(rel_preds.shape).long())

    # Tensors with 1s in locations of correct predictions
    #NOTE this could be optimized later to avoid sparse matrices
    correct_heads = head_preds.eq(head_targets).float()
    correct_rels = rel_preds.eq(rel_targets).float()

    # We get per-sentence averages, then average across the batch
    UAS = correct_heads.sum(1, True) # (b,l) -> (b,1)
    UAS = UAS - 1 if root_included else UAS
    UAS /= (sent_lens -1 if root_included else sent_lens)
    if not keep_dim:
        UAS = UAS.sum() / b

    LAS = (correct_heads * correct_rels).sum(1, True)
    LAS = LAS - 1 if root_included else LAS
    LAS /= (sent_lens -1 if root_included else sent_lens)
    if not keep_dim:
        LAS = LAS.sum() / b

    return UAS, LAS
