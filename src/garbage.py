#Collection of old code I'm nervous to delete forever

def syn_task_loader(syn_data, batch_size=None):
    sents_sorted = syn_data['sents_sorted']
    l2n = syn_data['l2n']
    l2b = syn_data['l2b']

    # NOTE Currently, we have no way of ensuring that every sentence is seen during training
    len_dist, len_array = get_lengths_distribution(l2n=l2n)
    while True:
        batch_lengths = sample_lengths(distribution=len_dist, lengths_array=len_array, batch_size=batch_size)
        batch_pairs = sample_pairs(sample_lengths=batch_lengths, l2b=l2b, granularity=syn_data['granularity'])

        s1 = []
        s2 = []
        target_scores = []
        for i, j, score in batch_pairs:
            s1.append(sents_sorted[i])
            s2.append(sents_sorted[j])
            target_scores.append(score)

        prepared_s1 = prepare_batch_sdp(s1)
        prepared_s2 = prepare_batch_sdp(s2)
        target_scores = torch.Tensor(target_scores)

        yield (prepared_s1, prepared_s2, target_scores)


#def sdp_corpus_stats(data, stats_pkl='../data/sdp_corpus_stats/sdp_corpus_stats.pkl', stats_readable ='../data/sdp_corpus_stats/readable_stats.txt', device=None):
def sdp_corpus_stats(data, stats_dir=None, min_length=2, max_len=40, device=None):
    if not os.path.isdir(stats_dir):
        os.mkdir(stats_dir)
    components_dir = os.path.join(stats_dir, 'components')
    if not os.path.isdir(components_dir):
        os.mkdir(components_dir)

    init_stats = True

    stats = {}

    data_sorted = sorted(data, key = lambda s : s.shape[0])
    bucket_dicts = build_bucket_dicts(data_sorted)
    l2n = bucket_dicts['l2n']
    i2c = bucket_dicts['i2c']
    l2c = bucket_dicts['l2c']

    # Create a copy of l2c devoid of useless lengths
    l2c_cleaned = copy.deepcopy(l2c)
    for l, n in l2n.items():
        if n < min_length:
            l2c_cleaned.pop(l)
        elif n > max_len:
            l2c_cleaned.pop(l)

    print('l2c has {len(l2c)} many keys after removal of lengths with less than 2 entries.')

    if init_stats:
        distribution, lengths_array = get_lengths_distribution(l2n, l2c_cleaned)

        l2r = length_to_results(data_sorted, l2c=l2c, device=device)

        with open(os.path.join(components_dir, 'l2r.pkl'), 'wb') as pkl:
            pickle.dump(l2r, pkl)

        l2t_UAS, ud = get_score_tensors(data_sorted, l2c=l2c, l2r=l2r, score_type='UAS', device=device)
        l2t_LAS, ld = get_score_tensors(data_sorted, l2c=l2c, l2r=l2r, score_type='LAS', device=device)

        avgs_UAS, ou = l2t_to_l2avg(l2t_UAS)
        avgs_LAS, ol = l2t_to_l2avg(l2t_LAS)

        stats['l2t_UAS'] = l2t_UAS
        stats['l2t_LAS'] = l2t_LAS
        stats['avgs_UAS'] = avgs_UAS
        stats['avgs_LAS'] = avgs_LAS
        stats['ou'] = ou
        stats['ol'] = ol
        stats['ud'] = ud
        stats['ld'] = ld

        with open(stats_pkl, 'wb') as f:
            pickle.dump(stats, f)
    else:
        with open(stats_pkl, 'rb') as f:
            stats = pickle.load(f)

    l2t_UAS = stats['l2t_UAS']
    l2t_LAS = stats['l2t_LAS'] 
    avgs_UAS = stats['avgs_UAS'] 
    avgs_LAS = stats['avgs_LAS'] 
    ou = stats['ou']
    ol = stats['ol']
    ud = stats['ud']
    ld = stats['ld']

    print('UAS averages: ', avgs_UAS)
    print('LAS averages: ', avgs_LAS)

    print('UAS overall: ', ou)
    print('LAS overall: ', ol)

    unique_len_counter = 0
    for n in l2n.values():
        if n <= 1:
            unique_len_counter += 1

    print('Unique lengths: ', unique_len_counter)

    with open(stats_readable, 'w') as f:
        f.write('Sentence length\tNum of length\tAvg UAS\t Avg LAS\n')
        for l, c in l2c.items():
            n = len(range(c[0], c[1]))
            f.write('{:10}\t{:10}\t{:10.3f}\t{:10.3f}\n'.format(
                l,
                n,
                avgs_UAS[l],
                avgs_LAS[l]))

        f.write(f'\nUnique lengths: {unique_len_counter}')
        f.write(f'\nNum dupes (by UAS): {ud}')
        f.write(f'\nNum dupes (by LAS): {ld}')


def l2t_to_l2avg(l2t):
    l2avg = {}
    avg_list = []

    for l, t in l2t.items():
        #nonzeros = t.flatten().index_select(0, t.nonzero().flatten())
        #avg = torch.mean(nonzeros).item()
        divisor = (t.shape[0] * (t.shape[0] - 1)) / 2
        avg = (t.sum() / divisor).item()
        avg_list.append(avg)
        l2avg[l] = avg
    
    overall_avg = torch.tensor(avg_list).mean().item()

    return l2avg, overall_avg


def build_l2b(sents_sorted, l2p=None, granularity=None, score_type=None, include_zeros=None):
    l2b = {}

    score_idx = 2 if score_type == 'UAS' else 3

    for l, pair_list in tqdm(l2p.items(), ascii=True, desc=f'Progress in building l2b', ncols=80):
        buckets = {}
        sorted_pair_list = sorted(pair_list, key = lambda p: p[score_idx], reverse=True)
        for x in np.arange(0, 1, granularity):
            bucket = []
            while(sorted_pair_list != [] 
                    and sorted_pair_list[-1][score_idx] <= x + granularity):
                curr_p = sorted_pair_list[-1]
                if curr_p[score_idx] != 0 or include_zeros:
                    bucket.append((curr_p[0], curr_p[1], curr_p[score_idx]))

                sorted_pair_list.pop(-1)

            buckets[x] = bucket

        l2b[l] = buckets

    return l2b


def get_lengths_distribution(l2n=None):
    distribution = np.zeros(len(l2n))
    lengths_array = np.zeros(len(l2n))

    total_sentences = 0
    for n in l2n.values():
        total_sentences += n

    for i, (l,n) in enumerate(l2n.items()):
        distribution[i] = n / total_sentences
        lengths_array[i] = l

    idx = np.argsort(lengths_array)
    lengths_array = lengths_array[idx]
    distribution = distribution[idx]

    return distribution, lengths_array


def sample_lengths(distribution=None, lengths_array=None, batch_size=None):
    sample = np.random.multinomial(1, distribution, size=batch_size)

    sample_idxs = np.argmax(sample, axis=1) # Takes one-hots to integers

    sample_lengths = lengths_array.take(sample_idxs)

    return sample_lengths


def sample_pairs(sample_lengths=None, l2b=None, granularity=None):
    sample_pairs = []
    quartiles = np.arange(0, 1, granularity)
    quart_samples = np.random.multinomial(1, [0.25]*4, size=len(sample_lengths))
    quart_samples = np.argmax(quart_samples, axis=1) # Takes one-hots to integers

    for l, q in zip(sample_lengths, quart_samples):
        quartile = quartiles[q]
        bucket = l2b[l][quartile]

        sample = np.random.multinomial(1, [1/len(bucket)]*len(bucket))
        sample_pair = bucket[np.argmax(sample)]

        sample_pairs.append(sample_pair)

    return sample_pairs


def forward_pos(parser, batch, args=None, data=None):
    device = data['device']
    parser.train()

    arc_targets = batch['arc_targets']
    rel_targets = batch['rel_targets']
    pos_targets = batch['pos'].to(device)
    sent_lens = batch['sent_lens'].to(device)
    
    lstm_input, indices, lens_sorted = parser.Embeddings(batch['words'].to(device), sent_lens)
    #lstm_input, indices, lens_sorted = parser.Embeddings(batch['words'].to(device), sent_lens, pos=batch['pos'].to(device))
    outputs = parser.SyntacticRNN(lstm_input)
    logits = parser.POSMLP(unsort(outputs, indices))

    loss_pos = losses.loss_pos(logits, pos_targets).cpu()
    
    #loss *= args.lr_syn

    return loss_pos

def get_paired_idx(idx: list, cutoffs: dict):
    '''
        produces a list of indices, paired to an index in idx, of a
        sentence of equal length
    '''
    paired_idx = []
    for i in idx:
        c = cutoffs[i]
        paired_i = random.randrange(c[0], c[1])
        is_unique_length = (c[0] == c[1] - 1)
        while (paired_i == i) and not is_unique_length:
            paired_i = random.randrange(c[0], c[1])
        paired_idx.append(paired_i)

    return paired_idx


def get_syntactic_scores(s1_batch, s2_batch, device=None):
    ''' Not a great name for the function: must change this later... '''
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    results = utils.attachment_scoring(
            arc_preds=s1_batch['arc_targets'].to(device), 
            rel_preds=s1_batch['rel_targets'].to(device), 
            arc_targets=s2_batch['arc_targets'].to(device), 
            rel_targets=s2_batch['rel_targets'].to(device), 
            sent_lens=s1_batch['sent_lens'].to(device), 
            include_root=False,
            keep_dim=True)

    return results


def build_l2p(sents_sorted, l2c=None):
    l2p = {}

    for l, c in tqdm(l2c.items(), ascii=True, desc=f'Progress in building l2p', ncols=80):
        idxs = list(range(c[0], c[1]))
        pairs = []

        for i, idx_i in enumerate(idxs[:-2]):
            s1_batch = []
            s2_batch = []
            for idx_j in idxs[i+1:]:
                s1_batch.append(sents_sorted[idx_i])
                s2_batch.append(sents_sorted[idx_j])

            results = get_syntactic_scores(
                    prepare_batch_sdp(s1_batch),
                    prepare_batch_sdp(s2_batch))

            uas_batch = results['UAS'].flatten().tolist() # (chunk_size) shaped tensor
            las_batch = results['LAS'].flatten().tolist()

            for idx_j, uas, las in zip(idxs[i+1:], uas_batch, las_batch):
                pairs.append( (idx_i, idx_j, uas, las) )

        l2p[l] = pairs
        print(f'Pairs size is {len(pairs)}')

    return l2p


#def length_to_results(data_sorted, l2c=None, device=None, chunk_size=100) -> dict:
#    ''' 'results', here, refers to the output of my get_syntactic_scores function '''
#
#    l2r = {}
#
#    for l, c in tqdm(l2c.items(), ascii=True, desc=f'Progress in building l2r', ncols=80):
#        idxs = list(range(c[0], c[1]))
#        n = len(idxs)
#
#        UAS_chunks = []
#        LAS_chunks = []
#        for i, idx_i in enumerate(idxs[:-2]):
#            s1_batch = []
#            s2_batch = []
#            for idx_j in idxs[i+1:]:
#                s1_batch.append(data_sorted[idx_i])
#                s2_batch.append(data_sorted[idx_j])
#        
#            results = get_syntactic_scores(
#                    prepare_batch_sdp(s1_batch),
#                    prepare_batch_sdp(s2_batch),
#                    device=device)
#
#            UAS_chunks.append(results['UAS'].flatten()) # (chunk_size) shaped tensor
#            LAS_chunks.append(results['LAS'].flatten())
#
#        # Stack up results from batched attachment scoring
#        UAS_l = torch.cat(UAS_chunks, dim=0)
#        LAS_l = torch.cat(LAS_chunks, dim=0)
#        expected_len = (n * (n-1)) / 2
#        if UAS_l.shape[0] != expected_len:
#            print(f'Expected len: {expected_len}, UAS_l len: {UAS_l.shape[0]}')
#            raise Exception
#        elif LAS_l.shape[0] != expected_len:
#            print(f'Expected len: {expected_len}, LAS_l len: {LAS_l.shape[0]}')
#            raise Exception
#
#        l2r[l] = {'UAS': UAS_l, 'LAS': LAS_l}
#
#    return l2r

def build_cutoff_dicts(sents_sorted: list) -> dict:
    '''
        inputs:
            sents_sorted - list of np arrays (conllu-formatted sentences)

        returns:
            a dictionary i2c, keys are indices in sorted data, values are lists with 2 elements,
            the first index in sents_sorted of a sentence of that length and the 
            (non-inclusive) final index
    '''
    i2c = dict()
    l2c = defaultdict(list)
    l2n = defaultdict(int)

    l_prev = sents_sorted[0].shape[0]
    l_max = sents_sorted[-1].shape[0]
    l2c[l_prev].append(0)
    l2n[l_prev] += 1
    for i, s in enumerate(sents_sorted[1:], start=1):
        l = s.shape[0]
        l2n[l] += 1
        if l > l_prev:
            l2c[l_prev].append(i)
            l2c[l].append(i)
        l_prev = l
    l2c[l_max].append(len(sents_sorted))

    for c in l2c.values():
        for i in range(c[0], c[1]):
            i2c[i] = c

    if len(i2c) != len(sents_sorted):
        print(f'i2c {len(i2c)} != sents_sorted {len(sents_sorted)}')
        raise Exception


    return {'l2c': dict(l2c), 'i2c': i2c, 'l2n': dict(l2n)}
#XXX in forward for biaffine parser
#if(words.shape[0] > 1):
#    pos = pos.index_select(0, indices)
#pos_tags = self.pe(pos)
#pos_tags = self.pe_drop(pos_tags)
#XXX


#XXX in __init__
#self.pe = nn.Embedding(
#    pos_vocab_size,
#    pos_e_size,
#    padding_idx=padding_idx).to(device)
#self.pe_drop = nn.Dropout(p=embedding_dropout).to(device)
#XXX

#                        if args.train_mode == 3:
#                            batch = next(loader_sdp_train)
#                            loss_par = forward_syntactic_parsing(
#                                    parser, 
#                                    batch=batch, 
#                                    args=args, 
#                                    data=data)
#                            loss_pos = forward_pos(
#                                    parser,
#                                    batch=batch,
#                                    args=args,
#                                    data=data)
#                            loss = loss_par + loss_pos


#def compare(conllus=[], stags=[]):
#    for c, s in zip(conllus, stags):
#
#        c_sents = conllu_to_sents(c)
#        s_sents = stag_to_sents(s)
#
#        while(len(c_sents) != 0):
#            s1 = c_sents[0]
#            s2 = s_sents[0]
#
#            if len(s1) != len(s2):
#                print(f'Problem in files {c} and {s}\nProblem sentences:\n{s1}\n{s2}')
#                breakpoint()
#                #break
#
#            c_sents.pop(0)
#            s_sents.pop(0)

#        for i in range(len(s_sents)):
#            s1 = c_sents[i]
#            s2 = s_sents[i]
#
#            for j in range(len(s1)):
#                if s1[j][1] != s2[j][0]:
#                    print(f'Problem in files {c} and {s}\nProblem sentences:\n{s1}\n{s2}')
#                    breakpoint()
#                    break


def loss_pos(logits, target_pos, pad_idx=-1):
    loss_nn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    losses = loss_nn(logits.view(-1, logits.shape[-1]), target_pos.flatten())

    return losses.mean()


