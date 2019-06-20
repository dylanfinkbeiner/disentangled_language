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
