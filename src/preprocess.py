import logging
import os
import pickle
from tqdm import tqdm
import copy
from collections import Counter

import matplotlib.pyplot as plt
from argparse import ArgumentParser

import data_utils

LOG_DIR = '../log'
DATA_DIR = '../data'

#CORPORA_DIR = '/corpora'
CORPORA_DIR = '/home/AD/dfinkbei/corpora'
STS_DIR = f'{DATA_DIR}/sts'
STS_INPUT = os.path.join(STS_DIR, 'input')
STS_GS = os.path.join(STS_DIR, 'gs')
WSJ_DIR = os.path.join(CORPORA_DIR, 'wsj')
CCG_DIR = os.path.join(CORPORA_DIR, 'ccgbank')
#BROWN_DIR = f'{CORPORA_DIR}/brown/dependencies'
BROWN_DIR = '../data/brown'
PARANMT_DIR = os.path.join(DATA_DIR, 'paranmt_5m')

log = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
log.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'preprocess.log'))
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)


class DataPaths:
    def __init__(self, filtered=False, glove_d=None, truncated=False):
        if truncated and glove_d == None:
            raise Exception

        fs = 'filtered' if filtered else 'unfiltered' # Filtered status
        trunc = 'intersectPTB' if truncated else ''

        voc = 'ptbvocab'
        if glove_d is not None and glove_d > 0:
            voc = f'glove{glove_d}{trunc}vocab'
    
        # Directories
        sdp_data_dir = os.path.join(DATA_DIR, 'sdp_processed')
        ss_data_dir = os.path.join(DATA_DIR, 'ss_processed')
        stag_data_dir = os.path.join(DATA_DIR, 'stag_processed')
        if not os.path.isdir(sdp_data_dir):
            os.mkdir(sdp_data_dir)
        if not os.path.isdir(ss_data_dir):
            os.mkdir(ss_data_dir)
        if not os.path.isdir(stag_data_dir):
            os.mkdir(stag_data_dir)
        if not os.path.isdir(os.path.join(PARANMT_DIR, 'pkl')):
            os.mkdir(os.path.join(PARANMT_DIR, 'pkl'))
        if not os.path.isdir(os.path.join(PARANMT_DIR, 'tagged')):
            os.mkdir(os.path.join(PARANMT_DIR, 'tagged'))
        if not os.path.isdir(os.path.join(STS_DIR, 'tagged')):
            os.mkdir(os.path.join(STS_DIR, 'tagged'))
    
        self.ptb_vocabs = os.path.join(sdp_data_dir, f'vocabs_{fs}.pkl')
        self.data_ptb = os.path.join(sdp_data_dir, f'data_ptb_{fs}_{voc}.pkl')
        print(f'PTB path is {self.data_ptb}')
        self.counts_wsj = os.path.join(sdp_data_dir, f'counts_wsj_{fs}.pkl')
        self.data_brown = os.path.join(sdp_data_dir, f'data_brown_{fs}_{voc}.pkl')
        self.data_brown_predicted = os.path.join(sdp_data_dir, f'data_brown_predictedpos_{fs}_{voc}.pkl')
        print(f'Brown path is {self.data_brown}')

        self.stag_vocabs = os.path.join(stag_data_dir, f'stag_vocabs.pkl')
        self.data_stag = os.path.join(stag_data_dir, f'data_ptb_stag_{voc}.pkl')

        self.glove_data = os.path.join(DATA_DIR, 'glove', f'glove_{glove_d}_data.pkl')
        self.glove_w2v = os.path.join(DATA_DIR, 'glove', f'glove_{glove_d}_w2v.pkl')
        self.ss_train_base = os.path.join(PARANMT_DIR, 'pkl', f'{fs}_{voc}_')
        self.ss_test = os.path.join(ss_data_dir, f'ss_test_{fs}_{voc}.pkl')
    

def preprocess(args):
    paths = DataPaths(
            filtered=args.filter, 
            glove_d=args.glove_d,
            truncated=args.truncated) 

    x2i, i2x = {}, {}
    # ptb vocabs are eternally those constructed from training set, nothing to worry about
    if os.path.exists(paths.ptb_vocabs):
        with open(paths.ptb_vocabs, 'rb') as f:
            x2i, i2x = pickle.load(f)

    if args.pretrained_voc:
        with open(paths.glove_data, 'rb') as pkl:
            embedding_data = pickle.load(pkl)

        #if args.truncated:
        #    ptb_words = set(x2i['word'].keys())
        #    glove_words = set(embedding_data['w2i'].keys())
        #    just_glove = glove_words - ptb_words

        #    ##XXX
        #    #just_ptb = ptb_words - paranmt_words
        #    #print(len(just_ptb))
        #    #sdp_data_dir = os.path.join(DATA_DIR, 'sdp_processed')
        #    #patt = os.path.join(sdp_data_dir, f'data_ptb_unfiltered_glove100vocab.pkl')
        #    #with open(patt, 'rb') as f:
        #    #    _, counts = pickle.load(f)
        #    #cts = []
        #    #xs=[]
        #    #for w in just_ptb:
        #    #    cts.append((w, counts[w]))
        #    #    xs.append(counts[w])
        #    #
        #    #ctssort = sorted(cts, key=lambda x: x[1])
        #    #breakpoint()
        #    #exit()
        #    #XXX

        #    print(f'{glove_words} many glove words, {just_glove} many unique to glove')
        #    count_before = len(glove_words)
        #    for w in just_glove:
        #        embedding_data['i2w'].pop(embedding_data['w2i'][w], None)
        #        embedding_data['w2i'].pop(w, None)


        ## Here's where things would go wrong if we didn't pop the just-glove words off
        #x2i['word'].update(embedding_data['w2i'])
        #i2x['word'].update(embedding_data['i2w'])



    if args.sdp != []:
        if 'ptb' in args.sdp:
            log.info(f'Initializing Penn Treebank WSJ data (including vocabs).')
            ptb_conllus = sorted(
                    [os.path.join(WSJ_DIR, f) for f in os.listdir(WSJ_DIR)])

            raw_data_ptb, x2i_ptb, i2x_ptb, word_counts = data_utils.build_ptb_dataset(
                    ptb_conllus, 
                    filter_sents=args.filter)
            with open(paths.ptb_vocabs, 'wb') as f:
                pickle.dump((x2i_ptb, i2x_ptb), f)
            x2i.update(x2i_ptb)
            i2x.update(i2x_ptb)
            if args.pretrained_voc:
                x2i['word'].update(embedding_data['w2i'])
                i2x['word'].update(embedding_data['i2w'])

            data_ptb = {}
            for split, raw_data in raw_data_ptb.items():
                data_ptb[split] = data_utils.numericalize_sdp(raw_data, x2i)
            with open(paths.data_ptb, 'wb') as f:
                pickle.dump((data_ptb, word_counts), f)

        if 'brown' in args.sdp:
            log.info(f'Initializing Brown corpus data.')
            brown_conllus = [os.path.join(BROWN_DIR, f) for f in os.listdir(BROWN_DIR)]
            data_brown, word_counts = data_utils.build_sdp_dataset(
                    brown_conllus, 
                    x2i=x2i, 
                    filter_sents=args.filter)

            wc = word_counts['cf']
            total = sum(wc.values())
            unique = list(wc)
            print(total)
            print(len(unique))
            with open(paths.data_brown, 'wb') as f:
                pickle.dump(data_brown, f)
            
            if args.predict_brown_pos:
                data_brown_predicted, _ =  data_utils.build_sdp_dataset(
                        brown_conllus, 
                        x2i=x2i, 
                        filter_sents=args.filter,
                        predict_pos=True)
                with open(paths.data_brown_predicted, 'wb') as f:
                    pickle.dump(data_brown_predicted, f)


    # NOTE This logic is very weird
    if args.glove_d and not args.pretrained_voc:
        w2v = data_utils.build_pretrained_w2v(word_v_file=f'../data/glove/glove.6B.{args.glove_d}d.top100k.txt')
        with open(paths.glove_w2v, 'wb') as pkl:
            pickle.dump(w2v, pkl)

        glove_data = data_utils.build_embedding_data(w2v, w2i=x2i['word'], i2w=i2x['word'])
        with open(paths.glove_data, 'wb') as pkl:
            pickle.dump(glove_data, pkl)


    if 'train' in args.ss:
        log.info(f'Initializing semantic similarity train data.')

        chunks_txt = sorted(list(os.listdir(os.path.join(PARANMT_DIR, 'txt'))))
        for chunk in chunks_txt:
            print(f'Processing chunk {chunk}')
            raw_sents_path = os.path.join(PARANMT_DIR, 'tagged', f'{os.path.splitext(chunk)[0]}-tagged.pkl')
            if args.pos_only:
                raw_sent_pairs = data_utils.paraphrase_to_sents(os.path.join(PARANMT_DIR, 'txt', chunk))

                with open(raw_sents_path, 'wb') as pkl:
                    pickle.dump(raw_sent_pairs, pkl)
            else:
                with open(raw_sents_path, 'rb') as pkl:
                    raw_sent_pairs = pickle.load(pkl)

                train_path_chunk = paths.ss_train_base + f'{os.path.splitext(chunk)[0]}.pkl'
                train_ss_chunk = data_utils.build_ss_dataset(
                        raw_sent_pairs, 
                        gs='', 
                        x2i=x2i,
                        filter_sents=args.filter)
                with open(train_path_chunk, 'wb') as pkl:
                    pickle.dump(train_ss_chunk, pkl)


    if 'test' in args.ss:
        log.info(f'Initializing semantic similarity test data.')

        test_ss = {}
        years = os.listdir(STS_INPUT)
        for year in years:
            raw_sents_path = os.path.join(STS_DIR, 'tagged', f'{year}-tagged.pkl')
            if args.pos_only:
                raw_sent_pairs = data_utils.paraphrase_to_sents(os.path.join(STS_INPUT, year))
                with open(raw_sents_path, 'wb') as pkl:
                    pickle.dump(raw_sent_pairs, pkl)
            else:
                with open(raw_sents_path, 'rb') as pkl:
                    raw_sent_pairs = pickle.load(pkl)
                test_ss[year] = data_utils.build_ss_dataset(
                        raw_sent_pairs,
                        gs=os.path.join(STS_GS, year),
                        x2i=x2i,
                        filter_sents=args.filter)
        with open(paths.ss_test, 'wb') as pkl:
            pickle.dump(test_ss, pkl)


    if args.stag:
        log.info(f'Initializing supertagging data.')
        stag_files = sorted(
                [os.path.join(CCG_DIR, f) for f in os.listdir(CCG_DIR)])

        raw_stag_sents, s2i, i2s = data_utils.build_stag_dataset(stag_files=stag_files)
        with open(paths.stag_vocabs, 'wb') as f:
            pickle.dump((s2i, i2s), f)
        x2i['stag'] = s2i
        #i2x['stag'] = i2s
        if args.pretrained_voc:
            x2i['word'].update(embedding_data['w2i'])
            #i2x['word'].update(embedding_data['i2w'])

        data_stag = {}
        for split, raw_sents in raw_stag_sents.items():
            data_stag[split] = data_utils.numericalize_stag(raw_sents, x2i)
        with open(paths.data_stag, 'wb') as f:
            pickle.dump(data_stag , f)
    

    print('Finished!')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-ss', help='Initialize a part (or all) of the semantic similarity data.', dest='ss', nargs='*', type=str, default=[])
    parser.add_argument('--pos', action='store_true', dest='pos_only', default=False)

    parser.add_argument('-sdp', help='Initialize a part (or all) of the syntactic parsing data.', dest='sdp', nargs='*', type=str, default=[])

    parser.add_argument('-stag', help='Initialize supertagging data.', dest='stag', action='store_true')

    pretrained_emb = parser.add_mutually_exclusive_group()
    pretrained_emb.add_argument('-g', help='Initialize data corresponding to glove word embeddings.', dest='glove_d', type=int, default=None)
    parser.add_argument('--bpos', action='store_true', dest='predict_brown_pos', default=False)
    parser.add_argument('--pevoc', action='store_true', help='Shall we use the word vocabulary derived from the chosen pretrained embeddings?', dest='pretrained_voc', default=False)
    parser.add_argument('--trunc', action='store_true', help='Vocab at the intersection of Glove and PTB?', dest='truncated')

    parser.add_argument('-f', help='Should sentences be filtered?', action='store_true', dest='filter', default=False)

    args = parser.parse_args()

    preprocess(args)

