import os
import sys
import subprocess

from main import predict_rels

from data_utils import conllu_to_sents

DATA

GOLD_CONLLU = ''
PREDICTED_CONLLU = ''

# Load model, load sentences in, parse sentence, write back out to a predicted_conllu file using dicts 


def main():

    # Get vocabs
    vocabs_pkl = os.path.join(DATA_DIR, 
            f'{os.path.splitext(CONLLU_FILE)[0]}_vocabs.pkl')

    data_pkl = os.path.join(DATA_DIR, 
            f'{os.path.splitext(CONLLU_FILE)[0]}_data.pkl')

    # Get data
    with open(vocabs_pkl, 'rb') as f:
        x2i_maps, i2x_maps = pickle.load(f)
    with open(data_pkl) as f:
        data_test = pickle.load(f)
    
    if not os.path.exists(vocabs_pkl) \
            or not os.path.exists(data_sdp_pkl) \
        print('Vocabs and/or data not initialized. Exiting.')
        exit()

    # Get loader
    data_loader = sdp_data_loader(data_test, 1, shuffle=False)

    # Get model

    parser.eval()
    with open(PREDICTED_CONLLU, 'w') as f:
        with torch.no_grad():
            for words, pos, sent_len in data_loader:

                _, S_rel, head_preds = parser(words, pos, sent_len)

                rel_preds = predict_rels(S_rel, sent_len)

                 = unnumericalize(words, pos, head_preds, rel_preds)

                for i in range(sent_len):
                    f.write(words[i], pos[i], head_preds[i], rel_preds[i], '\n')
                    f.write('\n')
