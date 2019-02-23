import os
import sys
import subprocess

from main import predict_rels


CONLLU_FILE = ''
CONLLU_OUT = ''

# Load model, load sentences in, parse sentence, write back out to a predicted_conllu file using dicts 


def main():

    #Get data
    vocabs_pkl = os.path.join(DATA_DIR, 
            f'{os.path.splitext(CONLLU_FILE)[0]}_vocabs.pkl')

    #Get loader
    sdp_data_loader(data, shuffle=False)
    batch_size = 1

    #Get model

    with open(PREDICTED_CONLLU, 'w') as f:

        for words, pos, sent_len in data_loader:

            _, S_rel, head_preds = parser(words, pos, sent_len)

            rel_preds = predict_rels(S_rel, sent_len)

             = unnumericalize(words, pos, head_preds, rel_preds)

            for i in range(sent_len):
                f.write(words[i], pos[i], head_preds[i], rel_preds[i], '\n')
