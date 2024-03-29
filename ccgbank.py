import os
from tqdm import tqdm 

AUTO_DIR = '/corpora/ccgbank_1_1/data/AUTO'
OUT_DIR = '/home/AD/dfinkbei/corpora/ccgbank'

if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

def autos_to_merged():
    for i in tqdm(range(25), ascii=True, desc=f'Progress on ccgbank merging', ncols=80):
        chunk_dir = f'{i:02d}'
        curr_dir = os.path.join(AUTO_DIR, chunk_dir)
    
        merged_sents = []
        auto_list = sorted([a for a in os.listdir(curr_dir) if os.path.splitext(a)[-1] == '.auto'])
        for a in auto_list:
            with open(os.path.join(curr_dir, a), 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if l[0] == '(':
                        sent = auto_to_word_tag_pairs(l)
                        merged_sents.append(sent)


        out_file = f'{chunk_dir}.stag'
        with open(os.path.join(OUT_DIR, out_file), 'w') as f:
            for s in merged_sents:
                for pair in s:
                    f.write(f'{pair[0]} {pair[1]} {pair[2]}\n')
                f.write('\n')

def auto_to_word_tag_pairs(sent):
    leaves = []
    for i, _ in enumerate(sent[:-1]):
        if sent[i] + sent[i+1] == '<L':
            leaf_string = ''
            for inner in sent[i+2:]:
                if inner != '>':
                    leaf_string += inner
                else:
                    break
            leaf_list = leaf_string.strip().split(' ')
            word_tag = (leaf_list[3], leaf_list[2], leaf_list[0])

            leaves.append(word_tag)

    return leaves

if __name__ == '__main__':
    autos_to_merged()


