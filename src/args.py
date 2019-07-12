import os

from argparse import ArgumentParser
#from configparser import ConfigParser

LOG_DIR = '../log'
DATA_DIR = '../data'
WEIGHTS_DIR = '../weights'
CONFIG_DIR = '../config'
EXPERIMENTS_DIR = '../experiments'

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)
if not os.path.isdir(CONFIG_DIR):
    os.mkdir(CONFIG_DIR)
if not os.path.isdir(EXPERIMENTS_DIR):
    os.mkdir(EXPERIMENTS_DIR)


def get_args():
    #config_parser = ArgumentParser(add_help=False)
    #config_parser.add_argument('-c', '--config_file', help='Provide filename of a configuration file')

    ## Parsing known args allows the same command line string to provide config filename as well as rest of args
    #args, remaining_argv = config_parser.parse_known_args()

    #defaults = {}

    #if args.config_file:
    #    config = ConfigParser()
    #    config.read([os.path.join(CONFIG_DIR, args.config_file + '.cfg')])
    #    defaults.update(dict(config.items('Defaults')))

    parser = ArgumentParser()

    parser.add_argument('model', help='Name of model', default='default_model')
    parser.add_argument('-seed', type=int, dest='seed', default=7)
    parser.add_argument('-cuda', type=int, dest='cuda', default=0)
    parser.add_argument('-auto', action='store_true', dest='autopilot')

    # Evaluation options
    sdp_eval = parser.add_mutually_exclusive_group()
    sdp_eval.add_argument('-e', help='Evaluate, and provide flags for datasets to evaluate on.', nargs='*', type=int)
    sdp_eval.add_argument('-ef', help='Evaluate, and provide filename to evaluate on.')
    parser.add_argument('-esem', action='store_true', dest='evaluate_semantic', default=False)
    parser.add_argument('-estag', action='store_true', dest='evaluate_stag', default=False)

    # Data options
    parser.add_argument('-filter', help='Should sentences be filtered?', action='store_true', dest='filter') 
    parser.add_argument('-gloved', help='Should we use glove data, and if so what size embeddings?', dest='glove_d', type=int, default=None)
    parser.add_argument('-trunc', action='store_true', help='yadda?', dest='truncated')

    # Model hyperparameters
    parser.add_argument('-we', help='Size of word embeddings.', dest='we', type=int, default=100)
    parser.add_argument('-pe', help='Size of pos embeddings.', dest='pe', type=int, default=0)
    parser.add_argument('-zw', action='store_true', help='Init word embeds as 0 matrix?', dest='zero_we')
    parser.add_argument('-dropstyle', help='Unif? or Freq?', dest='drop_style', type=str, default='freq')
    parser.add_argument('-sumfb', help='Should we sum the averaged forward and backward hiddens in average_hiddens()?', action='store_true', dest='sum_f_b', default=False)
    parser.add_argument('-synh', help='Number of units of hidden state dedicated to syntactic content.', dest='syn_h', type=int, default=69)
    parser.add_argument('-semh', help='Number of units of hidden state dedicated to semantic content.', dest='sem_h', type=int, default=69)
    parser.add_argument('-finh', help='Size of hidden state in final LSTM.', dest='final_h', type=int, default=400)
    parser.add_argument('-synl', help='Number of syntactic LSTM layers.', dest='syn_nlayers', type=int, default=2)
    parser.add_argument('-seml', help='Number of semantic LSTM layers.', dest='sem_nlayers', type=int, default=1)
    parser.add_argument('-finl', help='Number of final LSTM layers.', dest='final_nlayers', type=int, default=1)

    # Learning rates / scaling hyperparameters
    parser.add_argument('-lr', help='Learning rate.', dest='lr', type=float, default=1e-3)
    parser.add_argument('-lrsdp', help='Learning rate scaling syntactic dependency parsing.', dest='lr_sdp', type=float, default=1)
    parser.add_argument('-lrstag', help='Learning rate scaling supertagging loss.', dest='lr_stag', type=float, default=1)
    parser.add_argument('-lrsem', help='Learning rate scaling semantic similarity loss.', dest='lr_sem', type=float, default=1)

    # Dropout
    parser.add_argument('-wd', help='Word dropout rate (alpha parameter in case of freq-based).', dest='word_dropout', type=float, default=0.)
    parser.add_argument('-pd', help='POS tag dropout rate.', dest='pos_dropout', type=float, default=0.)
    parser.add_argument('-edrop', help='', dest='embedding_dropout', type=float, default=0.33)
    parser.add_argument('-ldrop', help='', dest='lstm_dropout', type=float, default=0.33)
    parser.add_argument('-sdrop', action='store_true', help='Init word embeds as 0 matrix?', dest='semantic_dropout')
    parser.add_argument('-wdsem', action='store_true', help='Do word dropout in semantic task?', dest='wd_sem')
    parser.add_argument('-wdstag', action='store_true', help='Do word dropout in supertag task?', dest='wd_stag')


    # Training hyperparameters
    parser.add_argument('-sdpbs', help='Batch size, syntactic dependency parsing.', type=int, dest='sdp_bs', default=100)
    parser.add_argument('-stagbs', help='Batch size, supertagging.', type=int, dest='stag_bs', default=100)
    parser.add_argument('-sembs', help='Batch size, semantic similarity.', type=int, dest='sem_bs', default=100)
    parser.add_argument('-M', help='Number of batches in a megabatch.', type=int, dest='M', default=1)
    parser.add_argument('-esp', help='Early stop point.', type=int, dest='earlystop_pt', default=5)
    parser.add_argument('-epochs', help='Number of epochs in training.', type=int, default=1)
    parser.add_argument('-margin', help='Margin in semantic similarity objective function.', dest='margin', type=float, default=0.4)
    parser.add_argument('-nchunks', help='Number of 100k-sentence-pair chunks of SS data from the filtered ParaNMT-50m dataset.', type=int, dest='n_chunks', default=1)
    parser.add_argument('-scramble', help='Probability with which a given paraphrase pair will get scrambled in semantic training.', dest='scramble', type=float, default=0.3)
    parser.add_argument('-2negs', help='Shall we get a negative sample for both sentences in a paraphrase pair?', action='store_true', dest='two_negs')
    parser.add_argument('-tunk', help='Shall we train the <unk> token embedding, or always drop it out during training?', action='store_true', dest='train_unk')
    parser.add_argument('-layerdrop', help='Drop entire layer with this prob.', dest='layer_drop', type=float, default=0.)

    # Adversarial
    parser.add_argument('-advstag', help='Adversarial supertagging?', action='store_true', dest='adv_stag')
    parser.add_argument('-startepoch', help='On what epoch (1,2,...) should adversarial regularization begin?', type=int, dest='start_epoch', default=2)
    parser.add_argument('-scaleAS', help='Scale adversarial supertagging loss.', dest='scale_adv_stag', type=float, default=1.)

    # Train mode
    #parser.add_argument('-tm', help='Training mode setting.', dest='train_mode', type=int, default=-1)
    parser.add_argument('-tm', help='List of training mode flags.', dest='train_mode', nargs='+', type=int, default=[0])

    # Check weights
    parser.add_argument('-w', action='store_true', help='Just check the weights of the transformation to FinalRNN hiddens.', dest='w')

    # Save configuration?
    parser.add_argument('-sc', help='Name of new file to save configuration to.', default='')

    # Warning: overrides argument-level defaults
    #parser.set_defaults(**defaults)

    #args = parser.parse_args(remaining_argv)
    args = parser.parse_args()

    # Save configuration
    if args.sc:
        new_file_path = os.path.join(CONFIG_DIR, args.sc + '.cfg')
        if os.path.exists(new_file_path):
            reply = input('Config file already exists. Overwrite? [y/n] ')
            if reply != 'y' :
                exit()
        args_dict = dict(vars(args))
        args_dict.pop('model')
        args_dict.pop('train_mode')
        args_dict.pop('e')
        args_dict.pop('ef')
        new_defaults = {'Defaults' : args_dict}
        new_config = ConfigParser()
        new_config.read_dict(new_defaults)
        with open(new_file_path, 'w') as f:
            new_config.write(f)

    return args


if __name__ == '__main__':
    args = get_args()
    print('Epochs: ', args.epochs)
    print("ef ", args.ef)
    print("e ", args.e)
    print("syn_size ", args.syn_size)
    print("lr_syn ", args.lr_syn)
