import os

from argparse import ArgumentParser
from configparser import ConfigParser

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
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument('-c', '--config_file', help='Provide filename of a configuration file')

    # Parsing known args allows the same command line string to provide config filename as well as rest of args
    args, remaining_argv = config_parser.parse_known_args()

    defaults = {}

    if args.config_file:
        config = ConfigParser()
        config.read([os.path.join(CONFIG_DIR, args.config_file + '.cfg')])
        defaults.update(dict(config.items('Defaults')))

    parser = ArgumentParser()

    parser.add_argument('model', help='Name of model', default='default_model')
    parser.add_argument('--seed', type=int, dest='seed', default=7)

    # Evaluation options
    evalgroup = parser.add_mutually_exclusive_group()
    evalgroup.add_argument('-e', help='Evaluate, and provide flags for datasets to evaluate on.', nargs='*', type=int)
    evalgroup.add_argument('-ef', help='Evaluate, and provide filename to evaluate on.')

    parser.add_argument('--initdata', action='store_true', default=False)
    parser.add_argument('--initmodel', action='store_true', default=False)

    # Model hyperparameters
    parser.add_argument('-H', '--hsize', help='Size of LSTM hidden state.', dest='h_size', type=int, default=400)
    parser.add_argument('--synsize', help='Number of units of hidden state dedicated to syntactic content.', dest='syn_size', type=int, default=200)
    parser.add_argument('--lrsyn', help='Learning rate for optimization during syntactic task.', dest='lr_syn', type=float, default=2e-3)
    parser.add_argument('--lrsem', help='Learning rate for optimization during semantic task.', dest='lr_sem', type=float, default=2e-3)

    # Training hyperparameters
    parser.add_argument('-b', '--batchsize', help='Size of batch', type=int, dest='batchsize', default=100)
    parser.add_argument('--sembatchsize', help='Size of batch for semantic similarity task.', type=int, dest='sem_batchsize', default=100)
    parser.add_argument('--synbatchsize', help='Size of batch for syntactic parsing task.', type=int, dest='syn_batchsize', default=100)
    parser.add_argument('-M', '--megasize', help='Number of batches in a megabatch.', type=int, dest='M', default=1)
    parser.add_argument('--epochs', help='Number of epochs in training.', type=int, default=5)
    parser.add_argument('--margin', help='Margin in semantic similarity objective function.', type=float, default=0.4)

    # Train mode
    parser.add_argument('-tm', help='Training mode setting.', dest='train_mode', type=int, default=-1)

    # Save configuration?
    parser.add_argument('-sc', help='Name of new file to save configuration to.', default='')

    # Warning: overrides argument-level defaults
    parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_argv)

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
    print("hsize ", args.h_size)
    print("syn_size ", args.syn_size)
    print("lr_syn ", args.lr_syn)
