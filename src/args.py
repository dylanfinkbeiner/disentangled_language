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
    #parser.add_argument('-e', action='store_true', dest='eval', default=False)
    evalgroup = parser.add_mutually_exclusive_group()
    evalgroup.add_argument('-e', help='Evaluate, and provide flags for datasets to evaluate on.', nargs='*', type=int, default=None)
    evalgroup.add_argument('-ef', help='Evaluate, and provide filename to evaluate on.', default=None)

    parser.add_argument('-initdata', action='store_true', default=False)
    parser.add_argument('-initmodel', action='store_true', default=False)

    # Model hyperparameters
    parser.add_argument('-H', '--hsize', help='Size of LSTM hidden state.', dest='hsize', type=int, default=400)
    #parser.add_argument('--semsize', help='Number of units of hidden state dedicated to semantic content.', dest='semsize', type=int, default=200)
    parser.add_argument('--synsize', help='Number of units of hidden state dedicated to syntactic content.', dest='synsize', type=int, default=200)

    # Training hyperparameters
    parser.add_argument('-b', '--batchsize', help='Size of batch', type=int, dest='batchsize', default=100)
    parser.add_argument('--sembatchsize', help='Size of batch for semantic similarity task.', type=int, dest='sembatchsize', default=100)
    parser.add_argument('--synbatchsize', help='Size of batch for syntactic similarity task.', type=int, dest='synbatchsize', default=100)
    parser.add_argument('-M', '--megasize', help='Number of batches in a megabatch.', type=int, dest='M', default=1)
    parser.add_argument('--epochs', help='Number of epochs in training.', type=int, default=5)
    parser.add_argument('--margin', help='Margin in semantic similarity objective function.', type=float, default=0.4)

    # Train mode
    parser.add_argument('-tm', '--trainingmode', help='Training mode setting.', type=int, default=-1)

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
        args_dict.pop('trainingmode')
        args_dict.pop('eval')
        args_dict.pop('evalmode')
        new_defaults = {'Defaults' : args_dict}
        new_config = ConfigParser()
        new_config.read_dict(new_defaults)
        with open(new_file_path, 'w') as f:
            new_config.write(f)

    return args


if __name__ == '__main__':
    args = get_args()
    print('Epochs: ', args.epochs)
    print("model ", args.ef)
