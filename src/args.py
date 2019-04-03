import os

from argparse import ArgumentParser
from configparser import ConfigParser

LOG_DIR = '../log'
DATA_DIR = '../data'
WEIGHTS_DIR = '../weights'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

def get_args():
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument('-c', '--config', help='Provide filename of a configuration file')

    # Parsing known args allows the same command line string to provide config filename as well as rest of args
    args, remaining_argv = config_parser.parse_known_args()

    defaults = {}

    if args.config:
        config = ConfigParser()
        config.read([args.config])
        defaults.update(dict(config.items('Defaults')))

    #parser = ArgumentParser(parents=[config_parser])
    parser = ArgumentParser()

    parser.add_argument('model', help='Name of model', default='default_model')
    parser.add_argument('--seed', type=int, dest='seed', default=7)

    # Evaluation options
    parser.add_argument('-e', '--eval', action='store_true', dest='eval', default=False)
    parser.add_argument('-emode', help='Evaluation mode setting.', type=int, dest='evalmode', default=0)

    parser.add_argument('-initdata', action='store_true', default=False)
    parser.add_argument('-initmodel', action='store_true', default=False)

    # Model hyperparameters
    parser.add_argument('-H', '--hsize', help='Size of LSTM hidden state.', dest='hsize', type=int, default=400)
    parser.add_argument('--semsize', help='Number of units of hidden state dedicated to semantic content.', dest='semsize', type=int, default=200)
    parser.add_argument('--synsize', help='Number of units of hidden state dedicated to syntactic content.', dest='synsize', type=int, default=200)

    # Training hyperparameters
    parser.add_argument('-b', '--batchsize', help='Size of batch', type=int, dest='batchsize', default=100)
    parser.add_argument('--sembatchsize', help='Size of batch for semantic similarity task.', type=int, dest='sembatchsize', default=100)
    parser.add_argument('--synbatchsize', help='Size of batch for syntactic similarity task.', type=int, dest='synbatchsize', default=100)
    parser.add_argument('-M', '--megasize', help='Number of batches in a megabatch.', type=int, dest='M', default=1)
    parser.add_argument('--epochs', help='Number of epochs in training.', type=int, default=5)
    parser.add_argument('--margin', help='Margin in semantic similarity objective function.', type=float, default=0.4)

    # Train mode
    parser.add_argument('-tm', '--trainingmode', help='Training mode setting.', type=int, default=0)

    # Overrides argument-level defaults
    parser.set_defaults(**defaults)

    return parser.parse_args(remaining_argv)


if __name__ == '__main__':
    args = get_args()
    print('Epochs: ', args.epochs)
    print("model ", args.model)
