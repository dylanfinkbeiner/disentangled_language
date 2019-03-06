from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    parser.add_argument("-seed", type=int, dest='seed', default=7)

    parser.add_argument('-e', action="store_true", dest='eval', default=False)
    parser.add_argument('-initdata', action="store_true", default=False)
    parser.add_argument('-initmodel', action="store_true", default=False)

    # File names
    parser.add_argument("-model", help="Name of model", default='default')

    # Model hyperparameters
    parser.add_argument("-numhidden", help="Number of epochs in training", dest='numhidden', type=int, default=400)

    # Training hyperparameters
    parser.add_argument("-batchsize", help="Size of batch", type=int, dest='batchsize', default=100)
    parser.add_argument("-M", help="Number of batches in a megabatch", type=int, dest='M', default=1)
    parser.add_argument("-epochs", help="Number of epochs in training", type=int, default=5)
    parser.add_argument("-margin", help="Margin in objective function", type=float, default=0.4)

    # Train mode
    parser.add_argument("-mode", help="Training mode.", type=int, default=0)

    return parser.parse_args()
