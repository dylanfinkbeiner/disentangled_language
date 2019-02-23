from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    parser.add_argument("-seed", type=int, dest='seed', default=7)

    parser.add_argument('-e', action="store_true", dest='eval', default=False)
    parser.add_argument('-initdata', action="store_true", default=True)
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



    #parser.add_argument("-dim", help="Dimension of model", type=int, default=300)
    ##parser.add_argument("-wordfile", help="Word embedding file", default='../data/paragram_sl999_czeng.txt')
    #parser.add_argument("-save", help="Whether to pickle model", type=int, default=0)
    #parser.add_argument("-samplingtype", help="Type of Sampling used: MAX, MIX, or RAND", default="MAX")

    #parser.add_argument("-eta", help="Learning rate", type=float, default=0.001)
    #parser.add_argument("-learner", help="Either AdaGrad or Adam", default="adam")
    #parser.add_argument("-model", help="Which model to use between wordaverage, maxpool, (bi)lstmavg, (bi)lstmmax")
    #parser.add_argument("-scramble", type=float, help="Rate of scrambling", default=0.3)
    #parser.add_argument("-max", type=int, help="Maximum number of examples to use (<= 0 means use all data)", default=0)
    #parser.add_argument("-loadmodel", help="Name of pickle file containing model", default=None)
    #parser.add_argument("-data", help="Name of data file containing paraphrases", default=None)
    #parser.add_argument("-wordtype", help="Either words or 3grams", default="words")
    #parser.add_argument("-random_embs", help="Whether to use random embeddings "
    #                                                 "and not pretrained embeddings", type = int, default=0)
    #parser.add_argument("-mb_batchsize", help="Size of megabatch", type=int, default=40)
    #parser.add_argument("-axis", help="Axis on which to concatenate hidden "
    #                                          "states (1 for sequence, 2 for embeddings)", type=int, default=1)
    #parser.add_argument("-combination_method", help="Type of combining models (either add or concat)")
    #parser.add_argument("-combination_type", help="choices are ngram-word, ngram-lstm, "
    #                                               "ngram-word-lstm, word-lstm")

    return parser.parse_args()
