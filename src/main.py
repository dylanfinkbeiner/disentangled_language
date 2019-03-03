if __name__ == '__main__':
    args = get_args()

    MODEL_NAME = f'{args.model}'
    NUM_EPOCHS = args.epochs

    if(not args.eval):
        # Train model
        train(args)

    else:
        # Evaluate model
        pass
