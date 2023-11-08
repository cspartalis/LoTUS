import argparse

def set_config():
    '''
    Set the command-line arguments that a user can provide to the main script.

    Returns:
    class 'argparse.Namespace': Command-line arguments

    Example:
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', type=int, help="An example argument")
    args = parser.parse_args()
    print(args.foo)
    '''

    # Create parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cudnn", type=str, default="benchmark")
    parser.add_argument("--dataset", type=str, help="dataset to use [mnist, cifar-10, cifar-100, imagenet]")
    parser.add_argument("--model", type=str, help="model to use [resnet18, allcnn, vgg19]")
    parser.add_argument("--train", type=str, default="original", help="training method to use [original, retrain]")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--loss", type=str, default="cross_entropy", help="loss function to use [cross_entropy, weighted_cross_entropy]")
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer to use [sgd, adam]")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight regularization")
    parser.add_argument("--lr_scheduler", type=str, default="step", help="learning rate scheduler to use [step, cosine]")
    parser.add_argument("--scheduler_step1", type=int, default=30, help="first step for step learning rate scheduler")
    parser.add_argument("--scheduler_step2", type=int, default=50, help="second step for step learning rate scheduler")
    parser.add_argument("--early_stopping", type=int, default=10, help="0 means no early stopping, otherwise the number of epochs to wait")

    # Parse the arguments
    args = parser.parse_args()

    # Return arguments
    return args