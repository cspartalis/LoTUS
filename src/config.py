import argparse


def _str2bool(v):
    """
    Convert a string representation of a boolean value to its corresponding boolean value.

    Args:
        v (str): The string representation of the boolean value.

    Returns:
        bool: The corresponding boolean value.
    """
    return v.lower() in "true"


def set_config():
    """
    Set the command-line arguments that a user can provide to the main script.

    Returns:
    class 'argparse.Namespace': Command-line arguments

    Example:
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', type=int, help="An example argument")
    args = parser.parse_args()
    print(args.foo)
    """

    # fmt: off
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cudnn", type=str, default="slow")
    parser.add_argument("--dataset", type=str, help="dataset to use [mnist, cifar-10, cifar-100, imagenet, mufac, tissuemnist]")
    parser.add_argument("--model", type=str, help="model to use [resnet18, vgg19, allcnn, vit]")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loss", type=str, default="cross_entropy", help="loss function to use [cross_entropy, weighted_cross_entropy]")
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer to use [sgd, adam]")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight regularization")
    parser.add_argument("--is_lr_scheduler", type=_str2bool, help="whether to use a learning rate scheduler")
    parser.add_argument("--warmup_epochs", type=int, help="number of epochs to warm up the learning rate")
    parser.add_argument("--is_early_stop", type=_str2bool, help="early stopping when training")
    parser.add_argument("--patience", type=int, help="number of epochs to wait before early stopping")
    parser.add_argument("--method", type=str, default=None, help="method to use for unlearning [finetuning, neggrad, relabel, unrolling, boundary_shring, zapping]")
    parser.add_argument("--is_class_unlearning", type=_str2bool, help="whether to unlearn a class")
    parser.add_argument("--class_to_forget", type=str, default=None, help="class to forget")
    parser.add_argument("--is_zapping", type=_str2bool)
    parser.add_argument("--subset_size", type=float, default=0.3)
    # MLflow arguments
    parser.add_argument("--registered_model", default=None, type=str)

    # Parse the arguments
    args = parser.parse_args()

    # fmt: on
    # Return arguments
    return args
