"""
This file contains the implementation of the unlearning methods.
"""


class UnlearningBaseClass:
    """
    Class representing an unlearning method.

    Args:
        dl (dict): Data loader object containing different datasets.
        batch_size (int): Batch size.
        num_classes (int): Number of classes in the dataset.
        model: Model to train.
        epochs (int): Number of epochs to train.
        loss_fn: Loss function.
        optimizer: Optimizer.
        lr_scheduler: Learning rate scheduler (it could be None).
        acc_forget_retrain: Accuracy threshold for retraining.
        is_early_stop (bool): Whether to use early stopping.
    """

    def __init__(
        self,
        dl,
        batch_size,
        num_classes,
        model,
        epochs,
        dataset,
    ) -> None:
        self.dl = dl
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = model
        self.epochs = epochs
        self.dataset = dataset
