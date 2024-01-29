"""
This module contains classes and function to load and process data.

Raises:
    ValueError: If the dataset is not implemented yet.

Returns:
    tuple: A tuple containing the data loaders and dataset.
"""

import os

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms

from imagenet_utils import TinyImageNet
from medmnist_utils import PneumoniaMNIST, TissueMNIST
from mufac_utils import MUFAC
from seed import set_work_init_fn  # pylint: disable=import-error

DATA_DIR = os.path.expanduser("~/data/")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnlearningDataLoader:
    """
    Base class for data loaders.

    Args:
            dataset (str): Name of the dataset to load.
            batch_size (int): Number of samples per batch to load.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
            tuple: A tuple containing the data loaders and dataset sizes.
    """

    def __init__(self, dataset, batch_size, seed=0, frac_per_class_forget=0.1):
        """
        Initializes a DataUtils object.

        Args:
                dataset: The dataset to use.
                batch_size: The batch size to use.
                seed: The random seed to use.
                percentage_per_class_forget: The number of samples per class to forget.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.frac_per_class_forget = frac_per_class_forget
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.forget_loader = None
        self.retain_loader = None
        self.classes = None
        self.input_channels = None
        self.image_size = None
        self.label_to_class = None
        self.class_to_idx = None
        self.idx_to_class = None

    def load_data(self):
        """
        Loads the specified dataset and returns the corresponding dataloaders and dataset sizes.

        Returns:
        dataloaders (dict): A dictionary containing the dataloaders for the train and validation sets.
        dataset_sizes (dict): A dictionary containing the sizes of the train and validation sets.
        """

        data_transforms = {
            "cifar-train": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            ),
            "cifar-val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            ),
            "imagenet-train": transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                    ),
                ]
            ),
            "imagenet-val": transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                    ),
                ]
            ),
            "mnist-train": transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.1307], [0.3081]),
                ]
            ),
            "mnist-val": transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize([0.1307], [0.3081]),
                ]
            ),
            "pcam-train": transforms.Compose(
                [
                    transforms.Resize(96),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "pcam-val": transforms.Compose(
                [
                    transforms.Resize(96),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "mufac-train": transforms.Compose(
                [
                    transforms.Resize(128),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                ]
            ),
            "mufac-val": transforms.Compose(
                [
                    transforms.Resize(128),
                    transforms.ToTensor(),
                ]
            ),
            "tissuemnist-train": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.103], [0.03]),
                ]
            ),
            "tissuemnist-val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.1], [0.03]),
                ]
            ),
            "pneumoniamnist-train": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5719], [0.02]),
                ]
            ),
            "pneumoniamnist-val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.57], [0.02]),
                ]
            ),
            "pneumoniamnist-test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.56], [0.02]),
                ]
            ),
        }

        if self.dataset == "mufac":
            self.input_channels = 3
            self.image_size = 128
            data_train = MUFAC(
                meta_data_path=DATA_DIR
                + "./custom_korean_family_dataset_resolution_128/custom_train_dataset.csv",
                image_directory=DATA_DIR
                + "./custom_korean_family_dataset_resolution_128/train_images",
                transform=data_transforms["mufac-train"],
            )
            data_val = MUFAC(
                meta_data_path=DATA_DIR
                + "./custom_korean_family_dataset_resolution_128/custom_val_dataset.csv",
                image_directory=DATA_DIR
                + "./custom_korean_family_dataset_resolution_128/val_images",
                transform=data_transforms["mufac-val"],
            )
            data_test = MUFAC(
                meta_data_path=DATA_DIR
                + "./custom_korean_family_dataset_resolution_128/custom_test_dataset.csv",
                image_directory=DATA_DIR
                + "./custom_korean_family_dataset_resolution_128/test_images",
                transform=data_transforms["mufac-val"],
            )

        elif self.dataset == "cifar-10":
            self.input_channels = 3
            self.image_size = 32
            data_train = datasets.CIFAR10(
                root=DATA_DIR,
                transform=data_transforms["cifar-train"],
                train=True,
                download=True,
            )

            held_out = datasets.CIFAR10(
                root=DATA_DIR,
                transform=data_transforms["cifar-val"],
                train=False,
                download=True,
            )
        elif self.dataset == "cifar-100":
            self.input_channels = 3
            self.image_size = 32
            data_train = datasets.CIFAR100(
                root=DATA_DIR,
                transform=data_transforms["cifar-train"],
                train=True,
                download=True,
            )

            held_out = datasets.CIFAR100(
                root=DATA_DIR,
                transform=data_transforms["cifar-val"],
                train=False,
                download=True,
            )
        elif self.dataset == "imagenet":
            self.input_channels = 3
            self.image_size = 64
            data_train = TinyImageNet(
                DATA_DIR + "tiny-imagenet-200",
                is_train=True,
                transform=data_transforms["imagenet-train"],
            )
            held_out = TinyImageNet(
                DATA_DIR + "tiny-imagenet-200",
                is_train=False,
                transform=data_transforms["imagenet-val"],
            )
        elif self.dataset == "mnist":
            self.input_channels = 1
            self.image_size = 28
            data_train = datasets.MNIST(
                root=DATA_DIR,
                transform=data_transforms["mnist-train"],
                train=True,
                download=True,
            )
            held_out = datasets.MNIST(
                root=DATA_DIR,
                transform=data_transforms["mnist-val"],
                train=False,
                download=True,
            )
        elif self.dataset == "pcam":
            self.input_channels = 3
            self.image_size = 96
            data_train = datasets.PCAM(
                root=DATA_DIR,
                transform=data_transforms["pcam-train"],
                split="train",
                download=True,
            )
            data_val = datasets.PCAM(
                root=DATA_DIR,
                transform=data_transforms["pcam-val"],
                split="val",
                download=True,
            )
            data_test = datasets.PCAM(
                root=DATA_DIR,
                transform=data_transforms["pcam-val"],
                split="test",
                download=True,
            )
        elif self.dataset == "tissuemnist":
            self.input_channels = 1
            self.image_size = 28
            data_train = TissueMNIST(
                root=DATA_DIR,
                transform=data_transforms["tissuemnist-train"],
                split="train",
                download=True,
            )
            data_val = TissueMNIST(
                root=DATA_DIR,
                transform=data_transforms["tissuemnist-val"],
                split="val",
                download=True,
            )
            data_test = TissueMNIST(
                root=DATA_DIR,
                transform=data_transforms["tissuemnist-val"],
                split="test",
                download=True,
            )
        elif self.dataset == "pneumoniamnist":
            self.input_channels = 1
            self.image_size = 28
            data_train = PneumoniaMNIST(
                root=DATA_DIR,
                transform=data_transforms["pneumoniamnist-train"],
                split="train",
                download=True,
            )
            data_val = PneumoniaMNIST(
                root=DATA_DIR,
                transform=data_transforms["pneumoniamnist-val"],
                split="val",
                download=True,
            )
            data_test = PneumoniaMNIST(
                root=DATA_DIR,
                transform=data_transforms["pneumoniamnist-test"],
                split="test",
                download=True,
            )
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

        self.classes = data_train.classes

        # Stratified splitting held-out set to test and val sets.
        if (
            self.dataset != "pcam"
            and self.dataset != "mufac"
            and self.dataset != "tissuemnist"
            and self.dataset != "pneumoniamnist"
        ):
            labels = held_out.targets
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.5, random_state=self.seed
            )
            val_idx, test_idx = next(sss.split(held_out, labels))
            data_val = torch.utils.data.Subset(held_out, val_idx)
            data_test = torch.utils.data.Subset(held_out, test_idx)

        image_datasets = {"train": data_train, "val": data_val, "test": data_test}
        # change list to Tensor as the input of the models
        dataloaders = {}
        dataloaders["train"] = torch.utils.data.DataLoader(
            image_datasets["train"],
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )
        dataloaders["val"] = torch.utils.data.DataLoader(
            image_datasets["val"],
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )
        dataloaders["test"] = torch.utils.data.DataLoader(
            image_datasets["test"],
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )

        if self.dataset == "mufac":
            # Fixed split for MUFAC, bc there are photos of the same person many
            # times and in different ages. We want to make sure that there is no
            # overalpping.
            train_data_forget = Subset(data_train, list(range(0, 1500)))
            train_data_retain = Subset(data_train, list(range(1500, len(data_train))))
        else:
            train_data_forget, train_data_retain = self._split_data_forget_retain(
                data_train
            )
        image_datasets["forget"] = train_data_forget
        image_datasets["retain"] = train_data_retain
        dataloaders["forget"] = torch.utils.data.DataLoader(
            train_data_forget,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )
        dataloaders["retain"] = torch.utils.data.DataLoader(
            train_data_retain,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )

        dataset_sizes = {
            x: len(image_datasets[x])
            for x in ["train", "forget", "retain", "val", "test"]
        }

        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.test_loader = dataloaders["test"]
        self.forget_loader = dataloaders["forget"]
        self.retain_loader = dataloaders["retain"]

        if self.dataset == "mufac":
            self.label_to_class = data_train.label_to_class
            self.class_to_idx = data_train.class_to_idx
            self.idx_to_class = data_train.idx_to_class
        # else:
        #     # TODO: Check if this can be applied to other datasets.
        #     raise ValueError(f"Dataset {self.dataset} not supported.")

        return dataloaders, dataset_sizes

    def _split_data_forget_retain(self, data_train):
        """
        Spit train data to forget and retain sets.

        Args:
            data_train (dataste): Training dataset.

        Split:
            train_data_forget (subset): Subset with forget samples.
            train_data_retain (subset): Subset with retain samples.
        """

        train_data_forget = []
        train_data_retain = []
        classes = data_train.classes
        if data_train.targets is type(torch.Tensor):
            targets = data_train.targets.detach().clone()
        else:
            targets = torch.Tensor(data_train.targets)
        for class_name in classes:
            indices = torch.where(targets == data_train.class_to_idx[class_name])[0]
            num_indices = len(indices)
            num_forget_samples_per_class = int(num_indices * self.frac_per_class_forget)
            indices_forget = indices[:num_forget_samples_per_class]
            indices_retain = indices[num_forget_samples_per_class:]
            train_data_forget.append(
                torch.utils.data.Subset(data_train, indices_forget)
            )
            train_data_retain.append(
                torch.utils.data.Subset(data_train, indices_retain)
            )
        train_data_forget = torch.utils.data.ConcatDataset(train_data_forget)
        train_data_retain = torch.utils.data.ConcatDataset(train_data_retain)
        return train_data_forget, train_data_retain

    def get_samples_per_class(self, split: str):
        """
        Returns a dictionary with the number of samples per class for a given split.

        Args:
            split (str): The split to get the samples for. Must be one of "train", "val", "test", "forget", or "retain".

        Returns:
            dict: A dictionary with the number of samples per class for the given split.
        """
        samples_per_class = {}
        # switch case condintion for split
        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        elif split == "forget":
            loader = self.forget_loader
        elif split == "retain":
            loader = self.retain_loader
        else:
            raise ValueError(f"Split {split} not supported.")

        samples_per_class = {i: 0 for i in range(len(self.classes))}
        for _, labels in loader:
            for label in labels:
                samples_per_class[label.item()] += 1
        return samples_per_class

    def _get_mock_forget_dataset(self, original_model, alpha=0):
        """
        This function returns the inputs and targets of the mocked forget samples.
        Args:
            alpha (float): ε[0,1/num_classes]
              If alpha is 0, then the target probabilities are all equal to 1/num_classes.
              If alpha is 1/num_classes, then the target probability of the mock_target is 1 and all the others are 0.
        """
        # Re-assign targets of forget samples to be the second most probable class
        original_model.eval()
        forget_inputs = []
        mock_forget_targets = []
        with torch.inference_mode():
            for input, target in self.forget_loader.dataset:
                input = input.to(DEVICE)
                with torch.no_grad():
                    outputs = original_model(input.unsqueeze(0))

                    # Mock target should be predicted with the highest probability possible
                    if target != outputs.argsort()[0][-1]:
                        mock_target = outputs.argsort()[0][-1]
                    else:
                        mock_target = outputs.argsort()[0][-2]

                    mock_target = torch.tensor(mock_target)

                    # ### This code snippet transform hard targets to soft targets
                    # num_classes = outputs.shape[1]
                    # assert alpha >= 0 and alpha <= 1 / num_classes
                    # soft_mock_target = torch.zeros(num_classes) + 1 / num_classes
                    # # Add alpha/num_classes to target tensors where the value is 1
                    # soft_mock_target[mock_target] += alpha * (num_classes - 1)
                    # soft_mock_target[soft_mock_target == 1 / num_classes] -= alpha
                    # ###

                    forget_inputs.append(input)
                    mock_forget_targets.append(mock_target)
        forget_inputs = torch.stack(forget_inputs).cpu()
        mock_forget_targets = torch.stack(mock_forget_targets).cpu()
        mock_forget_dataset = torch.utils.data.TensorDataset(
            forget_inputs, mock_forget_targets
        )
        return mock_forget_dataset

    def _get_retain_dataset(self):
        retain_inputs, retain_targets = [], []
        for input, target in self.retain_loader.dataset:
            retain_inputs.append(input)
            retain_targets.append(torch.tensor(target))
        retain_inputs = torch.stack(retain_inputs)
        retain_targets = torch.stack(retain_targets)
        retain_dataset = torch.utils.data.TensorDataset(retain_inputs, retain_targets)
        return retain_dataset

    def get_mixed_dataloader(self, original_model):
        """
        This function returns a mixed dataloader with the
        original retain samples and the mock forget samples.
        """
        mock_forget_dataset = self._get_mock_forget_dataset(original_model)
        retain_dataset = self._get_retain_dataset()
        mixed_dataset = ConcatDataset([mock_forget_dataset, retain_dataset])

        mixed_dataloader = torch.utils.data.DataLoader(
            mixed_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )
        return mixed_dataloader

    def get_mock_forget_dataloader(self, original_model, alpha=0):
        """
        This function returns a dataloader with the
        mock forget samples.
        """
        mock_forget_dataset = self._get_mock_forget_dataset(original_model, alpha)
        mock_forget_dataloader = torch.utils.data.DataLoader(
            mock_forget_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )
        return mock_forget_dataloader
