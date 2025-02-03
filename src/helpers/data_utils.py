"""
This module contains classes and function to load and process data.

Raises:
    ValueError: If the dataset is not implemented yet.

Returns:
    tuple: A tuple containing the data loaders and dataset.
"""

import os

import torch
import torchvision
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms

from helpers.seed import set_work_init_fn  # pylint: disable=import-error

DATA_DIR = os.path.expanduser("~/data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnlearningDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        image_size,
        seed,
        is_vit=False,
        frac_per_class_forget=0.1,
        is_class_unlearning=False,
        class_to_forget="rocket",
        using_CIFAKE=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.frac_per_class_forget = frac_per_class_forget
        self.image_size = image_size
        self.is_vit = is_vit
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.forget_loader = None
        self.retain_loader = None
        self.classes = None
        self.input_channels = None
        self.label_to_class = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.is_class_unlearning = is_class_unlearning
        self.class_to_forget = class_to_forget
        self.using_CIFAKE = using_CIFAKE

    def load_data(self):
        """
        Loads the specified dataset and returns the corresponding dataloaders and dataset sizes.

        Returns:
        dataloaders (dict): A dictionary containing the dataloaders for the train and validation sets.
        dataset_sizes (dict): A dictionary containing the sizes of the train and validation sets.
        """

        ########################################
        # Define data transforms
        ########################################

        data_transforms = {
            "cifar-train": transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
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
            "mufac-train": transforms.Compose(
                [
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
                    transforms.ToTensor(),
                ]
            ),
            "imagenet": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "tiny-imagenet-train": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(122.4786, 114.2755, 101.3963),
                        std=(70.4924, 68.5679, 71.8127),
                    ),
                ]
            ),
            "tiny-imagenet-val": transforms.Compose(
                [
                    # transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(122.4786, 114.2755, 101.3963),
                        std=(70.4924, 68.5679, 71.8127),
                    ),
                ]
            ),
        }

        # Resize the images
        if self.is_vit:
            for key in data_transforms.keys():
                data_transforms[key] = transforms.Compose(
                    [transforms.Resize(self.image_size)]
                    + list(data_transforms[key].transforms)
                )
        else:
            data_transforms["cifar-train"] = transforms.Compose(
                [transforms.RandomCrop(32, padding=4)]
                + list(data_transforms["cifar-train"].transforms)
            )
            data_transforms["tiny-imagenet-train"] = transforms.Compose(
                [transforms.RandomCrop(64, padding=8, padding_mode="edge")]
                + list(data_transforms["tiny-imagenet-train"].transforms)
            )

        ########################################
        # Load data
        ########################################

        if self.dataset == "mufac":
            from helpers.mufac_utils import MUFAC

            self.input_channels = 3
            data_train = MUFAC(
                meta_data_path=DATA_DIR
                + "/custom_korean_family_dataset_resolution_128_clean/custom_train_dataset.csv",
                image_directory=DATA_DIR
                + "/custom_korean_family_dataset_resolution_128_clean/train_images",
                transform=data_transforms["mufac-train"],
            )
            data_val = MUFAC(
                meta_data_path=DATA_DIR
                + "/custom_korean_family_dataset_resolution_128_clean/custom_val_dataset.csv",
                image_directory=DATA_DIR
                + "/custom_korean_family_dataset_resolution_128_clean/val_images",
                transform=data_transforms["mufac-val"],
            )
            data_test = MUFAC(
                meta_data_path=DATA_DIR
                + "/custom_korean_family_dataset_resolution_128_clean/custom_test_dataset.csv",
                image_directory=DATA_DIR
                + "/custom_korean_family_dataset_resolution_128_clean/test_images",
                transform=data_transforms["mufac-val"],
            )
        elif self.dataset == "cifar-10":
            self.input_channels = 3
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
            data_train = torchvision.datasets.ImageNet(
                root=DATA_DIR + "/pytorch_imagenet1k",
                split="train",
                transform=data_transforms["imagenet"],
            )
            held_out = datasets.ImageNet(
                root=DATA_DIR + "/pytorch_imagenet1k",
                split="val",
                transform=data_transforms["imagenet"],
            )
        elif self.dataset == "tiny-imagenet":
            self.input_channels = 3
            from helpers.tiny_imagenet_utils import (
                TrainTinyImageNetDataset,
                TestTinyImageNetDataset,
            )

            data_train = TrainTinyImageNetDataset(
                transform=data_transforms["tiny-imagenet-train"]
            )
            held_out = TestTinyImageNetDataset(
                transform=data_transforms["tiny-imagenet-val"]
            )
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

        self.classes = data_train.classes
        if self.dataset == "mufac":
            self.label_to_class = data_train.label_to_class
            self.class_to_idx = data_train.class_to_idx
            self.idx_to_class = data_train.idx_to_class
        else:
            # Split the held-out set to test and val sets, in a stratified manner.
            labels = held_out.targets
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.5, random_state=self.seed
            )
            val_idx, test_idx = next(sss.split(held_out, labels))
            data_val = torch.utils.data.Subset(held_out, val_idx)
            data_test = torch.utils.data.Subset(held_out, test_idx)

        ################################################
        # Uncomment if CIFAKE is used as calibration set
        if self.using_CIFAKE:
            data_val = datasets.ImageFolder(
                root="~/data/cifake_classes",
                transform=data_transforms["cifar-val"],
            )

        ################################################

        ########################################
        # Create data loaders
        ########################################

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

        if self.is_class_unlearning == False:
            # Fixed split for MUFAC
            if self.dataset == "mufac":
                data_forget = Subset(data_train, list(range(0, 1062)))
                data_retain = Subset(data_train, list(range(1062, len(data_train))))
            elif self.dataset == "imagenet":
                data_forget, data_retain = self._split_data_forget_retain_imagenet(
                    data_train
                )
            else:
                data_forget, data_retain = self._split_data_forget_retain(data_train)
        else:
            if self.dataset != "mufac":
                data_forget, data_retain = (
                    self._split_data_forget_retain_class_unlearning(
                        data_train, self.class_to_forget
                    )
                )
            else:
                raise NotImplementedError(
                    "Dataset {self.dataset} does not support class unlearning."
                )

        image_datasets["forget"] = data_forget
        image_datasets["retain"] = data_retain
        dataloaders["forget"] = torch.utils.data.DataLoader(
            data_forget,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )
        dataloaders["retain"] = torch.utils.data.DataLoader(
            data_retain,
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

        return dataloaders, dataset_sizes

    def _split_data_forget_retain_class_unlearning(self, data_train, class_to_forget):
        """
        Splits the training data into two subsets: one containing the data points
        of the specified class to forget, and the other containing the remaining data points.
        Args:
            data_train (Dataset): The training dataset.
            class_to_forget (int): The class label to forget.
        Returns:
            Tuple[Subset, Subset]: A tuple containing two subsets:
                - The first subset contains the data points of the class to forget.
                - The second subset contains the remaining data points.
        """
        forget_indices = []
        retain_indices = []
        classes = data_train.classes

        if isinstance(data_train.targets, torch.Tensor):
            targets = data_train.targets.detach().clone()
        else:
            targets = data_train.targets
        for i, target in enumerate(targets):
            if classes[target] == class_to_forget:
                forget_indices.append(
                    i
                )  # Append the index instead of the data_train element
            else:
                retain_indices.append(
                    i
                )  # Append the index instead of the data_train element
        train_data_forget = Subset(
            data_train, forget_indices
        )  # Create Subset using the forget indices
        train_data_retain = Subset(
            data_train, retain_indices
        )  # Create Subset using the retain indices
        return train_data_forget, train_data_retain

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

    def _split_data_forget_retain_imagenet(self, data_train):
        """
        Spit train data to forget and retain sets only for the imagenet experiments.

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
            indices = []
            for class_name_element in class_name:
                indices.extend(
                    torch.where(targets == data_train.class_to_idx[class_name_element])[
                        0
                    ]
                )
            indices = torch.tensor(indices)
            indices_forget = indices[:5]
            indices_retain = indices[5:50]
            train_data_forget.append(
                torch.utils.data.Subset(data_train, indices_forget)
            )
            train_data_retain.append(
                torch.utils.data.Subset(data_train, indices_retain)
            )
        train_data_forget = torch.utils.data.ConcatDataset(train_data_forget)
        train_data_retain = torch.utils.data.ConcatDataset(train_data_retain)
        print(f"Number of samples in forget set: {len(train_data_forget)}")
        print(f"Number of samples in retain set: {len(train_data_retain)}")
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
