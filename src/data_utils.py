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

from seed import set_work_init_fn  # pylint: disable=import-error

DATA_DIR = os.path.expanduser("~/data/")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class UnlearningDataLoader:
    def __init__(self, dataset, batch_size, image_size, seed, frac_per_class_forget=0.1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.frac_per_class_forget = frac_per_class_forget
        self.image_size = image_size
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
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            ),
            "cifar-val": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            ),
            "mnist-val": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.1307], [0.3081]),
                ]
            ),
            "mufac-train": transforms.Compose(
                [
                    transforms.Resize(self.image_size),
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
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                ]
            ),
        }

        if self.dataset == "mufac":
            from mufac_utils import MUFAC
            self.input_channels = 3
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
        elif self.dataset == "mnist":
            self.input_channels = 1
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
        elif self.dataset == "pneumoniamnist":
            self.input_channels = 1
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
        if self.dataset != "mufac" and self.dataset != "pneumoniamnist":
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

    def _get_mock_forget_dataset(self, original_model):
        """
        This function returns the inputs and targets of the mocked forget samples.
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

    def get_mock_forget_dataloader(self, original_model):
        """
        This function returns a dataloader with the
        mock forget samples.
        """
        mock_forget_dataset = self._get_mock_forget_dataset(original_model)
        mock_forget_dataloader = torch.utils.data.DataLoader(
            mock_forget_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=set_work_init_fn(self.seed),
            num_workers=4,
        )
        return mock_forget_dataloader
