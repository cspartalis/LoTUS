"""
This module contains classes and function to load and process data.

Raises:
    ValueError: If the dataset is not implemented yet.

Returns:
    tuple: A tuple containing the data loaders and dataset.
"""

import os

import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from seed import set_seed, set_work_init_fn # pylint: disable=import-error

SEED = 0
RNG = set_seed(SEED)

DATA_DIR = os.path.expanduser("~/data/")


class TinyImageNet(Dataset):
    """
    A PyTorch dataset class for Tiny-ImageNet-200.

    Args:
        root (str): Root directory path.
        is_train (bool, optional): If True, creates a dataset from the training set, otherwise creates a dataset from the validation set. Defaults to True.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``. Defaults to None.

    Attributes:
        is_train (bool): If True, creates a dataset from the training set, otherwise creates a dataset from the validation set.
        root_dir (str): Root directory path.
        transform (callable): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``.
        train_dir (str): Path to the training directory.
        val_dir (str): Path to the validation directory.
        classes (list): List of class names.
        len_dataset (int): Length of the dataset.
        idx_to_class (dict): Dictionary mapping target indices to class names.
        class_to_idx (dict): Dictionary mapping class names to target indices.
        val_img_to_class (dict): Dictionary mapping image filenames to their corresponding class labels for the validation dataset.
        images (list): List of image paths and their corresponding target indices.
        set_nids (set): Set of wnids.
        class_to_label (dict): Dictionary mapping wnids to class labels.
        targets (list): List of target indices.
    """

    def __init__(self, root, is_train=True, transform=None):
        """
        Initializes a dataset object.
        Requirement: Please download Tiny-ImageNet-200 from
        https://github.com/rmccorm4/Tiny-Imagenet-200
        and extract it to ~/data/tiny-imagenet-200!

        Args:
                root (str): Root directory path.
                is_train (bool, optional): If True, creates a dataset from the training set, otherwise creates a dataset from the validation set. Defaults to True.
                transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``. Defaults to None.
        """

        self.is_train = is_train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if self.is_train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset()

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, "r", encoding="utf-8") as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, "r", encoding="utf-8") as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

        self.targets = [i[1] for i in self.images]

    def _create_class_idx_dict_train(self) -> None:
        """
        Creates a dictionary mapping target indices to class names and vice versa for the training dataset.
        """

        self.classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        self.classes = sorted(self.classes)
        num_images = 0
        for _, __, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.idx_to_class = {i: self.classes[i] for i in range(len(self.classes))}
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

    def _create_class_idx_dict_val(self) -> None:
        """
        Creates a dictionary mapping image filenames to their corresponding class labels for the validation dataset.
        Also creates dictionaries mapping class labels to target indices and vice versa.
        """
        # val_image_dir = os.path.join(self.val_dir, "images")
        # images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, "r", encoding="utf-8") as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self) -> None:
        """
        Create a dataset of image paths and their corresponding target indices.
        """
        self.images = []
        if self.is_train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if self.is_train:
                            item = (path, self.class_to_idx[tgt])
                        else:
                            item = (
                                path,
                                self.class_to_idx[self.val_img_to_class[fname]],
                            )
                        self.images.append(item)

    def return_label(self, idx) -> list:
        """
        Returns the label corresponding to the given index.

        Args:
                idx (list): A list of indices.

        Returns:
                list: A list of labels corresponding to the given indices.
        """
        return [self.class_to_label[self.idx_to_class[i.item()]] for i in idx]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return self.len_dataset

    def __getitem__(self, idx) -> tuple:
        """
        Args:
                idx (int): The index of the sample to return.

        Returns:
                tuple: A tuple containing the sample and its corresponding target.
        """
        img_path, tgt = self.images[idx]
        with open(img_path, "rb") as f:
            sample = Image.open(img_path)
            sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

    def __class_to_idx__(self, class_name):
        """
        Given a class name, returns the corresponding index in the class_to_idx dictionary.

        Args:
        - class_name (str): the name of the class

        Returns:
        - int: the index of the class in the class_to_idx dictionary
        """
        return self.class_to_idx[class_name]


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
        }

        if self.dataset == "cifar-10":
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
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

        # data_val, data_test = torch.utils.data.random_split(
        #     held_out, [0.5, 0.5], generator=RNG
        # )

        # Stratified splitting held-out set to test and val sets.
        labels = held_out.targets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
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
        self.classes = data_train.classes
        if data_train.targets is type(torch.Tensor):
            targets = data_train.targets.detach().clone()
        else:
            targets = torch.Tensor(data_train.targets)
        for class_name in self.classes:
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
