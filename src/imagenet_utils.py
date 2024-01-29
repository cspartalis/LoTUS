"""
source: https://github.com/rmccorm4/Tiny-Imagenet-200
"""

import os
from torch.utils.data import Dataset
from PIL import Image


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
