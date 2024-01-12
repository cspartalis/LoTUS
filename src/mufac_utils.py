import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MUFAC(Dataset):
    """
    MUFAC (Machine Unlearning Facial Age Classification) dataset class.
    This class represents a dataset for facial age classification using the MUFAC dataset.
    It inherits from the PyTorch Dataset class.

    Args:
        meta_data_path (str): The path to the metadata file containing image paths and age classes.
        image_directory (str): The directory where the images are stored.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        retain (bool, optional): Whether to retain the dataset after loading it into memory. Defaults to False.
        forget (bool, optional): Whether to forget the dataset after loading it into memory. Defaults to False.
    """
    def __init__(self, meta_data_path, image_directory, transform=None):
        self.meta_data = pd.read_csv(meta_data_path)
        self.image_directory = image_directory
        self.transform = transform

        # Process the metadata.
        image_age_list = self.parsing(self.meta_data)

        self.image_age_list = image_age_list
        self.idx_to_class = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
        }
        self.label_to_class = {
            0: "0-6 years old",
            1: "7-12 years old",
            2: "13-19 years old",
            3: "20-30 years old",
            4: "31-45 years old",
            5: "46-55 years old",
            6: "56-66 years old",
            7: "67-80 years old",
        }
        self.class_to_idx = {v: k for k, v in self.label_to_class.items()}  # Inversion of label_to_class
        self.classes = sorted(list(self.label_to_class.keys()))
        self.targets = [self.idx_to_class[age_class] for _, age_class in self.image_age_list]

    def parsing(self, meta_data):
        """
        Internal method to parse the metadata and extract image paths and age classes.

        Args:
            meta_data (pd.DataFrame): The metadata dataframe.

        Returns:
            list: A list of image paths and age classes.
        """
        image_age_list = []
        # iterate all rows in the metadata file
        for idx, row in meta_data.iterrows():
            image_path = row["image_path"]
            age_class = row["age_class"]
            image_age_list.append([image_path, age_class])
        return image_age_list

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.image_age_list)

    def __getitem__(self, idx) -> tuple:
        """
        Returns the image and label at the given index.

        Args:
            idx (int): The index of the image and label to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        image_path, age_class = self.image_age_list[idx]
        img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.idx_to_class[age_class]

        if self.transform:
            img = self.transform(img)

        return img, label


class MUCAC(Dataset):
    pass
