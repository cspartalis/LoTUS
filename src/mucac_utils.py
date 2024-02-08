"""
https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/multi_label_classification/Machine_Unlearning_MUCAC_Multi_Label_Base_Models.ipynb
"""

import glob
import os

import torch
from PIL import Image


class MUCAC(torch.utils.data.Dataset):

    def __init__(
        self,
        split,
        transform=None,
        root="/home/cspartalis/data/",
    ):
        self.split = split
        self.transform = transform
        self.root = root
        self.identities = {}
        path = root + "CelebAMask-HQ/"
        identity_path = path + "CelebA-HQ-identity.txt"
        with open(identity_path, "r") as f:
            for line in f.readlines():
                file_name, identity = line.split()
                self.identities[file_name] = identity

        attributes_map = {"gender": 21, "smiling": 32, "young": 40}
        self.label_map = {}
        attribute_path = path + "CelebA-HQ-attribute.txt"
        with open(attribute_path) as f:
            lines = f.readlines()
            for line in lines[2:]:
                splited = line.strip().split()
                file_name = splited[0]
                self.label_map[file_name] = {
                    attr: int(splited[idx]) for attr, idx in attributes_map.items()
                }

        self.sample_key = list(self.label_map.keys())[0]

        self.source_root = path + "CelebA-HQ-img"
        self.image_path_list = glob.glob(os.path.join(self.source_root, "*"))

        self.train_index = 190
        self.retain_index = 1250
        self.unseen_index = 1627

        self.image_paths = []
        self.labels = []

        if self.split == "train":
            self._train_split()
        elif self.split == "val":
            self._val_split()
        elif self.split == "retain":
            self._retain_split()
        elif self.split == "forget":
            self._forget_split()
        elif self.split == "test":
            self._test_split()
        else:
            raise ValueError("Invalid split")

        self.classes = list(attributes_map.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        # identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        # Stack the targets into a single tensor
        targets = torch.tensor([gender, smiling, young]).type(torch.FloatTensor)

        return image, targets

    def _train_split(self):
        for image_path in self.image_path_list:
            file_name = image_path.split("/")[-1]
            identity = int(self.identities[file_name])
            if identity >= self.train_index and identity < self.unseen_index:
                gender, smiling, young = self._target_transform(file_name)
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def _val_split(self):
        for image_path in self.image_path_list:
            file_name = image_path.split("/")[-1]
            identity = int(self.identities[file_name])
            if identity < self.train_index:
                gender, smiling, young = self._target_transform(file_name)
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def _retain_split(self):
        for image_path in self.image_path_list:
            file_name = image_path.split("/")[-1]
            identity = int(self.identities[file_name])
            if identity < self.unseen_index and identity >= self.retain_index:
                gender, smiling, young = self._target_transform(file_name)
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def _forget_split(self):
        for image_path in self.image_path_list:
            file_name = image_path.split("/")[-1]
            identity = int(self.identities[file_name])
            if identity >= self.train_index and identity < self.retain_index:
                gender, smiling, young = self._target_transform(file_name)
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def _test_split(self):
        for image_path in self.image_path_list:
            file_name = image_path.split("/")[-1]
            identity = int(self.identities[file_name])
            if identity < self.unseen_index:
                continue
            gender, smiling, young = self._target_transform(file_name)
            self.labels.append((gender, identity, smiling, young))
            self.image_paths.append(image_path)

    def _target_transform(self, file_name):
        gender = int(self.label_map[file_name]["gender"])
        if gender == -1:
            gender = 0
        smiling = int(self.label_map[file_name]["smiling"])
        if smiling == -1:
            smiling = 0
        young = int(self.label_map[file_name]["young"])
        if young == -1:
            young = 0
        return gender, smiling, young
