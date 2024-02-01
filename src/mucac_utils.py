"""
https://github.com/ndb796/MachineUnlearning/blob/main/02_MUCAC/multi_label_classification/Machine_Unlearning_MUCAC_Multi_Label_Base_Models.ipynb
"""
import torch
import os

class MUCAC(torch.utils.data.Dataset):
    def __init__(self, root='~/data/', transform=None, split):
        self.root = root
        self.transform = transform
        self.split = split
        self.identities = {}
        self.path = os.path.join(root, "CelebAMask-HQ")
        with open(self.path + 'CelebA-HQ-identity.txt', 'r') as f:
            for line in f.readlines():
                file_name, identity = line.split()
                self.identities[file_name] = identity
        
        self.attribute_path = os.path.join(root, "CelebAMask-HQ-attribute.txt")
        self.attribute_map = {
            "gender": 21,
            "smiling": 32,
            "young": 40
        }
        self.label_map = {}
        with open(self.attribute_path, 'r') as f:
            splited = line.strip().split()
            file_name = splited[0]
            self.label_map[file_name] = {attr: int(splited[idx]) for attr, idx in self.attributes_map.items()}

        self.sample_key = list(self.label_map.keys())[0]
        self.label



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for image_path in self.image_path_list:
            file_name = image_path.split("/")[-1]
            identity = int(self.identities[file_name])
            if identity >= train_index and identity < unseen_index:
                gender = int(label_map[file_name]["gender"])
                if gender == -1:
                    gender = 0
                smiling = int(label_map[file_name]["smiling"])
                if smiling == -1:
                    smiling = 0
                young = int(label_map[file_name]["young"])
                if young == -1:
                    young = 0
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)
