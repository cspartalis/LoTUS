import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode

DATA_DIR = os.path.expanduser("~/data")


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        
        self.transform = transform
        self.id_to_idx = {} # id_to_idx = {'n02124075': 0, ..., 'n13037406': 199}
        for i, line in enumerate(open(DATA_DIR+'/tiny-imagenet-200/wnids.txt', 'r')):
            self.id_to_idx[line.replace('\n', '')] = i

        self.id_to_class_name = {}
        for i, line in enumerate(open(DATA_DIR+'/tiny-imagenet-200/words.txt', 'r')):
            a = line.split('\t')
            id, class_name = a[0],a[1].split(',')[0].split('\n')[0]
            self.id_to_class_name[id] = class_name 
        
        self.class_to_idx = {}
        for key, value in self.id_to_idx.items():
            class_name = self.id_to_class_name[key]
            self.class_to_idx[class_name] = value
        
        self.classes = list(self.class_to_idx.keys())

        self.filenames = glob.glob(DATA_DIR+"/tiny-imagenet-200/train/*/*/*.JPEG")
        self.targets = [self.id_to_idx[img_path.split('/')[6]] for img_path in self.filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_to_idx[img_path.split('/')[6]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.id_to_idx = {}
        for i, line in enumerate(open(DATA_DIR+'/tiny-imagenet-200/wnids.txt', 'r')):
            self.id_to_idx[line.replace('\n', '')] = i
         
        self.id_to_class_name = {}
        for i, line in enumerate(open(DATA_DIR+'/tiny-imagenet-200/words.txt', 'r')):
            a = line.split('\t')
            id, class_name = a[0],a[1].split(',')[0].split('\n')[0]
            self.id_to_class_name[id] = class_name 
        
        self.class_to_idx = {}
        for key, value in self.id_to_idx.items():
            class_name = self.id_to_class_name[key]
            self.class_to_idx[class_name] = value
        
        self.classes = list(self.class_to_idx.keys())

        self.filenames = glob.glob(DATA_DIR+"/tiny-imagenet-200/val/images/*.JPEG")
        self.cls_dict = {}
        for i, line in enumerate(open(DATA_DIR+'/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dict[img] = self.id_to_idx[cls_id]
        self.targets = list(self.cls_dict.values())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.cls_dict[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label
