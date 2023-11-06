"""
This file is used to debug the data_utils.py file.
"""
from data_utils import UnlearningDataLoader

ud = UnlearningDataLoader("imagenet", 128)
dataloaders, dataset_sizes = ud.load_data()

print(dataset_sizes)
