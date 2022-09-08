from sqlite3 import DatabaseError
import torch
import PIL
import torchvision
from torch.utils.data import Dataset


class whu_dataset(Dataset):
    def __init__(self):
        super(whu_dataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
