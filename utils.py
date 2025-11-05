import glob
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from PIL import Image


class ImageFolder(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(image_paths + "*")
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x, 0

    def __len__(self):
        return len(self.image_paths)

