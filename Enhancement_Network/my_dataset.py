from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import torch
import torchvision
import random
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

class MyDataSet(Dataset):
    def __init__(self, images_visible_path: list, images_infrared_path: list, transform=None):
        self.images_visible_path = images_visible_path
        self.images_infrared_path = images_infrared_path
        self.transform = transform

    def __len__(self):
        return len(self.images_visible_path)

    def __getitem__(self, item):
        img_visible = Image.open(self.images_visible_path[item]).convert("RGB")
        if img_visible.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_visible_path[item]))
        img_infrared = Image.open(self.images_infrared_path[item]).convert("RGB")
        if img_infrared.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_infrared_path[item]))
        # img_infrared = img_infrared.split()[0]
        if self.transform is not None:
            img_visible, img_infrared = self.transform(img_visible, img_infrared)
            # [img_visible, img_infrared] = transform_augment([img_visible, img_infrared], split=self.split, min_max=(-1, 1))
        img_name = os.path.splitext(os.path.basename(self.images_visible_path[item]))[0]
        return img_visible, img_infrared, img_name  # train: 3, 128, 128; test: 3, H, W
    @staticmethod
    def collate_fn(batch):
        image_visible, image_infrared, img_name= tuple(zip(*batch))

        image_visible = torch.stack(image_visible, dim=0)
        image_infrared = torch.stack(image_infrared, dim=0)
        # img_name = torch.stack(img_name, dim=0)

        return image_visible, image_infrared, list(img_name)
