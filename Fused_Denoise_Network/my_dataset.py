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
import torchvision.transforms as T

class transform():
    def __init__(self, mode="train"):
        self.data_transform = {
            "train": T.Compose([
                T.RandomCrop(120),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.ToTensor()
            ]),
            "val": T.Compose([
                T.ToTensor()
            ])
        }
        self.transform = self.data_transform[mode]

    def __call__(self, img_visible, img_visible_r, img_infrared):
        # Apply transforms to each image individually
        img_visible = self.transform(img_visible)
        img_visible_r = self.transform(img_visible_r)
        img_infrared = self.transform(img_infrared)
        return img_visible, img_visible_r, img_infrared
class MyDataSet(Dataset):
    def __init__(self, images_visible_path: list, images_visible_r_path: list, images_infrared_path: list, transform=None):
        self.images_visible_path = images_visible_path
        self.images_visible_r_path = images_visible_r_path
        self.images_infrared_path = images_infrared_path
        self.transform = transform

    def __len__(self):
        return len(self.images_visible_path)

    def __getitem__(self, item):
        img_visible = Image.open(self.images_visible_path[item]).convert("RGB")
        if img_visible.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_visible_path[item]))
        img_visible_r = Image.open(self.images_visible_r_path[item]).convert("RGB")
        if img_visible_r.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_visible_r_path[item]))
        img_infrared = Image.open(self.images_infrared_path[item]).convert("RGB")
        if img_infrared.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_infrared_path[item]))
        # img_infrared = img_infrared.split()[0]
        if self.transform is not None:
            # img_visible,  img_infrared = self.transform(img_visible, img_infrared)
            img_visible_l, img_visible_r, img_infrared = self.transform(img_visible,img_visible_r, img_infrared)
            # img_visible = self.transform(img_visible)
            # img_visible_r = self.transform(img_visible_r)
            # img_infrared = self.transform(img_infrared)
            # [img_visible, img_infrared] = transform_augment([img_visible, img_infrared], split=self.split, min_max=(-1, 1))
        img_name = os.path.splitext(os.path.basename(self.images_visible_path[item]))[0]
        return img_visible_l, img_visible_r, img_infrared, img_name  # train: 3, 128, 128; test: 3, H, W  都是3通道
    @staticmethod
    def collate_fn(batch):
        image_visible_l, image_visible_r, image_infrared, img_name = tuple(zip(*batch))

        image_visible_l = torch.stack(image_visible_l, dim=0)
        image_visible_r = torch.stack(image_visible_r, dim=0)
        image_infrared = torch.stack(image_infrared, dim=0)
        return image_visible_l, image_visible_r, image_infrared, list(img_name)
class transform_test():
    def __init__(self, mode="train"):
        self.data_transform = {
            "train": T.Compose([
                T.RandomCrop(120),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.ToTensor()
            ]),
            "val": T.Compose([
                T.ToTensor()
            ])
        }
        self.transform = self.data_transform[mode]

    def __call__(self, img_visible, img_visible_r, img_infrared, img_vis):
        # Apply transforms to each image individually
        img_visible = self.transform(img_visible)
        img_visible_r = self.transform(img_visible_r)
        img_infrared = self.transform(img_infrared)
        img_vis = self.transform(img_vis)
        return img_visible, img_visible_r, img_infrared, img_vis
class MyDataSet_test(Dataset):
    def __init__(self, images_visible_path: list, images_visible_r_path: list, images_infrared_path: list, images_vis_path: list, transform=None):
        self.images_visible_path = images_visible_path
        self.images_visible_r_path = images_visible_r_path
        self.images_infrared_path = images_infrared_path
        self.images_vis_path = images_vis_path
        self.transform = transform

    def __len__(self):
        return len(self.images_visible_path)

    def __getitem__(self, item):
        img_visible = Image.open(self.images_visible_path[item]).convert("RGB")
        if img_visible.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_visible_path[item]))
        img_visible_r = Image.open(self.images_visible_r_path[item]).convert("RGB")
        if img_visible_r.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_visible_r_path[item]))
        img_infrared = Image.open(self.images_infrared_path[item]).convert("RGB")
        if img_infrared.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_infrared_path[item]))
        img_vis = Image.open(self.images_vis_path[item]).convert("RGB")
        if img_vis.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_vis_path[item]))
        # img_infrared = img_infrared.split()[0]
        if self.transform is not None:
            # img_visible,  img_infrared = self.transform(img_visible, img_infrared)
            img_visible_l, img_visible_r, img_infrared, img_vis = self.transform(img_visible,img_visible_r, img_infrared, img_vis)
            # img_visible = self.transform(img_visible)
            # img_visible_r = self.transform(img_visible_r)
            # img_infrared = self.transform(img_infrared)
            # [img_visible, img_infrared] = transform_augment([img_visible, img_infrared], split=self.split, min_max=(-1, 1))
        img_name = os.path.splitext(os.path.basename(self.images_visible_path[item]))[0]
        return img_visible_l, img_visible_r, img_infrared, img_vis, img_name # train: 3, 128, 128; test: 3, H, W  都是3通道
    @staticmethod
    def collate_fn(batch):
        image_visible_l, image_visible_r, image_infrared, image_vis, img_name= tuple(zip(*batch))

        image_visible_l = torch.stack(image_visible_l, dim=0)
        image_visible_r = torch.stack(image_visible_r, dim=0)
        image_infrared = torch.stack(image_infrared, dim=0)
        image_vis = torch.stack(image_vis, dim=0)
        return image_visible_l, image_visible_r, image_infrared, image_vis, list(img_name)
