import torch
from torch.utils.data import Dataset

import torchvision.transforms as T
from PIL import Image

import os
import matplotlib.pyplot as plt
import numpy as np

HIGH_RES_SIZE = 96
LOW_RES_SIZE = HIGH_RES_SIZE // 4

img_transforms = T.Compose([
    T.RandomCrop((HIGH_RES_SIZE, HIGH_RES_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
])

high_res_transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

low_res_transform = T.Compose([
    T.Resize((LOW_RES_SIZE, LOW_RES_SIZE), interpolation=Image.BICUBIC),
    T.ToTensor(),
    # T.Normalize(mean=[0., 0., 0.], std=[1, 1, 1]),
])


class DIV2K(Dataset):
    def __init__(
        self,
        img_dir: str
    ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        img = Image.open(img_path)
        img = img_transforms(img)
        low_res = low_res_transform(img)
        high_res = high_res_transform(img)

        return low_res, high_res


if __name__ == "__main__":
    dataset = DIV2K(
        "/Users/muhammadarham/Drive/MLProjects/SRGAN/data/DIV2K/DIV2K_train_HR/DIV2K_train_HR"
    )

    low_res, high_res = dataset.__getitem__(5)
    print(low_res.shape, high_res.shape)
