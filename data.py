import json
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils import compact


class CaterDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_videos: Optional[int],
        cater_transforms: Callable,
        split: str = "val",
    ):
        super().__init__()
        self.data_root = data_root
        self.cater_transforms = cater_transforms
        self.max_num_videos = max_num_videos
        self.data_path = os.path.join(data_root, "cater_with_masks", split)
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.video_length = 33
        self.nFrames = 6
        self.length = len(os.listdir(os.path.join(self.data_path, "images")))//self.video_length
        
        if self.split == "val":
            self.length = self.max_num_videos
            self.nFrames = 6
            
        n_windows = self.video_length//self.nFrames
        
        self.paths = {}
        
        if self.split == "train":
            for w in range(n_windows):
                for index in range(self.length):
                    self.paths[index+self.length*w] = []
                    n_video = str(index).zfill(6)
                    image_paths = [os.path.join(self.data_path, "images", f'{n_video}_{iFrame}.jpg') for iFrame in range(self.nFrames*w, self.nFrames*(w+1))]
                    self.paths[index+self.length*w] = image_paths
         
        else:
            for index in range(self.length):
                self.paths[index] = []
                n_video = str(index).zfill(6)
                image_paths = [os.path.join(self.data_path, "images", f'{n_video}_{str(iFrame).zfill(2)}.jpg') for iFrame in range(self.nFrames)]
                self.paths[index] = image_paths

    def __getitem__(self, index: int):
        image_paths = self.paths[index]
        imgs = [Image.open(image_path) for image_path in image_paths]
        imgs = [img.convert("RGB") for img in imgs]
        imgs = torch.stack([self.cater_transforms(img) for img in imgs])
        return imgs

    def __len__(self):
        return len(self.paths)


class CaterDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        cater_transforms: Callable,
        num_workers: int,
        num_train_videos: Optional[int] = None,
        num_val_videos: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.cater_transforms = cater_transforms
        self.num_workers = num_workers
        self.num_train_videos = num_train_videos
        self.num_val_videos = num_val_videos

        self.train_dataset = CaterDataset(
            data_root=self.data_root,
            max_num_videos=self.num_train_videos,
            cater_transforms=self.cater_transforms,
            split="train",
        )
        self.val_dataset = CaterDataset(
            data_root=self.data_root,
            max_num_videos=self.num_val_videos,
            cater_transforms=self.cater_transforms,
            split="val",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
