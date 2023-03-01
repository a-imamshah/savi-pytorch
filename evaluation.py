import numpy as np
import os

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pytorch_lightning as pl
from PIL import Image


from model import SAViModel
from params import SAViParams
from utils import rescale
params = SAViParams()


img_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(rescale),  # rescale between -1 and 1
        transforms.Resize(params.resolution),
    ]
)

mask_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(),  # rescale between -1 and 1
        transforms.Resize(params.resolution),
    ]
)

class CaterDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        max_num_videos: Optional[int],
        img_transforms: Callable,
        mask_transforms: Callable,
        split: str = "val",
    ):
        super().__init__()
        self.data_root = data_root
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms,
        self.max_num_videos = max_num_videos
        self.data_path = os.path.join(data_root, "cater_with_masks", split)
        self.split = split
        self.nFrames = 2
        self.nObjects = 11
        self.length = len(os.listdir(os.path.join(self.data_path, "images")))//33
        
        if self.split == "val":
            self.length = self.max_num_videos
            self.nFrames = 33
        
        self.paths = {}
        for index in range(self.length):
            self.paths[index] = {'images': None, 'masks': []}
            n_video = str(index).zfill(6)
            
            image_paths = [os.path.join(self.data_path, "images", f'{n_video}_{str(iFrame).zfill(2)}.jpg') for iFrame in range(self.nFrames)]
            mask_paths_outer = []

            for iFrame in range(self.nFrames):
                mask_paths_inner = []
                for iMask in range(self.nObjects):
                    mask_paths_inner.append(os.path.join(self.data_path, "masks", f'{n_video}_{str(iFrame).zfill(2)}_{str(iMask).zfill(2)}.jpg'))
                mask_paths_outer.append(mask_paths_inner)
                
            self.paths[index]['images'] = image_paths
            self.paths[index]['masks'] = mask_paths_outer
            
            

    def __getitem__(self, index: int):
        
        image_paths = self.paths[index]['images']
        imgs = [Image.open(image_path) for image_path in image_paths]
        imgs = [img.convert("RGB") for img in imgs]
        imgs = torch.stack([self.img_transforms(img) for img in imgs])
        
        mask_paths_outer = self.paths[index]['masks']
        masks_allframes = []
        for masks_paths_list in mask_paths_outer:
            masks = []
            for masks_path in masks_paths_list:
                if os.path.exists(masks_path):
                    mask = Image.open(masks_path)
                    mask = mask.convert("RGB")
                else:
                    mask = np.zeros((params.resolution[0], params.resolution[1], 3))
                mask = mask_transforms(mask)
                masks.append(mask)
            masks = torch.stack(masks)
            masks_allframes.append(masks)
        masks_allframes = torch.stack(masks_allframes)
        
        return (imgs, masks_allframes)

    def __len__(self):
        return self.length


val_dataset = CaterDataset(
            data_root=params.data_root,
            max_num_videos=params.num_val_videos,
            img_transforms=img_transforms,
            mask_transforms=mask_transforms,
            split="val"
        )


def adjusted_rand_index(true_mask, pred_mask, exclude_background=True):
    """
    compute the ARI for a single image. N.b. ARI 
    is invariant to permutations of the cluster IDs.
    See https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index.
    true_mask: LongTensor of shape [N, num_entities, 1, H, W]
        background == 0
        object 1 == 1
        object 2 == 2
        ...
    pred_mask: FloatTensor of shape [N, K, 1, H, W]  (mask probs)
    Returns: ari [N]
    """
    N, _, H, W = true_mask.shape
    max_num_entities = 11

    true_group_ids = true_mask.view(N, H * W).long()
    true_mask_oh = torch.nn.functional.one_hot(true_group_ids).float()
    # exclude background
    if exclude_background:
        true_mask_oh[...,0] = 0

    # take argmax across slots for predicted masks
    pred_mask = pred_mask.squeeze(2)  # [N, K, H, W]
    pred_groups = pred_mask.shape[1]
    pred_mask = torch.argmax(pred_mask, dim=1)  # [N, H, W]
    pred_group_ids = pred_mask.view(N, H * W).long()
    pred_group_oh = torch.nn.functional.one_hot(pred_group_ids, pred_groups).float()
    
    n_points = H*W
    
    if n_points <= max_num_entities and n_points <= pred_groups:
        raise ValueError(
                "adjusted_rand_index requires n_groups < n_points. We don't handle "
                "the special cases that can occur when you have one cluster "
                "per datapoint")

    n_points = torch.sum(true_mask_oh, dim=[1,2])  # [N]
    nij = torch.einsum('bji,bjk->bki', pred_group_oh, true_mask_oh)
    a = torch.sum(nij, 1)
    b = torch.sum(nij, 2)

    rindex = torch.sum(nij * (nij - 1), dim=[1,2])
    aindex = torch.sum(a * (a - 1), 1)
    bindex = torch.sum(b * (b - 1), 1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    # check if both single cluster; nij matrix has only 1 nonzero entry
    check_single_cluster = torch.sum( (nij > 0).int(), dim=[1,2])  # [N]
    check_single_cluster = (1 == check_single_cluster).int()
    ari[ari != ari] = 0  # remove Nan
    ari = check_single_cluster * torch.ones_like(ari) + (1 - check_single_cluster) * ari

    return ari



def compute_ari(model):
    device = torch.device('cuda')
    model = model.to(device)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    ari_log = []
    for step, (imgs, gt_masks) in enumerate(val_dataloader):
        gt_masks = torch.squeeze(gt_masks, dim=0)
        gt_masks = torch.argmax(gt_masks, dim=1)
        recon_combined, recons, pred_masks, slots_all = model(imgs.to(device))
        pred_masks = torch.squeeze(pred_masks, dim=0)
        pred_masks = pred_masks.permute((1,0,2,3,4))
        ari = adjusted_rand_index(gt_masks.to(device), pred_masks.to(device))
        ari_log.append(torch.mean(ari))

    return torch.mean(torch.stack(ari_log))