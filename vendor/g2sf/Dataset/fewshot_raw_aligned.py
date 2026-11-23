"""Original RGB/GT coordinates with read-only inverse-padded cleaned XYZ."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from Dataset.mvtec3d import MVTecDataset
from Dataset.mvtec3d_util import read_tiff, resize_organized_pcd


PROCESSED_ROOT = None
RAW_ROOT = None


def invert_padding(cloud, height: int, width: int):
    """Undo Preprocess.pad_cropped_pc, never resize or crop foreground."""
    if cloud.shape[0] < height or cloud.shape[1] < width:
        raise ValueError('Processed XYZ is smaller than raw grid')
    top, left = (cloud.shape[0] - height) // 2, (cloud.shape[1] - width) // 2
    bottom, right = top + height, left + width
    borders = [cloud[:top], cloud[bottom:], cloud[top:bottom, :left], cloud[top:bottom, right:]]
    if any(np.any(b != 0) for b in borders):
        raise ValueError('Discarded padding contains real points; unsupported alignment')
    return cloud[top:bottom, left:right]


class RawAlignedDataset(MVTecDataset):
    def __init__(self, args, is_train=True, shot=4):
        if Path(args.dataset_path).resolve() != RAW_ROOT:
            raise ValueError('RawAlignedDataset requires original raw dataset root')
        super().__init__(args, is_train=is_train)
        if is_train:
            if any(x != 'good' for x in self.defect_type):
                raise ValueError('Non-normal training data')
            if len(self.image) < shot:
                raise ValueError('Not enough normal supports')
            for field in ['image', 'pcd', 'label', 'mask', 'defect_type']:
                setattr(self, field, getattr(self, field)[:shot])
        self.pcd = [str(PROCESSED_ROOT / Path(path).relative_to(RAW_ROOT)) for path in self.pcd]
        if not all(Path(path).is_file() for path in self.pcd):
            raise ValueError('Missing cleaned XYZ identity')

    def __getitem__(self, index):
        image_path, cloud_path = self.image[index], self.pcd[index]
        raw = Image.open(image_path).convert('RGB')
        cloud = invert_padding(read_tiff(cloud_path), raw.height, raw.width)
        image = self.image_transform(raw)
        cloud = resize_organized_pcd(cloud, height=224, width=224).float()
        if self.label[index] == 0:
            mask = torch.zeros(1, 224, 224)
        else:
            original_mask = Image.open(self.mask[index]).convert('L')
            if original_mask.size != raw.size:
                raise ValueError('Official mask and RGB are on different grids')
            mask = (self.gt_transform(original_mask) > .5).float()
        indicator = np.all(cloud.numpy() != 0, axis=0).astype(np.uint8)
        for _ in range(self.morph_epoch):
            indicator = cv2.dilate(indicator, np.ones((self.morph_size, self.morph_size), np.uint8))
        indicator = self.patch_transform(Image.fromarray(indicator))
        if self.is_train:
            return image, cloud, self.label[index], mask, indicator
        return image, cloud, self.label[index], mask, Path(image_path).stem, self.defect_type[index], indicator
