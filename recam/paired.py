import numpy as np
import torch
from .data import PairedData


class ReferencePairs(PairedData):
    def __init__(self, dataset, category, shot, raw, prepared, train, normal_root):
        super().__init__(dataset, category, shot, raw, prepared, train)
        self.normal_root = normal_root

    def __getitem__(self, index):
        image, cloud, *tail = super().__getitem__(index)
        if self.dataset == 'eyecandies':
            from pathlib import Path
            stem = Path(self.image[index]).name.removesuffix('_image_4.png')
            split = 'train' if self.train else 'test_public'
            path = self.normal_root/self.category/split/(stem+'.npz')
            with np.load(path) as data:
                normal = torch.from_numpy(data['normal224'].copy()).permute(2,0,1).float()/255
            image['author_normal'] = cloud['author_normal'] = normal
        return image, cloud, *tail
