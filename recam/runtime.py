"""Initialize both frozen encoders and the matching real-stream mapper pair locally."""
import hashlib
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT / 'vendor/g2sf'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seed_everything():
    import numpy as np
    import open3d as o3d
    import torch
    torch.set_num_threads(2)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    o3d.utility.random.seed(0)
    if not torch.cuda.is_available() or not o3d.core.cuda.is_available():
        raise RuntimeError('This implementation needs CUDA-enabled PyTorch and CUDA-enabled Open3D.')


def build(dataset, category, shot):
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path[:0] = [str(VENDOR), str(PROJECT / 'vendor/dinov2')]
    seed_everything()
    import timm
    from safetensors.torch import load_file
    from Model import fewshot_recam_real
    from .model import make_model
    fewshot_recam_real.ASSETS = PROJECT/'weights/mappers'/dataset/f'{shot}shot'
    create_model = timm.create_model

    def bundled_encoder(name, *args, **kwargs):
        if name == 'vit_base_patch8_224.dino' and kwargs.get('pretrained'):
            kwargs['pretrained'] = False
            encoder = create_model(name, *args, **kwargs)
            encoder.load_state_dict(load_file(str(PROJECT/'weights/backbones/dinov1_vitb8_dino.safetensors')))
            return encoder
        return create_model(name, *args, **kwargs)

    with patch.object(timm, 'create_model', bundled_encoder):
        return make_model(dataset, category, shot)
