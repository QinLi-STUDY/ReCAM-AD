"""Normal-only few-shot private memories with an optional frozen CLIP prior.

This is a new experimental method, not a reproduction of CMDS-AD or G2SF.
CLIP uses the unmodified OpenAI pretrained checkpoint through the local,
MIT-derived CLIP implementation. No target anomaly-finetuned weights are used.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from Dataset.mvtec3d import BaseDataset, MVTecDataset
from Engine.feature import Features
from Util import get_coreset_idx


CLIP_ROOT = Path(__file__).resolve().parent / 'unused_clip'
CLIP_CHECKPOINT = CLIP_ROOT / 'ViT-L-14-336px.pt'
NORMAL_PROMPTS = (
    'a photo of a normal {}.', 'a photo of an intact {}.',
    'a photo of a flawless {}.', 'a photo of a {} without defects.',
)
ABNORMAL_PROMPTS = (
    'a photo of a damaged {}.', 'a photo of a broken {}.',
    'a photo of a defective {}.', 'a photo of a {} with defects.',
)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class SelectedNormalDataset(MVTecDataset):
    """Apply original transforms ONLY to the native first-K support images."""
    def __init__(self, args, shot: int):
        BaseDataset.__init__(self, args, split='train')
        self.is_train = True
        directory = Path(self.dir) / 'good'
        images = sorted((directory / 'rgb').glob('*.png'))[:shot]
        if len(images) != shot:
            raise ValueError('Insufficient paired normal images')
        clouds = [directory / 'xyz' / f'{p.stem}.tiff' for p in images]
        if not all(p.is_file() for p in clouds):
            raise ValueError('Normal RGB/XYZ mismatch')
        self.image, self.pcd = list(map(str, images)), list(map(str, clouds))
        self.label, self.mask, self.defect_type = [0] * shot, [None] * shot, ['good'] * shot


def local_clip_modules():
    name = '_g2sf_fewshot_raw_clip'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(CLIP_ROOT)]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    return importlib.import_module(name + '.model'), importlib.import_module(name + '.tokenizer')


class RawFrozenCLIP(torch.nn.Module):
    """Native 336px QuickGELU CLIP; both semantic anchors and visual memories."""
    def __init__(self, category: str, device: str = 'cuda'):
        super().__init__()
        model_module, tokenizer_module = local_clip_modules()
        raw = torch.jit.load(str(CLIP_CHECKPOINT), map_location='cpu')
        state = raw.state_dict()
        self.model = model_module.build_model_from_openai_state_dict(
            state, quick_gelu=True, cast_dtype=torch.float16,
        ).to(device).eval()
        del raw, state
        self.model.requires_grad_(False)
        self.tokenize = tokenizer_module.tokenize
        self.device = device
        name = category.replace('_', ' ')
        with torch.inference_mode():
            normal = self.model.encode_text(self.tokenize([p.format(name) for p in NORMAL_PROMPTS]).to(device)).float()
            abnormal = self.model.encode_text(self.tokenize([p.format(name) for p in ABNORMAL_PROMPTS]).to(device)).float()
            self.register_buffer('anchors', F.normalize(torch.stack([
                F.normalize(normal, dim=-1).mean(0), F.normalize(abnormal, dim=-1).mean(0)
            ]), dim=-1))
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406], device=device)[None, :, None, None])
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225], device=device)[None, :, None, None])
        self.register_buffer('clip_mean', torch.tensor([.48145466, .4578275, .40821073], device=device)[None, :, None, None])
        self.register_buffer('clip_std', torch.tensor([.26862954, .26130258, .27577711], device=device)[None, :, None, None])

    @torch.inference_mode()
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = image.to(self.device)
        image = image * self.rgb_std + self.rgb_mean
        image = F.interpolate(image, (336, 336), mode='bicubic', align_corners=False).clamp(0, 1)
        image = ((image - self.clip_mean) / self.clip_std).half()
        global_features, tokens = self.model.encode_image(image, [24])
        visual = self.model.visual
        patches = tokens[-1][:, 1:]
        patches = F.normalize((visual.ln_post(patches) @ visual.proj).float(), dim=-1)
        global_features = F.normalize(global_features.float(), dim=-1)
        # Projecting final patch tokens uses CLIP's projection but is not a claim
        # that CLIP was pretrained with a dense pixel-text objective.
        semantics = (patches @ self.anchors.T / .07).softmax(-1)[..., 1]
        global_semantics = (global_features @ self.anchors.T / .07).softmax(-1)[..., 1]
        return {'CLIP': patches[0], 'semantic': semantics[0],
                'global_semantic': global_semantics[0], 'global_clip': global_features[0]}


class PrivateTextMemory:
    """Freeze encoders, fit ONLY selected support banks and crossfit scales."""
    def __init__(self, args, category: str, include_clip: bool = True):
        self.args = args
        self.native = Features(args)
        self.native.deep_feature_extractor.eval().requires_grad_(False)
        self.clip = RawFrozenCLIP(category, args.device) if include_clip else None
        self.banks, self.origins, self.scales = {}, {}, {}
        self.rgb_mean, self.rgb_std = None, None
        self.semantic_center = .5
        self.semantic_scale = 1.
        self.fit_records = []

    @torch.inference_mode()
    def extract(self, image: torch.Tensor, cloud: torch.Tensor) -> dict[str, torch.Tensor]:
        private = self.native.get_features((image, cloud))
        private = {k: v.to(self.args.device).float() for k, v in private.items()}
        result = {**private, 'valid': private['PCD'].abs().sum(-1) > 0}
        if self.clip is not None:
            result.update(self.clip(image))
        for k, value in result.items():
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite {k} features')
        return result

    def normalized(self, feature: dict) -> dict:
        result = dict(feature)
        result['IMG'] = (feature['IMG'] - self.rgb_mean) / self.rgb_std
        return result

    @staticmethod
    def nearest(query: torch.Tensor, bank: torch.Tensor, cosine: bool = False) -> torch.Tensor:
        if bank.shape[0] == 0:
            raise ValueError('Empty normal reference bank')
        result = []
        for q in query.split(512):
            if cosine:
                distance = (1 - q @ bank.T).clamp_min(0)
            else:
                distance = torch.cdist(q.contiguous(), bank.contiguous())
            result.append(distance.min(-1).values)
        return torch.cat(result)

    @staticmethod
    def grid56(values: torch.Tensor, side: int) -> torch.Tensor:
        return F.interpolate(values.view(1, 1, side, side), (56, 56), mode='bilinear', align_corners=False)[0, 0]

    def distances(self, feature: dict, exclude_image: int | None = None) -> dict:
        normalized = self.normalized(feature)
        result = {}
        for key, bank in self.banks.items():
            if exclude_image is not None:
                bank = bank[self.origins[key] != exclude_image]
            distance = self.nearest(normalized[key], bank, cosine=key == 'CLIP')
            side = int(distance.numel() ** .5)
            result[key] = self.grid56(distance, side)
        return result

    @torch.inference_mode()
    def fit(self, loader) -> dict:
        support = [self.extract(data[0], data[1]) for data in loader]
        if len(support) < 2:
            raise ValueError('This crossfit prototype requires at least two normal images')
        rgb_all = torch.cat([f['IMG'] for f in support])
        self.rgb_mean, self.rgb_std = rgb_all.mean(), rgb_all.std().clamp_min(1e-6)
        keys = ['IMG', 'PCD'] + (['CLIP'] if self.clip else [])
        for key in keys:
            all_features = torch.cat([self.normalized(f)[key] for f in support])
            selected = get_coreset_idx(all_features.cpu(), n=max(32, int(.1 * len(all_features))),
                                       eps=.9, float16=True, print_=False).long().to(self.args.device)
            self.banks[key] = all_features[selected]
            tokens = support[0][key].shape[0]
            self.origins[key] = selected // tokens
            if self.origins[key].unique().numel() < 2:
                raise ValueError('Reference coreset has fewer than two source images')
        crossfit = {key: [] for key in keys}
        semantic_values = []
        for image_index, feature in enumerate(support):
            maps = self.distances(feature, exclude_image=image_index)
            valid = feature['valid'].view(56, 56)
            for key in keys:
                crossfit[key].append(maps[key][valid])
            if self.clip:
                semantics = self.grid56(feature['semantic'], 24)
                semantic_values.append(semantics[valid])
        for key in keys:
            values = torch.cat(crossfit[key])
            self.scales[key] = values.mean().clamp_min(1e-6)
        if semantic_values:
            values = torch.cat(semantic_values)
            self.semantic_center = values.median()
            self.semantic_scale = (values - self.semantic_center).abs().median().clamp_min(.05)
        return {
            'support_count': len(support), 'bank_sizes': {k: len(v) for k, v in self.banks.items()},
            'scales': {k: float(v) for k, v in self.scales.items()},
            'semantic_center': float(self.semantic_center), 'semantic_scale': float(self.semantic_scale),
            'frozen': not any(p.requires_grad for p in self.native.deep_feature_extractor.parameters()),
            'epochs': 0, 'training_kind': 'frozen memories + normal-only leave-image-out scale fit',
            'fullshot_weights_banks_or_statistics_used': False,
        }

    @torch.inference_mode()
    def predict(self, feature: dict, variant: str) -> tuple[torch.Tensor, dict]:
        raw = self.distances(feature)
        maps = {k: v / self.scales[k] for k, v in raw.items()}
        valid = feature['valid'].view(56, 56)
        for key in maps:
            maps[key] = maps[key] * valid
        private = (maps['IMG'] * maps['PCD']).clamp_min(0).sqrt()
        if variant == 'private':
            fused = private
        elif variant == 'clip_memory':
            fused = private + .25 * maps['CLIP']
        elif variant == 'text_evidence':
            semantic = self.grid56(feature['semantic'], 24)
            residual = ((semantic - self.semantic_center) / self.semantic_scale).clamp(0, 4) * valid
            fused = private + .25 * maps['CLIP'] + .1 * residual
        else:
            raise ValueError(variant)
        maps['PRIVATE'] = private
        if self.clip:
            maps['TEXT'] = self.grid56(feature['semantic'], 24) * valid
        if not torch.isfinite(fused).all() or fused.min() < 0:
            raise ValueError('Invalid fused scores')
        return fused, maps

    def checkpoint(self) -> dict:
        return {'banks': {k: v.cpu() for k, v in self.banks.items()},
                'origins': {k: v.cpu() for k, v in self.origins.items()},
                'scales': {k: v.cpu() for k, v in self.scales.items()},
                'rgb_mean': self.rgb_mean.cpu(), 'rgb_std': self.rgb_std.cpu(),
                'semantic_center': float(self.semantic_center), 'semantic_scale': float(self.semantic_scale),
                'normal_prompts': NORMAL_PROMPTS, 'abnormal_prompts': ABNORMAL_PROMPTS}
