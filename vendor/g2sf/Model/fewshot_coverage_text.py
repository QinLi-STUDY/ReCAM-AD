"""Support coverage controls inspired by AnomalyDINO, not its reproduction."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from Model.models import PointTransformer
from Model import interpolating_points
from Model.fewshot_private_text import PrivateTextMemory, RawFrozenCLIP


DINOV2_ROOT = Path(__file__).resolve().parents[2] / 'dinov2'
DINOV2_WEIGHT = Path(__file__).resolve().parents[3] / 'weights/backbones/dinov2_vitb14_pretrain.pth'


class CoverageTextMemory(PrivateTextMemory):
    def __init__(self, args, category: str, include_clip=True):
        self.args = args
        sys.path.insert(0, str(DINOV2_ROOT))
        from dinov2.hub.backbones import dinov2_vitb14
        self.rgb = dinov2_vitb14(pretrained=False)
        self.rgb.load_state_dict(torch.load(DINOV2_WEIGHT, map_location='cpu', weights_only=True), strict=True)
        self.rgb.to(args.device).eval().requires_grad_(False)
        point = PointTransformer(group_size=args.group_size, num_group=args.num_group)
        point.load_model_from_ckpt(str(DINOV2_WEIGHT.with_name('pointmae_pretrain.pth')))
        point.to(args.device).eval().requires_grad_(False)
        self.native = SimpleNamespace(deep_feature_extractor=point)
        self.clip = RawFrozenCLIP(category, args.device) if include_clip else None
        self.banks, self.origins, self.scales = {}, {}, {}
        self.rgb_mean, self.rgb_std = 0., 1.
        self.semantic_center, self.semantic_scale = .5, 1.
        self.geometry_weight = .1

    @torch.inference_mode()
    def rgb_features(self, image):
        image = F.interpolate(image.to(self.args.device), (448, 448), mode='bicubic', align_corners=False)
        layers = self.rgb.get_intermediate_layers(image, n=[3, 7, 11], reshape=False, norm=True)
        return F.normalize(torch.cat([F.normalize(v[0].float(), dim=-1) for v in layers], dim=-1), dim=-1)

    @torch.inference_mode()
    def extract(self, image, cloud):
        point = cloud[0].permute(1, 2, 0).reshape(-1, 3).to(self.args.device)
        valid = torch.all(point != 0, dim=1)
        nonzero = point[valid].T.unsqueeze(0).contiguous()
        if nonzero.shape[-1] < self.args.group_size:
            raise ValueError('Too few foreground points')
        xyz_features, centers, _, _ = self.native.deep_feature_extractor(nonzero)
        interpolated = interpolating_points(nonzero.double(), centers.permute(0, 2, 1).double(), xyz_features.double()).float()
        full = torch.zeros(1, interpolated.shape[1], 224 * 224, device=self.args.device)
        full[:, :, valid] = interpolated
        full = F.adaptive_avg_pool2d(F.avg_pool2d(full.view(1, -1, 224, 224), 3, stride=1), (56, 56))
        pg = full[0].flatten(1).T
        result = {'IMG': self.rgb_features(image), 'PCD': F.normalize(pg, dim=-1),
                  'valid': pg.abs().sum(-1) > 0}
        if self.clip:
            result.update(self.clip(image))
        if any(not torch.isfinite(v).all() for v in result.values()):
            raise ValueError('Invalid frozen features')
        return result

    def normalized(self, feature):
        return feature

    def distances(self, feature, exclude_image=None):
        result = {}
        for key, bank in self.banks.items():
            if exclude_image is not None:
                bank = bank[self.origins[key] != exclude_image]
            distance = self.nearest(feature[key], bank, cosine=True)
            result[key] = self.grid56(distance, int(distance.numel() ** .5))
        return result

    @torch.inference_mode()
    def fit(self, loader):
        support, rgb_views = [], []
        for image_index, data in enumerate(loader):
            feature = self.extract(data[0], data[1])
            support.append(feature)
            valid32 = F.interpolate(feature['valid'].view(1, 1, 56, 56).float(), (32, 32), mode='nearest')[0, 0].bool()
            for rotation in range(4):
                rgb = feature['IMG'] if rotation == 0 else self.rgb_features(torch.rot90(data[0], rotation, [-2, -1]))
                mask = torch.rot90(valid32, rotation, [-2, -1]).flatten()
                rgb_views.append((rgb[mask], image_index))
        if len(support) != 4:
            raise ValueError('Registered native first-four protocol')
        self.banks['IMG'] = torch.cat([x[0] for x in rgb_views])
        self.origins['IMG'] = torch.cat([torch.full((len(x[0]),), x[1], device=self.args.device) for x in rgb_views])
        for key in ['PCD'] + (['CLIP'] if self.clip else []):
            values, origins = [], []
            for image_index, feature in enumerate(support):
                side = int(feature[key].shape[0] ** .5)
                valid = F.interpolate(feature['valid'].view(1, 1, 56, 56).float(), (side, side), mode='nearest').flatten().bool()
                values.append(feature[key][valid])
                origins.append(torch.full((int(valid.sum()),), image_index, device=self.args.device))
            self.banks[key], self.origins[key] = torch.cat(values), torch.cat(origins)
        crossfit, per_image_mean, semantic = {}, {}, []
        for i, feature in enumerate(support):
            maps = self.distances(feature, exclude_image=i)
            valid = feature['valid'].view(56, 56)
            for key, values in maps.items():
                crossfit.setdefault(key, []).append(values[valid])
                per_image_mean.setdefault(key, []).append(values[valid].mean())
            if self.clip:
                semantic.append(self.grid56(feature['semantic'], 24)[valid])
        for key, values in crossfit.items():
            self.scales[key] = torch.cat(values).quantile(.9).clamp_min(1e-6)
        variability = {}
        for key, means in per_image_mean.items():
            means = torch.stack(means)
            variability[key] = means.std(unbiased=True) / means.mean().clamp_min(1e-6)
        # Normal-only cross-image stability; no claims of independent patch calibration.
        self.geometry_weight = (variability['IMG'] / (variability['IMG'] + variability['PCD'] + .1)).clamp(.05, .5)
        if semantic:
            values = torch.cat(semantic)
            self.semantic_center = values.median()
            self.semantic_scale = (values - self.semantic_center).abs().median().clamp_min(.05)
        return {'support_count': len(support), 'rgb_augmented_views': len(rgb_views), 'original_source_images': 4,
                'bank_sizes': {k: len(v) for k, v in self.banks.items()},
                'scales': {k: float(v) for k, v in self.scales.items()},
                'normal_cross_image_variability': {k: float(v) for k, v in variability.items()},
                'geometry_weight': float(self.geometry_weight), 'frozen': True, 'epochs': 0,
                'training_kind': 'frozen full foreground memories + normal-only stability fit',
                'fullshot_weights_banks_or_statistics_used': False,
                'changed_method_settings': {'rgb_backbone': 'DINOv2B14', 'rgb_resolution': 448,
                                            'bank_fraction': 1., 'metric': 'cosine', 'support_rotations': 4}}

    @torch.inference_mode()
    def predict(self, feature, variant):
        aliases = {'private': 'coverage_visual', 'clip_memory': 'coverage_clip', 'text_evidence': 'coverage_text'}
        variant = aliases.get(variant, variant)
        raw = self.distances(feature)
        valid = feature['valid'].view(56, 56)
        maps = {k: v / self.scales[k] * valid for k, v in raw.items()}
        fused = maps['IMG'] + self.geometry_weight * (maps['PCD'] - 1).clamp_min(0)
        maps['PRIVATE'] = fused.clone()
        if self.clip:
            maps['TEXT'] = self.grid56(feature['semantic'], 24) * valid
        if variant in ['coverage_clip', 'coverage_text']:
            fused = fused + .1 * (maps['CLIP'] - 1).clamp_min(0)
        if variant == 'coverage_text':
            text = ((maps['TEXT'] - self.semantic_center) / self.semantic_scale).clamp(0, 4) * valid
            fused = fused + .05 * text
        if variant not in ['coverage_visual', 'coverage_clip', 'coverage_text'] or not torch.isfinite(fused).all():
            raise ValueError('Variant/finite check failed')
        return fused, maps

    def checkpoint(self):
        result = {'banks': {k: v.cpu() for k, v in self.banks.items()},
                  'origins': {k: v.cpu() for k, v in self.origins.items()},
                  'scales': {k: v.cpu() for k, v in self.scales.items()},
                  'geometry_weight': float(self.geometry_weight),
                  'semantic_center': float(self.semantic_center), 'semantic_scale': float(self.semantic_scale)}
        return result
