"""Native XYZ surface evidence and source-excluded paired memory.

Surface-normal inputs are inspired by GPFlow/CMDS-AD; neither is reproduced.
Only the selected four normal images fit memory and calibration statistics.
"""
import torch
import torch.nn.functional as F

from Model.fewshot_coverage_masked import MaskedCoverageTextMemory


@torch.inference_mode()
def surface_field(cloud):
    """Camera-oriented normals on the original organized XYZ grid."""
    valid = torch.all(cloud != 0, dim=1, keepdim=True)
    mass = F.avg_pool2d(valid.float(), 3, 1, 1)
    smooth = F.avg_pool2d(cloud * valid, 3, 1, 1) / mass.clamp_min(1 / 9)
    du = smooth[:, :, 1:-1, 2:] - smooth[:, :, 1:-1, :-2]
    dv = smooth[:, :, 2:, 1:-1] - smooth[:, :, :-2, 1:-1]
    normal = F.normalize(torch.cross(du, dv, dim=1), dim=1, eps=1e-12)
    center = smooth[:, :, 1:-1, 1:-1]
    normal = torch.where((normal * center).sum(1, keepdim=True) > 0, -normal, normal)
    interior_valid = valid[:, :, 1:-1, 1:-1].clone()
    interior_valid &= valid[:, :, 1:-1, :-2] & valid[:, :, 1:-1, 2:]
    interior_valid &= valid[:, :, :-2, 1:-1] & valid[:, :, 2:, 1:-1]
    normal = F.pad(normal * interior_valid, (1, 1, 1, 1))
    return (normal + 1) / 2, valid


class SurfaceMemory(MaskedCoverageTextMemory):
    def __init__(self, args, category, include_clip=True):
        super().__init__(args, category, include_clip=include_clip)
        self.support_features = []
        self.collect_support = False

    @torch.inference_mode()
    def extract(self, image, cloud):
        cloud = cloud.to(self.args.device)
        field, valid = surface_field(cloud)
        field = F.interpolate(field, (448, 448), mode='bicubic', align_corners=False).clamp(0, 1)
        mean = field.new_tensor([.485, .456, .406])[None, :, None, None]
        std = field.new_tensor([.229, .224, .225])[None, :, None, None]
        rgb = self.rgb_features(image)
        geometry = self.rgb_features((field - mean) / std)
        foreground = F.adaptive_max_pool2d(valid.float(), (56, 56))[0, 0].bool().flatten()
        feature = {'IMG': rgb, 'PCD': geometry, 'valid': foreground,
                   'JOINT': torch.cat([rgb, geometry], -1) / (2 ** .5)}
        if self.clip:
            feature.update(self.clip(image))
        if any(not torch.isfinite(v).all() for v in feature.values()):
            raise ValueError('Nonfinite native surface evidence')
        if self.collect_support:
            self.support_features.append(feature)
        return feature

    @torch.inference_mode()
    def fit(self, loader):
        self.support_features = []
        self.collect_support = True
        fitted = super().fit(loader)
        self.collect_support = False
        pairs, origins = [], []
        for index, feature in enumerate(self.support_features):
            valid = F.interpolate(feature['valid'].view(1, 1, 56, 56).float(), (32, 32),
                                  mode='nearest').flatten().bool()
            pairs.append(feature['JOINT'][valid])
            origins.append(torch.full((int(valid.sum()),), index, device=self.args.device))
        self.banks['JOINT'] = torch.cat(pairs)
        self.origins['JOINT'] = torch.cat(origins)
        joint, excess = [], []
        for index, feature in enumerate(self.support_features):
            maps = self.distances(feature, exclude_image=index)
            valid = feature['valid'].view(56, 56)
            joint.append(maps['JOINT'][valid])
            excess.append((maps['JOINT'] - .5 * (maps['IMG'] + maps['PCD'])).clamp_min(0)[valid])
        self.scales['JOINT'] = torch.cat(joint).quantile(.9).clamp_min(1e-6)
        self.excess_scale = torch.cat(excess).quantile(.9).clamp_min(1e-6)
        fitted.update({'bank_sizes': {k: len(v) for k, v in self.banks.items()},
                       'paired_rgb_views': 4, 'joint_excess_scale': float(self.excess_scale),
                       'geometry_weight': float(self.geometry_weight),
                       'training_kind': 'frozen native-surface memories + source-excluded statistics'})
        fitted['changed_method_settings'].update({'geometry_backbone': 'shared frozen DINOv2B14',
             'geometry_input': 'camera-oriented surface normals from native organized XYZ',
             'pointmae_used_in_predictions': False, 'normal_rgb_resolution': 448,
             'geometry_augmentations': 1, 'paired_augmentations': 1})
        self.support_features = []
        return fitted

    @torch.inference_mode()
    def predict(self, feature, variant):
        aliases = {'private': 'surface_residual', 'clip_memory': 'surface_product',
                   'text_evidence': 'surface_joint'}
        variant = aliases.get(variant, variant)
        raw = self.distances(feature)
        valid = feature['valid'].view(56, 56)
        maps = {key: value / self.scales[key] * valid for key, value in raw.items()}
        residual = maps['IMG'] + self.geometry_weight * (maps['PCD'] - 1).clamp_min(0) * valid
        maps['PRIVATE'] = residual.clone()
        excess = (raw['JOINT'] - .5 * (raw['IMG'] + raw['PCD'])).clamp_min(0)
        maps['RELATION'] = excess / self.excess_scale * valid
        if self.clip:
            maps['TEXT'] = self.grid56(feature['semantic'], 24) * valid
        if variant == 'surface_residual':
            score = residual
        elif variant == 'surface_product':
            score = (maps['IMG'] * maps['PCD']).clamp_min(0).sqrt()
        elif variant == 'surface_joint':
            score = residual + .25 * (maps['RELATION'] - 1).clamp_min(0) * valid
        else:
            raise ValueError(variant)
        return score, maps

    def checkpoint(self):
        result = super().checkpoint()
        result['joint_excess_scale'] = float(self.excess_scale)
        result['geometry_input'] = 'native XYZ normals, no PointMAE features'
        return result
