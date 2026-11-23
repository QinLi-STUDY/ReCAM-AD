"""ReCAM real-stream residual scoring and normal-reference statistics.

The frozen mapper architectures and weights come from the CMDS-AD release.
"""
import ast
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.fewshot_private_text import file_sha, PrivateTextMemory
ASSETS = None  # Set to the selected dataset and shot by the release loader.


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'code/external/CMDS-AD'


def normalize_recam_variant(variant):
    """Accept legacy scoring names used by earlier release scripts."""
    return {
        'cmds_real': 'recam_real',
        'cmds_whiten': 'recam_whiten',
        'cmds_private': 'recam_private',
    }.get(variant, variant)


def released_classes():
    # Load exact author class definitions without running wandb/CLI/figure code.
    path = RELEASE / 'test_anomaly_fusion_mvtec3dad.py'
    tree = ast.parse(path.read_text())
    names = {'TripleLayerExtractor', 'CoordinateAttention', 'HierarchicalGatedMapper'}
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names], type_ignores=[])
    if len(selected.body) != 3:
        raise ValueError('Author architecture definitions changed')
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'timm': timm}
    exec(compile(selected, str(path), 'exec'), namespace)
    return namespace


def square_pad(tensor):
    height, width = tensor.shape[-2:]
    size = max(height, width)
    top, left = (size - height) // 2, (size - width) // 2
    return F.pad(tensor, (left, size - width - left, top, size - height - top))


class ReCAMRealMemory:
    def __init__(self, category):
        if not o3c.cuda.is_available():
            raise RuntimeError('Open3D CUDA required for released normal estimator')
        self.category = category
        manifest = ASSETS / category / 'manifest.json'
        self.assets = json.loads(manifest.read_text())
        classes = released_classes()
        self.extractor = classes['TripleLayerExtractor']('cuda').eval().requires_grad_(False)
        self.mappers = {}
        for direction, identity in self.assets.items():
            checkpoint = ROOT.parents[1] / identity['file']
            if file_sha(checkpoint) != identity['sha256']:
                raise ValueError('Author checkpoint identity changed')
            model = classes['HierarchicalGatedMapper']().cuda()
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            self.mappers[direction] = model.eval().requires_grad_(False)
        path = RELEASE / 'processing/pc2sn_o3d_gpu_mvtec3dad.py'
        spec = importlib.util.spec_from_file_location('_recam_real_cleaning_readonly', path)
        self.cleaning = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.cleaning)
        self.mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
        self.banks, self.origins, self.scales, self.residual_stats = {}, {}, {}, {}

    @torch.inference_mode()
    def normals(self, cloud):
        array = np.nan_to_num(cloud[0].permute(1, 2, 0).cpu().numpy(), nan=0.)
        height, width = array.shape[:2]
        clean = self.cleaning.remove_plane_cpu(array, .002 if self.category == 'dowel' else .005)
        clean = self.cleaning.connected_components_cleaning_cpu(clean)
        points = clean.reshape(-1, 3).astype(np.float32)
        device = o3c.Device('CUDA:0')
        pc = o3d.t.geometry.PointCloud(device)
        pc.point.positions = o3c.Tensor(points, device=device)
        pc.estimate_normals(max_nn=30, radius=.01)
        pc.orient_normals_towards_camera_location(o3c.Tensor([0., 0., 0.], dtype=o3c.float32, device=device))
        normal = -pc.point.normals.cpu().numpy().reshape(height, width, 3)
        normal[np.all(np.isclose(clean, 0), axis=2)] = [0., 0., 1.]
        colors = (np.clip((normal + 1) / 2, 0, 1) * 255).astype(np.uint8)
        if height != width:
            size = max(height, width)
            top, left = (size - height) // 2, (size - width) // 2
            colors = cv2.copyMakeBorder(colors, top, size - height - top, left, size - width - left,
                                       cv2.BORDER_CONSTANT, value=(127, 127, 255))
        colors = cv2.resize(colors, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(colors.copy()).permute(2, 0, 1)[None].cuda().float() / 255
        return (image - self.mean) / self.std

    @torch.inference_mode()
    def extract(self, image, cloud):
        image = image.cuda()
        normals = self.normals(cloud)
        rgb_layers, shape_layers = self.extractor(image), self.extractor(normals)
        targets = {'IMG': F.normalize(torch.cat(rgb_layers, 1).flatten(2).transpose(1, 2)[0].float(), dim=-1),
                   'PCD': F.normalize(torch.cat(shape_layers, 1).flatten(2).transpose(1, 2)[0].float(), dim=-1)}
        pred_rgb = self.mappers['3d_to_2d'](shape_layers)[0][0]
        pred_shape = self.mappers['2d_to_3d'](rgb_layers)[0][0]
        residual = {'IMG': targets['IMG'] - F.normalize(pred_rgb.float(), dim=-1),
                    'PCD': targets['PCD'] - F.normalize(pred_shape.float(), dim=-1)}
        # Analytical Point interpolation support; explicit approximation, no features.
        xyz = F.interpolate(square_pad(cloud.cuda()), (224, 224), mode='bilinear', align_corners=False)
        valid = torch.all(xyz != 0, dim=1, keepdim=True).float()
        valid = F.adaptive_avg_pool2d(F.avg_pool2d(valid, 3, stride=1), (224, 224))[0, 0] > 0
        feature = {'valid': valid, 'residual': residual,
                   'raw': {key: .5 * value.square().sum(-1).view(224, 224) * valid for key, value in residual.items()}}
        # Memory is on original native28 patch lattice; no full224 memory expansion.
        feature['private'] = {key: F.normalize(F.interpolate(value.T.view(1, 2304, 224, 224), (28, 28),
                                 mode='bilinear', align_corners=False)[0].flatten(1).T, dim=-1)
                              for key, value in targets.items()}
        return feature

    def private_distances(self, feature, exclude=None):
        valid28 = F.interpolate(feature['valid'][None, None].float(), (28, 28), mode='nearest').flatten().bool()
        result = {}
        for key, tokens in feature['private'].items():
            bank = self.banks[key]
            if exclude is not None:
                bank = bank[self.origins[key] != exclude]
            values = torch.zeros(784, device='cuda')
            values[valid28] = PrivateTextMemory.nearest(tokens[valid28], bank, cosine=True)
            result[key] = F.interpolate(values.view(1, 1, 28, 28), (224, 224), mode='bilinear', align_corners=False)[0, 0] * feature['valid']
        return result

    @torch.inference_mode()
    def fit(self, loader):
        support, views = [], {'IMG': [], 'PCD': []}
        accumulated = {key: {'n': 0, 'sum': torch.zeros(2304, device='cuda'), 'square': torch.zeros(2304, device='cuda')}
                       for key in ['IMG', 'PCD']}
        real = {key: [] for key in ['IMG', 'PCD']}
        for index, data in enumerate(loader):
            feature = self.extract(data[0], data[1])
            foreground = feature['valid'].flatten()
            for key, value in feature['residual'].items():
                values = value[foreground]
                accumulated[key]['n'] += len(values)
                accumulated[key]['sum'] += values.sum(0)
                accumulated[key]['square'] += values.square().sum(0)
                real[key].append(feature['raw'][key][feature['valid']])
            valid28 = F.interpolate(feature['valid'][None, None].float(), (28, 28), mode='nearest')[0, 0].bool()
            for key in ['IMG', 'PCD']:
                views[key].append((feature['private'][key][valid28.flatten()], index))
            for rotation in [1, 2, 3]:
                layers = self.extractor(torch.rot90(data[0].cuda(), rotation, [-2, -1]))
                tokens = F.normalize(F.interpolate(torch.cat(layers, 1), (28, 28), mode='bilinear', align_corners=False)[0].flatten(1).T, dim=-1)
                views['IMG'].append((tokens[torch.rot90(valid28, rotation, [-2, -1]).flatten()], index))
            # Drop large224 residual tensors; only memory features/normal maps retained.
            support.append({'valid': feature['valid'], 'private': feature['private'], 'raw': feature['raw']})
        if len(support) != 4:
            raise ValueError('Exactly four normal support images required')
        for key, value in accumulated.items():
            mean = value['sum'] / value['n']
            variance = (value['square'] / value['n'] - mean.square()).clamp_min(0)
            self.residual_stats[key] = {'mean': mean, 'variance': (.9 * variance + .1 * variance.mean()).clamp_min(1e-8)}
            self.scales['real_' + key] = torch.cat(real[key]).quantile(.9).clamp_min(1e-6)
        for key, values in views.items():
            self.banks[key] = torch.cat([value[0] for value in values])
            self.origins[key] = torch.cat([torch.full((len(value[0]),), value[1], device='cuda') for value in values])
        private = {key: [] for key in ['IMG', 'PCD']}
        for index, feature in enumerate(support):
            maps = self.private_distances(feature, exclude=index)
            for key, value in maps.items():
                private[key].append(value[feature['valid']])
        for key, values in private.items():
            self.scales['private_' + key] = torch.cat(values).quantile(.9).clamp_min(1e-6)
        return {'support_count': 4, 'original_source_images': 4, 'frozen': True, 'epochs': 0,
                'fullshot_weights_banks_or_statistics_used': False,
                'training_kind': 'author4shot trained mapper initialization + local normal residual covariance and memory fit',
                'author_mapper_training_redone_locally': False, 'author_sd_normal_augmentation_prior_used': True,
                'baseline_reproduction': 'ReCAM real-stream residual scoring with released mappers',
                'bank_sizes': {key: len(value) for key, value in self.banks.items()},
                'scales': {key: float(value) for key, value in self.scales.items()},
                'mapper_identities': self.assets,
                'author_inference_source_sha': file_sha(RELEASE / 'test_anomaly_fusion_mvtec3dad.py'),
                'normal_statistics_independent_images': 4,
                'changed_method_settings': {'backbone': 'raw frozen DINOv1B8 native224', 'layer_indices': [4, 7, 11],
                    'normal_estimator': 'released Open3DCUDA radius.01/maxnn30/camera-away blueBG',
                    'normal_preprocessing': 'released CPU RANSAC/DBSCAN + squarepadding/colorquantization/cubic224',
                    'mask': 'analytical square-padded Point support approximation',
                    'mapper_and_score_resolution': 224, 'estimated_marigold_stream_used': False,
                    'clip_or_text_used': False, 'residual_covariance_shrinkage': .1}}

    @torch.inference_mode()
    def predict(self, feature, variant):
        variant = normalize_recam_variant(variant)
        valid = feature['valid']
        if variant == 'recam_real':
            maps = feature['raw']
            score = maps['IMG'] * maps['PCD']
        elif variant == 'recam_whiten':
            maps = {}
            for key, value in feature['residual'].items():
                stats = self.residual_stats[key]
                maps[key] = ((value - stats['mean']).square() / stats['variance']).mean(-1).view(224, 224) * valid
            score = maps['IMG'] * maps['PCD']
        elif variant == 'recam_private':
            maps = {key: value / self.scales['real_' + key] for key, value in feature['raw'].items()}
            private = self.private_distances(feature)
            extra = (private['IMG'] / self.scales['private_IMG'] * private['PCD'] / self.scales['private_PCD']).clamp_min(0).sqrt()
            score = maps['IMG'] * maps['PCD'] + .1 * (extra - 1).clamp_min(0) * valid
            maps['PRIVATE'] = extra
        else:
            raise ValueError(variant)
        if not torch.isfinite(score).all():
            raise ValueError('Nonfinite real-stream score')
        return score, maps

    def checkpoint(self):
        return {'assets': self.assets, 'banks': {key: value.cpu() for key, value in self.banks.items()},
                'origins': {key: value.cpu() for key, value in self.origins.items()},
                'scales': {key: value.cpu() for key, value in self.scales.items()},
                'residual_stats': {key: {name: value.cpu() for name, value in stats.items()} for key, stats in self.residual_stats.items()}}
