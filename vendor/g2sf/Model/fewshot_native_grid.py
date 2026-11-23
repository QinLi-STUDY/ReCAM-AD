"""Resolution control: native 56x56 frozen features, not a new backbone."""
import torch
import torch.nn.functional as F

from Model.fewshot_surface_memory import SurfaceMemory, surface_field


class NativeGridMemory(SurfaceMemory):
    @torch.inference_mode()
    def rgb_features(self, image):
        image = F.interpolate(image.to(self.args.device), (784, 784), mode='bicubic', align_corners=False)
        tokens = self.rgb.prepare_tokens_with_masks(image)
        selected = []
        for index, block in enumerate(self.rgb.blocks):
            tokens = block(tokens)
            if index in [3, 7, 11]:
                patches = self.rgb.norm(tokens)[:, 1 + self.rgb.num_register_tokens:]
                selected.append(F.normalize(patches[0].float(), dim=-1))
        if len(selected) != 3 or any(value.shape != (3136, 768) for value in selected):
            raise ValueError('Native score-grid token shape mismatch')
        return F.normalize(torch.cat(selected, -1), dim=-1)

    @torch.inference_mode()
    def extract(self, image, cloud):
        cloud = cloud.to(self.args.device)
        field, _ = surface_field(cloud)
        field = F.interpolate(field, (784, 784), mode='bicubic', align_corners=False).clamp(0, 1)
        mean = field.new_tensor([.485, .456, .406])[None, :, None, None]
        std = field.new_tensor([.229, .224, .225])[None, :, None, None]
        rgb, normal = self.rgb_features(image), self.rgb_features((field - mean) / std)
        native224 = F.interpolate(cloud, (224, 224), mode='nearest')
        foreground = torch.all(native224 != 0, dim=1, keepdim=True).float()
        foreground = F.adaptive_avg_pool2d(F.avg_pool2d(foreground, 3, stride=1), (56, 56))
        feature = {'IMG': rgb, 'PCD': normal, 'valid': (foreground[0, 0] > 0).flatten(),
                   'JOINT': torch.cat([rgb, normal], -1) / (2 ** .5)}
        if self.clip:
            feature.update(self.clip(image))
        if any(not torch.isfinite(value).all() for value in feature.values()):
            raise ValueError('Invalid native-grid frozen feature')
        return feature

    @torch.inference_mode()
    def fit(self, loader):
        support, rgb_views = [], []
        for index, data in enumerate(loader):
            feature = self.extract(data[0], data[1]); support.append(feature)
            valid = feature['valid'].view(56, 56)
            for rotation in range(4):
                rgb = feature['IMG'] if rotation == 0 else self.rgb_features(torch.rot90(data[0], rotation, [-2, -1]))
                rgb_views.append((rgb[torch.rot90(valid, rotation, [-2, -1]).flatten()], index))
        if len(support) != 4:
            raise ValueError('Native first4 only')
        self.banks['IMG'] = torch.cat([features for features, _ in rgb_views])
        self.origins['IMG'] = torch.cat([torch.full((len(features),), index, device=self.args.device)
                                        for features, index in rgb_views])
        for key in ['PCD', 'JOINT'] + (['CLIP'] if self.clip else []):
            values, origins = [], []
            for index, feature in enumerate(support):
                side = int(feature[key].shape[0] ** .5)
                valid = F.interpolate(feature['valid'].view(1, 1, 56, 56).float(), (side, side),
                                      mode='nearest').flatten().bool()
                values.append(feature[key][valid])
                origins.append(torch.full((int(valid.sum()),), index, device=self.args.device))
            self.banks[key], self.origins[key] = torch.cat(values), torch.cat(origins)
        crossfit, means, semantic, excess = {}, {}, [], []
        for index, feature in enumerate(support):
            maps = self.distances(feature, exclude_image=index)
            valid = feature['valid'].view(56, 56)
            for key, values in maps.items():
                crossfit.setdefault(key, []).append(values[valid])
                means.setdefault(key, []).append(values[valid].mean())
            excess.append((maps['JOINT'] - .5 * (maps['IMG'] + maps['PCD'])).clamp_min(0)[valid])
            if self.clip:
                semantic.append(self.grid56(feature['semantic'], 24)[valid])
        self.scales = {key: torch.cat(values).quantile(.9).clamp_min(1e-6)
                       for key, values in crossfit.items()}
        variability = {key: torch.stack(values).std(unbiased=True) / torch.stack(values).mean().clamp_min(1e-6)
                       for key, values in means.items()}
        self.geometry_weight = (variability['IMG'] / (variability['IMG'] + variability['PCD'] + .1)).clamp(.05, .5)
        self.excess_scale = torch.cat(excess).quantile(.9).clamp_min(1e-6)
        if semantic:
            semantic = torch.cat(semantic)
            self.semantic_center = semantic.median()
            self.semantic_scale = (semantic - self.semantic_center).abs().median().clamp_min(.05)
        return {'support_count': 4, 'original_source_images': 4, 'rgb_augmented_views': 16,
                'normal_views': 4, 'paired_views': 4, 'frozen': True, 'epochs': 0,
                'fullshot_weights_banks_or_statistics_used': False,
                'training_kind': 'frozen native56-grid memories and source-excluded calibration',
                'bank_sizes': {key: len(bank) for key, bank in self.banks.items()},
                'scales': {key: float(value) for key, value in self.scales.items()},
                'geometry_weight': float(self.geometry_weight), 'joint_excess_scale': float(self.excess_scale),
                'normal_cross_image_variability': {key: float(value) for key, value in variability.items()},
                'changed_method_settings': {'rgb_backbone': 'DINOv2B14', 'layers': [4, 8, 12],
                     'rgb_resolution': 784, 'normal_rgb_resolution': 784, 'native_grid': 56,
                     'bank_fraction': 1., 'pointmae_used_in_predictions': False,
                     'geometry_input': 'same native organized XYZ finite-difference normals',
                     'foreground_mask': 'original Point interpolation support at56',
                     'memory_compute_budget_increased': True}}

    def predict(self, feature, variant):
        aliases = {'grid_residual': 'surface_residual', 'grid_product': 'surface_product',
                   'grid_relation': 'surface_joint'}
        return super().predict(feature, aliases.get(variant, variant))
