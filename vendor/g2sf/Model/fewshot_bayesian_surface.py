"""Learned sparse GP coefficients; frozen encoders, normal-only source split."""
import math
import torch
import torch.nn.functional as F

from Model.fewshot_native_grid import NativeGridMemory


class SparseGP:
    def __init__(self, source, target, tau):
        self.source, self.tau = source, tau.clamp_min(1e-4)
        self.mean = target.mean(0)
        variance = target.var(0, unbiased=True)
        self.output_variance = (.9 * variance + .1 * variance.mean()).clamp_min(1e-8)
        kernel = torch.exp((source @ source.T - 1) / self.tau)
        kernel = (kernel + kernel.T) / 2
        self.cholesky, info = torch.linalg.cholesky_ex(kernel + .1 * torch.eye(len(source), device=source.device))
        if int(info) != 0:
            raise ValueError('GP covariance is not numerically positive definite')
        self.alpha = torch.cholesky_solve(target - self.mean, self.cholesky)

    def predict(self, source, target):
        errors, uncertainties = [], []
        for begin in range(0, len(source), 256):
            query = source[begin:begin + 256]
            kernel = torch.exp((query @ self.source.T - 1) / self.tau)
            mean = self.mean + kernel @ self.alpha
            solved = torch.linalg.solve_triangular(self.cholesky, kernel.T, upper=False)
            uncertainty = (1.1 - solved.square().sum(0)).clamp_min(.1)
            residual = (target[begin:begin + 256] - mean).square()
            error = (residual / (uncertainty[:, None] * self.output_variance)).mean(-1)
            errors.append(error); uncertainties.append(uncertainty)
        return torch.cat(errors), torch.cat(uncertainties)

    def checkpoint(self):
        return {key: getattr(self, key).cpu() for key in
                ['source', 'tau', 'mean', 'output_variance', 'cholesky', 'alpha']}


class BayesianSurfaceMemory(NativeGridMemory):
    def __init__(self, args, category, include_clip=True):
        super().__init__(args, category, include_clip=include_clip)
        self.category = category
        self.image_scores = []

    def _models(self, anchors, exclude_image=None):
        kept = [feature for index, feature in enumerate(anchors) if index != exclude_image]
        banks = {key: torch.cat([feature[key] for feature in kept]) for key in ['IMG', 'PCD']}
        return {name: SparseGP(banks[source], banks[target], self.scales[source])
                for source, target, name in [('IMG', 'PCD', 'GP_SHAPE'), ('PCD', 'IMG', 'GP_RGB')]}

    def gp_scores(self, feature, models):
        valid = feature['valid']
        maps = {}
        for source, target, name in [('IMG', 'PCD', 'GP_SHAPE'), ('PCD', 'IMG', 'GP_RGB')]:
            error, uncertainty = models[name].predict(feature[source][valid], feature[target][valid])
            error_map = torch.zeros(3136, device=self.args.device)
            variance_map = torch.zeros_like(error_map)
            error_map[valid], variance_map[valid] = error, uncertainty
            maps[name], maps[name + '_UNCERTAINTY'] = error_map.view(56, 56), variance_map.view(56, 56)
        return maps

    @torch.inference_mode()
    def fit(self, loader):
        fitted = super().fit(loader)
        support = [self.extract(data[0], data[1]) for data in loader]
        anchors, indices = [], []
        for feature in support:
            valid = feature['valid'].nonzero().flatten()
            selected = valid[torch.linspace(0, len(valid) - 1, min(64, len(valid)),
                                           device=valid.device).round().long()]
            indices.append(selected.cpu().tolist())
            anchors.append({key: feature[key][selected] for key in ['IMG', 'PCD']})
        if len(anchors) != 4:
            raise ValueError('Only four sources for Bayesian fit')
        crossfit = {}
        for index, feature in enumerate(support):
            models = self._models(anchors, exclude_image=index)
            scores = self.gp_scores(feature, models)
            for key in ['GP_SHAPE', 'GP_RGB']:
                crossfit.setdefault(key, []).append(scores[key][feature['valid'].view(56, 56)])
        self.gp_scales = {key: torch.cat(values).quantile(.9).clamp_min(1e-6)
                          for key, values in crossfit.items()}
        self.gp_models = self._models(anchors)
        fitted.update({'fusion_gp_coefficients_learned': True, 'encoder_frozen': True,
                       'gp_anchor_indices_by_source': indices,
                       'gp_anchor_count': sum(len(value['IMG']) for value in anchors),
                       'gp_normal_calibration_sources': 3, 'gp_test_reference_sources': 4,
                       'gp_noise_variance': .1, 'gp_covariance_pooled_shrinkage': .1,
                       'gp_scales': {key: float(value) for key, value in self.gp_scales.items()},
                       'training_kind': 'frozen encoders + normal-only learned GP coefficients (closed form)'})
        return fitted

    @torch.inference_mode()
    def predict(self, feature, variant):
        aliases = {'private': 'gp_shape', 'clip_memory': 'gp_pair', 'text_evidence': 'gp_tail'}
        variant = aliases.get(variant, variant)
        score, maps = super().predict(feature, 'grid_product')
        gp = self.gp_scores(feature, self.gp_models)
        valid = feature['valid'].view(56, 56)
        maps.update({key: value / self.gp_scales[key] if key in self.gp_scales else value
                     for key, value in gp.items()})
        score = score + .1 * (maps['GP_SHAPE'] - 1).clamp_min(0) * valid
        if variant == 'gp_pair':
            score = score + .1 * (maps['GP_RGB'] - 1).clamp_min(0) * valid
        elif variant not in ['gp_shape', 'gp_tail']:
            raise ValueError(variant)
        if not torch.isfinite(score).all():
            raise ValueError('Nonfinite Bayesian score')
        if variant == 'gp_tail':
            from recam.metrics import smooth
            dense = F.interpolate(score.view(1,1,56,56),(224,224),mode='bilinear',align_corners=False)
            processed = smooth(dense[0,0],'mvtec',self.category)
            pixel_valid = F.interpolate(valid.float().view(1, 1, 56, 56), (224, 224), mode='nearest')[0, 0].bool()
            values = processed[pixel_valid]
            denominator = processed[processed != 0].mean().clamp_min(1e-12).sqrt()
            self.image_scores.append(float(values.topk(max(1, math.ceil(.01 * len(values)))).values.mean() / denominator))
        return score, maps

    def checkpoint(self):
        result = super().checkpoint()
        result['gp_models'] = {key: value.checkpoint() for key, value in self.gp_models.items()}
        result['gp_scales'] = {key: float(value) for key, value in self.gp_scales.items()}
        return result
