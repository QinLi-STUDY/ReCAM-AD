"""Two normal priors: author conditional mapper and frozen marginal memory.

Calibrated energies are heuristic proxies, not exact likelihood ratios.
"""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from Model.fewshot_recam_real import normalize_recam_variant
from Model.fewshot_recam_prepared import ReCAMPreparedMemory
from Model.fewshot_bayesian_surface import BayesianSurfaceMemory
from Model.fewshot_coverage_text import DINOV2_WEIGHT
from Model.fewshot_private_text import file_sha


class ViewLoader:
    def __init__(self, loader, image, cloud):
        self.loader, self.image, self.cloud = loader, image, cloud

    def __iter__(self):
        for value in self.loader:
            yield (value[0][self.image], value[1][self.cloud], *value[2:])


class HybridDensityMemory:
    def __init__(self, category):
        self.conditional = ReCAMPreparedMemory(category)
        config = SimpleNamespace(device='cuda', group_size=128, num_group=512)
        self.marginal = BayesianSurfaceMemory(config, category, include_clip=False)

    @torch.inference_mode()
    def extract(self, image, cloud):
        return {'conditional': self.conditional.extract(image['real'], cloud['prepared']),
                'marginal': self.marginal.extract(image['fine'], cloud['native'])}

    @torch.inference_mode()
    def fit(self, loader):
        conditional_fit = self.conditional.fit(ViewLoader(loader, 'real', 'prepared'))
        marginal_fit = self.marginal.fit(ViewLoader(loader, 'fine', 'native'))
        support = [self.extract(data[0], data[1]) for data in loader]
        anchors = []
        for feature in support:
            m = feature['marginal']; valid = m['valid'].nonzero().flatten()
            indices = valid[torch.linspace(0, len(valid) - 1, min(64, len(valid)), device='cuda').round().long()]
            anchors.append({key: m[key][indices] for key in ['IMG', 'PCD']})
        values = {'conditional': [], 'marginal': []}
        for index, feature in enumerate(support):
            c, _ = self.conditional.predict(feature['conditional'], 'recam_real')
            values['conditional'].append(c[feature['conditional']['valid']])
            m = feature['marginal']; mask = m['valid'].view(56, 56)
            distances = self.marginal.distances(m, exclude_image=index)
            base = (distances['IMG'] / self.marginal.scales['IMG'] * distances['PCD'] / self.marginal.scales['PCD']).clamp_min(0).sqrt()
            gps = self.marginal.gp_scores(m, self.marginal._models(anchors, exclude_image=index))
            for key in ['GP_SHAPE', 'GP_RGB']:
                base = base + .1 * (gps[key] / self.marginal.gp_scales[key] - 1).clamp_min(0) * mask
            values['marginal'].append(base.square()[mask])
        self.scales = {key: torch.cat(v).quantile(.9).clamp_min(1e-8) for key, v in values.items()}
        return {'support_count': 4, 'original_source_images': 4, 'frozen': True, 'epochs': 0,
                'fullshot_weights_banks_or_statistics_used': False,
                'training_kind': 'author4shot conditional weights plus locally fit normal marginal GP and joint-energy calibration',
                'conditional_fit': conditional_fit, 'marginal_fit': marginal_fit,
                'fused_energy_scales': {key: float(v) for key, v in self.scales.items()},
                'normal_calibration': {'marginal_bank_and_gp_source_excluded': True,
                                       'author_conditional_weights_saw_all_four_sources': True},
                'dinov2_raw_checkpoint_sha': file_sha(DINOV2_WEIGHT),
                'changed_method_settings': {'backbones': ['DINOv1B8 author mapper', 'raw DINOv2B14 memory'],
                    'resolutions': [224, 784], 'same_support_count': 4, 'extra_backbone_and_compute': True,
                    'estimated_marigold_stream_used': False, 'text_used': False,
                    'normal_geometry_layouts': ['author squarepadded', 'native inversepadding'],
                    'fused_normal_energy_calibration': 'sourceLOO q90 for marginal; four training normal q90 for conditional'}}

    @torch.inference_mode()
    def predict(self, feature, variant):
        variant = normalize_recam_variant(variant)
        aliases = {'recam_real': 'hybrid_sum', 'recam_whiten': 'hybrid_product', 'recam_private': 'hybrid_sum'}
        variant = aliases.get(variant, variant)
        conditional, _ = self.conditional.predict(feature['conditional'], 'recam_real')
        marginal, _ = self.marginal.predict(feature['marginal'], 'gp_pair')
        conditional = conditional / self.scales['conditional']
        marginal = F.interpolate(marginal.square().view(1, 1, 56, 56), (224, 224), mode='bilinear', align_corners=False)[0, 0] / self.scales['marginal']
        if variant == 'hybrid_sum':
            score = .5 * (conditional + marginal)
        elif variant == 'hybrid_product':
            score = (conditional * marginal).clamp_min(0).sqrt()
        else:
            raise ValueError(variant)
        if not torch.isfinite(score).all():
            raise ValueError('Nonfinite hybrid energy')
        return score, {'CONDITIONAL': conditional, 'MARGINAL': marginal}

    def checkpoint(self):
        return {'conditional': self.conditional.checkpoint(), 'marginal': self.marginal.checkpoint(),
                'fused_energy_scales': {key: value.cpu() for key, value in self.scales.items()}}
