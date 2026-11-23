"""Learn RGB visibility from normal-only geometry positives/trusted exterior."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_dilation
from Model.fewshot_recam_real import normalize_recam_variant
from Model.fewshot_hybrid_fidelity_residual import ResidualFidelityMemory
from Model.fewshot_recam_input_correct import InputCorrectHybrid
from Model.fewshot_private_text import PrivateTextMemory


def trusted_exterior(valid):
    array = valid.view(56, 56).cpu().numpy()
    coordinates = np.argwhere(array)
    hull = ConvexHull(coordinates)
    yy, xx = np.meshgrid(np.arange(56), np.arange(56), indexing='ij')
    grid = np.stack([yy, xx], -1)
    interior = np.ones((56, 56), dtype=bool)
    for equation in hull.equations:
        interior &= (grid @ equation[:2] + equation[2]) <= 1e-6
    return torch.from_numpy(~binary_dilation(interior, iterations=1)).to(valid.device).flatten()


class RGBVisibilityMemory(ResidualFidelityMemory):
    def confidences(self, feature, exclude=None):
        fg = self.marginal.banks['IMG']
        bg = self.background_bank
        if exclude is not None:
            fg = fg[self.marginal.origins['IMG'] != exclude]
            bg = bg[self.background_origins != exclude]
        dfg = PrivateTextMemory.nearest(feature['IMG'], fg, cosine=True)
        dbg = PrivateTextMemory.nearest(feature['IMG'], bg, cosine=True)
        return (dbg / (dfg + dbg).clamp_min(1e-8)).clamp(0, 1), dfg

    @torch.inference_mode()
    def fit(self, loader):
        result = super().fit(loader)
        support = [self.extract(data[0], data[1])['marginal'] for data in loader]
        trusted = [trusted_exterior(feature['valid']) for feature in support]
        self.visibility_enabled = all(int(mask.sum()) >= 32 for mask in trusted)
        if self.visibility_enabled:
            self.background_bank = torch.cat([feature['IMG'][mask] for feature, mask in zip(support, trusted)])
            self.background_origins = torch.cat([torch.full((int(mask.sum()),), index, device='cuda') for index, mask in enumerate(trusted)])
            confidence = [self.confidences(feature, exclude=index)[0] for index, feature in enumerate(support)]
            bg_values = torch.cat([value[mask] for value, mask in zip(confidence, trusted)])
            self.visibility_threshold = bg_values.quantile(.999).clamp_min(.5)
            foreground = []
            for feature, value in zip(support, confidence):
                eroded = F.avg_pool2d(feature['valid'].float().view(1, 1, 56, 56), 3, 1, 1)[0, 0] == 1
                if eroded.any():
                    foreground.append(value[eroded.flatten()])
            fg_values = torch.cat(foreground) if foreground else torch.empty(0, device='cuda')
            recall = float((fg_values > self.visibility_threshold).float().mean()) if len(fg_values) else 0.
            bg_fpr = float((bg_values > self.visibility_threshold).float().mean())
            self.visibility_enabled = recall >= .8
        else:
            self.background_bank = torch.empty((0, 2304), device='cuda')
            self.background_origins = torch.empty(0, dtype=torch.long, device='cuda')
            self.visibility_threshold = torch.tensor(1., device='cuda')
            recall, bg_fpr = 0., 0.
        result['normal_rgb_visibility'] = {'enabled': self.visibility_enabled,
            'trusted_background_count_by_source': [int(mask.sum()) for mask in trusted],
            'bank_size': len(self.background_bank), 'source_excluded_calibration': True,
            'confidence_threshold': float(self.visibility_threshold), 'normal_fg_recovery': recall,
            'normal_bg_false_positive_proxy': bg_fpr, 'only_four_normal_sources': True,
            'not_a_population_fpr_guarantee': True,
            'fallback_is_normal_only_not_test_selected': True}
        result['changed_method_settings']['fusion'] = 'normal-fidelity residual with learned RGB visibility at missing 3D support'
        return result

    @torch.inference_mode()
    def predict(self, feature, variant):
        variant = normalize_recam_variant(variant)
        if variant not in ['fidelity_rgb_visibility', 'recam_real', 'recam_whiten', 'recam_private']:
            raise ValueError(variant)
        if not self.visibility_enabled:
            return super().predict(feature, 'fidelity_residual')
        m = feature['marginal']
        confidence, distance = self.confidences(m)
        geometry = m['valid'].view(56, 56)
        neighborhood = F.max_pool2d(geometry.float()[None, None], 3, 1, 1)[0, 0] > 0
        rgb_valid = geometry | (neighborhood & (confidence.view(56, 56) > self.visibility_threshold))
        r = distance.view(56, 56) / self.marginal.scales['IMG'] * rgb_valid
        r = F.interpolate(r[None, None], (224, 224), mode='bilinear', align_corners=False)[0, 0]
        # Original conditional/GP input validity remains unchanged.
        joint, maps = InputCorrectHybrid.predict(self, feature, 'correct_hybrid')
        weight = self.conditional_weight
        support = F.interpolate(rgb_valid.float()[None, None], (224, 224), mode='bilinear', align_corners=False)[0, 0] > 0
        valid = support | feature['conditional']['valid']
        score = (weight * joint + (1 - weight) * r) * valid
        if not torch.isfinite(score).all() or (score[~valid] != 0).any():
            raise ValueError('Invalid RGB visibility score')
        return score, {**maps, 'RGB_MARGINAL': r}

    def checkpoint(self):
        result = super().checkpoint()
        result.update({'rgb_visibility_enabled': self.visibility_enabled,
                       'background_bank': self.background_bank.cpu(),
                       'background_origins': self.background_origins.cpu(),
                       'visibility_threshold': self.visibility_threshold.cpu()})
        return result
