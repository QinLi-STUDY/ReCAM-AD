"""Retain the joint expert; normal fidelity controls its RGB fallback."""
import torch
import torch.nn.functional as F
from Model.fewshot_recam_real import normalize_recam_variant
from Model.fewshot_hybrid_fidelity import FidelityHybridMemory
from Model.fewshot_recam_input_correct import InputCorrectHybrid


class ResidualFidelityMemory(FidelityHybridMemory):
    @torch.inference_mode()
    def fit(self, loader):
        result = super().fit(loader)
        result['changed_method_settings']['fusion'] = 'normal-fidelity weighted original joint expert plus RGB expert'
        return result

    @torch.inference_mode()
    def predict(self, feature, variant):
        variant = normalize_recam_variant(variant)
        if variant not in ['fidelity_residual', 'recam_real', 'recam_whiten', 'recam_private']:
            raise ValueError(variant)
        joint, maps = InputCorrectHybrid.predict(self, feature, 'correct_hybrid')
        distances = self.marginal.distances(feature['marginal'])
        r = F.interpolate((distances['IMG'] / self.marginal.scales['IMG'])[None, None],
                          (224, 224), mode='bilinear', align_corners=False)[0, 0].clamp_min(0)
        rgb_support = F.interpolate(feature['marginal']['valid'].float().view(1, 1, 56, 56),
                                    (224, 224), mode='bilinear', align_corners=False)[0, 0] > 0
        valid = rgb_support | feature['conditional']['valid']
        weight = self.conditional_weight
        score = (weight * joint + (1 - weight) * r) * valid
        if not torch.isfinite(score).all() or (score[~valid] != 0).any():
            raise ValueError('Invalid fidelity residual')
        return score, {**maps, 'RGB_MARGINAL': r}
