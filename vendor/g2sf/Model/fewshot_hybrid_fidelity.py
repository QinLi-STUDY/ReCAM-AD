"""Normal angular-error fidelity gates for two separately calibrated experts."""
import torch
import torch.nn.functional as F
from Model.fewshot_recam_real import normalize_recam_variant
from Model.fewshot_recam_input_correct import InputCorrectHybrid


class FidelityHybridMemory(InputCorrectHybrid):
    @torch.inference_mode()
    def fit(self, loader):
        result = super().fit(loader)
        conditional_error = (self.conditional.scales['real_IMG'] * self.conditional.scales['real_PCD']).sqrt()
        marginal_error = self.marginal.scales['IMG']
        self.conditional_weight = (marginal_error / (marginal_error + conditional_error).clamp_min(1e-8)).clamp(1e-6, 1 - 1e-6)
        result['normal_fidelity_gate'] = {'conditional_weight': float(self.conditional_weight),
            'conditional_angular_error_proxy': float(conditional_error),
            'rgb_angular_error_proxy': float(marginal_error),
            'learned_from_normal_only': True, 'conditional_teacher_saw_support': True,
            'rgb_source_excluded': True, 'not_an_unconditional_blue_guarantee': True}
        result['changed_method_settings']['fusion'] = 'normal angular-fidelity gate; arithmetic or weighted geometric'
        result['changed_method_settings']['text_used'] = False
        return result

    @torch.inference_mode()
    def predict(self, feature, variant):
        variant = normalize_recam_variant(variant)
        aliases = {'recam_real': 'fidelity_mean', 'recam_whiten': 'fidelity_geometric', 'recam_private': 'fidelity_mean'}
        variant = aliases.get(variant, variant)
        c, _ = self.conditional.predict(feature['conditional'], 'recam_real')
        c = (c / self.scales['conditional']).clamp_min(0).sqrt()
        distances = self.marginal.distances(feature['marginal'])
        r = F.interpolate((distances['IMG'] / self.marginal.scales['IMG'])[None, None],
                          (224, 224), mode='bilinear', align_corners=False)[0, 0].clamp_min(0)
        weight = self.conditional_weight
        if variant == 'fidelity_mean':
            score = weight * c + (1 - weight) * r
        elif variant == 'fidelity_geometric':
            score = c.pow(weight) * r.pow(1 - weight)
        else:
            raise ValueError(variant)
        rgb_support = F.interpolate(feature['marginal']['valid'].float().view(1, 1, 56, 56),
                                    (224, 224), mode='bilinear', align_corners=False)[0, 0] > 0
        valid = rgb_support | feature['conditional']['valid']
        score = score * valid
        if not torch.isfinite(score).all() or (score[~valid] != 0).any():
            raise ValueError('Invalid fidelity-gated normal energy')
        return score, {'CONDITIONAL': c, 'RGB_MARGINAL': r}

    def checkpoint(self):
        result = super().checkpoint()
        result['normal_conditional_weight'] = self.conditional_weight.cpu()
        return result
