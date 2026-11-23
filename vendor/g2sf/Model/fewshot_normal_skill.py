"""Reject conditional experts worse than a normal-only constant predictor."""
import torch
import torch.nn.functional as F
from Model.fewshot_recam_real import normalize_recam_variant
from Model.fewshot_rgb_visibility import RGBVisibilityMemory


class NormalSkillMemory(RGBVisibilityMemory):
    @torch.inference_mode()
    def fit(self, loader):
        result = super().fit(loader)
        target, sums = [], []
        for data in loader:
            feature = self.extract(data[0], data[1])
            layers = self.conditional.extractor(data[0]['real'].cuda())
            tokens = F.normalize(torch.cat(layers, 1).flatten(2).transpose(1, 2)[0].float(), dim=-1)
            tokens = tokens[feature['conditional']['valid'].flatten()]
            target.append(tokens)
            sums.append(tokens.sum(0))
        errors = []
        for index, tokens in enumerate(target):
            vector = F.normalize(torch.stack([value for j, value in enumerate(sums) if j != index]).sum(0), dim=0)
            errors.append((1 - tokens @ vector).clamp_min(0))
        baseline = torch.cat(errors).quantile(.9).clamp_min(1e-6)
        teacher = self.conditional.scales['real_IMG']
        skill = 1 - teacher / baseline
        old_weight = self.conditional_weight.clone()
        if skill < 0:
            self.conditional_weight = self.conditional_weight.new_tensor(0.)
        result['normal_constant_skill_gate'] = {
            'constant_source_excluded': True, 'teacher_source_excluded': False,
            'teacher_training_bias_disclosed': True, 'target_space': 'same DINOv1 unit224 RGB target',
            'normal_sources': len(target), 'quantile': .9,
            'constant_error_quantile': float(baseline), 'teacher_error_quantile': float(teacher),
            'normal_relative_quantile_skill': float(skill),
            'old_conditional_weight': float(old_weight), 'conditional_weight': float(self.conditional_weight),
            'not_r_squared_or_anomaly_performance_guarantee': True,
            'rejected_using_normal_only': bool(skill < 0)}
        result['changed_method_settings']['fusion'] = 'normal constant-skill rejection + fidelity residual + RGB visibility'
        return result

    @torch.inference_mode()
    def predict(self, feature, variant):
        variant = normalize_recam_variant(variant)
        if variant not in ['fidelity_skill_visibility', 'recam_real', 'recam_whiten', 'recam_private']:
            raise ValueError(variant)
        return super().predict(feature, 'fidelity_rgb_visibility')
