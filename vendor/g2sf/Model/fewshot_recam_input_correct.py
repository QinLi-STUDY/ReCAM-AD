"""Restore the author's [0,1] normal-input contract; not a novel mechanism."""
import torch
from Model.fewshot_recam_real import normalize_recam_variant
from Model.fewshot_hybrid_aligned import AlignedFitConditional, AlignedHybridMemory


class InputCorrectConditional(AlignedFitConditional):
    @torch.inference_mode()
    def normals(self, cloud):
        normalized = super().normals(cloud)
        colors = normalized * self.std + self.mean
        if colors.min() < -1e-6 or colors.max() > 1 + 1e-6:
            raise ValueError('Author normal inputs must lie within [0,1]')
        return colors.clamp(0, 1)


class InputCorrectHybrid(AlignedHybridMemory):
    def __init__(self, category):
        super().__init__(category)
        self.conditional.__class__ = InputCorrectConditional

    @torch.inference_mode()
    def fit(self, loader):
        result = super().fit(loader)
        result['changed_method_settings']['author_normal_input'] = 'Resize + ToTensor [0,1]; NO ImageNet normalization'
        result['changed_method_settings']['prior_normal_input_error_corrected'] = True
        return result

    @torch.inference_mode()
    def predict(self, feature, variant):
        variant = normalize_recam_variant(variant)
        if variant == 'correct_real':
            return self.conditional.predict(feature['conditional'], 'recam_real')
        aliases = {'correct_hybrid': 'hybrid_product', 'recam_real': 'hybrid_product',
                   'recam_whiten': 'hybrid_product', 'recam_private': 'hybrid_product'}
        return super().predict(feature, aliases.get(variant, variant))
