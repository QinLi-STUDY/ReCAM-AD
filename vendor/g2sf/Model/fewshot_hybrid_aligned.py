"""Align NORMAL-target residuals to the RGB-target physical pixel frame."""
import torch
import torch.nn.functional as F
from Model.fewshot_recam_prepared import ReCAMPreparedMemory
from Model.fewshot_hybrid_density import HybridDensityMemory


class AlignedConditionalMemory(ReCAMPreparedMemory):
    @torch.inference_mode()
    def extract_aligned(self, image, prepared, native):
        feature = super().extract(image, prepared)
        height, width = native.shape[-2:]
        ph, pw = prepared.shape[-2:]
        if ph != pw or ph < height or pw < width:
            raise ValueError('Unsupported prepared-to-native geometry')
        if height == ph and width == pw:
            return feature  # Exactly the old computation, no interpolation.
        top, left = (ph - height) // 2, (pw - width) // 2
        yy, xx = torch.meshgrid(torch.arange(224, device='cuda').float(), torch.arange(224, device='cuda').float(), indexing='ij')
        grid = torch.stack([2 * (left + (xx + .5) * width / 224) / pw - 1,
                            2 * (top + (yy + .5) * height / 224) / ph - 1], -1)[None]

        def warp(value):
            return F.grid_sample(value, grid, mode='bilinear', padding_mode='border', align_corners=False)

        xyz = F.interpolate(native.cuda(), (224, 224), mode='bilinear', align_corners=False)
        mask = torch.all(xyz != 0, dim=1, keepdim=True).float()
        valid = F.adaptive_avg_pool2d(F.avg_pool2d(mask, 3, stride=1), (224, 224))[0, 0] > 0
        # Recover unmasked residuals: the old padded mask must not erase RGB pixels.
        raw_rgb = .5 * feature['residual']['IMG'].square().sum(-1).view(224, 224)
        raw_normal = .5 * feature['residual']['PCD'].square().sum(-1).view(224, 224)
        feature['raw'] = {'IMG': raw_rgb * valid, 'PCD': warp(raw_normal[None, None])[0, 0] * valid}
        feature['residual']['PCD'] = warp(feature['residual']['PCD'].T.view(1, 2304, 224, 224))[0].flatten(1).T
        private = F.interpolate(feature['private']['PCD'].T.view(1, 2304, 28, 28), (224, 224), mode='bilinear', align_corners=False)
        feature['private']['PCD'] = F.normalize(F.interpolate(warp(private), (28, 28), mode='bilinear', align_corners=False)[0].flatten(1).T, dim=-1)
        feature['valid'] = valid
        return feature


class AlignedViewLoader:
    """Pass native layout solely as calibration-frame metadata to conditional fit."""
    def __init__(self, loader): self.loader = loader

    def __iter__(self):
        for value in self.loader:
            yield value[0]['real'], value[1], *value[2:]


class AlignedFitConditional(AlignedConditionalMemory):
    @torch.inference_mode()
    def extract(self, image, cloud):
        if isinstance(cloud, dict):
            return self.extract_aligned(image, cloud['prepared'], cloud['native'])
        return super().extract(image, cloud)


class AlignedHybridMemory(HybridDensityMemory):
    def __init__(self, category):
        super().__init__(category)
        # Keep already-loaded author modules/stat containers; change only dispatch.
        self.conditional.__class__ = AlignedFitConditional

    @torch.inference_mode()
    def extract(self, image, cloud):
        return {'conditional': self.conditional.extract(image['real'], cloud),
                'marginal': self.marginal.extract(image['fine'], cloud['native'])}

    @torch.inference_mode()
    def fit(self, loader):
        # Hybrid parent constructs ViewLoader(real,prepared); intercept conditional
        # fit with the original loader so the native shape is available every time.
        original_fit = self.conditional.fit
        self.conditional.fit = lambda ignored_loader: original_fit(AlignedViewLoader(loader))
        try:
            result = super().fit(loader)
        finally:
            self.conditional.fit = original_fit
        result['changed_method_settings']['conditional_normal_error_frame'] = 'explicit padded-to-native pixel warp; RGB error unchanged'
        result['changed_method_settings']['foreground_mask_frame'] = 'native RGB-aligned prepared XYZ'
        result['changed_method_settings']['square_identity'] = 'H==W==preparedS returns exact old feature'
        return result

    @torch.inference_mode()
    def predict(self, feature, variant):
        aliases = {'hybrid_aligned_product': 'hybrid_product', 'hybrid_aligned_sum': 'hybrid_sum'}
        return super().predict(feature, aliases.get(variant, variant))
