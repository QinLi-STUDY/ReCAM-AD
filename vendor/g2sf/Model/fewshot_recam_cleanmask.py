"""Foreground derives from cleaned geometry, not the raw background plane."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from Model.fewshot_recam_real import ReCAMRealMemory, square_pad


class ReCAMCleanMemory(ReCAMRealMemory):
    def __init__(self, category):
        super().__init__(category)
        self._component_cleaning = self.cleaning.connected_components_cleaning_cpu
        self.cleaning = SimpleNamespace(remove_plane_cpu=self.cleaning.remove_plane_cpu,
                                       connected_components_cleaning_cpu=self.capture_clean)

    def capture_clean(self, array):
        clean = self._component_cleaning(array)
        self.latest_clean_cloud = torch.from_numpy(clean.copy()).permute(2, 0, 1)[None].cuda()
        return clean

    @torch.inference_mode()
    def extract(self, image, cloud):
        feature = super().extract(image, cloud)
        xyz = F.interpolate(square_pad(self.latest_clean_cloud), (224, 224), mode='bilinear', align_corners=False)
        support = torch.all(xyz != 0, dim=1, keepdim=True).float()
        feature['valid'] = F.adaptive_avg_pool2d(F.avg_pool2d(support, 3, stride=1), (224, 224))[0, 0] > 0
        if not feature['valid'].any():
            raise ValueError('No cleaned foreground support')
        feature['raw'] = {key: value * feature['valid'] for key, value in feature['raw'].items()}
        return feature

    @torch.inference_mode()
    def fit(self, loader):
        result = super().fit(loader)
        result['changed_method_settings']['mask'] = 'analytical square-padded CLEANED-XYZ Point support approximation'
        return result
