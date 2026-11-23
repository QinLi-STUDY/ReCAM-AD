"""Apply foreground identity on each native feature grid before interpolation."""
import torch
import torch.nn.functional as F

from Model.fewshot_coverage_text_compat import CompatibleCoverageTextMemory


class MaskedCoverageTextMemory(CompatibleCoverageTextMemory):
    def distances(self, feature, exclude_image=None):
        result = {}
        for key, bank in self.banks.items():
            if exclude_image is not None:
                bank = bank[self.origins[key] != exclude_image]
            query = feature[key]
            side = int(query.shape[0] ** .5)
            foreground = F.interpolate(feature['valid'].view(1, 1, 56, 56).float(),
                                       (side, side), mode='nearest').flatten().bool()
            distance = torch.zeros(query.shape[0], device=query.device)
            distance[foreground] = self.nearest(query[foreground], bank, cosine=True)
            result[key] = self.grid56(distance, side)
        return result
