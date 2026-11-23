"""Read-only local DINOv2 API compatibility, identical registered layer choices."""
import torch
import torch.nn.functional as F

from Model.fewshot_coverage_text import CoverageTextMemory


class CompatibleCoverageTextMemory(CoverageTextMemory):
    @torch.inference_mode()
    def rgb_features(self, image):
        image = F.interpolate(image.to(self.args.device), (448, 448), mode='bicubic', align_corners=False)
        tokens = self.rgb.prepare_tokens_with_masks(image)
        selected = []
        for index, block in enumerate(self.rgb.blocks):
            tokens = block(tokens)
            if index in [3, 7, 11]:
                patches = self.rgb.norm(tokens)[:, 1 + self.rgb.num_register_tokens:]
                selected.append(F.normalize(patches[0].float(), dim=-1))
        if len(selected) != 3 or any(v.shape != (1024, 768) for v in selected):
            raise ValueError('DINOv2 layer/token shape mismatch')
        return F.normalize(torch.cat(selected, dim=-1), dim=-1)
