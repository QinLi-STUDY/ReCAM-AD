"""Apply the audited GPU normal estimator to existing prepared geometry."""
import torch
from Model.fewshot_recam_sparse_normals import ReCAMSparseNormalMemory


class ReCAMPreparedMemory(ReCAMSparseNormalMemory):
    @torch.inference_mode()
    def normals(self, cloud):
        self.latest_clean_cloud = cloud.cuda()
        clean = cloud[0].permute(1, 2, 0).cpu().numpy()
        return self.normals_from_clean(clean)

    @torch.inference_mode()
    def fit(self, loader):
        result = super().fit(loader)
        result['changed_method_settings']['normal_preprocessing'] = 'existing read-only G2SF prepared XYZ; Open3DCUDA real normals/author colors retained'
        result['changed_method_settings']['repeat_cpu_preprocessing'] = False
        result['baseline_reproduction'] = 'ReCAM real-stream residual scoring with prepared XYZ'
        return result
