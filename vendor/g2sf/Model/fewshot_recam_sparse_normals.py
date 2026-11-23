"""Remove zero points which are outside every valid point's normal radius."""
import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
from Model.fewshot_recam_cleanmask import ReCAMCleanMemory


class ReCAMSparseNormalMemory(ReCAMCleanMemory):
    @torch.inference_mode()
    def normals_from_clean(self, clean):
        height, width = clean.shape[:2]
        points = clean.reshape(-1, 3).astype(np.float32)
        valid = ~np.all(np.isclose(points, 0), axis=1)
        selected = points[valid]
        if not len(selected) or np.any(np.linalg.norm(selected, axis=-1) <= .01):
            raise ValueError('Zero-background points might enter the foreground normal radius')
        device = o3c.Device('CUDA:0')
        pc = o3d.t.geometry.PointCloud(device)
        pc.point.positions = o3c.Tensor(selected, device=device)
        pc.estimate_normals(max_nn=30, radius=.01)
        pc.orient_normals_towards_camera_location(o3c.Tensor([0., 0., 0.], dtype=o3c.float32, device=device))
        normals = np.zeros_like(points); normals[:, 2] = 1.
        normals[valid] = -pc.point.normals.cpu().numpy()
        colors = (np.clip((normals.reshape(height, width, 3) + 1) / 2, 0, 1) * 255).astype(np.uint8)
        if height != width:
            size = max(height, width)
            top, left = (size - height) // 2, (size - width) // 2
            colors = cv2.copyMakeBorder(colors, top, size - height - top, left, size - width - left,
                                       cv2.BORDER_CONSTANT, value=(127, 127, 255))
        colors = cv2.resize(colors, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(colors.copy()).permute(2, 0, 1)[None].cuda().float() / 255
        return (image - self.mean) / self.std

    @torch.inference_mode()
    def normals(self, cloud):
        array = np.nan_to_num(cloud[0].permute(1, 2, 0).cpu().numpy(), nan=0.)
        clean = self.cleaning.remove_plane_cpu(array, .002 if self.category == 'dowel' else .005)
        clean = self.cleaning.connected_components_cleaning_cpu(clean)
        return self.normals_from_clean(clean)

    @torch.inference_mode()
    def fit(self, loader):
        result = super().fit(loader)
        result['changed_method_settings']['normal_estimator_zero_points_omitted'] = True
        result['changed_method_settings']['omission_radius_separation_checked_each_input'] = True
        return result
