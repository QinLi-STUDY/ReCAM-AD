"""Normal supports and raw test identities; no test statistic enters fitting."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .runtime import PROJECT, sha
from .categories import CLASSES

MEAN, STD = [.485, .456, .406], [.229, .224, .225]


def eye_conversion_module():
    path = PROJECT / 'vendor/g2sf/code/external/CMDS-AD/processing/pc2sn_o3d_gpu_eyecandies.py'
    spec = importlib.util.spec_from_file_location('_eye_author_conversion', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_eye(raw_root, prepared_root):
    """Vectorize author's projection; cache numeric XYZ and physical normals."""
    import open3d as o3d
    import open3d.core as o3c
    import yaml
    author = eye_conversion_module()
    prepared_root.mkdir(parents=True, exist_ok=True)
    device = o3c.Device('CUDA:0')
    records = []
    for category in CLASSES['eyecandies']:
        for split in ('train', 'test_public'):
            directory = raw_root / category / split / 'data'
            inputs = sorted(directory.glob('*_depth.png'))
            if split == 'train':
                inputs = inputs[:4]
            elif len(inputs) != 50:
                raise RuntimeError(f'EyeCandies expected 50 public tests: {category}: {len(inputs)}')
            for depth_path in inputs:
                stem = depth_path.name.removesuffix('_depth.png')
                rgb = directory / f'{stem}_image_4.png'
                info = directory / f'{stem}_info_depth.yaml'
                pose_file = directory / f'{stem}_pose.txt'
                cache = prepared_root / category / split / f'{stem}.npz'
                identity = {'rgb': sha(rgb), 'depth': sha(depth_path), 'info': sha(info), 'pose': sha(pose_file)}
                record = {'category': category, 'split': split, 'id': stem, 'input_sha256': identity,
                          'cache': str(cache.relative_to(prepared_root))}
                if not cache.is_file():
                    with info.open() as stream:
                        normalization = yaml.safe_load(stream)['normalization']
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
                    depth = depth / 65535. * (normalization['max'] - normalization['min']) + normalization['min']
                    h, w = depth.shape
                    if (h, w) != (512, 512):
                        raise RuntimeError('Author geometry assumes 512 square grid')
                    pose = np.loadtxt(pose_file)
                    intrinsic = np.array([[711.11, 0, w / 2, 0], [0, 711.11, h / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                    vectors = np.stack([xx.ravel(), yy.ravel(), np.ones(h*w), 1 / depth.ravel()], 0)
                    raw = (depth.reshape(-1, 1) * (np.linalg.inv(intrinsic @ pose) @ vectors).T)[:, :3]
                    mapped, physical = author.process_point_cloud_split_paths(raw)
                    bg = np.all(physical == 0, axis=1)
                    # The author's split-path routine maps zero rotated background to
                    # a nonzero translation. Explicitly preserve background identity.
                    mapped[bg] = 0
                    physical = author.connected_components_cleaning_cpu(physical.reshape(h, w, 3))
                    valid = ~np.all(np.isclose(physical.reshape(-1, 3), 0), axis=1)
                    mapped[~valid] = 0
                    if valid.sum() < 128:
                        raise RuntimeError(f'EyeCandies foreground too small: {rgb}')
                    camera_pose = np.eye(4)
                    camera_pose[:pose.shape[0], :pose.shape[1]] = pose
                    camera = np.linalg.inv(camera_pose)[:3, 3].astype(np.float32)
                    pc = o3d.t.geometry.PointCloud(device)
                    pc.point.positions = o3c.Tensor(physical.reshape(-1, 3)[valid].astype(np.float32), device=device)
                    pc.estimate_normals(max_nn=30, radius=.04)
                    pc.orient_normals_towards_camera_location(o3c.Tensor(camera, device=device))
                    fg = -pc.point.normals.cpu().numpy()[:, [0, 2, 1]]
                    fg[:, 1] *= -1
                    normals = np.zeros((h*w, 3), dtype=np.float32)
                    normals[:, 2] = 1
                    normals[valid] = fg
                    colors = (np.clip((normals.reshape(h, w, 3) + 1) / 2, 0, 1) * 255).astype(np.uint8)
                    colors = cv2.resize(colors, (224, 224), interpolation=cv2.INTER_CUBIC)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary, xyz=mapped.reshape(h, w, 3).astype(np.float32), normal224=colors,
                                        input_identity=np.array(json.dumps(identity, sort_keys=True)))
                    temporary.replace(cache)
                else:
                    with np.load(cache) as data:
                        if json.loads(str(data['input_identity'])) != identity:
                            raise RuntimeError(f'Prepared cache inputs changed: {cache}')
                record['cache_sha256'] = sha(cache)
                records.append(record)
        print(f'DATA_PREPARED eyecandies {category} train_supports=4 test_public=50', flush=True)
    manifest = {'dataset': 'eyecandies', 'rgb_view': 4, 'train_selection': 'sorted first4 maximum',
                'test_split': 'test_public', 'test_images': 500, 'source_module_sha256': sha(Path(author.__file__)),
                'original_data_modified': False, 'normal_radius_physical': .04, 'normal_max_nn': 30,
                'background_translation_corrected': True, 'save_figure': False, 'records': records}
    (prepared_root / 'manifest.json').write_text(json.dumps(manifest, indent=2))


class PairedData(Dataset):
    def __init__(self, dataset, category, shot, raw_root, prepared_root, train):
        self.dataset, self.category, self.shot, self.train = dataset, category, shot, train
        self.raw_root, self.prepared_root = Path(raw_root), Path(prepared_root)
        if dataset == 'mvtec':
            from Dataset.mvtec3d import MVTecDataset
            args = SimpleNamespace(dataset_path=str(raw_root), class_name=category, img_size=224)
            original = MVTecDataset(args, is_train=train)
            self.image, self.pcd, self.label, self.mask, self.defect_type = (list(getattr(original, k)) for k in ('image', 'pcd', 'label', 'mask', 'defect_type'))
            if train:
                for key in ('image', 'pcd', 'label', 'mask', 'defect_type'):
                    setattr(self, key, getattr(self, key)[:shot])
            self.pcd = [str(self.prepared_root / Path(p).relative_to(self.raw_root)) for p in self.pcd]
        else:
            split = 'train' if train else 'test_public'
            folder = self.raw_root / category / split / 'data'
            paths = sorted(folder.glob('*_image_4.png'))
            self.image = [str(x) for x in (paths[:shot] if train else paths)]
            self.pcd = [str(self.prepared_root / category / split / (Path(x).name.removesuffix('_image_4.png') + '.npz')) for x in self.image]
            self.mask = [None if train else str(Path(x).with_name(Path(x).name.replace('_image_4.png', '_mask.png'))) for x in self.image]
            self.label = [0 if train else int(np.any(np.asarray(Image.open(x)))) for x in self.mask]
            self.defect_type = ['good' if label == 0 else 'bad' for label in self.label]
            if not train and len(self.image) != 50:
                raise RuntimeError('Missing public test images')
        if train and (len(self.image) != shot or any(self.label)):
            raise RuntimeError('Wrong normal support count')
        if not all(Path(x).is_file() for x in self.pcd):
            raise RuntimeError('Prepare geometry before running experiments')
        self.real = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
        self.fine = T.Compose([T.Resize((784, 784), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(MEAN, STD)])
        self.gt_transform = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        with Image.open(self.image[index]) as file:
            rgb = file.convert('RGB')
        image = {'real': self.real(rgb), 'fine': self.fine(rgb)}
        if self.dataset == 'mvtec':
            import tifffile
            from Dataset.fewshot_raw_aligned import invert_padding
            prepared = np.asarray(tifffile.imread(self.pcd[index]), dtype=np.float32)
            native = invert_padding(prepared, rgb.height, rgb.width)
        else:
            with np.load(self.pcd[index]) as data:
                prepared = data['xyz'].copy()
                image['author_normal'] = torch.from_numpy(data['normal224'].copy()).permute(2, 0, 1).float() / 255
            native = prepared
        cloud = {'prepared': torch.from_numpy(np.array(prepared, copy=True)).permute(2, 0, 1),
                 'native': torch.from_numpy(np.array(native, copy=True)).permute(2, 0, 1)}
        if self.dataset == 'eyecandies':
            cloud['author_normal'] = image['author_normal']
        mask = torch.zeros(1, 224, 224)
        if not self.train and self.label[index]:
            with Image.open(self.mask[index]) as file:
                if file.size != rgb.size:
                    raise RuntimeError('RGB / annotation coordinates differ')
                mask = (self.gt_transform(file.convert('L')) > .5).float()
        tail = (self.label[index], mask, torch.zeros(1, 224, 224)) if self.train else (
            self.label[index], mask, Path(self.image[index]).stem, self.defect_type[index], torch.zeros(1, 224, 224))
        return image, cloud, *tail

    def identity(self):
        return {'dataset': self.dataset, 'category': self.category, 'shot': self.shot,
                'selection': f'native sorted first{self.shot}', 'original_source_images': len(self),
                'samples': [{'rgb': str(Path(rgb).relative_to(self.raw_root)), 'rgb_sha': sha(rgb), 'xyz_sha': sha(xyz)}
                            for rgb, xyz in zip(self.image, self.pcd)]}


class SpatialCalibrationData(Dataset):
    """Four spatial folds of ONE source, never counted as four-shot."""
    def __init__(self, original):
        assert original.train and len(original) == 1
        self.original = original

    def __len__(self):
        return 4

    def __getitem__(self, fold):
        image, cloud, *tail = self.original[0]
        image['calibration_fold'] = torch.tensor(fold)
        return image, cloud, *tail
