import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .runtime import PROJECT, sha

PROFILE = 'author_mapped_xyz_radius01_origin'


def author_module(filename, name):
    path = PROJECT / 'vendor/g2sf/code/external/CMDS-AD/processing' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_converter():
    """Execute unchanged author's converter, replacing only its disk image I/O."""
    import open3d.core as o3c
    import cv2
    conversion = author_module('pc2sn_o3d_gpu_mvtec3dad.py', '_eye_generic_readonly')
    context = {}
    original_plane = conversion.get_plane_eq

    def plane(points, ransac_n_pts=50):
        if len(points) < ransac_n_pts:
            context['no_plane'] = True
            return None  # Author returns (None,None), which is not a plane.
        return original_plane(points, ransac_n_pts)

    def read(_path):
        return context['xyz'].copy()

    def write(_path, bgr):
        context['normal224'] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return True

    class CVProxy:
        imwrite = staticmethod(write)
        def __getattr__(self, name):
            return getattr(cv2, name)

    conversion.get_plane_eq = plane
    conversion.tifffile = SimpleNamespace(imread=read)
    conversion.cv2 = CVProxy()

    def convert(xyz):
        context.clear(); context['xyz'] = xyz
        conversion.convert_mvtec_tiff_to_normal_o3d_gpu(
            'eyecandies_numeric_xyz.tiff', 'captured_not_written.png',
            radius=.01, max_nn=30, target_size=(224,224), device=o3c.Device('CUDA:0'))
        if 'normal224' not in context:
            raise RuntimeError('Author numeric converter failed')
        return context['normal224'], bool(context.get('no_plane'))

    return convert, sha(conversion.__file__)


def original_mapped_xyz(directory, stem):
    """Vectorized exact preprocess_eyecandies projection/background sentinel."""
    import cv2
    import yaml
    with (directory / f'{stem}_info_depth.yaml').open() as stream:
        limits = yaml.safe_load(stream)['normalization']
    depth = cv2.imread(str(directory / f'{stem}_depth.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = depth / 65535. * (limits['max']-limits['min']) + limits['min']
    h,w = depth.shape
    pose = np.loadtxt(directory / f'{stem}_pose.txt')
    intrinsic = np.array([[711.11,0,w/2,0],[0,711.11,h/2,0],[0,0,1,0],[0,0,0,1]])
    yy,xx = np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    # Author fills a float64 camera_vectors array using scalar 1/depth. Keep
    # the reciprocal in float64 rather than introducing a float32 vector roundoff.
    vectors = np.stack([xx.ravel(),yy.ravel(),np.ones(h*w),1/depth.ravel().astype(np.float64)])
    raw = (depth.reshape(-1,1)*(np.linalg.inv(intrinsic @ pose) @ vectors).T)[:,:3]
    dz,dy = raw[256,1]-raw[-256,1], raw[256,2]-raw[-256,2]
    norm = np.hypot(dz,dy)
    start = np.array([0,raw[-256,1],raw[-256,2]])
    c,s = dy/norm,dz/norm
    rotation = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    rotated = (rotation @ (raw-start).T).T
    background = (rotated[:,1] > -.02) | (rotated[:,2] > 1.8) | (rotated[:,0] > 1) | (rotated[:,0] < -1)
    rotated[background] = -start
    mapped = (rotation.T @ rotated.T).T + start
    return (mapped[:,[0,2,1]] * [.1,-.1,.1]).reshape(h,w,3)


def prepare_four(category,raw_root,cache_root):
    from .runtime import seed_everything
    seed_everything()
    convert,source_sha=make_converter()
    preprocessing_sha=sha(PROJECT/'vendor/g2sf/code/external/CMDS-AD/processing/preprocess_eyecandies.py')
    records=[]
    for split in ('train','test_public'):
        directory=raw_root/category/split/'data'
        paths=sorted(directory.glob('*_image_4.png'))
        if split=='train':paths=paths[:4]
        if len(paths)!=(4 if split=='train' else 50):raise RuntimeError('Wrong data population')
        for rgb in paths:
            stem=rgb.name.removesuffix('_image_4.png')
            identity={suffix:sha(directory/f'{stem}_{suffix}') for suffix in ('image_4.png','depth.png','info_depth.yaml','pose.txt')}
            identity.update(profile=PROFILE,converter_sha=source_sha,preprocessing_sha=preprocessing_sha)
            path=cache_root/category/split/f'{stem}.npz'
            reused=path.exists()
            if not reused:
                normal,no_plane=convert(original_mapped_xyz(directory,stem))
                if normal.shape!=(224,224,3) or normal.dtype!=np.uint8:raise RuntimeError('Wrong numeric normal')
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,normal224=normal,identity=np.array(json.dumps(identity,sort_keys=True)),no_edge_plane=np.array(no_plane))
                tmp.replace(path)
            with np.load(path) as z:
                if json.loads(str(z['identity']))!=identity:raise RuntimeError('Cache input changed')
                no_plane=bool(z['no_edge_plane'])
            records.append(dict(split=split,stem=stem,sha256=sha(path),reused=reused,no_edge_plane=no_plane,input_identity=identity))
    (cache_root/category/'manifest_4shot.json').write_text(json.dumps(dict(profile=PROFILE,records=records,save_figure=False),indent=2))


