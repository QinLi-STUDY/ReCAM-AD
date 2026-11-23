"""Verify all release blobs and the dataset/category/shot assignments."""
import hashlib
import json
from pathlib import Path
from recam.categories import CLASSES


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'weights/SHA256.json').read_text())
    paths={str(p.relative_to(root)) for p in (root/'weights/backbones').iterdir() if p.is_file()}
    if len(paths)!=3:
        raise ValueError('Expected the three initialization checkpoints')
    for ds,cats in CLASSES.items():
        for k in (1,2,4):
            for cat in cats:
                group=Path(ds)/f'{k}shot'/cat
                state=Path('weights/states')/group
                cal=json.loads((root/state/'calibration.json').read_text())
                support=json.loads((root/state/'support.json').read_text())
                if any((d['dataset'],d['shot'],d['category'])!=(ds,k,cat) for d in (cal,support)) or len(support['samples'])!=k:
                    raise ValueError(f'Calibration or support mismatch in {group}')
                paths.add(str(state/'normal_memory.pt'))
                maps=json.loads((root/'weights/mappers'/group/'manifest.json').read_text())
                if set(maps)!={'2d_to_3d','3d_to_2d'}:
                    raise ValueError(f'Mapper pair incomplete in {group}')
                for direction,m in maps.items():
                    name=str(Path('weights/mappers')/group/(direction+'.pth'))
                    if name!=m['file'] or manifest[name]['sha256']!=m['sha256']:
                        raise ValueError(f'Mapper identity mismatch: {name}')
                    paths.add(name)
    if paths!=manifest.keys():
        raise ValueError('Release manifest differs from the required model inventory')
    for name,info in manifest.items():
        f=root/name
        if not f.is_file() or f.stat().st_size!=info['bytes']:
            raise SystemExit(f'{name} is missing or has the wrong size. If using a Git clone, run git lfs pull.')
        with f.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=info['sha256']:
                raise SystemExit(f'SHA-256 mismatch: {name}')
    print(f'Verified {len(paths)} binary weights and all 60 calibration/support groups.')


if __name__=='__main__':main()
