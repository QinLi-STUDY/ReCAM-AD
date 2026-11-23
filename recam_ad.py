"""Evaluate a published checkpoint, refit from K normals, or prepare EyeCandies."""
import argparse
import json
from pathlib import Path


def main():
    from recam.categories import CLASSES
    from recam.runtime import PROJECT
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare-eye')
    for cmd in [prep, sub.add_parser('fit'), sub.add_parser('eval')]:
        cmd.add_argument('--raw', type=Path, required=True)
        cmd.add_argument('--prepared', type=Path, required=True)
        cmd.add_argument('--eye-normals', type=Path, help='EyeCandies author-profile normal cache')
        if cmd is prep:
            continue
        cmd.add_argument('--dataset',required=True,choices=CLASSES)
        cmd.add_argument('--category',required=True)
        cmd.add_argument('--shot',required=True,type=int,choices=[1,2,4])
        cmd.add_argument('--state-root',type=Path,help='Root above dataset/SHOTshot/category')
    sub.choices['eval'].add_argument('--output',type=Path,default=Path('outputs'))
    sub.choices['eval'].add_argument('--save-maps',action='store_true',help='Save smoothed localization maps and matching query IDs')
    args=p.parse_args()
    eye = args.command=='prepare-eye' or args.dataset=='eyecandies'
    if eye and args.eye_normals is None:
        p.error('EyeCandies requires --eye-normals')
    if args.command=='prepare-eye':
        from recam.data import prepare_eye
        from recam.eye_normals import prepare_four
        from recam.runtime import seed_everything
        seed_everything()
        prepare_eye(args.raw,args.prepared)
        for category in CLASSES['eyecandies']:
            prepare_four(category,args.raw,args.eye_normals)
        return
    if args.category not in CLASSES[args.dataset]:
        p.error('Unknown category for selected dataset')
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from recam.runtime import build
    from recam.paired import ReferencePairs
    from recam.detector import Detector
    from recam.metrics import smooth,normalized,evaluate_normalized
    from recam.state import restore
    group=Path(args.dataset)/f'{args.shot}shot'/args.category
    root=args.state_root or (PROJECT/'weights/states' if args.command=='eval' else Path('fitted'))
    state=root/group
    if args.command=='fit' and state.exists():
        p.error('Fit destination exists; choose a fresh --state-root to preserve the saved calibration.')
    model=build(args.dataset,args.category,args.shot)
    dataset=ReferencePairs(args.dataset,args.category,args.shot,args.raw,args.prepared,args.command=='fit',args.eye_normals)
    if args.command=='fit':
        import os,tempfile,shutil
        from recam.calibrate import fit
        cal=fit(model,dataset)
        state.parent.mkdir(parents=True,exist_ok=True)
        tmp=Path(tempfile.mkdtemp(dir=state.parent))
        try:
            torch.save(model.checkpoint(),tmp/'normal_memory.pt')
            (tmp/'calibration.json').write_text(json.dumps(cal,indent=2))
            (tmp/'support.json').write_text(json.dumps(dataset.identity(),indent=2))
            os.rename(tmp,state)
        finally:
            if tmp.exists(): shutil.rmtree(tmp)
        print(state)
        return
    cal=json.loads((state/'calibration.json').read_text())
    if (cal['dataset'],cal['shot'],cal['category'])!=(args.dataset,args.shot,args.category):
        raise ValueError('Checkpoint and requested evaluation group differ')
    restore(model,torch.load(state/'normal_memory.pt',map_location='cuda',weights_only=True))
    detector=Detector(model,cal)
    ids=[];labels=[];gt=[];scores=[];pro=[];pixels=[];maps=[]
    for image,cloud,label,mask,stem,defect,_ in DataLoader(dataset,batch_size=1,shuffle=False):
        head,dense=detector.predict(image,cloud)
        scores.append(normalized(smooth(head,args.dataset,args.category))[0])
        loc=smooth(dense,args.dataset,args.category)
        _,a,b=normalized(loc);pro.append(a);pixels.append(b)
        if args.save_maps: maps.append(loc.cpu().numpy())
        gt.append(mask[0,0].numpy());labels.append(int(label.item()));ids.append(f'{defect[0]}/{stem[0]}')
    values=evaluate_normalized(np.asarray(scores),np.asarray(pro),np.asarray(pixels),np.asarray(gt),np.asarray(labels))
    out=args.output/group;out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(dict(calibration=cal,test_images=len(ids),metrics=values),indent=2))
    if args.save_maps: np.savez_compressed(out/'maps.npz',ids=ids,image_scores=scores,localization=maps)
    print(json.dumps(values),flush=True)


if __name__=='__main__':main()
