"""Original six-metric convention; vectorized exact official PRO threshold logic."""
from pathlib import Path
import importlib.util
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score

SOURCE=Path(__file__).resolve().parents[1]
KEYS=('auc','seg_auc','aupro','aupro_0p1','aupro_0p05','aupro_0p01')


def official():
    p=SOURCE/'vendor/g2sf/code/external/CMDS-AD/utils/metrics_utils.py'
    spec=importlib.util.spec_from_file_location('_frozen_completion_official_pro',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fast_pro(gt,prediction):
    """Same 8-connectivity, order statistics, tie rule and 99 thresholds.

    Only the Python per-positive-pixel while loop is replaced by searchsorted.
    FPR intentionally follows original order-statistic positions, including ties.
    """
    background=[];components=[]
    for mask,score in zip(gt,prediction):
        regions,n=label(mask,structure=np.ones((3,3),dtype=int))
        background.append(score[regions==0].astype(np.float64))
        for k in range(1,n+1):components.append(np.sort(score[regions==k].copy()))
    ok=np.sort(np.concatenate(background))
    positions=np.linspace(0,len(ok)-1,num=99,dtype=int)
    thresholds=ok[positions]
    overlaps=np.zeros(99,dtype=np.float64)
    for component in components:
        overlaps += 1.0-np.searchsorted(component,thresholds,side='right')/len(component)
    overlaps/=len(components)
    fpr=np.r_[1.,1.-(positions+1)/len(ok)][::-1]
    pro=np.r_[1.,overlaps][::-1]
    module=official()
    areas=[float(module.trapezoid(fpr,pro,x_max=q)/q) for q in (.3,.1,.05,.01)]
    return areas,(fpr,pro)


def smooth(raw,dataset,category):
    import torch.nn.functional as F
    small=category in {'bagel','cable_gland','dowel','foam','tire'} if dataset=='mvtec' else category!='HazelnutTruffle'
    value=raw.view(1,1,224,224)
    for k,n in ((3 if small else 5,5),(5 if small else 7,3)):
        kernel=value.new_ones(1,1,k,k)/(k*k)
        for _ in range(n):value=F.conv2d(value,kernel,padding=k//2)
    return value[0,0]


def normalized(processed):
    import torch
    pos=processed[processed!=0]
    mean=pos.mean().clamp_min(1e-12) if len(pos) else processed.new_tensor(1.)
    score=float((processed/mean.sqrt()).max())
    return score,(processed/mean).cpu().numpy(),(processed/processed.mean().clamp_min(1e-12).sqrt()).cpu().numpy()


def evaluate_raw(image_raw,dense_raw,gt,labels,dataset,category):
    import torch
    scores=[];pro=[];pixels=[]
    with torch.inference_mode():
        for head,dense in zip(image_raw,dense_raw):
            hp=smooth(torch.as_tensor(head,device='cuda'),dataset,category)
            dp=hp if np.array_equal(head,dense) else smooth(torch.as_tensor(dense,device='cuda'),dataset,category)
            scores.append(normalized(hp)[0]);_,p,x=normalized(dp);pro.append(p);pixels.append(x)
    values=evaluate_normalized(np.asarray(scores),np.asarray(pro),np.asarray(pixels),gt,labels)
    return values,np.asarray(scores)


def evaluate_normalized(scores,pro,pixels,gt,labels):
    areas,_=fast_pro(gt,pro)
    return dict(auc=float(roc_auc_score(labels,scores)),seg_auc=float(roc_auc_score(gt.ravel(),pixels.ravel())),
                aupro=areas[0],aupro_0p1=areas[1],aupro_0p05=areas[2],aupro_0p01=areas[3])


