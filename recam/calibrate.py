"""Fit NRC using K normal pairs, including the source-excluded marginal energy scale."""
import torch
from torch.utils.data import DataLoader
from .detector import energy
from .relative_prior import temperature


@torch.inference_mode()
def fit(model, support):
    from .model import CalibrationLoader, CONTEXT, finalize_fit
    from .data import SpatialCalibrationData
    units = SpatialCalibrationData(support) if support.shot == 1 else support
    loader = CalibrationLoader(DataLoader(units, batch_size=1, shuffle=False))
    fit = finalize_fit(model, model.fit(loader), support.dataset, support.shot)
    memory = model.marginal
    sizes = [len(idx) for idx in fit['marginal_fit']['gp_anchor_indices_by_source']]
    rgb = memory.gp_models['GP_SHAPE'].source
    geom = memory.gp_models['GP_RGB'].source
    assert sum(sizes) == len(rgb) == len(geom)
    anchors = [dict(IMG=x, PCD=y) for x,y in zip(rgb.split(sizes),geom.split(sizes))]
    measurements = []
    for index, (image, cloud, label, *_) in enumerate(loader):
        if bool(label.item()):
            raise ValueError('NRC requires normal support pairs')
        m = memory.extract(image['fine'],cloud['native'])
        gp = memory.gp_scores(m, memory._models(anchors, exclude_image=index))
        values = energy(memory,m,memory.distances(m,exclude_image=index),gp,.2)
        measurements.append(values[m['valid'].view(56,56)])
    CONTEXT.regions.clear()
    memory._spatial_cache.clear()
    stats = fit['normal_constant_skill_gate']
    a, w = stats['old_conditional_weight'], stats['conditional_weight']
    skill = stats['normal_relative_quantile_skill']
    q = temperature(a,skill)
    return dict(dataset=support.dataset,category=support.category,shot=support.shot,
        fidelity=a,weight=w,skill=skill,temperature=q,use_joint_image=support.shot>1 or q>=.5,
        marginal_scale=float(torch.cat(measurements).quantile(.9).clamp_min(1e-8)),gp_coefficient=.2)
