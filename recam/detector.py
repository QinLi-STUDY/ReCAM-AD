"""Use the published normal-only calibration and the original final ReCAM-AD route."""
import torch
from .relative_prior import tempered_joint
from .task_decoupled import TaskDecoupledEvidence, up


def energy(memory, feature, distances, gp, beta):
    valid = feature['valid'].view(56, 56)
    dr = distances['IMG'] / memory.scales['IMG']
    dg = distances['PCD'] / memory.scales['PCD']
    response = (dr * dg).clamp_min(0).sqrt()
    for direction in ('GP_SHAPE','GP_RGB'):
        excess = (gp[direction] / memory.gp_scales[direction] - 1).clamp_min(0)
        response = response + beta * excess * valid
    return response.square()


class Detector:
    def __init__(self, model, calibration):
        self.model = model
        self.cal = calibration
        self.router = TaskDecoupledEvidence(model, {'normal_constant_skill_gate':
            {'old_conditional_weight': calibration['fidelity']}})

    @torch.inference_mode()
    def predict(self, image, cloud):
        feature = self.model.extract(image, cloud)
        e = self.router.evidence(feature)
        memory = self.model.marginal
        m = feature['marginal']
        gp = memory.gp_scores(m, memory.gp_models)
        cal = self.cal
        marginal = up(energy(memory, m, memory.distances(m), gp, cal['gp_coefficient'])) / cal['marginal_scale']
        croot = e['joint'] / e['marginal'].clamp_min(1e-20).sqrt()
        j = croot * marginal.sqrt()
        if cal['dataset'] == 'mvtec':
            a = cal['fidelity']
            head = torch.maximum(a*j, (1-a)*e['rgb_geometry']) * e['support_geometry']
            dense = (a*j + (1-a)*e['rgb_dilation']) * e['support_dilation']
            fallback = e['rgb_geometry'] * e['support_geometry']
        else:
            w = cal['weight']
            prior = tempered_joint(j, marginal, cal['temperature'])
            head = dense = torch.maximum(w*prior, (1-w)*e['rgb_visible']) * e['support_visible']
            fallback = e['rgb_visible'] * e['support_visible']
        return (head if cal['use_joint_image'] else fallback), dense
