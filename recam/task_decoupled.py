"""Task-decoupled response routing; original model and states stay immutable.

All routing constants are fixed across datasets/categories/shots. No labels,
ground-truth masks or evaluation statistics are accepted by this module.
"""
import torch
import torch.nn.functional as F


def up(value):
    return F.interpolate(value.view(1, 1, 56, 56), (224, 224),
                         mode='bilinear', align_corners=False)[0, 0]


class TaskDecoupledEvidence:
    VARIANTS = ('full_tdf', 'no_skill', 'no_visibility', 'mean_image',
                'shared_skill', 'shared_visibility', 'original_shared',
                'original_max', 'no_conditional_fixed_gate', 'fixed_weight',
                'simple_dilation', 'no_gp')

    def __init__(self, model, fit, no_gp_scale=None):
        self.model = model
        self.weight = model.conditional_weight
        self.fidelity_weight = self.weight.new_tensor(
            fit['normal_constant_skill_gate']['old_conditional_weight'])
        self.no_gp_scale = no_gp_scale

    @torch.inference_mode()
    def evidence(self, feature):
        from Model.fewshot_private_text import PrivateTextMemory
        model, m = self.model, feature['marginal']
        valid = m['valid'].view(56, 56)
        distances = model.marginal.distances(m)
        rgb = distances['IMG'] / model.marginal.scales['IMG']
        geom = distances['PCD'] / model.marginal.scales['PCD']
        base = (rgb * geom).clamp_min(0).sqrt()
        gp = model.marginal.gp_scores(m, model.marginal.gp_models)
        native = base
        for key in ('GP_SHAPE', 'GP_RGB'):
            native = native + .1 * (gp[key] / model.marginal.gp_scales[key] - 1).clamp_min(0) * valid
        conditional = model.conditional.predict(feature['conditional'], 'recam_real')[0] / model.scales['conditional']
        marginal = up(native.square()) / model.scales['marginal']
        joint = (conditional * marginal).clamp_min(0).sqrt()
        r_geom = up(rgb).clamp_min(0)
        support_geom = (up(valid.float()) > 0) | feature['conditional']['valid']
        all_dist = PrivateTextMemory.nearest(m['IMG'], model.marginal.banks['IMG'], cosine=True).view(56, 56)
        neighborhood = F.max_pool2d(valid.float()[None, None], 3, 1, 1)[0, 0] > 0
        if model.visibility_enabled:
            bg_dist = PrivateTextMemory.nearest(m['IMG'], model.background_bank, cosine=True).view(56, 56)
            confidence = (bg_dist / (all_dist + bg_dist).clamp_min(1e-8)).clamp(0, 1)
            visible = valid | (neighborhood & (confidence > model.visibility_threshold))
            r_visible = up(all_dist / model.marginal.scales['IMG'] * visible)
            support_visible = (up(visible.float()) > 0) | feature['conditional']['valid']
        else:
            r_visible, support_visible = r_geom, support_geom
        dilation = up(all_dist / model.marginal.scales['IMG'] * neighborhood)
        dilation_support = (up(neighborhood.float()) > 0) | feature['conditional']['valid']
        result = dict(joint=joint, marginal=marginal, rgb_geometry=r_geom,
                      rgb_visible=r_visible, support_geometry=support_geom,
                      support_visible=support_visible, rgb_dilation=dilation,
                      support_dilation=dilation_support)
        if self.no_gp_scale is not None:
            no_gp_m = up(base.square()) / self.no_gp_scale
            result['joint_no_gp'] = (conditional * no_gp_m).clamp_min(0).sqrt()
        return result

    def route(self, evidence):
        e = evidence
        j, r, v = e['joint'], e['rgb_geometry'], e['rgb_visible']
        sg, sv = e['support_geometry'], e['support_visible']
        w, a = self.weight, self.fidelity_weight
        image = torch.maximum(w*j, (1-w)*r) * sg
        dense = (a*j + (1-a)*v) * sv
        original = (w*j + (1-w)*v) * sv
        original_max = torch.maximum(w*j, (1-w)*v) * sv
        pairs = {
            'full_tdf': (image, dense),
            'no_skill': (torch.maximum(a*j, (1-a)*r)*sg, dense),
            'no_visibility': (image, (a*j+(1-a)*r)*sg),
            'mean_image': ((w*j+(1-w)*r)*sg, dense),
            'shared_skill': (image, original),
            'shared_visibility': (original_max, dense),
            'original_shared': (original, original),
            'original_max': (original_max, original_max),
            # Neutralize conditional ENERGY while holding the original gate fixed.
            # This is not advertised as a completely mapper-free model.
            'no_conditional_fixed_gate': (
                torch.maximum(w*e['marginal'].clamp_min(0).sqrt(), (1-w)*r)*sg,
                (a*e['marginal'].clamp_min(0).sqrt()+(1-a)*v)*sv),
            'fixed_weight': (torch.maximum(.5*j, .5*r)*sg, (.5*j+.5*v)*sv),
            'simple_dilation': (image, (a*j+(1-a)*e['rgb_dilation'])*e['support_dilation']),
        }
        if 'joint_no_gp' in e:
            j0 = e['joint_no_gp']
            pairs['no_gp'] = (torch.maximum(w*j0, (1-w)*r)*sg, (a*j0+(1-a)*v)*sv)
        for name, pair in pairs.items():
            for value in pair:
                if value.shape != (224, 224) or not torch.isfinite(value).all() or (value < 0).any():
                    raise RuntimeError('Invalid task response: '+name)
        return pairs

    @torch.inference_mode()
    def predict(self, feature):
        return self.route(self.evidence(feature))['full_tdf']
