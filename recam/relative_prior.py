"""Normal-cost relative evidence tempering, no query-label input."""
import torch


def temperature(fidelity_weight, normal_skill):
    return max(0., 2*float(fidelity_weight)-1) * max(0., min(1., float(normal_skill)))


def tempered_joint(joint, marginal, value):
    # Exactly neutral conditional energy when value=0; exactly original at 1.
    if value == 0:
        return marginal.clamp_min(0).sqrt()
    if value == 1:
        return joint
    return joint.clamp_min(0).pow(value) * marginal.clamp_min(0).pow((1-value)/2)


def responses(joint, marginal, rgb_geometry, rgb_visible, fidelity_weight, gated_weight, normal_skill):
    q=temperature(fidelity_weight,normal_skill)
    result={}
    for prefix,exponent in [('relative',q),('untempered',1.),('neutral',0.)]:
        j=tempered_joint(joint,marginal,exponent)
        shared=gated_weight*j+(1-gated_weight)*rgb_visible
        result[prefix+'_shared']=(shared,shared)
        result[prefix+'_tdf']=(torch.maximum(fidelity_weight*j,(1-fidelity_weight)*rgb_geometry),
                               fidelity_weight*j+(1-fidelity_weight)*rgb_visible)
    return result,q
