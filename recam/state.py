def restore(model, state):
    from Model.fewshot_bayesian_surface import SparseGP
    for name in ('banks', 'origins', 'scales', 'residual_stats'):
        setattr(model.conditional, name, state['conditional'][name])
    marginal = state['marginal']
    for name in ('banks', 'origins', 'scales', 'geometry_weight', 'semantic_center', 'semantic_scale'):
        setattr(model.marginal, name, marginal[name])
    model.marginal.excess_scale = marginal['joint_excess_scale']
    model.marginal.gp_scales = marginal['gp_scales']
    model.marginal.gp_models = {}
    for name, values in marginal['gp_models'].items():
        gp = SparseGP.__new__(SparseGP)
        for attribute, value in values.items():
            setattr(gp, attribute, value)
        model.marginal.gp_models[name] = gp
    model.scales = state['fused_energy_scales']
    model.conditional_weight = state['normal_conditional_weight']
    model.visibility_enabled = bool(state['rgb_visibility_enabled'])
    for name in ('background_bank', 'background_origins', 'visibility_threshold'):
        setattr(model, name, state[name])
