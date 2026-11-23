"""Generalize the audited fit to 2 sources or disclosed single-source blocks.

4-shot fit and inference operations remain unchanged. Runtime adaptations are
compiled from exact audited source; source files and released state stay intact.
"""
import ast
import inspect
import textwrap
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from Model.fewshot_normal_skill import NormalSkillMemory
from Model.fewshot_recam_input_correct import InputCorrectConditional
from Model.fewshot_native_grid import NativeGridMemory


CONTEXT = SimpleNamespace(fold=None, regions={})


def fold_mask(side, fold, device):
    yy, xx = torch.meshgrid(torch.arange(56, device=device), torch.arange(56, device=device), indexing='ij')
    labels = 2 * ((yy // 7) % 2) + ((xx // 7) % 2)
    mask = labels == fold
    if side != 56:
        mask = F.interpolate(mask.float()[None, None], (side, side), mode='nearest')[0, 0].bool()
    return mask


class CalibrationLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for item in self.loader:
            CONTEXT.fold = int(item[0]['calibration_fold'][0]) if 'calibration_fold' in item[0] else None
            yield item
        CONTEXT.fold = None


def generalize_fit(cls, count, anchors):
    """Only adapt source count, metadata and the 1-shot total anchor budget."""
    original = inspect.unwrap(cls.fit)
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'super' and not node.args:
                node.args = [ast.Name(id='_OriginalClass', ctx=ast.Load()), ast.Name(id='self', ctx=ast.Load())]
            if isinstance(node.func, ast.Name) and node.func.id == 'min' and len(node.args) == 2 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 64:
                node.args[0] = ast.Constant(value=anchors)
            return node

        def visit_Compare(self, node):
            self.generic_visit(node)
            if isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name) and node.left.func.id == 'len':
                for i, value in enumerate(node.comparators):
                    if isinstance(value, ast.Constant) and value.value == 4:
                        node.comparators[i] = ast.Constant(value=count)
            return node

        def visit_Dict(self, node):
            self.generic_visit(node)
            counters = {'support_count': count, 'original_source_images': count, 'normal_statistics_independent_images': count,
                        'rgb_augmented_views': 4 * count, 'normal_views': count, 'paired_views': count,
                        'gp_normal_calibration_sources': count-1, 'gp_test_reference_sources': count,
                        'same_support_count': count}
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value in counters:
                    node.values[i] = ast.Constant(value=counters[key.value])
            return node
    tree = ast.fix_missing_locations(Rewrite().visit(tree))
    namespace = {**original.__globals__, '_OriginalClass': cls}
    exec(compile(tree, f'<multishot:{cls.__name__}.fit>', 'exec'), namespace)
    cls.fit = namespace['fit']


class Conditional(InputCorrectConditional):
    @torch.inference_mode()
    def normals(self, cloud):
        if self.dataset_name == 'eyecandies':
            return self.author_normal.cuda()
        return super().normals(cloud)

    @torch.inference_mode()
    def extract(self, image, cloud):
        if self.dataset_name == 'eyecandies':
            self.author_normal = cloud['author_normal']
            self.latest_clean_cloud = cloud['prepared'].cuda()
        cache = self._spatial_cache
        if CONTEXT.fold is not None and cache:
            full = cache['full']
        else:
            full = super().extract(image, cloud)
            if CONTEXT.fold is not None:
                cache['full'] = full
        if CONTEXT.fold is None:
            return full
        mask = full['valid'] & fold_mask(224, CONTEXT.fold, full['valid'].device)
        return {**full, 'valid': mask, 'raw': {k: v * mask for k, v in full['raw'].items()}}


class Marginal(NativeGridMemory):
    @torch.inference_mode()
    def extract(self, image, cloud):
        if CONTEXT.fold is not None and self._spatial_cache:
            full = self._spatial_cache['full']
        else:
            full = super().extract(image, cloud)
            if CONTEXT.fold is not None:
                self._spatial_cache['full'] = full
        if CONTEXT.fold is None:
            return full
        valid = full['valid'] & fold_mask(56, CONTEXT.fold, full['valid'].device).flatten()
        CONTEXT.regions[valid.data_ptr()] = (full['valid'], CONTEXT.fold)
        return {**full, 'valid': valid}


def make_model(dataset, category, shot):
    from Model.fewshot_recam_real import ReCAMRealMemory
    from Model.fewshot_bayesian_surface import BayesianSurfaceMemory
    from Model.fewshot_hybrid_density import HybridDensityMemory
    import Model.fewshot_rgb_visibility as visibility
    # The process runs exactly one class/shot; patch fit dispatch only here.
    count, anchors = (4, 16) if shot == 1 else (shot, 64)
    if shot != 4:
        for cls in (ReCAMRealMemory, NativeGridMemory, BayesianSurfaceMemory, HybridDensityMemory):
            generalize_fit(cls, count, anchors)
    original_exterior = visibility.trusted_exterior
    if shot == 1:
        def spatial_exterior(valid):
            full, fold = CONTEXT.regions[valid.data_ptr()]
            return original_exterior(full) & fold_mask(56, fold, valid.device).flatten()
        visibility.trusted_exterior = spatial_exterior
    model = NormalSkillMemory(category)
    model.conditional.__class__ = Conditional
    model.conditional.dataset_name = dataset
    model.conditional._spatial_cache = {}
    # Preserve GP methods in the MRO while replacing only feature extraction.
    class GPWithFold(BayesianSurfaceMemory):
        extract = Marginal.extract
    # Marginal.extract uses super() tied to Marginal; call original native extractor
    # explicitly to keep GP dispatch without changing its fit/predict.
    def marginal_extract(self, image, cloud):
        if CONTEXT.fold is not None and self._spatial_cache:
            full = self._spatial_cache['full']
        else:
            full = NativeGridMemory.extract(self, image, cloud)
            if CONTEXT.fold is not None:
                self._spatial_cache['full'] = full
        if CONTEXT.fold is None:
            return full
        valid = full['valid'] & fold_mask(56, CONTEXT.fold, full['valid'].device).flatten()
        CONTEXT.regions[valid.data_ptr()] = (full['valid'], CONTEXT.fold)
        return {**full, 'valid': valid}
    GPWithFold.extract = torch.inference_mode()(marginal_extract)
    model.marginal.__class__ = GPWithFold
    model.marginal._spatial_cache = {}
    return model


def finalize_fit(model, result, dataset, shot):
    CONTEXT.fold = None
    model.conditional._spatial_cache.clear()
    model.marginal._spatial_cache.clear()
    CONTEXT.regions.clear()
    result.update({'dataset': dataset, 'shot': shot, 'support_count': shot, 'original_source_images': shot,
                   'strict_matching_mapper_shot': True,
                   'calibration': 'source leave-one-out' if shot > 1 else 'four within-source spatial folds; same-image encoder context correlated',
                   'independent_calibration_sources': shot if shot > 1 else 1,
                   'spatial_folds_are_extra_shots': False,
                   'normal_fit_epochs': 0, 'epochs': 0,
                   'training_kind': f'author matched {dataset}/{shot}shot real mappers + local normal-only memory/GP fit',
                   'author_mapper_training_redone_locally': False,
                   'fullshot_weights_banks_or_statistics_used': False})
    for name in ('conditional_fit', 'marginal_fit'):
        result[name].update({'support_count': shot, 'original_source_images': shot,
                            'calibration_fold_count': 4 if shot == 1 else shot})
    result['conditional_fit']['normal_statistics_independent_images'] = shot
    result['conditional_fit']['training_kind'] = f'author matched {dataset}/{shot}shot real mappers + normal residual covariance and memory fit'
    result['normal_calibration'] = {'marginal_bank_and_gp_source_excluded': shot > 1,
                                   'single_shot_spatial_block_excluded': shot == 1,
                                   'author_conditional_weights_saw_selected_support': 'original IDs not independently verified'}
    for name in ('normal_constant_skill_gate', 'normal_rgb_visibility'):
        result[name]['original_source_images'] = shot
        result[name]['source_excluded_calibration'] = shot > 1
        result[name]['spatial_calibration'] = shot == 1
    result['normal_constant_skill_gate']['constant_source_excluded'] = shot > 1
    result['normal_constant_skill_gate']['normal_sources'] = shot
    result['normal_rgb_visibility']['only_four_normal_sources'] = shot == 4
    result['changed_method_settings']['same_support_count'] = shot
    result['changed_method_settings']['fused_normal_energy_calibration'] = ('sourceLOO q90 for marginal; selected training normal q90 for conditional'
        if shot > 1 else 'within-source spatial-fold q90 for marginal; one training normal q90 for conditional')
    result['bank_origin_semantics'] = 'spatial calibration fold within one original image' if shot == 1 else 'original source image'
    result['marginal_fit']['gp_normal_calibration_sources'] = shot-1 if shot > 1 else 1
    result['marginal_fit']['gp_test_reference_sources'] = shot
    if shot == 1:
        result['marginal_fit']['variability_calibration_unit'] = 'spatial fold of one image, not independent images'
    if dataset == 'eyecandies':
        settings = result['conditional_fit']['changed_method_settings']
        settings['normal_estimator'] = 'author EyeCandies physical CUDA normals radius.04/maxnn30; native camera center; author axis permutation; blue BG'
        settings['normal_preprocessing'] = 'numeric cache from author physical EyeCandies converter; explicit zero BG; no inference PNG rendering'
        settings['repeat_cpu_preprocessing'] = False
        settings['normal_radius_physical'] = .04
        result['changed_method_settings']['normal_geometry_layouts'] = ['square512 author-axis XYZ with explicit zero background', 'same native512 RGB coordinates']
    return result
