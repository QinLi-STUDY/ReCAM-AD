# ReCAM-AD

Reference-Calibrated Multimodal Anomaly Detection for few-shot RGB–3D industrial inspection. This repository contains the inference path and normal-reference calibration used for MVTec 3D-AD and EyeCandies at 1, 2 and 4 shots. The classwise reference results are in [reference_metrics.csv](reference_metrics.csv).

| Stage | Role | Implementation |
| --- | --- | --- |
| NRC · Normal Reference Calibration | Builds normal token banks and paired anchors, and calibrates evidence scales and the reliability of the conditional prior. | [Model](recam/model.py), [reference fit](recam/calibrate.py) |
| BRE · Bidirectional Relation Evidence | Computes and aligns the directional residuals between frozen cross-modal predictions and observed features. Their pixelwise product supplies the conditional evidence. | [Residuals](vendor/g2sf/Model/fewshot_recam_real.py), [camera alignment](vendor/g2sf/Model/fewshot_hybrid_aligned.py) |
| MCF · Marginal Coverage Fusion | Retrieves DINOv2 normal tokens, adds bidirectional kernel residuals and combines the resulting evidence with the conditional and RGB paths. | [Evidence](recam/task_decoupled.py), [final decision](recam/detector.py) |

The experiment fixes all neural weights. The local shot count refers to original paired normal training instances per category. Each released state includes the reference IDs and calibration used for inference. The upstream real-stream mapper checkpoints are matched by dataset, category and shot count. Their individual training IDs could not be independently verified.

## Weights

| Directory | Contents |
| --- | --- |
| [weights/backbones](weights/backbones) | DINOv1 ViT-B/8, DINOv2 ViT-B/14 and the Point-MAE initialization loaded by the inherited model constructor. The marginal geometry path encodes measured surface normals with DINOv2. |
| [weights/mappers](weights/mappers) | 120 released CMDS-AD mapper files, two directions for each dataset/category/shot. Each pair has a source and checksum manifest. |
| [weights/states](weights/states) | 60 fitted states. Each group contains normal memories, GP parameters and statistics in `normal_memory.pt`, final routing/scales in `calibration.json`, and the selected references in `support.json`. |

The [weight index](weights/SHA256.json) records byte sizes and SHA-256 digests. State files include the original checkpoint hash in the index. The packaged state metadata uses repository-relative mapper paths. The tensor values are preserved.

## Setup

Tested on Linux with Python 3.11 in the G2SF environment, PyTorch 2.5.1+cu118, torchvision 0.20.1+cu118 and CUDA-enabled Open3D 0.19.0. If using an existing G2SF environment, activate it and run the checks below. For a new environment:

```bash
conda create -n recam-ad python=3.11 -y
conda activate recam-ad
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -c "import torch, open3d as o; assert torch.cuda.is_available() and o.core.cuda.is_available()"
python verify_weights.py
```

When cloning from a Git LFS remote, run `git lfs install` and `git lfs pull` to retrieve the weights. The complete local release can be used offline with both datasets supplied separately.

## Evaluate the released states

For **MVTec 3D-AD**, point `--raw` at the original directory containing `bagel/train`, and `--prepared` at the matching G2SF directory containing `bagel/train/good/xyz`. RGB and ground truth come from the original dataset. Geometry comes from the prepared XYZs, with square padding removed to recover camera alignment. [G2SF prepared data](https://github.com/ctaoaa/G2SF#readme)

```bash
python recam_ad.py eval --dataset mvtec --category bagel --shot 4 \
  --raw /data/MVTEC_3D-AD --prepared '/data/mvtec_preprocessed/dataset/MVTEC 3D'
```

For **EyeCandies**, the raw root contains directories such as `CandyCane/train/data` and `CandyCane/test_public/data`. Prepare the numeric XYZs and conditional-branch surface normals once. Existing caches of the same profiles can be passed directly to evaluation.

```bash
python recam_ad.py prepare-eye --raw /data/Eyecandies \
  --prepared /data/eye_xyz --eye-normals /data/eye_author_normals

python recam_ad.py eval --dataset eyecandies --category CandyCane --shot 2 \
  --raw /data/Eyecandies --prepared /data/eye_xyz --eye-normals /data/eye_author_normals
```

The evaluator writes `outputs/DATASET/Kshot/CATEGORY/metrics.json`. The six metrics use a 0–1 scale: `auc` (image AUROC), `seg_auc` (pixel AUROC), `aupro` (PRO@30%) and `aupro_0p1`, `aupro_0p05`, `aupro_0p01` (PRO@10%, 5%, 1%). Use `--save-maps` to include query IDs, image scores and camera-aligned smoothed localization maps. The original category-specific smoothing, GP coefficient 0.2 and dataset-specific evaluation routes are implemented in the release.

To evaluate every category and shot for a dataset, iterate over the actual published state directories:

```bash
set -e
dataset=mvtec
data_args=(--raw /data/MVTEC_3D-AD --prepared '/data/mvtec_preprocessed/dataset/MVTEC 3D')
# For EyeCandies, use dataset=eyecandies and the three data paths shown above.
for shot in 1 2 4; do
  for state in "weights/states/$dataset/${shot}shot/"*/; do
    python recam_ad.py eval --dataset "$dataset" --category "$(basename "$state")" \
      --shot "$shot" "${data_args[@]}"
  done
done
```

## Refit the normal-reference state

`fit` uses only the selected normal pairs with the frozen mappers. One shot uses the original within-image spatial folds, and two/four shots use the original source-excluded marginal calibration. The default destination is `fitted/`. Use the same raw, prepared and EyeCandies normal roots at fit and evaluation time.

```bash
python recam_ad.py fit --dataset mvtec --category bagel --shot 4 \
  --raw /data/MVTEC_3D-AD --prepared '/data/mvtec_preprocessed/dataset/MVTEC 3D'
python recam_ad.py eval --dataset mvtec --category bagel --shot 4 --state-root fitted \
  --raw /data/MVTEC_3D-AD --prepared '/data/mvtec_preprocessed/dataset/MVTEC 3D'
```

## Credits and distribution

We use the released real-stream mappers from [CMDS-AD](https://github.com/Junhaocai27/CMDS-AD) and its [checkpoint release](https://huggingface.co/Junhaocai27/CMDS-AD), the [DINO](https://github.com/facebookresearch/dino) and [DINOv2](https://github.com/facebookresearch/dinov2) representations, and components from [G2SF](https://github.com/ctaoaa/G2SF) and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE). Third-party license notices are in [licenses](licenses). Refer to each upstream release for its weight terms.

For the repository owner: binary weights are covered by [.gitattributes](.gitattributes). Before the first commit to a new GitHub repository, install and initialize Git LFS, run `git add .`, and use `git lfs ls-files` to confirm the weight files are staged as LFS objects. The full weight inventory is approximately 20.4 GiB.
