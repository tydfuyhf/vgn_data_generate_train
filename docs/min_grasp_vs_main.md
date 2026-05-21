# min_grasp vs main

This branch keeps the VGN training and inference architecture comparable with
`main`, while changing the data generation policy used to create the
min-grasp datasets.

## Main Difference

`main` and `min_grasp` are intended to differ in data generation policy, not in
the model architecture.

The `min_grasp` branch changes `scripts/generate_data.py` so grasp points are
sampled with an object-balanced policy by default:

```bash
--sampling-policy object-balanced
--min-grasps-per-object 12
```

For each generated scene, the reconstructed point cloud is cropped by each
PyBullet object AABB. The generator guarantees a minimum number of grasp point
samples from each object crop, then fills the remaining samples from the full
scene point cloud.

The original full-scene sampling behavior is still available with:

```bash
--sampling-policy global
```

## What Should Stay Comparable

For fair comparison against checkpoints trained from `main` data, keep these
parts aligned between branches:

- VGN network architecture
- loss functions
- optimizer settings
- dataset tensor format
- training loop
- inference code

The goal is to compare checkpoints trained on different data policies, not
checkpoints trained by different model or inference code.

## Training Pipeline Notes

Raw min-grasp datasets must not be passed directly to `scripts/train_vgn.py`.
Raw `grasps.csv` files contain metric positions:

```text
scene_id,qx,qy,qz,qw,x,y,z,width,label
```

`scripts/train_vgn.py` expects a constructed dataset with voxel indices:

```text
scene_id,qx,qy,qz,qw,i,j,k,width,label
```

Use this order:

```text
raw tar extract -> construct_dataset.py -> filter_dataset_indices.py -> train_vgn.py
```

`scripts/filter_dataset_indices.py` removes constructed rows whose rounded
`i,j,k` indices are outside the voxel grid. The default resolution is 40.

For no-augmentation training, run:

```bash
python scripts/filter_dataset_indices.py "$CONSTRUCTED"
```

Only use `--augment-safe` when `scripts/train_vgn.py` is also run with
`--augment`. Do not use `--augment-safe` for no-augmentation training, because
it checks `k` as if augmentation will recenter z. That can leave real
out-of-range `k` rows and cause CUDA index out-of-bounds errors during
`rot_out[..., i, j, k]` selection.

## Seraph Notes

On Seraph, extract and train from `/local_datasets`, not directly from `/data`
NAS paths. NAS-backed dataloading can be slow and unstable for training jobs.

Useful generated min-grasp inputs:

```text
/data/allen516/vgn_generated/min_grasp/datasets/min_grasp_packed_60000_74772.tar.gz
/data/allen516/vgn_generated/min_grasp/datasets/min_grasp_pile_60000_74773.tar.gz
```

Reference successful checkpoints:

```text
/data/allen516/vgn_generated/checkpoints/vgn_conv_min_grasp_packed60k_epoch30_noaug.pt
/data/allen516/vgn_generated/checkpoints/vgn_conv_min_grasp_combined120k_epoch30_noaug.pt
```

Do not commit generated datasets, logs, checkpoints, `.pt`, `.tar.gz`, `.out`,
`.err`, `__pycache__`, or local `/local_datasets` contents.

When debugging jobs, check both stdout and stderr. If CUDA reports
`IndexKernel` or index out-of-bounds errors, inspect constructed `i,j,k` ranges
first.
