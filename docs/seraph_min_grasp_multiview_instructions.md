# Seraph Min-Grasp Multi-View Fuse Instructions

Use this as the project instruction/context when asking GPT to prepare Seraph
commands or job scripts for the min-grasp multi-view experiments.

## Goal

Generate and train three min-grasp variants without changing code:

- `2-view fuse`: one top-like view plus one EE-like view
- `3-view fuse`: one top-like view plus two yaw-spread EE-like views
- `4-view fuse`: one top-like view plus three yaw-spread EE-like views

The existing 2-view path must remain unchanged. The default
`scripts/generate_data.py` behavior is still:

```bash
--view-policy paired
```

which means one top-like view and one EE-like view.

## View Geometry

The synthetic camera is not sampled by directly randomizing Cartesian RPY.
Instead, camera position is sampled on a sphere around the workspace center:

```text
x = r * sin(theta) * cos(phi)
y = r * sin(theta) * sin(phi)
z = r * cos(theta)
```

Then `look_at()` orients the camera toward the workspace center.

Definitions:

- `theta`: polar angle from the world vertical z axis
- `phi`: azimuth/yaw direction around the world z axis in the xy plane
- `r`: distance from workspace center

Current ranges:

```text
top-like:
  theta = 0 to 15 deg from vertical
  phi   = 0 to 360 deg

EE-like:
  theta = 45 to 80 deg from vertical
  phi   = 0 to 360 deg, or yaw-spread for multi-view
```

Interpretation:

- `theta = 0 deg`: top-down view
- `theta = 45 deg`: oblique view
- `theta = 80 deg`: near side/front view

## Current View Policies

```text
paired:
  top 1 view:
    theta random in 0~15 deg
    phi random in 0~360 deg
  EE 1 view:
    theta random in 45~80 deg
    phi random in 0~360 deg

paired-3:
  top 1 view
  EE 2 views:
    theta random in 45~80 deg for each view
    phi spread by 180 deg inside the same scene

paired-4:
  top 1 view
  EE 3 views:
    theta random in 45~80 deg for each view
    phi spread by 120 deg inside the same scene

legacy:
  random number of views, 1~6
  theta random in 0~45 deg
  phi random in 0~360 deg
```

For `paired-3` and `paired-4`, the scene-level EE yaw phase is random, but the
EE views inside that scene are evenly separated. This avoids wasting views by
sampling multiple EE cameras from almost the same direction.

Examples:

```text
paired-3, random phase = 70 deg:
  EE phi = 70, 250 deg

paired-4, random phase = 30 deg:
  EE phi = 30, 150, 270 deg
```

## Recommended Experiment Size

Generation time is acceptable, so use a larger dataset than the original 60k.
For clean comparison:

```text
2-view packed 240k, train 50 epochs
3-view packed 240k, train 50 epochs
4-view packed 240k, train 50 epochs
```

Keep these the same across both runs:

- scene: `packed`
- object set: `packed/train`
- sampling policy: `object-balanced`
- minimum grasps per object: `12` unless intentionally changed
- training hyperparameters
- validation split

Use packed first because the demo setup is closer to packed tabletop scenes
than random piles.

## Raw Data Generation

Use the wrapper scripts. They allow overrides through environment variables and
extra CLI arguments.

2-view:

```bash
NUM_GRASPS=240000 mpirun -np 8 bash scripts/generate_min_grasp_2view_fuse.sh \
  /data/allen516/vgn_generated/min_grasp/raw/min_grasp_packed_2view_240k
```

3-view:

```bash
NUM_GRASPS=240000 mpirun -np 8 bash scripts/generate_min_grasp_3view_fuse.sh \
  /data/allen516/vgn_generated/min_grasp/raw/min_grasp_packed_3view_240k
```

4-view:

```bash
NUM_GRASPS=240000 mpirun -np 8 bash scripts/generate_min_grasp_4view_fuse.sh \
  /data/allen516/vgn_generated/min_grasp/raw/min_grasp_packed_4view_240k
```

Available safe overrides:

```bash
NUM_GRASPS=240000
MIN_GRASPS_PER_OBJECT=12
EE_PHI_SPAN_DEG=90
```

Example with overrides:

```bash
NUM_GRASPS=240000 MIN_GRASPS_PER_OBJECT=16 EE_PHI_SPAN_DEG=120 \
mpirun -np 8 bash scripts/generate_min_grasp_4view_fuse.sh \
  /data/allen516/vgn_generated/min_grasp/raw/min_grasp_packed_4view_240k_mg16_span120
```

Avoid passing duplicate options after the wrapper if the wrapper already sets
them. Prefer environment variable overrides for:

- `NUM_GRASPS`
- `MIN_GRASPS_PER_OBJECT`
- `EE_PHI_SPAN_DEG`

## Construct and Filter

Constructed datasets should be made from the raw directories, then filtered for
out-of-range voxel labels.

2-view:

```bash
python scripts/construct_dataset.py \
  /data/allen516/vgn_generated/min_grasp/raw/min_grasp_packed_2view_240k \
  /local_datasets/allen516/vgn_min_grasp/constructed/min_grasp_packed_2view_240k

python scripts/filter_dataset_indices.py \
  /local_datasets/allen516/vgn_min_grasp/constructed/min_grasp_packed_2view_240k
```

3-view:

```bash
python scripts/construct_dataset.py \
  /data/allen516/vgn_generated/min_grasp/raw/min_grasp_packed_3view_240k \
  /local_datasets/allen516/vgn_min_grasp/constructed/min_grasp_packed_3view_240k

python scripts/filter_dataset_indices.py \
  /local_datasets/allen516/vgn_min_grasp/constructed/min_grasp_packed_3view_240k
```

4-view:

```bash
python scripts/construct_dataset.py \
  /data/allen516/vgn_generated/min_grasp/raw/min_grasp_packed_4view_240k \
  /local_datasets/allen516/vgn_min_grasp/constructed/min_grasp_packed_4view_240k

python scripts/filter_dataset_indices.py \
  /local_datasets/allen516/vgn_min_grasp/constructed/min_grasp_packed_4view_240k
```

`construct_dataset.py` already fuses all saved views in each scene. No network
or dataset-loader change is required.

## Training

Train with the same hyperparameters for fair comparison.

Epochs are command-line configurable through `scripts/train_vgn.py --epochs`.
Use `EPOCHS` so the run description stays consistent when overriding epochs.
For example, set `EPOCHS=60` before the command to train for 60 epochs and name
the run with `epoch60`.

2-view:

```bash
EPOCHS=${EPOCHS:-50}
python scripts/train_vgn.py \
  --dataset /local_datasets/allen516/vgn_min_grasp/constructed/min_grasp_packed_2view_240k \
  --logdir /data/allen516/vgn_generated/runs \
  --description "vgn_conv_min_grasp_packed_2view_240k_epoch${EPOCHS}_noaug" \
  --epochs "${EPOCHS}" \
  --batch-size 32 \
  --lr 3e-4 \
  --val-split 0.1
```

3-view:

```bash
EPOCHS=${EPOCHS:-50}
python scripts/train_vgn.py \
  --dataset /local_datasets/allen516/vgn_min_grasp/constructed/min_grasp_packed_3view_240k \
  --logdir /data/allen516/vgn_generated/runs \
  --description "vgn_conv_min_grasp_packed_3view_240k_epoch${EPOCHS}_noaug" \
  --epochs "${EPOCHS}" \
  --batch-size 32 \
  --lr 3e-4 \
  --val-split 0.1
```

4-view:

```bash
EPOCHS=${EPOCHS:-50}
python scripts/train_vgn.py \
  --dataset /local_datasets/allen516/vgn_min_grasp/constructed/min_grasp_packed_4view_240k \
  --logdir /data/allen516/vgn_generated/runs \
  --description "vgn_conv_min_grasp_packed_4view_240k_epoch${EPOCHS}_noaug" \
  --epochs "${EPOCHS}" \
  --batch-size 32 \
  --lr 3e-4 \
  --val-split 0.1
```

## Notes for GPT

- Do not modify the 2-view default path.
- Do not change model architecture for these experiments.
- Do not change `construct_dataset.py`; it already fuses all views in a scene.
- Keep 2-view, 3-view, and 4-view runs identical except for view policy.
- Prefer packed-only first for demo alignment.
- If creating an sbatch script, use environment variable overrides instead of
  editing Python code.
