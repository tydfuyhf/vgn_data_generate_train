# VGN Data Generation and Training Fork

This repository is a small fork of the original Volumetric Grasping Network implementation:

- Original VGN repository: https://github.com/ethz-asl/vgn
- Original paper: M. Breyer, J. J. Chung, L. Ott, R. Siegwart, and J. Nieto, "Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter", CoRL 2020.

The goal of this fork is to keep the original VGN training pipeline intact while adapting synthetic data generation for a setup that uses both a top depth camera and an oblique wrist/end-effector depth camera.

## Main Changes

### Paired top + end-effector views

Original VGN generated synthetic depth views mostly from top-like camera angles. In our deployment, the input TSDF is expected to come from both:

- a top-like depth camera
- an oblique wrist/end-effector depth camera

So `scripts/generate_data.py` now renders **two views per scene by default**:

```text
Top-like view:
  theta: 0 to 15 degrees from vertical
  phi:   0 to 360 degrees

EE-like view:
  theta: 45 to 80 degrees from vertical
  phi:   0 to 360 degrees by default
```

Both views are saved into the same scene file. `scripts/construct_dataset.py` already integrates all depth images in a scene, so this produces a fused two-view TSDF without changing the VGN network.

New generation options:

```bash
--view-policy paired          # default, top-like + EE-like
--view-policy legacy          # original random view behavior
--ee-phi-center-deg <deg>     # optional EE yaw center
--ee-phi-span-deg <deg>       # EE yaw span around the center, default 90
```

### Training code mostly unchanged

The VGN model, loss functions, optimizer, dataset loader, and training loop are kept close to the original implementation. The training script only adds clearer path help and a log directory default suitable for our server setup.

### Packed vs pile

This fork does not force packed-only training. Scene type is still selected with:

```bash
--scene packed
--scene pile
```

The examples below use `packed` because our target tabletop setup is closer to upright packed scenes than fully random piles.

## Environment

Recommended environment for GPU training:

```text
Python 3.10
PyTorch with CUDA 11.8
Open3D 0.18.0
PyBullet
mpi4py / OpenMPI
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate vgn
pip install -e .
```

Quick CUDA check:

```bash
python - <<'PY'
import torch, open3d, pybullet
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("open3d:", open3d.__version__)
print("pybullet ok")
PY
```

## Basic Usage

### 1. Generate raw grasp data

Small sanity run:

```bash
python scripts/generate_data.py data/raw/packed_sanity \
  --scene packed \
  --object-set packed/train \
  --num-grasps 3000 \
  --ee-phi-span-deg 90
```

If the real EE camera yaw is known:

```bash
python scripts/generate_data.py data/raw/packed_sanity \
  --scene packed \
  --object-set packed/train \
  --num-grasps 3000 \
  --ee-phi-center-deg <actual_phi_deg> \
  --ee-phi-span-deg 90
```

Larger run:

```bash
mpirun -np 8 python scripts/generate_data.py data/raw/packed_full \
  --scene packed \
  --object-set packed/train \
  --num-grasps 60000 \
  --ee-phi-span-deg 90
```

### 2. Construct the training dataset

```bash
python scripts/construct_dataset.py \
  data/raw/packed_sanity \
  data/datasets/packed_sanity
```

For the full dataset:

```bash
python scripts/construct_dataset.py \
  data/raw/packed_full \
  data/datasets/packed_full
```

### 3. Train VGN

Short smoke test:

```bash
python scripts/train_vgn.py \
  --dataset data/datasets/packed_sanity \
  --augment \
  --epochs 5 \
  --batch-size 32
```

Full training:

```bash
python scripts/train_vgn.py \
  --dataset data/datasets/packed_full \
  --augment \
  --epochs 30 \
  --batch-size 32
```

## Notes

- `generate_data.py` is mostly CPU-bound because it runs PyBullet simulation.
- `train_vgn.py` uses the GPU through PyTorch.
- `construct_dataset.py` fuses all saved views in each scene into the final TSDF grid.
- Generated raw data, constructed datasets, runs, and checkpoints are ignored by Git.
- `data/urdfs/` is committed because data generation needs those assets.

Ignored generated outputs:

```text
data/raw/
data/datasets/
data/runs/
data/models/
*.pth
*.pt
```
