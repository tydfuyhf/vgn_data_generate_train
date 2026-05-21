# Seraph Min-Grasp Training

The min-grasp raw datasets cannot be passed directly to `scripts/train_vgn.py`.
Raw `grasps.csv` files use:

```text
scene_id,qx,qy,qz,qw,x,y,z,width,label
```

Training expects a constructed dataset with voxel indices:

```text
scene_id,qx,qy,qz,qw,i,j,k,width,label
```

Use this flow on Seraph:

```bash
tar -xzf /data/allen516/vgn_generated/min_grasp/datasets/min_grasp_packed_60000_74772.tar.gz -C /local_datasets/allen516/min_grasp/raw_packed --strip-components=1
python scripts/construct_dataset.py /local_datasets/allen516/min_grasp/raw_packed /local_datasets/allen516/min_grasp/packed_constructed
python scripts/filter_dataset_indices.py /local_datasets/allen516/min_grasp/packed_constructed
python scripts/train_vgn.py --dataset /local_datasets/allen516/min_grasp/packed_constructed --logdir /data/allen516/vgn_generated/runs
```

`construct_dataset.py` converts raw metric grasp positions into voxel-grid
indices and rebuilds scene TSDF grids. `filter_dataset_indices.py` then removes
constructed rows whose rounded `i,j,k` indices fall outside the default
resolution-40 voxel grid.

For no-augmentation training, run:

```bash
python scripts/filter_dataset_indices.py "$CONSTRUCTED"
```

Do not pass `--augment-safe` unless `scripts/train_vgn.py` will also receive
`--augment`. The augment-safe check treats `k` as the grid center because
augmentation shifts z during training. In no-augmentation training, that can
leave real out-of-bounds `k` values in `grasps.csv`, which can later trigger a
CUDA `IndexKernel` index out-of-bounds failure in `train_vgn.py` at the
`rot_out[..., i, j, k]` selection.

Run dataloading from `/local_datasets`, not directly from `/data` NAS storage.
For Slurm jobs, check both stdout and stderr. If CUDA reports an index
out-of-bounds error, inspect the constructed `i,j,k` ranges first.
