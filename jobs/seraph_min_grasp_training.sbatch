#!/bin/bash
#SBATCH --job-name=vgn-min-grasp
#SBATCH --account=grad
#SBATCH -p batch_grad
#SBATCH -w aurora-g6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -t 1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

source /data/allen516/anaconda3/etc/profile.d/conda.sh
conda activate vgn_np

REPO=/data/allen516/vgn_min_grasp_train
GENERATED=/data/allen516/vgn_generated/min_grasp/datasets
CHECKPOINT_ROOT=/data/allen516/vgn_generated/checkpoints
RUN_ROOT=/data/allen516/vgn_generated/runs
LOCAL_ROOT=/local_datasets/allen516/vgn_min_grasp

PACKED_TAR=${GENERATED}/min_grasp_packed_60000_74772.tar.gz
PILE_TAR=${GENERATED}/min_grasp_pile_60000_74773.tar.gz

PACKED_RAW=${LOCAL_ROOT}/raw/min_grasp_packed_60000_74772
PILE_RAW=${LOCAL_ROOT}/raw/min_grasp_pile_60000_74773
COMBINED_RAW=${LOCAL_ROOT}/raw/min_grasp_combined120k
PACKED_CONSTRUCTED=${LOCAL_ROOT}/constructed/min_grasp_packed60k
COMBINED_CONSTRUCTED=${LOCAL_ROOT}/constructed/min_grasp_combined120k

mkdir -p "${LOCAL_ROOT}/raw" "${LOCAL_ROOT}/constructed" "${CHECKPOINT_ROOT}" "${RUN_ROOT}"
cd "${REPO}"

extract_dataset() {
    local tar_path="$1"
    local dst="$2"
    rm -rf "${dst}"
    mkdir -p "${dst}"
    tar -xzf "${tar_path}" -C "${dst}" --strip-components=1
}

construct_filter_train() {
    local raw="$1"
    local constructed="$2"
    local description="$3"

    rm -rf "${constructed}"
    python scripts/construct_dataset.py "${raw}" "${constructed}"
    python scripts/filter_dataset_indices.py "${constructed}"
    python scripts/train_vgn.py \
        --dataset "${constructed}" \
        --logdir "${RUN_ROOT}" \
        --description "${description}" \
        --epochs 30 \
        --batch-size 32 \
        --lr 3e-4 \
        --val-split 0.1
}

# Packed-only 60k: raw tar extract -> construct -> index filter -> train.
extract_dataset "${PACKED_TAR}" "${PACKED_RAW}"
construct_filter_train \
    "${PACKED_RAW}" \
    "${PACKED_CONSTRUCTED}" \
    "vgn_conv_min_grasp_packed60k_epoch30_noaug"

# Combined 120k: merge raw packed + pile first, then construct once.
extract_dataset "${PILE_TAR}" "${PILE_RAW}"
rm -rf "${COMBINED_RAW}"
mkdir -p "${COMBINED_RAW}/scenes"

cp "${PACKED_RAW}/setup.json" "${COMBINED_RAW}/setup.json"
cp "${PACKED_RAW}/scenes/"*.npz "${COMBINED_RAW}/scenes/"
cp "${PILE_RAW}/scenes/"*.npz "${COMBINED_RAW}/scenes/"

head -n 1 "${PACKED_RAW}/grasps.csv" > "${COMBINED_RAW}/grasps.csv"
tail -n +2 "${PACKED_RAW}/grasps.csv" >> "${COMBINED_RAW}/grasps.csv"
tail -n +2 "${PILE_RAW}/grasps.csv" >> "${COMBINED_RAW}/grasps.csv"

construct_filter_train \
    "${COMBINED_RAW}" \
    "${COMBINED_CONSTRUCTED}" \
    "vgn_conv_min_grasp_combined120k_epoch30_noaug"
