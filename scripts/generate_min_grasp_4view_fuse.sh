#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?Usage: $0 <raw-output-root> [extra generate_data.py args...]}
shift || true

python scripts/generate_data.py "${ROOT}" \
    --scene packed \
    --object-set packed/train \
    --num-grasps "${NUM_GRASPS:-60000}" \
    --sampling-policy object-balanced \
    --min-grasps-per-object "${MIN_GRASPS_PER_OBJECT:-12}" \
    --view-policy paired-4 \
    --ee-phi-span-deg "${EE_PHI_SPAN_DEG:-90}" \
    "$@"
