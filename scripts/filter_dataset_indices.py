import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RESOLUTION = 40
BACKUP_NAME = "grasps.csv.bak_before_index_filter"


def main(args):
    csv_path = args.dataset / "grasps.csv"
    backup_path = args.dataset / BACKUP_NAME

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    idx = np.round(df[["i", "j", "k"]].to_numpy(dtype=np.float32)).astype(np.int64)
    valid = ((idx >= 0) & (idx < args.resolution)).all(axis=1)
    filtered = df.loc[valid].reset_index(drop=True)

    if backup_path.exists() and not args.overwrite_backup:
        raise FileExistsError(
            f"{backup_path} already exists. Pass --overwrite-backup to replace it."
        )

    csv_path.replace(backup_path)
    filtered.to_csv(csv_path, index=False)

    removed = len(df) - len(filtered)
    print(f"Loaded rows: {len(df)}")
    print(f"Valid rows: {len(filtered)}")
    print(f"Removed rows: {removed}")
    print(f"Backup: {backup_path}")
    print(f"Filtered CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        help="Constructed dataset directory containing grasps.csv",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RESOLUTION,
        help="Voxel grid resolution; VGN default is 40",
    )
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help=f"replace existing {BACKUP_NAME}",
    )
    args = parser.parse_args()
    main(args)
