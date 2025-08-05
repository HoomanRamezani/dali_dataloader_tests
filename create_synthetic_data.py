#!/usr/bin/env python3
# create_synthetic_data.py

import argparse
import shutil
import tarfile
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

warnings.filterwarnings("ignore", category=DeprecationWarning, module="zarr")

try:
    import pyarrow  # noqa: F401

    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

BASE_DIR = Path("/mnt/weka/")
SHARD_ROOT = BASE_DIR / "shards" / "00000000"
DS_DIR = BASE_DIR / "datasets" / "synthetic"


def make_shard(steps: int, cameras: int, height: int, width: int) -> Path:
    shard_uuid = uuid.uuid4().hex[:8]
    shard_dir = SHARD_ROOT / shard_uuid
    store_path = shard_dir / "steps.zarr"
    shard_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(store_path, mode="w")
    root.create_array(
        "image",
        shape=(steps, cameras, height, width, 3),
        chunks=(1, 1, height, width, 3),
        dtype=np.uint8,
    )
    root.create_array(
        "actions",
        shape=(steps, 14),
        chunks=(1, 14),
        dtype=np.float32,
    )
    root.create_array(
        "state",
        shape=(steps, 16),
        chunks=(1, 16),
        dtype=np.float32,
    )

    root["image"][:] = np.random.randint(0, 256, size=root["image"].shape, dtype=np.uint8)
    root["actions"][:] = np.random.randn(*root["actions"].shape).astype(np.float32)
    root["state"][:] = np.random.randn(*root["state"].shape).astype(np.float32)

    tar_path = shard_dir / "steps.zarr.pack"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(store_path, arcname="steps.zarr")

    shutil.rmtree(store_path)
    return tar_path


def write_metadata(shard_path: Path, steps: int):
    DS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "episode_id": 0,
        "shard_path": str(shard_path),
        "start_step": 0,
        "num_steps": steps,
    }
    df = pd.DataFrame([record])

    if _HAS_PARQUET:
        df.to_parquet(DS_DIR / "dataset.parquet", index=False)
    else:
        df.to_csv(DS_DIR / "dataset.parquet.csv", index=False)

    df.to_csv(DS_DIR / "inventory.txt", sep="\t", header=False, index=False)
    pd.DataFrame({"num_episodes": [1], "steps_per_episode": [steps]}).to_csv(
        DS_DIR / "report.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    print("⏳  Creating shard …", flush=True)
    shard_path = make_shard(args.steps, args.cameras, args.height, args.width)
    write_metadata(shard_path, args.steps)
    size_mb = shard_path.stat().st_size / (1024 * 1024)
    print(f"✅  Synthetic dataset ready ({size_mb:.1f} MB) at {BASE_DIR}")


if __name__ == "__main__":
    main()
