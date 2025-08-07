#!/usr/bin/env python3
"""
create_synthetic_data.py
------------------------
Create synthetic “packed-Zarr” shards for random-read benchmarking.

Each shard is written with the exact binary format understood by
PackedDirectoryStore (version 2 header).  No tarballs are produced.
"""

import argparse
import math
import os
import pickle
import re
import shutil
import struct
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

warnings.filterwarnings("ignore", category=DeprecationWarning, module="zarr")

ROOT = Path("/mnt/weka")
SHARDS_ROOT = ROOT / "shards2" / "00000000"   # keep single episode ID
DATASET_DIR = ROOT / "datasets2" / "synthetic"

# ───────────────────────────────────────── pack helper ─────────────────────────
def _write_header(f, manifest):
    """Write  b"pack"<u32 ver=2><u64 size><pickle(manifest)>  header."""
    f.write(b"pack")                 # signature (4 bytes)
    f.write(struct.pack("<I", 2))    # version 2 → pickle manifest
    payload = pickle.dumps(manifest, protocol=4)
    f.write(struct.pack("<Q", len(payload)))
    f.write(payload)


def pack_directory_store(store_dir: Path, *, remove: bool = True) -> Path:
    """
    Bundle `store_dir` (a Zarr directory) into `store_dir.with_suffix('.pack')`
    using the format that PackedDirectoryStore can open directly.
    """
    store_dir = store_dir.resolve()
    pack_path = store_dir.with_suffix(".pack")

    # Collect files and sizes
    file_sizes = {}
    for root, _, files in os.walk(store_dir):
        for fn in files:
            p = Path(root) / fn
            file_sizes[str(p.relative_to(store_dir))] = p.stat().st_size

    # Stable order: interleave chunks by numeric chunk index, meta files first
    chunk_re = re.compile(r"\d+")
    def sort_key(path):
        m = chunk_re.match(Path(path).name)
        return (int(m.group()) if m else -1, path)

    manifest, offset = {}, 0
    for name in sorted(file_sizes, key=sort_key):
        size = file_sizes[name]
        manifest[name] = (offset, size)
        offset += size

    # Write header + raw bytes
    with open(pack_path, "wb") as f:
        _write_header(f, manifest)
        for name in manifest:
            with open(store_dir / name, "rb") as g:
                shutil.copyfileobj(g, f)

    if remove:
        shutil.rmtree(store_dir)
    return pack_path
# ───────────────────────────────────────────────────────────────────────────────


def make_single_shard(steps: int, cams: int, H: int, W: int) -> Path:
    """Create one shard and return the .pack path (~500 MB for default params)."""
    shard_dir  = SHARDS_ROOT / uuid.uuid4().hex[:8]
    store_dir  = shard_dir / "steps.zarr"
    shard_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(store_dir, mode="w")
    root.create_array("image",
                      shape=(steps, cams, H, W, 3),
                      chunks=(1, 1, H, W, 3),
                      dtype="u1")
    root.create_array("actions", shape=(steps, 14), chunks=(1, 14), dtype="f4")
    root.create_array("state",   shape=(steps, 16), chunks=(1, 16), dtype="f4")

    rng = np.random.default_rng()
    root["image"][:]   = rng.integers(0, 256, size=root["image"].shape, dtype=np.uint8)
    root["actions"][:] = rng.standard_normal(root["actions"].shape, dtype=np.float32)
    root["state"][:]   = rng.standard_normal(root["state"].shape,   dtype=np.float32)

    # Convert to single-file packed store
    return pack_directory_store(store_dir, remove=True)


def approx_shard_size_mb(steps: int, cams: int, H: int, W: int) -> float:
    bytes_per_step = cams * H * W * 3            # images (uint8)
    bytes_per_step += (14 + 16) * 4              # actions + state
    return steps * bytes_per_step / 2**20 * 0.7  # ≈ Blosc compression factor


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--num-shards", type=int, default=50, help="# shards to create")
    p.add_argument("--steps",      type=int, default=4000, help="timesteps per shard")
    p.add_argument("--cameras",    type=int, default=2)
    p.add_argument("--height",     type=int, default=256)
    p.add_argument("--width",      type=int, default=320)
    args = p.parse_args()

    SHARDS_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    est_mb = approx_shard_size_mb(args.steps, args.cameras, args.height, args.width)
    print(f"≈{est_mb:.0f} MB per shard  →  ≈{est_mb*args.num_shards/1024:.1f} GB total")

    records = []
    for n in range(args.num_shards):
        print(f"⏳  ({n+1}/{args.num_shards}) writing shard …", end="", flush=True)
        pack_path = make_single_shard(args.steps, args.cameras, args.height, args.width)
        size_mb   = pack_path.stat().st_size / 2**20
        records.append({
            "episode_id": 0,
            "shard_path": str(pack_path),
            "start_step": 0,
            "num_steps":  args.steps,
        })
        print(f" done ({size_mb:.1f} MB)")

    df = pd.DataFrame(records)
    if pd.util._optional.import_optional_dependency("pyarrow", errors="ignore"):
        df.to_parquet(DATASET_DIR / "dataset.parquet", index=False)
    else:
        df.to_csv(DATASET_DIR / "dataset.parquet.csv", index=False)

    df.to_csv(DATASET_DIR / "inventory.txt", sep="\t", header=False, index=False)
    pd.DataFrame({"num_episodes": [1], "steps_per_episode": [args.steps]}).to_csv(
        DATASET_DIR / "report.csv", index=False)

    total_mb = sum(Path(r["shard_path"]).stat().st_size for r in records) / 2**20
    print(f"\n✅  Created {args.num_shards} shards "
          f"({total_mb/1024:.1f} GB) under {SHARDS_ROOT}")


if __name__ == "__main__":
    main()