#!/usr/bin/env python3
"""
create_synthetic_numpy.py
-------------------------
Generate simple *.npy* shards for random-read benchmarking.

Each shard directory  (<root>/shards/<uuid>/) contains:
    image.npy   shape (steps, cams, H, W, 3)   uint8
    actions.npy shape (steps, 14)              float32
    state.npy   shape (steps, 16)              float32
"""

from pathlib import Path
import argparse, uuid
import numpy as np

ROOT = Path("/mnt/weka")            # change if desired
SHARDS = ROOT / "shards_numpy"      # all shards live here


def make_shard(steps=4000, cams=2, H=256, W=320) -> Path:
    shard_dir = SHARDS / uuid.uuid4().hex[:8]
    shard_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    np.save(shard_dir / "image.npy",
            rng.integers(0, 256, size=(steps, cams, H, W, 3), dtype=np.uint8))
    np.save(shard_dir / "actions.npy",
            rng.standard_normal((steps, 14), dtype=np.float32))
    np.save(shard_dir / "state.npy",
            rng.standard_normal((steps, 16), dtype=np.float32))
    return shard_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-shards", type=int, default=50)
    p.add_argument("--steps",      type=int, default=4000)
    p.add_argument("--cameras",    type=int, default=2)
    p.add_argument("--height",     type=int, default=256)
    p.add_argument("--width",      type=int, default=320)
    args = p.parse_args()

    SHARDS.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_shards):
        print(f"[{i+1}/{args.num_shards}] writing shard …", end="", flush=True)
        make_shard(args.steps, args.cameras, args.height, args.width)
        print(" done")
    print("✅  synthetic shards written to", SHARDS)


if __name__ == "__main__":
    main()