#!/usr/bin/env python3
"""
create_fixed_record_shards.py
-----------------------------
Write raw-binary shards for pure random-read benchmarking.

Each shard folder:
    data.bin   – 8 000 records × 64 KiB  (≈500 MiB)
    meta.json  – little JSON with layout info
"""

import argparse, json, os, uuid, random
from pathlib import Path

ROOT   = Path("/mnt/weka")          # change if needed
SHARDS = ROOT / "shards_64k"        # parent directory

RECORD_BYTES = 64 * 1024            # 65 536
RECORDS_PER_SHARD = 8_000           # 8 000 × 64 KiB = 524 288 000

def write_shard():
    shard_dir = SHARDS / uuid.uuid4().hex[:8]
    shard_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    with open(shard_dir / "data.bin", "wb", buffering=0) as f:
        for _ in range(RECORDS_PER_SHARD):
            f.write(rng.randbytes(RECORD_BYTES))

    with open(shard_dir / "meta.json", "w") as mj:
        json.dump(dict(records=RECORDS_PER_SHARD,
                       record_bytes=RECORD_BYTES), mj)
    return shard_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=50)
    args = ap.parse_args()

    SHARDS.mkdir(parents=True, exist_ok=True)
    for n in range(args.num_shards):
        print(f"[{n+1}/{args.num_shards}] writing shard …", end="", flush=True)
        write_shard();  print(" done")
    print("✅  wrote", args.num_shards, "shards to", SHARDS)

if __name__ == "__main__":
    main()