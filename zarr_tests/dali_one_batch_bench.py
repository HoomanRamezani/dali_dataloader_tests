#!/usr/bin/env python3
"""
dali_one_batch_bench.py – production-style DALI random-read benchmark (multi-shard)

• Reads *all* steps.zarr.pack files that match the --shard glob.
• Each 64-KB chunk is fetched through PackedDirectoryStore → Zarr → mmap → DALI.
• ExternalInputIterator builds one global shuffled index, then shards it
  across GPUs exactly like prod.

Example
--------
python dali_one_batch_bench.py \
  --shard "/mnt/weka/shards/00000000/*/steps.zarr.pack" \
  --batch 256 --workers 16 --shuffle
"""

from __future__ import annotations

import argparse
import bisect
import glob
import math
import pickle
import random
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Sequence

import numpy as np
import torch
import zarr
from nvidia.dali import fn, pipeline_def, types

# ──────────────────────────────── PackedDirectoryStore ─────────────────────────
class PackedDirectoryStore(zarr.storage.Store):
    """
    Read-only Store for .pack files produced by *pack_directory_store* (v2).

    Format:
        0-3   b'pack'
        4-7   uint32  version (==2)
        8-15  uint64  pickled manifest length
        16-.. pickled manifest  (dict {name: (offset, size)})
        data  concatenated file bytes
    """

    _erasable = False
    _writeable = False

    def __init__(self, path: str | Path):
        super().__init__()
        self._path = str(path)
        self._fh = open(self._path, "rb")

        sig = self._fh.read(4)
        if sig != b"pack":
            raise ValueError("Not a .pack file (missing signature)")

        version = struct.unpack("<I", self._fh.read(4))[0]
        if version != 2:
            raise ValueError(f"Unsupported .pack version {version} (expected 2)")

        (size,) = struct.unpack("<Q", self._fh.read(8))
        manifest_bytes = self._fh.read(size)
        self._manifest: Dict[str, tuple[int, int]] = pickle.loads(manifest_bytes)
        self._data_offset = self._fh.tell()  # start of raw blob

    # zarr Store API -----------------------------------------------------------
    def __getitem__(self, key: str) -> bytes:           # type: ignore[override]
        off, sz = self._manifest[key]
        self._fh.seek(self._data_offset + off)
        return self._fh.read(sz)

    def __contains__(self, key: str) -> bool:
        return key in self._manifest

    def __iter__(self) -> Iterator[str]:
        return iter(self._manifest)

    def __len__(self) -> int:
        return len(self._manifest)

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    # disable writes -----------------------------------------------------------
    def __setitem__(self, key, value):
        raise zarr.errors.ReadOnlyError()

    def __delitem__(self, key):
        raise zarr.errors.ReadOnlyError()
# ───────────────────────────────────────────────────────────────────────────────


# ───────────────────────── helper: open shards & collect lengths ──────────────
def open_all_shards(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No file matches glob {pattern!r}")

    roots, lengths, cum = [], [], [0]
    for p in files:
        store = PackedDirectoryStore(p)
        root = zarr.open_group(store, mode="r")
        roots.append(root)
        n = len(root["image"])
        lengths.append(n)
        cum.append(cum[-1] + n)

    total = cum[-1]
    return roots, lengths, cum, total, files


# ───────────────────────── ExternalInputIterator ──────────────────────────────
class ExternalInputIterator:
    def __init__(
        self,
        roots: Sequence[zarr.Group],
        cumlens: Sequence[int],
        global_indices: Sequence[int],
        batch: int,
        rank: int,
        world: int,
    ):
        self.roots = roots
        self.cumlens = cumlens
        self.idx_pool = global_indices
        self.bs = batch

        start = len(self.idx_pool) * rank // world
        end = len(self.idx_pool) * (rank + 1) // world
        self.local_idx = self.idx_pool[start:end]
        self.n = len(self.local_idx)

    def __iter__(self):
        self.i = 0
        return self

    def _fetch_sample(self, global_idx: int):
        shard_idx = bisect.bisect_right(self.cumlens, global_idx) - 1
        local = global_idx - self.cumlens[shard_idx]
        root = self.roots[shard_idx]
        return (
            np.asarray(root["image"][local]),
            np.asarray(root["actions"][local]),
            np.asarray(root["state"][local]),
        )

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        imgs, acts, sts = [], [], []
        for _ in range(self.bs):
            gidx = self.local_idx[self.i % self.n]
            im, ac, st = self._fetch_sample(gidx)
            imgs.append(im)
            acts.append(ac)
            sts.append(st)
            self.i += 1
        return imgs, acts, sts

    next = __next__  # Py2 compat


# ───────────────────────────── DALI pipeline ──────────────────────────────────
@pipeline_def
def zarr_pipe(eii, device_read_ahead: int):
    imgs, acts, sts = fn.external_source(
        source=eii,
        num_outputs=3,
        batch=True,
        dtype=[types.UINT8, types.FLOAT, types.FLOAT],
        device_buffer_queue_depth=device_read_ahead,
    )
    return imgs, acts, sts


# ──────────────────────────────── main ────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(description="Multi-shard random-read benchmark")
    pa.add_argument("--shard", required=True,
                    help='glob like "/mnt/weka/shards/00000000/*/steps.zarr.pack"')
    pa.add_argument("--batch",   type=int, default=256, help="batch per GPU")
    pa.add_argument("--workers", type=int, default=4,   help="DALI CPU threads / GPU")
    pa.add_argument("--shuffle", action="store_true",   help="shuffle global index")
    pa.add_argument("--device-read-ahead", type=int, default=1, metavar="N",
                    help="GPU prefetch queue depth")
    args = pa.parse_args()

    roots, lens, cum, total, files = open_all_shards(args.shard)
    print(f"📦  {len(files)} shards → {total} samples")

    global_indices = list(range(total))
    if args.shuffle:
        random.shuffle(global_indices)

    n_gpu = max(1, min(1, torch.cuda.device_count()))

    pipes = []
    for dev in range(n_gpu):
        eii = ExternalInputIterator(roots, cum, global_indices,
                                    args.batch, dev, n_gpu)
        pipes.append(
            zarr_pipe(
                batch_size=args.batch,
                num_threads=args.workers,
                device_id=dev,
                eii=eii,
                device_read_ahead=args.device_read_ahead,
            )
        )
    for p in pipes:
        p.build()

    try:
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    except ImportError:
        from nvidia.dali.plugin.base_iterator import DALIGenericIterator
        from nvidia.dali.plugin.pytorch import LastBatchPolicy

    dali_it = DALIGenericIterator(
        pipes,
        ["image", "actions", "state"],
        size=total,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        dynamic_shape=True,
    )

    # warm-up & timing
    next(dali_it)
    t0 = time.perf_counter()
    batch = next(dali_it)
    for samp in batch:
        [t.as_cpu() for t in samp.values()]
    ms = (time.perf_counter() - t0) * 1e3

    tot_bs = args.batch * n_gpu
    print(f"✅  1 batch ({tot_bs} samples, {len(files)} shards) "
          f"in {ms:.2f} ms → {ms / tot_bs:.3f} ms/sample")


if __name__ == "__main__":
    main()