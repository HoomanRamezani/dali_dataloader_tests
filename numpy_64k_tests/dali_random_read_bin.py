#!/usr/bin/env python3
"""
dali_random_read_fixed.py
-------------------------
Random-read benchmark: one 64 KiB read per sample,
no mmap, no compression, optional O_DIRECT.
"""
from __future__ import annotations
import argparse, bisect, glob, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from nvidia.dali import fn, pipeline_def, types

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
except ImportError:
    from nvidia.dali.plugin.base_iterator import DALIGenericIterator
    from nvidia.dali.plugin.pytorch import LastBatchPolicy

# ── open all shards ──────────────────────────────────────────────────────────
def open_shards(pattern: str, direct: bool):
    dirs = sorted(Path(p).parent for p in glob.glob(pattern))
    if not dirs:
        sys.exit(f"No matches for {pattern!r}")

    flag = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fds, rec_bytes, cum = [], 0, [0]
    for d in dirs:
        meta = json.load(open(d / "meta.json"))
        rec_bytes = meta["record_bytes"]
        fds.append(os.open(d / "data.bin", flag))
        cum.append(cum[-1] + meta["records"])
    return fds, rec_bytes, cum

# ── iterator ─────────────────────────────────────────────────────────────────
class ExtIter:
    def __init__(self, fds, rec_bytes, cum, idx, bs, rank, world):
        self.fds, self.rec_bytes, self.cum = fds, rec_bytes, cum
        s = len(idx)*rank//world; e = len(idx)*(rank+1)//world
        self.idx = idx[s:e];  self.bs = bs;  self.n = len(self.idx); self.i = 0

    def __iter__(self): self.i = 0; return self

    def _fetch(self, g):
        shard = bisect.bisect_right(self.cum, g)-1
        offset = (g - self.cum[shard]) * self.rec_bytes
        buf = os.pread(self.fds[shard], self.rec_bytes, offset)
        return np.frombuffer(buf, np.uint8)              # 64 KiB → (65536,)

    def __next__(self):
        if self.i >= self.n: raise StopIteration
        batch = [self._fetch(self.idx[(self.i+k)%self.n]) for k in range(self.bs)]
        self.i += self.bs
        return batch        

    next = __next__

# ── DALI pipeline ────────────────────────────────────────────────────────────
@pipeline_def
def pipe(eii):
    tensor = fn.external_source(source=eii, batch=True, dtype=types.UINT8)
    return tensor

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True,
        help='glob like "/mnt/weka/shards_64k/*/data.bin"')
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--direct", action="store_true", help="O_DIRECT bypass cache")
    args = ap.parse_args()

    fds, rec_bytes, cum = open_shards(args.shard, args.direct)
    total = cum[-1]
    print(f"📦 {len(fds)} shards → {total} samples  "
          f"(record = {rec_bytes//1024} KiB, O_DIRECT={args.direct})")

    idx = list(range(total))
    if args.shuffle: random.shuffle(idx)

    n_gpu = max(1, torch.cuda.device_count())
    pipes = []
    for gpu in range(n_gpu):
        eii = ExtIter(fds, rec_bytes, cum, idx, args.batch, gpu, n_gpu)
        pipes.append(pipe(batch_size=args.batch,
                          num_threads=args.workers,
                          device_id=gpu,
                          eii=eii))
    for p in pipes: p.build()

    it = DALIGenericIterator(pipes, ["record"], size=total,
                             last_batch_policy=LastBatchPolicy.PARTIAL)

    next(it)                            # warm-up
    t0 = time.perf_counter()
    for sample in next(it):             # one timed batch
        v = sample["record"].cpu();  _ = v.numpy()
    ms = (time.perf_counter()-t0)*1e3
    print(f"✅ {args.batch*n_gpu} samples in {ms:.1f} ms  "
          f"→ {ms/(args.batch*n_gpu):.3f} ms/sample")

if __name__ == "__main__":
    main()