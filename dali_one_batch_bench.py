#!/usr/bin/env python3
"""
dali_one_batch_bench.py
---------------------------------
Benchmark ONE batch read from a .zarr.pack shard using DALI ExternalSource,
closely mirroring NVIDIA’s ExternalSource-with-PyTorch tutorial.

• Zarr ≥ 3 shards
• nvidia-dali-cuda*   • torch   • numpy   • numcodecs
"""

import argparse, glob, tarfile, tempfile, time, math, pathlib, numpy as np, zarr, torch
from nvidia.dali import fn, types, pipeline_def

# ────────── DALI ↔ PyTorch iterator (new & old wheels) ────────────────────────
try:  # DALI ≥ 1.30
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
except ImportError:  # older wheels
    from nvidia.dali.plugin.base_iterator import DALIGenericIterator
    from nvidia.dali.plugin.pytorch import LastBatchPolicy
# -----------------------------------------------------------------------------

def resolve_shard(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file matches pattern: {pattern}")
    return max(files, key=lambda p: pathlib.Path(p).stat().st_mtime)

def open_packed_zarr(pack_path: str) -> zarr.Group:
    tmp = pathlib.Path(
        tempfile.mkdtemp(dir="/dev/shm" if pathlib.Path("/dev/shm").exists() else None)
    )
    with tarfile.open(pack_path, "r") as tf:
        tf.extractall(tmp)                # → tmp/steps.zarr/
    return zarr.open_group(tmp / "steps.zarr", mode="r")

class ExternalInputIterator:
    def __init__(self, zroot, batch, rank, world):
        self.img, self.act, self.state = zroot["image"], zroot["actions"], zroot["state"]
        self.bs, self.rank, self.world = batch, rank, world
        n = self.img.shape[0]
        self.indices = list(range(n * rank // world, n * (rank + 1) // world))
        self._len = len(self.indices)

    def __iter__(self):
        self.i = 0
        np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.i >= self._len:
            self.__iter__(); raise StopIteration
        ims, acts, sts = [], [], []
        for _ in range(self.bs):
            idx = self.indices[self.i % self._len]
            ims .append(np.asarray(self.img[idx]))
            acts.append(np.asarray(self.act[idx]))
            sts .append(np.asarray(self.state[idx]))
            self.i += 1
        return ims, acts, sts

    next = __next__
    def __len__(self): return self._len

@pipeline_def
def zarr_pipe(eii):
    imgs, acts, sts = fn.external_source(
        source=eii,
        num_outputs=3,
        batch=True,
        dtype=[types.UINT8, types.FLOAT, types.FLOAT],
    )
    return imgs, acts, sts

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--shard", required=True, help="steps.zarr.pack (glob OK)")
    pa.add_argument("--batch", type=int, default=256)
    pa.add_argument("--workers", type=int, default=4)
    args = pa.parse_args()

    shard = resolve_shard(args.shard)
    print(f"🗂  Using shard: {shard}")
    zroot = open_packed_zarr(shard)

    n_gpu = torch.cuda.device_count() or 1
    image_shape = zroot["image"].shape
    num_samples = image_shape[0]
    sample_shape = image_shape[1:]
    total_batch = args.batch * n_gpu

    print(f"Detected GPUs           : {n_gpu}")
    print(f"Image dataset shape     : {image_shape}")
    print(f"Actions dataset shape   : {zroot['actions'].shape}")
    print(f"State dataset shape     : {zroot['state'].shape}")
    print(f"Per-sample image shape  : {sample_shape}")
    print(f"Batch size per GPU      : {args.batch}")
    print(f"Total batch size        : {total_batch}")
    print(f"Total dataset samples   : {num_samples}")

    if num_samples < n_gpu:
        print(f"⚠️  WARNING: Fewer samples ({num_samples}) than GPUs ({n_gpu})! "
              "At least some pipelines will have no data to process.")
    if num_samples < total_batch:
        print(f"⚠️  WARNING: Dataset smaller than a single step "
              f"({num_samples} < {total_batch})! Only one batch will be processed.")

    batches_per_epoch = math.ceil(num_samples / total_batch)
    print(f"Calculated batches/epoch: {batches_per_epoch}")

    pipes = []
    for dev in range(n_gpu):
        eii = ExternalInputIterator(zroot, args.batch, dev, n_gpu)
        pipes.append(
            zarr_pipe(
                batch_size=args.batch,
                num_threads=args.workers,
                device_id=dev,
                eii=eii,
            )
        )
    for p in pipes: p.build()

    dali_it = DALIGenericIterator(
        pipes,
        output_map=["image", "actions", "state"],
        size=num_samples,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        dynamic_shape=True,
    )

    if batches_per_epoch > 1:
        print("Doing warm-up batch (more than one batch per epoch)...")
        next(dali_it)
    else:
        print("Skipping warm-up batch (only one batch per epoch)...")

    t0 = time.perf_counter()
    try:
        batch = next(dali_it)
    except StopIteration:
        print("Caught StopIteration after warm-up, resetting iterator and retrying...")
        dali_it.reset()
        batch = next(dali_it)
    ms = (time.perf_counter() - t0) * 1e3

    for idx, sample in enumerate(batch):
        print(f"  Batch[{idx}] keys: {list(sample.keys())}")
        for key in sample:
            print(f"    {key} shape: {sample[key].shape}")

    print(f"✅  Read 1 batch ({args.batch} × {n_gpu}) in {ms:.2f} ms "
          f"({ms/(args.batch*n_gpu):.3f} ms/sample)")

if __name__ == "__main__":
    main()