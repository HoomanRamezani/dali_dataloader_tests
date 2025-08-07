# DALI 64k Tests

## How to run

```bash
cd numpy_64k_tests

# Create 25 shards, 500mb each, 64kb rows
python create_synthetic_bin.py --num-shards 45

# Read random rows from WEKA file system from all shards, direct flag can be on or off depending on how you want to test read_ahead.
python dali_random_read_bin.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 16 \
  --shuffle \
  --direct 
```
