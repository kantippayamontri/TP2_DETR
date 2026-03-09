# ActionFormer Benchmark Guide

This guide explains how to use the benchmark script to measure inference speed of the TP2_DETR ActionFormer model.

## Quick Start

```bash
# Activate conda environment
conda activate gap

# Navigate to ActionFormer directory
cd TP2_DETR/ActionFormer

# Run benchmark with default settings
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume ../../ckpt/Thumos14/best_model.pkl
```

## Command-Line Arguments

### Required Arguments

- `--cfg_path`: Path to configuration YAML file
- `--resume`: Path to checkpoint file (.pkl)

### Benchmark-Specific Arguments

- `--num_videos`: Number of videos to benchmark (default: 100)
- `--warm_up`: Number of warm-up iterations (default: 5)
- `--output_file`: Path to save results JSON (optional)

### Model Arguments (from config or command line)

- `--dataset_name`: Dataset name (Thumos14 or ActivityNet13)
- `--batch_size`: Batch size for inference (default: 1)
- `--device`: Device to use (cuda or cpu)
- `--num_workers`: Number of data loading workers

## Usage Examples

### Example 1: Basic Benchmark (Thumos14)

```bash
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume ../../ckpt/Thumos14/best_model.pkl \
    --num_videos 100 \
    --warm_up 5
```

### Example 2: ActivityNet13 Benchmark

```bash
python benchmark.py \
    --cfg_path ./config/ActivityNet13_CLIP_zs_50.yaml \
    --resume ../../ckpt/ActivityNet13/best_model.pkl \
    --num_videos 200 \
    --warm_up 10
```

### Example 3: Save Results to File

```bash
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume ../../ckpt/Thumos14/best_model.pkl \
    --num_videos 100 \
    --output_file benchmark_results.json
```

### Example 4: Different Batch Sizes

```bash
# Batch size 1 (default)
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume ../../ckpt/Thumos14/best_model.pkl \
    --batch_size 1

# Batch size 4
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume ../../ckpt/Thumos14/best_model.pkl \
    --batch_size 4
```

### Example 5: CPU Benchmark

```bash
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume ../../ckpt/Thumos14/best_model.pkl \
    --device cpu \
    --num_videos 50
```

## Output Explanation

The benchmark script outputs:

### During Execution

```
======================================================================
Starting Benchmark
======================================================================
Device: cuda
Number of videos: 100
Warm-up iterations: 5
Batch size: 1
======================================================================

Warm-up phase (5 iterations)...
  Warm-up: 5/5
✓ Warm-up complete

Benchmark phase...
----------------------------------------------------------------------
Batch 50: 0.1234s (8.10 videos/s) | Avg: 0.1250s | Progress: 50/100
```

### Final Results

```
======================================================================
Benchmark Results
======================================================================
Configuration:
  - Dataset: Thumos14
  - Model: TP2_DETR
  - Device: cuda
  - Batch size: 1
  - Number of videos: 100

Timing Statistics:
  - Total time: 12.5000s
  - Mean time per batch: 0.1250s
  - Std time per batch: 0.0050s
  - Min time per batch: 0.1150s
  - Max time per batch: 0.1450s
  - Median time per batch: 0.1240s

Performance Metrics:
  - FPS (videos/second): 8.00
  - Average latency: 125.00ms
  - Throughput: 8.00 videos/s
======================================================================
```

## Metrics Explained

- **FPS (Frames Per Second)**: Number of videos processed per second
- **Average Latency**: Average time to process one batch (in milliseconds)
- **Throughput**: Number of videos processed per second (same as FPS for batch_size=1)
- **Mean/Std/Min/Max/Median Time**: Statistical measures of inference time per batch

## Tips for Accurate Benchmarking

1. **Warm-up**: Always use warm-up iterations (5-10) to avoid cold-start overhead
2. **Sufficient samples**: Use at least 100 videos for stable statistics
3. **GPU synchronization**: The script automatically handles CUDA synchronization
4. **Consistent environment**: Run benchmarks with the same:
   - GPU model
   - CUDA version
   - PyTorch version
   - Batch size
5. **Multiple runs**: Run the benchmark 3 times and average the results

## Comparing Different Configurations

To compare different models or configurations:

```bash
# Benchmark configuration 1
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume ../../ckpt/Thumos14/model_v1.pkl \
    --output_file results_v1.json

# Benchmark configuration 2
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_75_8frame.yaml \
    --resume ../../ckpt/Thumos14/model_v2.pkl \
    --output_file results_v2.json

# Compare results
python -c "
import json
with open('results_v1.json') as f1, open('results_v2.json') as f2:
    r1, r2 = json.load(f1), json.load(f2)
    print(f'V1 FPS: {r1[\"statistics\"][\"fps\"]:.2f}')
    print(f'V2 FPS: {r2[\"statistics\"][\"fps\"]:.2f}')
"
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or number of videos

```bash
python benchmark.py ... --batch_size 1 --num_videos 50
```

### Issue: Checkpoint not found

**Solution**: Check the checkpoint path

```bash
ls -la ../../ckpt/Thumos14/
python benchmark.py ... --resume /full/path/to/checkpoint.pkl
```

### Issue: CUDA out of memory

**Solution**: Use CPU or reduce batch size

```bash
python benchmark.py ... --device cpu
# or
python benchmark.py ... --batch_size 1
```

## Advanced Usage

### Profiling with PyTorch Profiler

For detailed profiling, you can modify the benchmark script to use PyTorch's profiler:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    outputs = model(samples, classes, description_dict, targets, epoch=0)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Memory Usage Monitoring

To monitor GPU memory usage during benchmarking:

```bash
# In another terminal
watch -n 0.5 nvidia-smi
```

## Contact

For issues or questions about the benchmark script, please refer to the main project documentation.
