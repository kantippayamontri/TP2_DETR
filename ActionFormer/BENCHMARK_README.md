# ActionFormer Inference Speed Benchmark

## ✅ Created Files

1. **benchmark.py** - Main benchmark script
2. **BENCHMARK_GUIDE.md** - Detailed usage guide

## 🚀 Quick Start

```bash
# 1. Activate conda environment
conda activate gap

# 2. Navigate to ActionFormer directory
cd TP2_DETR/ActionFormer

# 3. Run benchmark (example with Thumos14)
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume /path/to/your/checkpoint.pkl \
    --num_videos 100 \
    --warm_up 5
```

## 📋 Required Arguments

- `--cfg_path`: Path to config YAML file (e.g., `./config/Thumos14_CLIP_zs_50_8frame.yaml`)
- `--resume`: Path to checkpoint file (e.g., `../../ckpt/Thumos14/best_model.pkl`)

## 📊 What It Measures

The benchmark script measures:

- **FPS (videos/second)**: How many videos can be processed per second
- **Average Latency**: Time to process one batch (in milliseconds)
- **Throughput**: Videos processed per second
- **Timing Statistics**: Mean, std, min, max, median inference times

## 🎯 Example Output

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

## 🔧 Common Options

```bash
# Benchmark 200 videos with 10 warm-up iterations
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume checkpoint.pkl \
    --num_videos 200 \
    --warm_up 10

# Save results to JSON file
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume checkpoint.pkl \
    --output_file benchmark_results.json

# Test with different batch sizes
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume checkpoint.pkl \
    --batch_size 4

# Run on CPU
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume checkpoint.pkl \
    --device cpu
```

## 📖 Full Documentation

See **BENCHMARK_GUIDE.md** for:

- Detailed usage examples
- All command-line options
- Troubleshooting tips
- Advanced profiling techniques

## ⚠️ Important Notes

1. **Checkpoint Path**: Make sure your checkpoint file exists at the specified path
2. **Config File**: Use the correct config file for your dataset (Thumos14 or ActivityNet13)
3. **Warm-up**: Always use warm-up iterations (5-10) for accurate results
4. **CUDA Sync**: The script automatically handles CUDA synchronization for accurate timing

## 🐛 Troubleshooting

### "Checkpoint not found" error

```bash
# Check if checkpoint exists
ls -la /path/to/checkpoint.pkl

# Use absolute path
python benchmark.py --resume /full/path/to/checkpoint.pkl ...
```

### Out of memory error

```bash
# Reduce batch size
python benchmark.py --batch_size 1 ...

# Or reduce number of videos
python benchmark.py --num_videos 50 ...
```

### Module not found error

```bash
# Make sure conda environment is activated
conda activate gap
```

## 📝 Example: Complete Benchmark Run

```bash
# Step 1: Activate environment
conda activate gap

# Step 2: Navigate to directory
cd TP2_DETR/ActionFormer

# Step 3: Check checkpoint exists
ls -la ../../ckpt/Thumos14/

# Step 4: Run benchmark
python benchmark.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume ../../ckpt/Thumos14/best_model.pkl \
    --num_videos 100 \
    --warm_up 5 \
    --output_file thumos14_benchmark.json

# Step 5: View results
cat thumos14_benchmark.json
```

## 🎓 Understanding the Results

- **Higher FPS = Faster inference**
- **Lower latency = Faster per-video processing**
- **Std time** shows consistency (lower is more consistent)
- Compare results across different:
  - Models
  - Batch sizes
  - Datasets
  - Hardware configurations

For more details, see **BENCHMARK_GUIDE.md**.
