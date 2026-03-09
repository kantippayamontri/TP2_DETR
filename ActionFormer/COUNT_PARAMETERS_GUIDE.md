# Model Parameter Count Guide

This guide explains how to count the number of parameters in the TP2_DETR/GAP model.

## Quick Start

```bash
# Activate conda environment
conda activate gap

# Navigate to ActionFormer directory
cd TP2_DETR/ActionFormer

# Count parameters (basic)
python count_parameters.py --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml

# Count parameters with detailed breakdown
python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --show_details
```

## Command-Line Arguments

### Required Arguments

- `--cfg_path`: Path to configuration YAML file

### Optional Arguments

- `--show_details`: Show detailed parameter breakdown by layer
- `--resume`: Load checkpoint to verify parameter count (optional)

## Usage Examples

### Example 1: Basic Parameter Count (Thumos14)

```bash
python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml
```

**Output:**

```
======================================================================
Model Parameter Count
======================================================================
Total parameters: 45,123,456 (45.12M)
Trainable parameters: 45,123,456 (45.12M)
Non-trainable parameters: 0
======================================================================

Parameter Count by Module:
----------------------------------------------------------------------
Module                         Total      Trainable         Frozen
----------------------------------------------------------------------
transformer                15,234,567     15,234,567              0
backbone                   12,345,678     12,345,678              0
text_encoder               10,123,456     10,123,456              0
...
----------------------------------------------------------------------
```

### Example 2: Detailed Breakdown

```bash
python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --show_details
```

This will show every layer with its shape and parameter count.

### Example 3: ActivityNet13 Model

```bash
python count_parameters.py \
    --cfg_path ./config/ActivityNet13_CLIP_zs_50.yaml
```

### Example 4: With Checkpoint Verification

```bash
python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    --resume ../../ckpt/Thumos14/best_model.pkl
```

This loads the checkpoint to verify the parameter count matches.

## Understanding the Output

### Main Summary

```
Total parameters: 45,123,456 (45.12M)
Trainable parameters: 45,123,456 (45.12M)
Non-trainable parameters: 0
```

- **Total parameters**: All parameters in the model
- **Trainable parameters**: Parameters that will be updated during training
- **Non-trainable parameters**: Frozen parameters (not updated during training)

### Module Breakdown

```
Module                         Total      Trainable         Frozen
----------------------------------------------------------------------
transformer                15,234,567     15,234,567              0
backbone                   12,345,678     12,345,678              0
text_encoder               10,123,456     10,123,456              0
```

Shows parameter count for each major component:

- **transformer**: Deformable DETR transformer
- **backbone**: Feature extraction backbone
- **text_encoder**: CLIP text encoder
- Other modules (decoder, embeddings, etc.)

### Component Statistics

```
Backbone parameters: 12,345,678 (12.35M)
Transformer parameters: 15,234,567 (15.23M)
Text encoder parameters: 10,123,456 (10.12M)
```

Breakdown by major architectural components.

## Comparing Different Configurations

To compare parameter counts across different configurations:

```bash
# Count parameters for different configs
python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    > params_thumos14_50.txt

python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_75_8frame.yaml \
    > params_thumos14_75.txt

# Compare
diff params_thumos14_50.txt params_thumos14_75.txt
```

## Tips

1. **Use CPU**: The script automatically uses CPU for parameter counting (faster and uses less memory)
2. **No checkpoint needed**: You don't need a trained checkpoint to count parameters
3. **Detailed view**: Use `--show_details` to see every layer (useful for debugging)
4. **Module breakdown**: The module breakdown helps identify which components are largest

## Common Questions

### Q: Why do I need to count parameters?

**A:** Parameter count is important for:

- Comparing model complexity
- Understanding memory requirements
- Reporting in papers/documentation
- Debugging model architecture

### Q: What's a typical parameter count for TP2_DETR?

**A:** Typical range is 40-60M parameters, depending on:

- Backbone architecture
- Transformer configuration
- Number of decoder layers
- Hidden dimensions

### Q: How do I reduce parameter count?

**A:** You can reduce parameters by:

- Using a smaller backbone
- Reducing transformer layers
- Decreasing hidden dimensions
- Using fewer attention heads

### Q: What if trainable ≠ total?

**A:** This means some parameters are frozen (non-trainable), which happens when:

- Fine-tuning with frozen backbone
- Using pre-trained components
- Selective layer training

## Troubleshooting

### Issue: Module not found error

**Solution**: Activate conda environment

```bash
conda activate gap
```

### Issue: Config file not found

**Solution**: Check the config path

```bash
ls -la ./config/
python count_parameters.py --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml
```

### Issue: Out of memory

**Solution**: The script uses CPU by default, so this shouldn't happen. If it does:

```bash
# Close other applications
# Or use a machine with more RAM
```

## Advanced Usage

### Save Results to File

```bash
python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    > model_parameters.txt
```

### Compare with Other Models

```bash
# Count parameters for your model
python count_parameters.py --cfg_path config1.yaml > model1_params.txt

# Count parameters for another model
python count_parameters.py --cfg_path config2.yaml > model2_params.txt

# Compare
echo "Model 1:"
grep "Total parameters" model1_params.txt
echo "Model 2:"
grep "Total parameters" model2_params.txt
```

### Extract Just the Number

```bash
# Get just the parameter count
python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    | grep "Total parameters" \
    | awk '{print $3}'
```

## Example: Complete Workflow

```bash
# Step 1: Activate environment
conda activate gap

# Step 2: Navigate to directory
cd TP2_DETR/ActionFormer

# Step 3: Count parameters
python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml

# Step 4: Save to file
python count_parameters.py \
    --cfg_path ./config/Thumos14_CLIP_zs_50_8frame.yaml \
    > thumos14_parameters.txt

# Step 5: View results
cat thumos14_parameters.txt
```

## Integration with Papers/Reports

When reporting parameter counts in papers:

```
Our TP2_DETR model contains 45.12M parameters, with:
- Backbone: 12.35M parameters
- Transformer: 15.23M parameters
- Text Encoder: 10.12M parameters
- Other components: 7.42M parameters
```

## Contact

For issues or questions about parameter counting, please refer to the main project documentation.
