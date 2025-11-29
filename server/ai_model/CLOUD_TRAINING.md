# Cloud GPU Training Guide

This guide explains how to train the custom AI code generation model on cloud GPU infrastructure for production-quality results.

## Hardware Requirements

| Model Size | GPU Memory | Recommended GPU | Training Time (50 epochs) |
|------------|------------|-----------------|--------------------------|
| tiny       | 4GB+       | RTX 3060        | ~30 minutes              |
| small      | 8GB+       | RTX 3070/4070   | ~1 hour                  |
| medium     | 16GB+      | RTX 4080/A10    | ~2-3 hours               |
| large      | 24GB+      | RTX 4090/A100   | ~4-6 hours               |
| xlarge     | 40GB+      | A100-40GB       | ~8-12 hours              |
| xxlarge    | 80GB+      | A100-80GB/H100  | ~24+ hours               |

## Cloud Provider Options

### AWS (Amazon Web Services)
```bash
# Launch EC2 instance with GPU
# Recommended: p4d.24xlarge (8x A100 40GB) or g5.xlarge (A10G 24GB)

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm numpy tiktoken

# Run training
python cloud_training.py --mode full --model-size large --gpu 0
```

### GCP (Google Cloud Platform)
```bash
# Launch Compute Engine with GPU
# Recommended: a2-highgpu-1g (A100 40GB) or n1-standard-8 + T4

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm numpy tiktoken

# Run training
python cloud_training.py --mode full --model-size large --gpu 0
```

### Lambda Labs (Most Cost-Effective)
```bash
# Launch instance with A100 or H100
# ~$1.10/hour for A100 40GB

# Clone your repository
git clone <your-repo>
cd <your-repo>/server/ai_model

# Install dependencies
pip install -r requirements_cloud.txt

# Run training
python cloud_training.py --mode full --model-size large
```

### RunPod / Vast.ai (Budget Option)
```bash
# Rent GPU by the hour
# A100 ~$0.80/hour, RTX 4090 ~$0.40/hour

# Setup
pip install -r requirements_cloud.txt

# Run training
python cloud_training.py --mode full --model-size medium
```

## Training Commands

### Quick Test (Verify Setup)
```bash
python cloud_training.py --mode quick --model-size small
```

### Full Training (Production)
```bash
python cloud_training.py \
    --mode full \
    --model-size large \
    --epochs 100 \
    --batch-size 32 \
    --lr 3e-4 \
    --max-length 2048 \
    --save-name my_code_model
```

### Multi-GPU Training
```bash
# 4 GPUs
torchrun --nproc_per_node=4 cloud_training.py \
    --mode full \
    --model-size xlarge \
    --distributed \
    --batch-size 16

# 8 GPUs (e.g., p4d.24xlarge)
torchrun --nproc_per_node=8 cloud_training.py \
    --mode full \
    --model-size xxlarge \
    --distributed \
    --batch-size 8
```

### Resume Training
```bash
python cloud_training.py \
    --mode full \
    --resume checkpoints/cloud_model/my_model/checkpoint_epoch_50
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-size` | medium | Model size (tiny/small/medium/large/xlarge/xxlarge) |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 16 | Batch size per GPU |
| `--lr` | 3e-4 | Learning rate |
| `--max-length` | 1024 | Maximum sequence length |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--gpu` | 0 | GPU device ID |
| `--distributed` | false | Enable multi-GPU training |
| `--no-amp` | false | Disable mixed precision |

## Expected Results

### Model Performance by Size

| Model Size | Parameters | Perplexity | Code Quality |
|------------|------------|------------|--------------|
| tiny       | ~1M        | 50-100     | Basic snippets |
| small      | ~10M       | 20-50      | Simple functions |
| medium     | ~50M       | 10-20      | Good completions |
| large      | ~125M      | 5-15       | Quality code |
| xlarge     | ~350M      | 3-10       | Production-ready |
| xxlarge    | ~1B        | 2-5        | Near-human quality |

### Training Data Recommendations

| Model Size | Minimum Samples | Recommended |
|------------|----------------|-------------|
| tiny       | 100            | 500+        |
| small      | 500            | 2,000+      |
| medium     | 2,000          | 10,000+     |
| large      | 10,000         | 50,000+     |
| xlarge     | 50,000         | 200,000+    |
| xxlarge    | 200,000        | 1,000,000+  |

## Adding More Training Data

To improve model quality, add more code samples:

```python
# In training_data.py, add to existing lists:

PYTHON_SAMPLES.extend([
    '''your code sample 1''',
    '''your code sample 2''',
    # ... more samples
])

# Or load from files:
import glob

for filepath in glob.glob('data/python/*.py'):
    with open(filepath) as f:
        PYTHON_SAMPLES.append(f.read())
```

### Data Sources

1. **Your Own Code**: Best quality, domain-specific
2. **Open Source Projects**: GitHub repos with permissive licenses
3. **Code Documentation**: Official docs with examples
4. **LeetCode Solutions**: Algorithm implementations
5. **Stack Overflow**: Verified answers (check license)

## Monitoring Training

### TensorBoard (Optional)
```bash
pip install tensorboard
tensorboard --logdir checkpoints/cloud_model/logs
```

### Watch GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Training Metrics

The training script saves:
- `training_history.json` - Loss/perplexity per epoch
- `best_model/` - Best validation checkpoint
- `final_model/` - Final epoch checkpoint

## Deploying Trained Model

After training, copy the checkpoint back to your Replit project:

```bash
# On cloud machine
tar -czvf trained_model.tar.gz checkpoints/cloud_model/best_model

# Transfer to your machine or cloud storage
scp trained_model.tar.gz user@your-server:~/

# Or upload to S3/GCS
aws s3 cp trained_model.tar.gz s3://your-bucket/models/
```

Then in your Replit project:

```python
from server.ai_model.inference import CodeGenerator

# Load the trained model
gen = CodeGenerator()
gen.load('path/to/best_model')

# Use for inference
result = gen.generate("def calculate_", max_tokens=100)
```

## Cost Estimates

| Provider | GPU | Hourly Cost | 100 Epochs (large) |
|----------|-----|-------------|-------------------|
| Lambda Labs | A100 40GB | $1.10 | ~$8-12 |
| RunPod | A100 40GB | $0.80 | ~$6-10 |
| Vast.ai | RTX 4090 | $0.40 | ~$3-5 |
| AWS | p4d.24xlarge | $32.77 | ~$250+ |
| GCP | a2-highgpu-1g | $3.67 | ~$30-45 |

**Recommendation**: Start with Lambda Labs or RunPod for cost-effective training.

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python cloud_training.py --batch-size 8

# Use gradient accumulation
python cloud_training.py --batch-size 4 --grad-accum 8

# Reduce sequence length
python cloud_training.py --max-length 512
```

### Slow Training
```bash
# Enable mixed precision (default)
python cloud_training.py  # AMP is on by default

# Increase batch size if GPU memory allows
python cloud_training.py --batch-size 32

# Use multiple GPUs
torchrun --nproc_per_node=4 cloud_training.py --distributed
```

### Loss Not Decreasing
- Add more training data
- Reduce learning rate: `--lr 1e-4`
- Increase warmup: modify `warmup_ratio` in code
- Check for data quality issues
