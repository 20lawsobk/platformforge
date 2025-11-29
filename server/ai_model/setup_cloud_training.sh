#!/bin/bash
# Cloud GPU Training Setup Script
# Run this on your cloud GPU instance to set up the training environment

set -e

echo "=============================================="
echo "Cloud GPU Training Setup"
echo "=============================================="

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: No NVIDIA GPU detected!"
    echo "Training will be very slow on CPU."
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Detect CUDA version and install appropriate PyTorch
echo ""
echo "Installing PyTorch with CUDA support..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "CUDA version: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == 11.* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio
    fi
else
    echo "CUDA not found, installing CPU version..."
    pip install torch torchvision torchaudio
fi

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip install numpy tqdm tiktoken

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python cloud_training.py --mode quick  # Quick test"
echo "  python cloud_training.py --mode full   # Full training"
echo ""
echo "For multi-GPU training:"
echo "  torchrun --nproc_per_node=4 cloud_training.py --distributed"
echo ""
