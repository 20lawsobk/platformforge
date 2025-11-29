#!/usr/bin/env python3
"""
Deploy training to RunPod Cloud GPU

This script packages and deploys training to RunPod's cloud GPUs.
Get your API key from: https://www.runpod.io/console/user/settings

Usage:
    export RUNPOD_API_KEY="your_api_key_here"
    python deploy_to_runpod.py --gpu "NVIDIA A100 80GB" --model-size large
"""

import os
import sys
import json
import time
import argparse
import base64
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip install requests")
    import requests


RUNPOD_API_URL = "https://api.runpod.io/graphql"

# Training script to run on cloud
TRAINING_SCRIPT = '''
#!/bin/bash
set -e

echo "=============================================="
echo "RunPod Cloud GPU Training"
echo "=============================================="
nvidia-smi

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm tiktoken

# Clone or download training files
cd /workspace

# Create training files
cat > transformer.py << 'TRANSFORMER_EOF'
{transformer_code}
TRANSFORMER_EOF

cat > tokenizer.py << 'TOKENIZER_EOF'
{tokenizer_code}
TOKENIZER_EOF

cat > training_data.py << 'DATA_EOF'
{training_data_code}
DATA_EOF

cat > cloud_training.py << 'CLOUD_EOF'
{cloud_training_code}
CLOUD_EOF

# Run training
python cloud_training.py --mode full --model-size {model_size} --epochs {epochs} --batch-size {batch_size}

echo "Training complete!"
echo "Checkpoints saved to /workspace/checkpoints/"

# Keep pod alive for file download
sleep 300
'''


def get_api_key():
    """Get RunPod API key from environment."""
    api_key = os.environ.get('RUNPOD_API_KEY')
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        print("Get your API key from: https://www.runpod.io/console/user/settings")
        sys.exit(1)
    return api_key


def read_file(filepath):
    """Read file content."""
    with open(filepath, 'r') as f:
        return f.read()


def create_pod(api_key, gpu_type, name="ai-training"):
    """Create a RunPod pod for training."""
    
    query = """
    mutation createPod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            runtime {
                uptimeInSeconds
            }
        }
    }
    """
    
    variables = {
        "input": {
            "cloudType": "SECURE",
            "gpuTypeId": gpu_type,
            "name": name,
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "volumeInGb": 50,
            "containerDiskInGb": 20,
            "minVcpuCount": 4,
            "minMemoryInGb": 16,
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(
        RUNPOD_API_URL,
        json={"query": query, "variables": variables},
        headers=headers
    )
    
    result = response.json()
    
    if "errors" in result:
        print(f"Error creating pod: {result['errors']}")
        return None
    
    return result.get("data", {}).get("podFindAndDeployOnDemand")


def get_pod_status(api_key, pod_id):
    """Get pod status."""
    
    query = """
    query getPod($podId: String!) {
        pod(input: { podId: $podId }) {
            id
            name
            runtime {
                uptimeInSeconds
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                }
            }
            desiredStatus
            lastStatusChange
        }
    }
    """
    
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(
        RUNPOD_API_URL,
        json={"query": query, "variables": {"podId": pod_id}},
        headers=headers
    )
    
    return response.json().get("data", {}).get("pod")


def list_available_gpus(api_key):
    """List available GPU types."""
    
    query = """
    query getGpuTypes {
        gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
        }
    }
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(
        RUNPOD_API_URL,
        json={"query": query},
        headers=headers
    )
    
    return response.json().get("data", {}).get("gpuTypes", [])


def main():
    parser = argparse.ArgumentParser(description='Deploy training to RunPod')
    parser.add_argument('--gpu', type=str, default='NVIDIA RTX A5000',
                        help='GPU type (e.g., "NVIDIA A100 80GB PCIe")')
    parser.add_argument('--model-size', type=str, default='medium',
                        choices=['tiny', 'small', 'medium', 'large', 'xlarge'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--list-gpus', action='store_true', help='List available GPUs')
    
    args = parser.parse_args()
    
    api_key = get_api_key()
    
    if args.list_gpus:
        print("\nAvailable GPUs:")
        print("-" * 60)
        gpus = list_available_gpus(api_key)
        for gpu in sorted(gpus, key=lambda x: x.get('memoryInGb', 0)):
            if gpu.get('secureCloud'):
                print(f"  {gpu['displayName']} ({gpu.get('memoryInGb', '?')}GB)")
        return
    
    print("\n" + "="*60)
    print("RUNPOD CLOUD DEPLOYMENT")
    print("="*60)
    print(f"GPU: {args.gpu}")
    print(f"Model Size: {args.model_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("="*60 + "\n")
    
    # Create pod
    print("Creating pod...")
    pod = create_pod(api_key, args.gpu, name=f"training-{args.model_size}")
    
    if pod:
        print(f"Pod created: {pod['id']}")
        print(f"Name: {pod['name']}")
        print("\nWaiting for pod to start...")
        
        # Wait for pod to be ready
        for i in range(60):
            status = get_pod_status(api_key, pod['id'])
            if status and status.get('desiredStatus') == 'RUNNING':
                print(f"\nPod is running!")
                runtime = status.get('runtime', {})
                ports = runtime.get('ports', [])
                for port in ports:
                    if port.get('isIpPublic'):
                        print(f"Connect: ssh root@{port['ip']} -p {port['publicPort']}")
                break
            print(".", end="", flush=True)
            time.sleep(5)
        
        print("\n")
        print("Next steps:")
        print("1. SSH into the pod")
        print("2. Upload training files or clone your repo")
        print("3. Run: python cloud_training.py --mode full")
        print("4. Download trained model from /workspace/checkpoints/")
    else:
        print("Failed to create pod. Check your API key and GPU availability.")


if __name__ == '__main__':
    main()
