#!/usr/bin/env python3
"""
Deploy training to Lambda Labs Cloud GPU

This script packages and deploys training to Lambda Labs cloud GPUs.
Get your API key from: https://cloud.lambdalabs.com/api-keys

Usage:
    export LAMBDA_API_KEY="your_api_key_here"
    python deploy_to_lambda.py --gpu gpu_1x_a100 --model-size large
"""

import os
import sys
import json
import time
import argparse

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip install requests")
    import requests


LAMBDA_API_URL = "https://cloud.lambdalabs.com/api/v1"


def get_api_key():
    """Get Lambda Labs API key from environment."""
    api_key = os.environ.get('LAMBDA_API_KEY')
    if not api_key:
        print("Error: LAMBDA_API_KEY environment variable not set")
        print("Get your API key from: https://cloud.lambdalabs.com/api-keys")
        sys.exit(1)
    return api_key


def list_instance_types(api_key):
    """List available instance types."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{LAMBDA_API_URL}/instance-types", headers=headers)
    return response.json().get("data", {})


def list_running_instances(api_key):
    """List running instances."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{LAMBDA_API_URL}/instances", headers=headers)
    return response.json().get("data", [])


def launch_instance(api_key, instance_type, region=None, ssh_key_names=None, name=None):
    """Launch a new instance."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "instance_type_name": instance_type,
        "region_name": region or "us-west-1",
        "ssh_key_names": ssh_key_names or [],
        "name": name or "ai-training"
    }
    
    response = requests.post(
        f"{LAMBDA_API_URL}/instance-operations/launch",
        headers=headers,
        json=data
    )
    
    return response.json()


def get_instance(api_key, instance_id):
    """Get instance details."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{LAMBDA_API_URL}/instances/{instance_id}", headers=headers)
    return response.json().get("data", {})


def terminate_instance(api_key, instance_id):
    """Terminate an instance."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{LAMBDA_API_URL}/instance-operations/terminate",
        headers=headers,
        json={"instance_ids": [instance_id]}
    )
    
    return response.json()


def list_ssh_keys(api_key):
    """List SSH keys."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{LAMBDA_API_URL}/ssh-keys", headers=headers)
    return response.json().get("data", [])


def add_ssh_key(api_key, name, public_key):
    """Add an SSH key."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{LAMBDA_API_URL}/ssh-keys",
        headers=headers,
        json={"name": name, "public_key": public_key}
    )
    
    return response.json()


def main():
    parser = argparse.ArgumentParser(description='Deploy training to Lambda Labs')
    parser.add_argument('--gpu', type=str, default='gpu_1x_a100',
                        help='Instance type (e.g., gpu_1x_a100, gpu_8x_a100)')
    parser.add_argument('--region', type=str, default='us-west-1',
                        help='Region (us-west-1, us-east-1, etc.)')
    parser.add_argument('--model-size', type=str, default='large',
                        choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--ssh-key', type=str, help='SSH key name to use')
    parser.add_argument('--list-gpus', action='store_true', help='List available instance types')
    parser.add_argument('--list-instances', action='store_true', help='List running instances')
    parser.add_argument('--terminate', type=str, help='Terminate instance by ID')
    
    args = parser.parse_args()
    
    api_key = get_api_key()
    
    if args.list_gpus:
        print("\nAvailable Instance Types:")
        print("-" * 70)
        types = list_instance_types(api_key)
        for name, info in types.items():
            desc = info.get('instance_type', {}).get('description', '')
            price = info.get('instance_type', {}).get('price_cents_per_hour', 0) / 100
            regions = info.get('regions_with_capacity_available', [])
            available = len(regions) > 0
            status = "Available" if available else "Unavailable"
            print(f"  {name}: {desc}")
            print(f"    Price: ${price:.2f}/hr | Status: {status}")
            if available:
                print(f"    Regions: {', '.join([r['name'] for r in regions])}")
            print()
        return
    
    if args.list_instances:
        print("\nRunning Instances:")
        print("-" * 70)
        instances = list_running_instances(api_key)
        if not instances:
            print("  No running instances")
        for inst in instances:
            print(f"  ID: {inst['id']}")
            print(f"  Name: {inst.get('name', 'N/A')}")
            print(f"  Type: {inst.get('instance_type', {}).get('name', 'N/A')}")
            print(f"  IP: {inst.get('ip', 'N/A')}")
            print(f"  Status: {inst.get('status', 'N/A')}")
            print()
        return
    
    if args.terminate:
        print(f"Terminating instance {args.terminate}...")
        result = terminate_instance(api_key, args.terminate)
        print(f"Result: {result}")
        return
    
    print("\n" + "="*60)
    print("LAMBDA LABS CLOUD DEPLOYMENT")
    print("="*60)
    print(f"Instance Type: {args.gpu}")
    print(f"Region: {args.region}")
    print(f"Model Size: {args.model_size}")
    print(f"Epochs: {args.epochs}")
    print("="*60 + "\n")
    
    # Check SSH keys
    ssh_keys = list_ssh_keys(api_key)
    if not ssh_keys:
        print("WARNING: No SSH keys found!")
        print("Add an SSH key at: https://cloud.lambdalabs.com/ssh-keys")
        print("Or use: python deploy_to_lambda.py --add-ssh-key")
        return
    
    ssh_key_names = [args.ssh_key] if args.ssh_key else [ssh_keys[0]['name']]
    print(f"Using SSH key: {ssh_key_names[0]}")
    
    # Launch instance
    print("\nLaunching instance...")
    result = launch_instance(
        api_key,
        args.gpu,
        region=args.region,
        ssh_key_names=ssh_key_names,
        name=f"training-{args.model_size}"
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        if "capacity" in str(result).lower():
            print("\nNo capacity available. Try:")
            print("  1. Different region: --region us-east-1")
            print("  2. Different GPU: --gpu gpu_1x_a10")
            print("  3. Check availability: --list-gpus")
        return
    
    instance_ids = result.get("data", {}).get("instance_ids", [])
    if not instance_ids:
        print(f"Failed to launch: {result}")
        return
    
    instance_id = instance_ids[0]
    print(f"Instance launched: {instance_id}")
    
    # Wait for instance to be ready
    print("Waiting for instance to be ready...")
    for i in range(60):
        instance = get_instance(api_key, instance_id)
        status = instance.get('status', 'unknown')
        ip = instance.get('ip', None)
        
        if status == 'active' and ip:
            print(f"\n{'='*60}")
            print("INSTANCE READY!")
            print("="*60)
            print(f"Instance ID: {instance_id}")
            print(f"IP Address: {ip}")
            print(f"SSH: ssh ubuntu@{ip}")
            print()
            print("To start training, SSH in and run:")
            print("  git clone <your-repo>")
            print("  cd <your-repo>/server/ai_model")
            print(f"  python cloud_training.py --mode full --model-size {args.model_size}")
            print()
            print("To terminate when done:")
            print(f"  python deploy_to_lambda.py --terminate {instance_id}")
            print("="*60)
            break
        
        print(".", end="", flush=True)
        time.sleep(5)
    else:
        print("\nTimeout waiting for instance. Check dashboard.")


if __name__ == '__main__':
    main()
