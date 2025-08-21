#!/usr/bin/env python3
"""
H100 Cluster Configuration and Benchmarking Setup
Checks available GPUs, system specs, and prepares environment for performance testing
"""

import subprocess
import json
import os
import sys
from pathlib import Path
import torch
import psutil

class H100ClusterAnalyzer:
    def __init__(self):
        self.gpu_info = {}
        self.system_info = {}
        
    def check_gpu_availability(self):
        """Check available NVIDIA GPUs and their specifications"""
        try:
            # Get GPU info using nvidia-ml-py
            import pynvml
            pynvml.nvmlInit()
            
            gpu_count = pynvml.nvmlDeviceGetCount()
            print(f"Found {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                self.gpu_info[i] = {
                    'name': name,
                    'memory_total': memory_info.total // (1024**3),  # GB
                    'memory_free': memory_info.free // (1024**3),   # GB
                    'is_h100': 'H100' in name
                }
                
                print(f"GPU {i}: {name}")
                print(f"  Memory: {self.gpu_info[i]['memory_total']} GB total, {self.gpu_info[i]['memory_free']} GB free")
                
        except ImportError:
            print("Warning: pynvml not available. Install with: pip install nvidia-ml-py")
            # Fallback to nvidia-smi
            self._check_gpu_nvidia_smi()
            
    def _check_gpu_nvidia_smi(self):
        """Fallback GPU check using nvidia-smi"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            for i, line in enumerate(result.stdout.strip().split('\n')):
                if line:
                    name, mem_total, mem_free = line.split(', ')
                    self.gpu_info[i] = {
                        'name': name,
                        'memory_total': int(mem_total) // 1024,  # Convert MB to GB
                        'memory_free': int(mem_free) // 1024,
                        'is_h100': 'H100' in name
                    }
                    
        except subprocess.CalledProcessError:
            print("Error: nvidia-smi not available or no GPUs found")
            
    def check_system_specs(self):
        """Check system specifications"""
        self.system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total // (1024**3),
            'cuda_available': torch.cuda.is_available(),
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
        
        print(f"\nSystem Specifications:")
        print(f"CPU cores: {self.system_info['cpu_count']}")
        print(f"System RAM: {self.system_info['memory_gb']} GB")
        print(f"PyTorch version: {self.system_info['torch_version']}")
        print(f"CUDA available: {self.system_info['cuda_available']}")
        if self.system_info['cuda_version']:
            print(f"CUDA version: {self.system_info['cuda_version']}")
            
    def check_h100_compatibility(self):
        """Check if system has H100 GPUs and is ready for benchmarking"""
        h100_count = sum(1 for gpu in self.gpu_info.values() if gpu['is_h100'])
        
        print(f"\nH100 Compatibility Check:")
        print(f"H100 GPUs found: {h100_count}")
        
        if h100_count == 0:
            print("⚠️  No H100 GPUs detected. Benchmarks will run on available hardware.")
        else:
            print(f"✅ {h100_count} H100 GPU(s) ready for benchmarking")
            
        # Check minimum requirements
        min_memory = 40  # GB per GPU for large models
        suitable_gpus = sum(1 for gpu in self.gpu_info.values() 
                          if gpu['memory_total'] >= min_memory)
        
        print(f"GPUs with ≥{min_memory}GB memory: {suitable_gpus}")
        
        return h100_count > 0, suitable_gpus
        
    def setup_benchmark_environment(self):
        """Create directory structure and configuration files for benchmarking"""
        base_dir = Path("h100_benchmarks")
        
        directories = [
            "results",
            "logs", 
            "models",
            "datasets",
            "scripts"
        ]
        
        for dir_name in directories:
            (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
        # Create requirements.txt
        requirements = [
            "torch>=2.0.0",
            "torchvision",
            "transformers>=4.20.0",
            "datasets",
            "nvidia-ml-py",
            "psutil",
            "wandb",
            "tensorboard",
            "matplotlib",
            "seaborn",
            "pandas",
            "numpy"
        ]
        
        with open(base_dir / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))
            
        print(f"\n✅ Benchmark environment created in: {base_dir.absolute()}")
        print("Install dependencies with: pip install -r h100_benchmarks/requirements.txt")
        
    def generate_config(self):
        """Generate configuration file for benchmarks"""
        config = {
            "cluster_info": {
                "gpu_count": len(self.gpu_info),
                "h100_count": sum(1 for gpu in self.gpu_info.values() if gpu['is_h100']),
                "total_gpu_memory": sum(gpu['memory_total'] for gpu in self.gpu_info.values()),
                "system_memory": self.system_info['memory_gb'],
                "cpu_cores": self.system_info['cpu_count']
            },
            "benchmark_config": {
                "batch_sizes": [32, 64, 128, 256],
                "models_to_test": ["resnet50", "bert-base", "gpt2", "vit-base"],
                "scaling_tests": [1, 2, 4, 8] if len(self.gpu_info) >= 8 else list(range(1, len(self.gpu_info) + 1)),
                "metrics_to_track": ["throughput", "memory_usage", "gpu_utilization", "training_time"]
            }
        }
        
        with open("h100_benchmarks/cluster_config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        print("✅ Configuration saved to: h100_benchmarks/cluster_config.json")

def main():
    print("H100 Cluster Analysis and Setup")
    print("=" * 40)
    
    analyzer = H100ClusterAnalyzer()
    
    # Check GPU availability
    analyzer.check_gpu_availability()
    
    # Check system specs
    analyzer.check_system_specs()
    
    # Check H100 compatibility
    has_h100, suitable_gpus = analyzer.check_h100_compatibility()
    
    # Setup benchmark environment
    analyzer.setup_benchmark_environment()
    
    # Generate configuration
    analyzer.generate_config()
    
    print("\n" + "=" * 40)
    print("Setup complete! Next steps:")
    print("1. Install dependencies: pip install -r h100_benchmarks/requirements.txt")
    print("2. Review cluster_config.json")
    print("3. Run benchmark scripts from h100_benchmarks/scripts/")

if __name__ == "__main__":
    main()
