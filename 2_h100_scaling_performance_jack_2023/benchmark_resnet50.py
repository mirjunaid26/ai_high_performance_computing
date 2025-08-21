#!/usr/bin/env python3
"""
ResNet-50 H100 GPU Scaling Benchmark
Measures training performance across different GPU configurations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
import time
import json
import argparse
from pathlib import Path
import os

class ResNet50Benchmark:
    def __init__(self, batch_size=256, num_epochs=5, image_size=224):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.results = {}
        
    def setup_distributed(self, rank, world_size):
        """Initialize distributed training"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        
    def cleanup_distributed(self):
        """Clean up distributed training"""
        dist.destroy_process_group()
        
    def create_synthetic_dataset(self, num_samples=50000):
        """Create synthetic ImageNet-like dataset for benchmarking"""
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples, image_size, num_classes=1000):
                self.num_samples = num_samples
                self.image_size = image_size
                self.num_classes = num_classes
                
            def __len__(self):
                return self.num_samples
                
            def __getitem__(self, idx):
                # Generate random image and label
                image = torch.randn(3, self.image_size, self.image_size)
                label = torch.randint(0, self.num_classes, (1,)).item()
                return image, label
                
        return SyntheticDataset(num_samples, self.image_size)
        
    def create_model(self):
        """Create ResNet-50 model"""
        model = torchvision.models.resnet50(weights=None, num_classes=1000)
        return model
        
    def benchmark_single_gpu(self, gpu_id=0):
        """Benchmark ResNet-50 on single GPU"""
        print(f"Running single GPU benchmark on GPU {gpu_id}")
        
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        # Create model and move to GPU
        model = self.create_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        
        # Create dataset and dataloader
        dataset = self.create_synthetic_dataset()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
        
        # Warmup
        model.train()
        for i, (images, labels) in enumerate(dataloader):
            if i >= 5:  # 5 warmup iterations
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        total_samples = 0
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_samples += images.size(0)
                
                if batch_idx >= 100:  # Limit iterations for benchmark
                    break
                    
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{self.num_epochs}: {epoch_time:.2f}s")
            
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # Calculate metrics
        throughput = total_samples / total_time
        memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
        
        results = {
            'gpu_count': 1,
            'batch_size': self.batch_size,
            'total_time': total_time,
            'throughput': throughput,
            'samples_per_second': throughput,
            'memory_gb': memory_used,
            'gpu_utilization': self._get_gpu_utilization(gpu_id)
        }
        
        print(f"Single GPU Results:")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Memory usage: {memory_used:.2f} GB")
        
        return results
        
    def benchmark_multi_gpu(self, world_size):
        """Setup multi-GPU benchmark"""
        mp.spawn(self._benchmark_multi_gpu_worker, 
                args=(world_size,), nprocs=world_size, join=True)
        
    def _benchmark_multi_gpu_worker(self, rank, world_size):
        """Worker function for multi-GPU benchmark"""
        print(f"Running multi-GPU benchmark: GPU {rank}/{world_size}")
        
        # Setup distributed training
        self.setup_distributed(rank, world_size)
        
        device = torch.device(f'cuda:{rank}')
        
        # Create model and wrap with DDP
        model = self.create_model().to(device)
        model = DDP(model, device_ids=[rank])
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        
        # Create dataset with distributed sampler
        dataset = self.create_synthetic_dataset()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                              sampler=sampler, num_workers=4, pin_memory=True)
        
        # Warmup
        model.train()
        for i, (images, labels) in enumerate(dataloader):
            if i >= 5:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Synchronize all processes
        dist.barrier()
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        total_samples = 0
        
        for epoch in range(self.num_epochs):
            sampler.set_epoch(epoch)
            epoch_start = time.time()
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_samples += images.size(0)
                
                if batch_idx >= 100:  # Limit iterations
                    break
                    
            if rank == 0:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1}/{self.num_epochs}: {epoch_time:.2f}s")
                
        torch.cuda.synchronize()
        dist.barrier()
        total_time = time.time() - start_time
        
        if rank == 0:
            # Calculate metrics (only on rank 0)
            throughput = (total_samples * world_size) / total_time
            memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)
            
            results = {
                'gpu_count': world_size,
                'batch_size': self.batch_size,
                'total_time': total_time,
                'throughput': throughput,
                'samples_per_second': throughput,
                'memory_gb': memory_used,
                'scaling_efficiency': throughput / (self.results.get('1_gpu', {}).get('throughput', throughput))
            }
            
            self.results[f'{world_size}_gpu'] = results
            
            print(f"Multi-GPU ({world_size}) Results:")
            print(f"  Throughput: {throughput:.2f} samples/sec")
            print(f"  Memory usage: {memory_used:.2f} GB")
            print(f"  Scaling efficiency: {results['scaling_efficiency']:.2f}x")
            
        self.cleanup_distributed()
        
    def _get_gpu_utilization(self, gpu_id):
        """Get GPU utilization percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0
            
    def run_scaling_benchmark(self, max_gpus=None):
        """Run complete scaling benchmark"""
        if max_gpus is None:
            max_gpus = torch.cuda.device_count()
            
        print(f"Starting ResNet-50 scaling benchmark (max {max_gpus} GPUs)")
        print(f"Batch size: {self.batch_size}, Epochs: {self.num_epochs}")
        
        # Single GPU baseline
        single_results = self.benchmark_single_gpu()
        self.results['1_gpu'] = single_results
        
        # Multi-GPU scaling
        for gpu_count in [2, 4, 8]:
            if gpu_count <= max_gpus:
                self.benchmark_multi_gpu(gpu_count)
                
        return self.results
        
    def save_results(self, filename="resnet50_scaling_results.json"):
        """Save benchmark results to file"""
        results_dir = Path("h100_benchmarks/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Results saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='ResNet-50 H100 Scaling Benchmark')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--max-gpus', type=int, default=None, help='Maximum GPUs to use')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
        
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Run benchmark
    benchmark = ResNet50Benchmark(
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    results = benchmark.run_scaling_benchmark(args.max_gpus)
    benchmark.save_results()
    
    # Print summary
    print("\n" + "="*50)
    print("ResNet-50 Scaling Benchmark Summary")
    print("="*50)
    
    for config, result in results.items():
        print(f"{config}: {result['throughput']:.2f} samples/sec")

if __name__ == "__main__":
    main()
