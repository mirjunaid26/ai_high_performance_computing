# AI High Performance Computing Research Repository

This repository contains research projects focused on analyzing and reproducing performance benchmarks for distributed deep learning frameworks on high-performance computing (HPC) clusters, with particular emphasis on NVIDIA H100 GPU architectures.

## Repository Overview

This repository serves as a comprehensive resource for:
- **Performance analysis** of state-of-the-art GPU architectures
- **Benchmarking frameworks** for deep learning workloads
- **Reproducible experiments** on HPC clusters
- **Scaling studies** for distributed training

## 📁 Project Structure

```
ai_high_performance_computing/
├── 1_performance_analysis_cnn_marcel_2025/
│   └── Large scale performance analysis of distributed deep learning frameworks for convolutional neural networks.pdf
└── 2_h100_scaling_performance_jack_2023/
    ├── NVIDIA_Hopper_H100_GPU_Scaling_Performance.pdf
    ├── analysis_plan.md
    ├── cluster_setup.py
    ├── benchmark_resnet50.py
    └── requirements.txt
```

## Research Projects

### 1. CNN Performance Analysis (2025)
**Focus**: Large-scale performance analysis of distributed deep learning frameworks for convolutional neural networks

**Key Areas**:
- Framework comparison (PyTorch, TensorFlow, JAX)
- Distributed training efficiency
- Memory optimization strategies
- Communication overhead analysis

### 2. H100 GPU Scaling Performance (2023)
**Focus**: NVIDIA H100 Hopper architecture scaling performance analysis and reproduction

**Key Features**:
- Multi-GPU scaling benchmarks
- Memory bandwidth utilization
- NVLink interconnect performance
- Power efficiency analysis

## NVIDIA H100 GPU Architecture

### Technical Specifications
- **Architecture**: Hopper (4th Gen Tensor Cores)
- **Memory**: 80GB HBM3 (3.35 TB/s bandwidth)
- **Compute**: 989 TFLOPS (FP16), 1979 TOPS (INT8)
- **Interconnect**: NVLink 4.0 (900 GB/s)
- **Process Node**: TSMC 4N (4nm)

### Key Advantages for HPC
- **Transformer Engine**: Optimized for large language models
- **Multi-Instance GPU (MIG)**: Up to 7 isolated instances
- **Confidential Computing**: Hardware-level security
- **Enhanced Memory**: 3x bandwidth improvement over A100

## HPC Cluster Computing Fundamentals

### What is HPC?
High Performance Computing (HPC) refers to the practice of aggregating computing power to deliver higher performance than typical desktop computers and workstations. In AI/ML contexts, HPC enables:

- **Parallel Processing**: Distributing workloads across multiple GPUs/nodes
- **Scalable Training**: Training larger models with massive datasets
- **Reduced Time-to-Solution**: Faster experimentation and iteration
- **Resource Optimization**: Efficient utilization of expensive hardware

### Cluster Architecture Components

#### 1. **Compute Nodes**
- High-performance servers with multiple GPUs
- Typically 4-8 H100 GPUs per node
- High-bandwidth memory and fast storage

#### 2. **Interconnect Network**
- **InfiniBand**: Low-latency, high-bandwidth networking (200-400 Gbps)
- **NVLink**: Direct GPU-to-GPU communication (900 GB/s per H100)
- **Ethernet**: Cost-effective option for less demanding workloads

#### 3. **Storage Systems**
- **Parallel File Systems**: Lustre, GPFS for high-throughput data access
- **NVMe Storage**: Fast local storage for temporary data
- **Object Storage**: S3-compatible systems for dataset storage

#### 4. **Job Scheduling**
- **SLURM**: Most common HPC job scheduler
- **PBS/Torque**: Alternative scheduling systems
- **Kubernetes**: Container orchestration for ML workloads

### GPU Scaling Patterns

#### Linear Scaling (Ideal)
```
1 GPU:  100 samples/sec
2 GPUs: 200 samples/sec (2.0x speedup)
4 GPUs: 400 samples/sec (4.0x speedup)
8 GPUs: 800 samples/sec (8.0x speedup)
```

#### Real-World Scaling
```
1 GPU:  100 samples/sec
2 GPUs: 190 samples/sec (1.9x speedup) - 95% efficiency
4 GPUs: 360 samples/sec (3.6x speedup) - 90% efficiency  
8 GPUs: 680 samples/sec (6.8x speedup) - 85% efficiency
```

**Scaling Bottlenecks**:
- Communication overhead between GPUs
- Memory bandwidth limitations
- Load balancing inefficiencies
- Framework-specific optimizations

## Getting Started

### Prerequisites
- NVIDIA H100 GPUs (recommended) or compatible CUDA GPUs
- CUDA Toolkit 11.8+
- Python 3.8+
- High-speed interconnect (NVLink/InfiniBand preferred)

### Quick Setup

1. **Clone the repository**:
```bash
git clone https://github.com/mirjunaid26/ai_high_performance_computing.git
cd ai_high_performance_computing
```

2. **Navigate to H100 project**:
```bash
cd 2_h100_scaling_performance_jack_2023
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Check cluster configuration**:
```bash
python cluster_setup.py
```

5. **Run ResNet-50 scaling benchmark**:
```bash
python benchmark_resnet50.py --batch-size 256 --epochs 5
```

## Benchmark Suite

### Available Benchmarks
- **ResNet-50**: Image classification scaling performance
- **BERT**: Natural language processing transformer scaling
- **GPT Models**: Large language model training efficiency
- **Vision Transformer**: Computer vision transformer scaling

### Performance Metrics
- **Throughput**: Samples/second, tokens/second
- **Memory Utilization**: GPU memory usage patterns
- **Scaling Efficiency**: Speedup relative to single GPU
- **Power Consumption**: Energy efficiency analysis
- **Communication Overhead**: Inter-GPU data transfer costs

## Configuration

### Cluster Configuration
The `cluster_setup.py` script automatically detects:
- Available H100 GPUs
- System memory and CPU specifications
- CUDA compatibility
- Network topology

### Benchmark Parameters
Key parameters for reproducible results:
- **Batch Size**: Optimized for GPU memory capacity
- **Model Size**: Scaled appropriately for available resources
- **Data Pipeline**: Efficient data loading and preprocessing
- **Mixed Precision**: FP16/BF16 for optimal performance

## Expected Results

### H100 Performance Targets
Based on NVIDIA specifications and research literature:

| Model | Single H100 | 2x H100 | 4x H100 | 8x H100 |
|-------|-------------|---------|---------|---------|
| ResNet-50 | ~2,000 img/s | ~3,800 img/s | ~7,200 img/s | ~13,600 img/s |
| BERT-Base | ~1,200 seq/s | ~2,280 seq/s | ~4,320 seq/s | ~8,160 seq/s |
| GPT-3 175B | Memory bound | Distributed | Multi-node | Multi-node |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-benchmark`)
3. Commit your changes (`git commit -am 'Add new benchmark'`)
4. Push to the branch (`git push origin feature/new-benchmark`)
5. Create a Pull Request

## References

### Research Papers
- "Large scale performance analysis of distributed deep learning frameworks for convolutional neural networks" (2025)
- "NVIDIA Hopper H100 GPU Scaling Performance" (2023)

### Technical Documentation
- [NVIDIA H100 Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/h100/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### HPC Resources
- [TOP500 Supercomputer Rankings](https://www.top500.org/)
- [SLURM Workload Manager](https://slurm.schedmd.com/)
- [InfiniBand Trade Association](https://www.infinibandta.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this research or collaboration opportunities:
- **Repository**: [ai_high_performance_computing](https://github.com/mirjunaid26/ai_high_performance_computing)
- **Issues**: Use GitHub Issues for bug reports and feature requests

---

**Note**: This repository is designed for research and educational purposes. Performance results may vary based on hardware configuration, software versions, and system optimization.
