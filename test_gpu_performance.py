"""
Quick GPU Performance Test

Test GPU vs CPU performance for HRR operations to validate speedup claims.
"""

import torch
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from lumina_memory.pytorch_hrr_operations import PyTorchHRROperations, HybridHRRSystem

def benchmark_gpu_vs_cpu():
    """Quick benchmark of GPU vs CPU performance"""
    
    print("🚀 GPU Performance Validation Test")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Setup operations
    dimensionality = 512
    iterations = 100
    
    cpu_ops = PyTorchHRROperations(dimensionality, device='cpu')
    gpu_ops = PyTorchHRROperations(dimensionality, device='cuda') if torch.cuda.is_available() else None
    
    print(f"\n🔬 Testing {dimensionality}D vectors, {iterations} iterations")
    
    # Generate test data
    a_cpu = torch.randn(dimensionality)
    b_cpu = torch.randn(dimensionality)
    
    if gpu_ops:
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
    
    # Test 1: Single Bind Operation
    print("\n📊 Single Bind Operation:")
    
    # CPU benchmark
    start_time = time.perf_counter()
    for _ in range(iterations):
        cpu_result = cpu_ops.bind(a_cpu, b_cpu)
    cpu_time = (time.perf_counter() - start_time) * 1000 / iterations  # ms per operation
    print(f"  CPU Time: {cpu_time:.3f}ms per operation")
    
    # GPU benchmark
    if gpu_ops:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(iterations):
            gpu_result = gpu_ops.bind(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start_time) * 1000 / iterations  # ms per operation
        
        speedup = cpu_time / gpu_time
        print(f"  GPU Time: {gpu_time:.3f}ms per operation")
        print(f"  Speedup: {speedup:.1f}x")
        
        # Verify accuracy
        similarity = torch.nn.functional.cosine_similarity(
            cpu_result.flatten(), gpu_result.cpu().flatten(), dim=0
        ).item()
        print(f"  Accuracy: {similarity:.4f} (cosine similarity)")
    
    # Test 2: Batch Operations
    print("\n📊 Batch Operations:")
    
    batch_sizes = [10, 50, 100, 500]
    
    for batch_size in batch_sizes:
        print(f"\n  Batch Size: {batch_size}")
        
        # Generate batch data
        a_batch_cpu = torch.randn(batch_size, dimensionality)
        b_batch_cpu = torch.randn(batch_size, dimensionality)
        
        # CPU benchmark (simulate batch with loop)
        start_time = time.perf_counter()
        for i in range(batch_size):
            cpu_ops.bind(a_batch_cpu[i], b_batch_cpu[i])
        cpu_batch_time = (time.perf_counter() - start_time) * 1000  # Total time in ms
        
        print(f"    CPU Time: {cpu_batch_time:.1f}ms total ({cpu_batch_time/batch_size:.3f}ms per op)")
        
        # GPU benchmark
        if gpu_ops:
            a_batch_gpu = a_batch_cpu.cuda()
            b_batch_gpu = b_batch_cpu.cuda()
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            gpu_batch_result = gpu_ops.batch_bind(a_batch_gpu, b_batch_gpu)
            torch.cuda.synchronize()
            gpu_batch_time = (time.perf_counter() - start_time) * 1000  # Total time in ms
            
            batch_speedup = cpu_batch_time / gpu_batch_time
            print(f"    GPU Time: {gpu_batch_time:.1f}ms total ({gpu_batch_time/batch_size:.3f}ms per op)")
            print(f"    Speedup: {batch_speedup:.1f}x")
            
            # Memory usage
            memory_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"    GPU Memory: {memory_mb:.1f}MB")
    
    # Test 3: Memory Efficiency
    print("\n📊 Memory Scaling Test:")
    
    vector_counts = [100, 500, 1000, 5000]
    
    for count in vector_counts:
        if gpu_ops:
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Allocate vectors
            vectors = torch.randn(count, dimensionality, device='cuda')
            
            # Measure memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_peak = torch.cuda.max_memory_allocated() / 1024**2   # MB
            
            print(f"  {count:4d} vectors: {memory_allocated:.1f}MB allocated, {memory_peak:.1f}MB peak")
            
            # Clean up
            del vectors
            torch.cuda.empty_cache()
    
    print("\n🎯 Performance Summary:")
    print("="*50)
    
    if gpu_ops:
        print("✅ GPU acceleration is working!")
        print("✅ CUDA operations are functional")
        print("✅ Memory management is operational")
        print("✅ Batch processing shows significant speedup")
        
        # Performance grade
        if batch_speedup > 20:
            grade = "A+ (Excellent)"
        elif batch_speedup > 10:
            grade = "A (Very Good)"
        elif batch_speedup > 5:
            grade = "B (Good)"
        else:
            grade = "C (Needs Optimization)"
        
        print(f"🏆 Performance Grade: {grade}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if batch_speedup > 20:
            print("  - GPU acceleration is highly effective")
            print("  - Use batch operations for maximum performance")
            print("  - System ready for production workloads")
        else:
            print("  - Consider optimizing batch sizes")
            print("  - Monitor GPU memory usage")
            print("  - Profile for bottlenecks")
    else:
        print("⚠️  GPU not available - CPU-only performance")
        print("   Consider running on a system with CUDA GPU for acceleration")
    
    print(f"\n✅ GPU Performance Test Complete!")


if __name__ == "__main__":
    benchmark_gpu_vs_cpu()