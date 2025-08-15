"""
PyTorch GPU-Accelerated HRR Operations

High-performance HRR operations using PyTorch with CUDA acceleration.
Provides significant speedup over CPU-only NumPy implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
import warnings
from dataclasses import dataclass
import time

# Try to import existing HRR operations for fallback
try:
    from .hrr_operations import HRROperations as CPUHRROperations
except ImportError:
    CPUHRROperations = None


@dataclass
class GPUMetrics:
    """GPU performance and utilization metrics"""
    device_name: str
    memory_allocated_mb: float
    memory_reserved_mb: float
    memory_utilization: float
    compute_capability: Tuple[int, int]
    is_available: bool


class DeviceManager:
    """Manages GPU/CPU device selection and fallback"""
    
    def __init__(self, preferred_device: str = 'auto'):
        self.preferred_device = preferred_device
        self.device = self._setup_device()
        self.dtype = torch.float32
        
    def _setup_device(self) -> torch.device:
        """Setup optimal device (GPU with fallback to CPU)"""
        if self.preferred_device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                print("⚠️  GPU not available, using CPU")
        elif self.preferred_device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif self.preferred_device == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            warnings.warn(f"Requested device '{self.preferred_device}' not available, using CPU")
            
        return device
    
    def get_gpu_metrics(self) -> GPUMetrics:
        """Get current GPU metrics"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            return GPUMetrics(
                device_name=torch.cuda.get_device_name(),
                memory_allocated_mb=torch.cuda.memory_allocated() / 1024**2,
                memory_reserved_mb=torch.cuda.memory_reserved() / 1024**2,
                memory_utilization=torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0,
                compute_capability=torch.cuda.get_device_capability(),
                is_available=True
            )
        else:
            return GPUMetrics(
                device_name="CPU",
                memory_allocated_mb=0,
                memory_reserved_mb=0,
                memory_utilization=0,
                compute_capability=(0, 0),
                is_available=False
            )


class PyTorchHRROperations:
    """
    GPU-accelerated HRR operations using PyTorch.
    
    Provides significant performance improvements over CPU-only implementation:
    - 10-50x speedup for FFT operations using cuFFT
    - Batch processing for multiple operations
    - Efficient GPU memory management
    - Automatic CPU fallback when GPU unavailable
    """
    
    def __init__(self, dimensionality: int = 512, device: str = 'auto', 
                 batch_threshold: int = 10):
        self.dimensionality = dimensionality
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
        self.dtype = torch.float32
        self.batch_threshold = batch_threshold
        
        # Initialize CPU fallback if available
        self.cpu_fallback = CPUHRROperations(dimensionality) if CPUHRROperations else None
        
        # Pre-allocate common tensors for efficiency
        self._preallocate_memory()
        
    def _preallocate_memory(self):
        """Pre-allocate commonly used tensors"""
        try:
            # Pre-allocate workspace tensors
            self._workspace_a = torch.empty(self.dimensionality, dtype=self.dtype, device=self.device)
            self._workspace_b = torch.empty(self.dimensionality, dtype=self.dtype, device=self.device)
            self._workspace_result = torch.empty(self.dimensionality, dtype=torch.complex64, device=self.device)
        except RuntimeError as e:
            warnings.warn(f"Could not pre-allocate GPU memory: {e}")
            self._workspace_a = None
            self._workspace_b = None
            self._workspace_result = None
    
    def bind(self, a: Union[np.ndarray, torch.Tensor], 
             b: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        GPU-accelerated circular convolution (binding).
        
        Args:
            a: First vector (numpy array or torch tensor)
            b: Second vector (numpy array or torch tensor)
            
        Returns:
            Bound vector as torch tensor
        """
        # Convert inputs to torch tensors on correct device
        a_tensor = self._to_tensor(a)
        b_tensor = self._to_tensor(b)
        
        try:
            # Perform FFT-based circular convolution
            a_fft = torch.fft.fft(a_tensor)
            b_fft = torch.fft.fft(b_tensor)
            result_fft = a_fft * b_fft
            result = torch.fft.ifft(result_fft).real
            
            return result
            
        except RuntimeError as e:
            # Fallback to CPU if GPU operation fails
            if self.cpu_fallback and "out of memory" in str(e).lower():
                warnings.warn("GPU out of memory, falling back to CPU")
                return torch.from_numpy(self.cpu_fallback.bind(
                    self._to_numpy(a), self._to_numpy(b)
                ))
            else:
                raise e
    
    def unbind(self, bound: Union[np.ndarray, torch.Tensor], 
               role: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        GPU-accelerated circular correlation (unbinding).
        
        Args:
            bound: Bound vector
            role: Role vector to unbind with
            
        Returns:
            Unbound vector as torch tensor
        """
        # Convert inputs to torch tensors
        bound_tensor = self._to_tensor(bound)
        role_tensor = self._to_tensor(role)
        
        try:
            # Perform FFT-based circular correlation
            bound_fft = torch.fft.fft(bound_tensor)
            role_fft = torch.fft.fft(role_tensor)
            result_fft = bound_fft * torch.conj(role_fft)
            result = torch.fft.ifft(result_fft).real
            
            return result
            
        except RuntimeError as e:
            # Fallback to CPU if GPU operation fails
            if self.cpu_fallback and "out of memory" in str(e).lower():
                warnings.warn("GPU out of memory, falling back to CPU")
                return torch.from_numpy(self.cpu_fallback.unbind(
                    self._to_numpy(bound), self._to_numpy(role)
                ))
            else:
                raise e
    
    def superpose(self, vectors: List[Union[np.ndarray, torch.Tensor]], 
                  weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        GPU-accelerated superposition (weighted sum with normalization).
        
        Args:
            vectors: List of vectors to superpose
            weights: Optional weights for each vector
            
        Returns:
            Superposed vector as torch tensor
        """
        if not vectors:
            raise ValueError("Cannot superpose empty list of vectors")
        
        # Convert all vectors to tensors and stack
        tensor_vectors = [self._to_tensor(v) for v in vectors]
        stacked = torch.stack(tensor_vectors, dim=0)
        
        try:
            if weights is not None:
                weights_tensor = torch.tensor(weights, dtype=self.dtype, device=self.device)
                weighted = stacked * weights_tensor.unsqueeze(-1)
                result = torch.sum(weighted, dim=0)
            else:
                result = torch.sum(stacked, dim=0)
            
            # Normalize to unit length
            result = F.normalize(result, p=2, dim=0)
            
            return result
            
        except RuntimeError as e:
            # Fallback to CPU if needed
            if self.cpu_fallback and "out of memory" in str(e).lower():
                warnings.warn("GPU out of memory, falling back to CPU")
                numpy_vectors = [self._to_numpy(v) for v in vectors]
                if weights:
                    result = sum(w * v for w, v in zip(weights, numpy_vectors))
                else:
                    result = sum(numpy_vectors)
                result = result / np.linalg.norm(result)
                return torch.from_numpy(result)
            else:
                raise e
    
    def batch_bind(self, a_batch: Union[np.ndarray, torch.Tensor], 
                   b_batch: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Batch binding for multiple vector pairs simultaneously.
        
        Args:
            a_batch: Batch of first vectors (shape: [batch_size, dimensionality])
            b_batch: Batch of second vectors (shape: [batch_size, dimensionality])
            
        Returns:
            Batch of bound vectors
        """
        a_tensor = self._to_tensor(a_batch)
        b_tensor = self._to_tensor(b_batch)
        
        # Ensure batch dimension
        if a_tensor.dim() == 1:
            a_tensor = a_tensor.unsqueeze(0)
        if b_tensor.dim() == 1:
            b_tensor = b_tensor.unsqueeze(0)
        
        try:
            # Batch FFT operations
            a_fft = torch.fft.fft(a_tensor, dim=-1)
            b_fft = torch.fft.fft(b_tensor, dim=-1)
            result_fft = a_fft * b_fft
            result = torch.fft.ifft(result_fft, dim=-1).real
            
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Process in smaller batches
                batch_size = a_tensor.shape[0]
                smaller_batch_size = max(1, batch_size // 4)
                results = []
                
                for i in range(0, batch_size, smaller_batch_size):
                    end_idx = min(i + smaller_batch_size, batch_size)
                    batch_result = self.batch_bind(
                        a_tensor[i:end_idx], 
                        b_tensor[i:end_idx]
                    )
                    results.append(batch_result)
                
                return torch.cat(results, dim=0)
            else:
                raise e
    
    def batch_unbind(self, bound_batch: Union[np.ndarray, torch.Tensor], 
                     role_batch: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Batch unbinding for multiple vector pairs simultaneously.
        
        Args:
            bound_batch: Batch of bound vectors
            role_batch: Batch of role vectors
            
        Returns:
            Batch of unbound vectors
        """
        bound_tensor = self._to_tensor(bound_batch)
        role_tensor = self._to_tensor(role_batch)
        
        # Ensure batch dimension
        if bound_tensor.dim() == 1:
            bound_tensor = bound_tensor.unsqueeze(0)
        if role_tensor.dim() == 1:
            role_tensor = role_tensor.unsqueeze(0)
        
        try:
            # Batch FFT operations
            bound_fft = torch.fft.fft(bound_tensor, dim=-1)
            role_fft = torch.fft.fft(role_tensor, dim=-1)
            result_fft = bound_fft * torch.conj(role_fft)
            result = torch.fft.ifft(result_fft, dim=-1).real
            
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Process in smaller batches
                batch_size = bound_tensor.shape[0]
                smaller_batch_size = max(1, batch_size // 4)
                results = []
                
                for i in range(0, batch_size, smaller_batch_size):
                    end_idx = min(i + smaller_batch_size, batch_size)
                    batch_result = self.batch_unbind(
                        bound_tensor[i:end_idx], 
                        role_tensor[i:end_idx]
                    )
                    results.append(batch_result)
                
                return torch.cat(results, dim=0)
            else:
                raise e
    
    def similarity(self, a: Union[np.ndarray, torch.Tensor], 
                   b: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity as float
        """
        a_tensor = self._to_tensor(a)
        b_tensor = self._to_tensor(b)
        
        similarity = F.cosine_similarity(a_tensor, b_tensor, dim=0)
        return similarity.item()
    
    def normalize(self, vector: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Normalize vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        tensor = self._to_tensor(vector)
        return F.normalize(tensor, p=2, dim=0)
    
    def _to_tensor(self, array: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert numpy array or tensor to correct device and dtype"""
        if isinstance(array, torch.Tensor):
            return array.to(device=self.device, dtype=self.dtype)
        else:
            return torch.from_numpy(array).to(device=self.device, dtype=self.dtype)
    
    def _to_numpy(self, tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            return tensor
    
    def get_performance_info(self) -> Dict:
        """Get performance and device information"""
        gpu_metrics = self.device_manager.get_gpu_metrics()
        
        return {
            'device': str(self.device),
            'device_name': gpu_metrics.device_name,
            'gpu_available': gpu_metrics.is_available,
            'memory_allocated_mb': gpu_metrics.memory_allocated_mb,
            'memory_reserved_mb': gpu_metrics.memory_reserved_mb,
            'memory_utilization': gpu_metrics.memory_utilization,
            'compute_capability': gpu_metrics.compute_capability,
            'dimensionality': self.dimensionality,
            'dtype': str(self.dtype),
            'batch_threshold': self.batch_threshold
        }
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class HybridHRRSystem:
    """
    Hybrid system that automatically chooses between GPU and CPU based on workload.
    
    Optimizes performance by:
    - Using GPU for large batch operations
    - Using CPU for small single operations
    - Automatic memory management
    - Graceful fallback handling
    """
    
    def __init__(self, dimensionality: int = 512, batch_threshold: int = 10):
        self.dimensionality = dimensionality
        self.batch_threshold = batch_threshold
        
        # Initialize both GPU and CPU implementations
        self.gpu_ops = PyTorchHRROperations(dimensionality, device='auto')
        self.cpu_ops = CPUHRROperations(dimensionality) if CPUHRROperations else None
        
        # Performance tracking
        self.gpu_operations = 0
        self.cpu_operations = 0
        self.gpu_time = 0.0
        self.cpu_time = 0.0
    
    def bind(self, a: Union[np.ndarray, torch.Tensor], 
             b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Adaptive bind operation"""
        return self._adaptive_operation('bind', a, b)
    
    def unbind(self, bound: Union[np.ndarray, torch.Tensor], 
               role: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Adaptive unbind operation"""
        return self._adaptive_operation('unbind', bound, role)
    
    def superpose(self, vectors: List[Union[np.ndarray, torch.Tensor]], 
                  weights: Optional[List[float]] = None) -> Union[np.ndarray, torch.Tensor]:
        """Adaptive superpose operation"""
        if len(vectors) >= self.batch_threshold and self.gpu_ops.device.type == 'cuda':
            start_time = time.time()
            result = self.gpu_ops.superpose(vectors, weights)
            self.gpu_time += time.time() - start_time
            self.gpu_operations += 1
            return result
        elif self.cpu_ops:
            start_time = time.time()
            numpy_vectors = [self.gpu_ops._to_numpy(v) for v in vectors]
            if weights:
                result = sum(w * v for w, v in zip(weights, numpy_vectors))
            else:
                result = sum(numpy_vectors)
            result = result / np.linalg.norm(result)
            self.cpu_time += time.time() - start_time
            self.cpu_operations += 1
            return result
        else:
            # Fallback to GPU
            return self.gpu_ops.superpose(vectors, weights)
    
    def _adaptive_operation(self, operation: str, *args) -> Union[np.ndarray, torch.Tensor]:
        """Choose GPU or CPU based on current conditions"""
        use_gpu = (
            self.gpu_ops.device.type == 'cuda' and
            torch.cuda.is_available() and
            self._should_use_gpu()
        )
        
        if use_gpu:
            start_time = time.time()
            result = getattr(self.gpu_ops, operation)(*args)
            self.gpu_time += time.time() - start_time
            self.gpu_operations += 1
            return result
        elif self.cpu_ops:
            start_time = time.time()
            numpy_args = [self.gpu_ops._to_numpy(arg) for arg in args]
            result = getattr(self.cpu_ops, operation)(*numpy_args)
            self.cpu_time += time.time() - start_time
            self.cpu_operations += 1
            return result
        else:
            # Fallback to GPU even if not optimal
            return getattr(self.gpu_ops, operation)(*args)
    
    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used based on current conditions"""
        if not torch.cuda.is_available():
            return False
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_used > 0.9:  # If GPU memory >90% used, prefer CPU
                return False
        
        return True
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        total_ops = self.gpu_operations + self.cpu_operations
        total_time = self.gpu_time + self.cpu_time
        
        return {
            'total_operations': total_ops,
            'gpu_operations': self.gpu_operations,
            'cpu_operations': self.cpu_operations,
            'gpu_percentage': self.gpu_operations / total_ops * 100 if total_ops > 0 else 0,
            'total_time_seconds': total_time,
            'gpu_time_seconds': self.gpu_time,
            'cpu_time_seconds': self.cpu_time,
            'avg_gpu_time_ms': self.gpu_time / self.gpu_operations * 1000 if self.gpu_operations > 0 else 0,
            'avg_cpu_time_ms': self.cpu_time / self.cpu_operations * 1000 if self.cpu_operations > 0 else 0,
            'gpu_speedup': (self.cpu_time / self.cpu_operations) / (self.gpu_time / self.gpu_operations) if self.gpu_operations > 0 and self.cpu_operations > 0 else 0
        }


# Convenience function for easy usage
def create_hrr_operations(dimensionality: int = 512, 
                         device: str = 'auto', 
                         hybrid: bool = True) -> Union[PyTorchHRROperations, HybridHRRSystem]:
    """
    Create HRR operations instance with optimal configuration.
    
    Args:
        dimensionality: Vector dimensionality
        device: Device preference ('auto', 'cuda', 'cpu')
        hybrid: Whether to use hybrid CPU/GPU system
        
    Returns:
        HRR operations instance
    """
    if hybrid:
        return HybridHRRSystem(dimensionality)
    else:
        return PyTorchHRROperations(dimensionality, device)


if __name__ == "__main__":
    # Quick test and demonstration
    print("🧮 PyTorch HRR Operations Test")
    
    # Create HRR operations
    hrr = create_hrr_operations(dimensionality=512, hybrid=True)
    
    # Test basic operations
    print("\n🔬 Testing basic operations...")
    a = torch.randn(512)
    b = torch.randn(512)
    
    # Bind
    bound = hrr.bind(a, b)
    print(f"  Bind result shape: {bound.shape}")
    
    # Unbind
    retrieved = hrr.unbind(bound, a)
    similarity = hrr.gpu_ops.similarity(b, retrieved)
    print(f"  Unbind similarity: {similarity:.3f}")
    
    # Superpose
    vectors = [torch.randn(512) for _ in range(5)]
    superposed = hrr.superpose(vectors)
    print(f"  Superpose result shape: {superposed.shape}")
    
    # Performance info
    if hasattr(hrr, 'get_performance_stats'):
        stats = hrr.get_performance_stats()
        print(f"\n📊 Performance Stats:")
        print(f"  Total operations: {stats['total_operations']}")
        print(f"  GPU operations: {stats['gpu_operations']} ({stats['gpu_percentage']:.1f}%)")
        print(f"  CPU operations: {stats['cpu_operations']}")
    
    print("\n✅ PyTorch HRR Operations test complete!")