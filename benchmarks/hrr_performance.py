"""
HRR Performance Benchmarking Suite

Comprehensive performance benchmarking for HRR operations to establish
concrete metrics and replace marketing claims with engineering data.
"""

import numpy as np
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import psutil
import gc
from pathlib import Path

# Import HRR operations
try:
    from src.lumina_memory.hrr_operations import HRROperations
except ImportError:
    # Mock implementation for development
    class HRROperations:
        def __init__(self, dimensionality: int):
            self.dimensionality = dimensionality
            
        def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            # FFT-based circular convolution
            return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
            
        def unbind(self, bound: np.ndarray, role: np.ndarray) -> np.ndarray:
            # FFT-based circular correlation
            return np.fft.ifft(np.fft.fft(bound) * np.conj(np.fft.fft(role))).real


@dataclass
class PerformanceMetrics:
    """Performance metrics for HRR operations"""
    operation: str
    dimensionality: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    operations_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class CapacityMetrics:
    """Capacity analysis metrics"""
    dimensionality: int
    max_items_90_percent: int
    max_items_80_percent: int
    max_items_70_percent: int
    theoretical_capacity: int
    capacity_efficiency: float  # actual/theoretical


@dataclass
class ScalabilityMetrics:
    """Scalability analysis metrics"""
    dimensionalities: List[int]
    bind_times_ms: List[float]
    unbind_times_ms: List[float]
    memory_usage_mb: List[float]
    scaling_factor: float  # how performance scales with dimensionality


class HRRPerformanceBenchmark:
    """
    Comprehensive HRR performance benchmarking suite.
    
    Provides concrete metrics to replace marketing claims:
    - "Ultra-fast processing" → Specific ms per operation
    - "Optimized algorithms" → Performance vs baseline comparisons
    - "Production ready" → Scalability and resource usage data
    """
    
    def __init__(self, dimensionality: int = 512):
        self.dimensionality = dimensionality
        self.hrr_ops = HRROperations(dimensionality)
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def benchmark_basic_operations(self, iterations: int = 1000) -> Dict[str, PerformanceMetrics]:
        """
        Benchmark basic HRR operations (bind, unbind, superpose).
        
        Args:
            iterations: Number of iterations for statistical accuracy
            
        Returns:
            Performance metrics for each operation
        """
        print(f"🔬 Benchmarking basic HRR operations ({iterations} iterations)...")
        
        # Generate test vectors
        vector_a = self._generate_random_vector()
        vector_b = self._generate_random_vector()
        bound_vector = self.hrr_ops.bind(vector_a, vector_b)
        
        results = {}
        
        # Benchmark bind operation
        print("  Testing bind operation...")
        bind_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            self.hrr_ops.bind(vector_a, vector_b)
            end_time = time.perf_counter()
            bind_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        results['bind'] = self._calculate_performance_metrics(
            'bind', bind_times, self.dimensionality
        )
        
        # Benchmark unbind operation
        print("  Testing unbind operation...")
        unbind_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            self.hrr_ops.unbind(bound_vector, vector_a)
            end_time = time.perf_counter()
            unbind_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        results['unbind'] = self._calculate_performance_metrics(
            'unbind', unbind_times, self.dimensionality
        )
        
        # Benchmark superposition
        print("  Testing superposition operation...")
        vectors = [self._generate_random_vector() for _ in range(10)]
        superpose_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            self._superpose_vectors(vectors)
            end_time = time.perf_counter()
            superpose_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        results['superpose'] = self._calculate_performance_metrics(
            'superpose', superpose_times, self.dimensionality
        )
        
        return results
    
    def benchmark_capacity_analysis(self, max_items: int = 100) -> CapacityMetrics:
        """
        Analyze superposition capacity with accuracy measurements.
        
        Args:
            max_items: Maximum number of items to test
            
        Returns:
            Capacity analysis metrics
        """
        print(f"📊 Analyzing superposition capacity (up to {max_items} items)...")
        
        # Generate role-filler pairs
        pairs = []
        for i in range(max_items):
            role = self._generate_random_vector()
            filler = self._generate_random_vector()
            pairs.append((role, filler))
        
        # Test capacity at different accuracy thresholds
        capacity_90 = 0
        capacity_80 = 0
        capacity_70 = 0
        
        for n_items in range(1, max_items + 1, 2):  # Test every 2 items
            accuracy = self._test_superposition_accuracy(pairs[:n_items])
            
            if accuracy >= 0.9 and capacity_90 == 0:
                capacity_90 = n_items
            if accuracy >= 0.8 and capacity_80 == 0:
                capacity_80 = n_items
            if accuracy >= 0.7 and capacity_70 == 0:
                capacity_70 = n_items
            
            # Stop if accuracy drops too low
            if accuracy < 0.6:
                break
        
        # Calculate theoretical capacity (0.1 * dimensionality)
        theoretical = int(0.1 * self.dimensionality)
        efficiency = capacity_90 / theoretical if theoretical > 0 else 0
        
        return CapacityMetrics(
            dimensionality=self.dimensionality,
            max_items_90_percent=capacity_90,
            max_items_80_percent=capacity_80,
            max_items_70_percent=capacity_70,
            theoretical_capacity=theoretical,
            capacity_efficiency=efficiency
        )
    
    def benchmark_scalability(self, dimensionalities: List[int] = None) -> ScalabilityMetrics:
        """
        Analyze performance scalability across different dimensionalities.
        
        Args:
            dimensionalities: List of dimensionalities to test
            
        Returns:
            Scalability analysis metrics
        """
        if dimensionalities is None:
            dimensionalities = [64, 128, 256, 512, 1024]
        
        print(f"📈 Analyzing scalability across dimensionalities: {dimensionalities}")
        
        bind_times = []
        unbind_times = []
        memory_usage = []
        
        for dim in dimensionalities:
            print(f"  Testing dimensionality {dim}...")
            
            # Create HRR operations for this dimensionality
            hrr_ops = HRROperations(dim)
            
            # Generate test vectors
            vector_a = np.random.randn(dim)
            vector_a = vector_a / np.linalg.norm(vector_a)
            vector_b = np.random.randn(dim)
            vector_b = vector_b / np.linalg.norm(vector_b)
            
            # Benchmark bind operation
            start_time = time.perf_counter()
            for _ in range(100):  # Average over 100 operations
                hrr_ops.bind(vector_a, vector_b)
            bind_time = (time.perf_counter() - start_time) * 10  # ms per operation
            bind_times.append(bind_time)
            
            # Benchmark unbind operation
            bound = hrr_ops.bind(vector_a, vector_b)
            start_time = time.perf_counter()
            for _ in range(100):  # Average over 100 operations
                hrr_ops.unbind(bound, vector_a)
            unbind_time = (time.perf_counter() - start_time) * 10  # ms per operation
            unbind_times.append(unbind_time)
            
            # Measure memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_usage.append(memory_mb)
            
            # Clean up
            del hrr_ops, vector_a, vector_b, bound
            gc.collect()
        
        # Calculate scaling factor (how performance scales with dimensionality)
        # Fit linear model: time = a * dimensionality + b
        dim_array = np.array(dimensionalities)
        bind_array = np.array(bind_times)
        scaling_factor = np.polyfit(dim_array, bind_array, 1)[0]  # Slope
        
        return ScalabilityMetrics(
            dimensionalities=dimensionalities,
            bind_times_ms=bind_times,
            unbind_times_ms=unbind_times,
            memory_usage_mb=memory_usage,
            scaling_factor=scaling_factor
        )
    
    def benchmark_memory_efficiency(self, vector_counts: List[int] = None) -> Dict[str, List[float]]:
        """
        Analyze memory efficiency for different numbers of vectors.
        
        Args:
            vector_counts: List of vector counts to test
            
        Returns:
            Memory usage metrics
        """
        if vector_counts is None:
            vector_counts = [10, 50, 100, 500, 1000, 5000]
        
        print(f"💾 Analyzing memory efficiency for vector counts: {vector_counts}")
        
        memory_usage = []
        vector_storage = []
        
        for count in vector_counts:
            print(f"  Testing {count} vectors...")
            
            # Create vectors
            vectors = [self._generate_random_vector() for _ in range(count)]
            
            # Measure memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Store vectors (simulate memory system storage)
            storage = {f"vector_{i}": vec for i, vec in enumerate(vectors)}
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before
            
            memory_usage.append(memory_delta)
            vector_storage.append(count * self.dimensionality * 4 / 1024 / 1024)  # 4 bytes per float32
            
            # Clean up
            del vectors, storage
            gc.collect()
        
        return {
            'vector_counts': vector_counts,
            'actual_memory_mb': memory_usage,
            'theoretical_memory_mb': vector_storage,
            'memory_overhead': [actual - theoretical for actual, theoretical in zip(memory_usage, vector_storage)]
        }
    
    def generate_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report.
        
        Returns:
            Complete performance analysis
        """
        print("📊 Generating comprehensive HRR performance report...")
        
        # Run all benchmarks
        basic_ops = self.benchmark_basic_operations(iterations=1000)
        capacity = self.benchmark_capacity_analysis(max_items=100)
        scalability = self.benchmark_scalability()
        memory_efficiency = self.benchmark_memory_efficiency()
        
        # Compile report
        report = {
            'metadata': {
                'dimensionality': self.dimensionality,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_gb': psutil.virtual_memory().total / 1024**3,
                    'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
                }
            },
            'basic_operations': {op: asdict(metrics) for op, metrics in basic_ops.items()},
            'capacity_analysis': asdict(capacity),
            'scalability_analysis': asdict(scalability),
            'memory_efficiency': memory_efficiency,
            'summary': self._generate_summary(basic_ops, capacity, scalability)
        }
        
        # Save report
        report_file = self.results_dir / f"hrr_performance_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved to: {report_file}")
        return report
    
    def _generate_random_vector(self) -> np.ndarray:
        """Generate random normalized vector"""
        vector = np.random.randn(self.dimensionality)
        return vector / np.linalg.norm(vector)
    
    def _superpose_vectors(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Create superposition of vectors"""
        result = sum(vectors)
        return result / np.linalg.norm(result)
    
    def _test_superposition_accuracy(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Test retrieval accuracy for superposition of role-filler pairs"""
        # Create superposition
        superposition = np.zeros(self.dimensionality)
        for role, filler in pairs:
            bound = self.hrr_ops.bind(role, filler)
            superposition += bound
        
        # Normalize
        superposition = superposition / np.linalg.norm(superposition)
        
        # Test retrieval accuracy
        correct_retrievals = 0
        for role, expected_filler in pairs:
            retrieved_filler = self.hrr_ops.unbind(superposition, role)
            similarity = np.dot(expected_filler, retrieved_filler) / (
                np.linalg.norm(expected_filler) * np.linalg.norm(retrieved_filler)
            )
            
            if similarity > 0.7:  # Threshold for correct retrieval
                correct_retrievals += 1
        
        return correct_retrievals / len(pairs)
    
    def _calculate_performance_metrics(self, operation: str, times_ms: List[float], 
                                     dimensionality: int) -> PerformanceMetrics:
        """Calculate performance metrics from timing data"""
        times_array = np.array(times_ms)
        
        # Get system resource usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        return PerformanceMetrics(
            operation=operation,
            dimensionality=dimensionality,
            mean_time_ms=float(np.mean(times_array)),
            std_time_ms=float(np.std(times_array)),
            min_time_ms=float(np.min(times_array)),
            max_time_ms=float(np.max(times_array)),
            operations_per_second=1000.0 / np.mean(times_array),
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent
        )
    
    def _generate_summary(self, basic_ops: Dict, capacity: CapacityMetrics, 
                         scalability: ScalabilityMetrics) -> Dict:
        """Generate performance summary"""
        bind_perf = basic_ops['bind']
        unbind_perf = basic_ops['unbind']
        
        return {
            'performance_grade': self._calculate_performance_grade(bind_perf, capacity),
            'key_metrics': {
                'bind_time_ms': bind_perf.mean_time_ms,
                'unbind_time_ms': unbind_perf.mean_time_ms,
                'total_cycle_ms': bind_perf.mean_time_ms + unbind_perf.mean_time_ms,
                'capacity_90_percent': capacity.max_items_90_percent,
                'capacity_efficiency': capacity.capacity_efficiency,
                'scaling_factor': scalability.scaling_factor
            },
            'targets_met': {
                'bind_unbind_under_1ms': (bind_perf.mean_time_ms + unbind_perf.mean_time_ms) < 1.0,
                'capacity_over_40_items': capacity.max_items_90_percent >= 40,
                'efficiency_over_70_percent': capacity.capacity_efficiency >= 0.7,
                'reasonable_scaling': scalability.scaling_factor < 0.01  # Linear scaling
            }
        }
    
    def _calculate_performance_grade(self, bind_perf: PerformanceMetrics, 
                                   capacity: CapacityMetrics) -> str:
        """Calculate overall performance grade"""
        total_time = bind_perf.mean_time_ms + bind_perf.mean_time_ms  # Approximate unbind time
        
        if total_time < 0.5 and capacity.max_items_90_percent >= 50:
            return "A+ (Excellent)"
        elif total_time < 1.0 and capacity.max_items_90_percent >= 40:
            return "A (Very Good)"
        elif total_time < 2.0 and capacity.max_items_90_percent >= 30:
            return "B (Good)"
        elif total_time < 5.0 and capacity.max_items_90_percent >= 20:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"


if __name__ == "__main__":
    # Run comprehensive benchmarking
    benchmark = HRRPerformanceBenchmark(dimensionality=512)
    report = benchmark.generate_performance_report()
    
    # Print summary
    print("\n🎯 HRR PERFORMANCE SUMMARY:")
    summary = report['summary']
    print(f"  Performance Grade: {summary['performance_grade']}")
    print(f"  Bind Time: {summary['key_metrics']['bind_time_ms']:.3f}ms")
    print(f"  Unbind Time: {summary['key_metrics']['unbind_time_ms']:.3f}ms")
    print(f"  Total Cycle: {summary['key_metrics']['total_cycle_ms']:.3f}ms")
    print(f"  Capacity (90%): {summary['key_metrics']['capacity_90_percent']} items")
    print(f"  Efficiency: {summary['key_metrics']['capacity_efficiency']:.1%}")
    
    print(f"\n✅ TARGETS MET:")
    for target, met in summary['targets_met'].items():
        status = "✅" if met else "❌"
        print(f"  {status} {target.replace('_', ' ').title()}: {met}")