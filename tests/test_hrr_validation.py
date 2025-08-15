"""
HRR Validation Test Suite

Comprehensive validation of Holographic Reduced Representations (HRR) operations
including bind/unbind correctness, superposition stress testing, and capacity limits.

This module transforms marketing claims into engineering reality with rigorous testing.
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple
import time
from dataclasses import dataclass

# Import HRR operations from the main system
try:
    from src.lumina_memory.hrr_operations import HRROperations
    from src.lumina_memory.xp_unit import XPUnit
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from lumina_memory.hrr_operations import HRROperations
    from lumina_memory.xp_unit import XPUnit


@dataclass
class HRRMetrics:
    """Metrics for HRR performance validation"""
    bind_unbind_accuracy: float
    superposition_capacity: int
    retrieval_accuracy_at_k: Dict[int, float]
    semantic_drift_rate: float
    temporal_decay_accuracy: float
    performance_ms_per_operation: float


class HRRValidationSuite:
    """
    Comprehensive HRR validation suite that provides concrete metrics
    to replace marketing claims with engineering evidence.
    """
    
    def __init__(self, dimensionality: int = 512):
        self.dimensionality = dimensionality
        self.hrr_ops = HRROperations(dimensionality)
        self.tolerance = 0.1  # 10% tolerance for floating point comparisons
        
    def test_bind_unbind_correctness(self, k_pairs: int = 10) -> float:
        """
        Test bind/unbind correctness for k role-filler pairs.
        
        Target: >95% accuracy for basic bind/unbind operations
        
        Args:
            k_pairs: Number of role-filler pairs to test
            
        Returns:
            Accuracy percentage (0.0 to 1.0)
        """
        correct_retrievals = 0
        total_tests = k_pairs
        
        for i in range(k_pairs):
            # Create random role and filler vectors
            role = self._generate_random_vector()
            filler = self._generate_random_vector()
            
            # Bind role and filler
            bound = self.hrr_ops.bind(role, filler)
            
            # Unbind to retrieve filler
            retrieved_filler = self.hrr_ops.unbind(bound, role)
            
            # Check similarity (should be high)
            similarity = self._cosine_similarity(filler, retrieved_filler)
            
            if similarity > 0.7:  # Threshold for "correct" retrieval
                correct_retrievals += 1
                
        accuracy = correct_retrievals / total_tests
        return accuracy
    
    def test_superposition_stress(self, n_items: int = 50, target_accuracy: float = 0.9) -> Dict[str, float]:
        """
        Test superposition capacity with increasing number of items.
        
        Target: ~51 items (0.1 * 512D) at 90% accuracy
        
        Args:
            n_items: Maximum number of items to test
            target_accuracy: Target retrieval accuracy
            
        Returns:
            Dictionary with capacity metrics
        """
        # Generate role-filler pairs
        pairs = []
        for i in range(n_items):
            role = self._generate_random_vector()
            filler = self._generate_random_vector()
            pairs.append((role, filler))
        
        # Test increasing superposition sizes
        results = {}
        for size in range(1, n_items + 1, 5):  # Test every 5 items
            # Create superposition of first 'size' pairs
            superposition = np.zeros(self.dimensionality)
            for j in range(size):
                role, filler = pairs[j]
                bound = self.hrr_ops.bind(role, filler)
                superposition += bound
            
            # Normalize superposition
            superposition = self._normalize_vector(superposition)
            
            # Test retrieval accuracy for all items in superposition
            correct_retrievals = 0
            for j in range(size):
                role, expected_filler = pairs[j]
                retrieved_filler = self.hrr_ops.unbind(superposition, role)
                similarity = self._cosine_similarity(expected_filler, retrieved_filler)
                
                if similarity > 0.7:
                    correct_retrievals += 1
            
            accuracy = correct_retrievals / size
            results[size] = accuracy
            
            # Stop if accuracy drops below target
            if accuracy < target_accuracy:
                break
        
        return results
    
    def test_capacity_limits(self) -> int:
        """
        Determine practical capacity limit for HRR superposition.
        
        Target: ~0.1 * D items per trace (51 items for 512D)
        
        Returns:
            Maximum number of items at 90% accuracy
        """
        stress_results = self.test_superposition_stress(n_items=100, target_accuracy=0.9)
        
        # Find maximum size with >90% accuracy
        max_capacity = 0
        for size, accuracy in stress_results.items():
            if accuracy >= 0.9:
                max_capacity = size
            else:
                break
                
        return max_capacity
    
    def test_retrieval_accuracy_curve(self, item_counts: List[int] = None) -> Dict[int, float]:
        """
        Generate retrieval accuracy curve for different superposition sizes.
        
        Args:
            item_counts: List of item counts to test
            
        Returns:
            Dictionary mapping item count to accuracy
        """
        if item_counts is None:
            item_counts = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        accuracy_curve = {}
        
        for count in item_counts:
            if count <= 100:  # Reasonable limit for testing
                stress_results = self.test_superposition_stress(n_items=count, target_accuracy=0.0)
                if count in stress_results:
                    accuracy_curve[count] = stress_results[count]
                    
        return accuracy_curve
    
    def test_semantic_drift_prevention(self, operations: int = 1000) -> float:
        """
        Test semantic drift over repeated operations.
        
        Target: <5% drift per 1000 operations
        
        Args:
            operations: Number of operations to perform
            
        Returns:
            Drift rate as percentage
        """
        # Create initial vector
        original = self._generate_random_vector()
        current = original.copy()
        
        # Perform repeated bind/unbind operations
        for i in range(operations):
            role = self._generate_random_vector()
            
            # Bind and immediately unbind (should preserve vector)
            bound = self.hrr_ops.bind(current, role)
            current = self.hrr_ops.unbind(bound, role)
            current = self._normalize_vector(current)
        
        # Calculate drift
        final_similarity = self._cosine_similarity(original, current)
        drift_rate = 1.0 - final_similarity
        
        return drift_rate
    
    def test_performance_benchmarks(self, operations: int = 1000) -> Dict[str, float]:
        """
        Benchmark HRR operation performance.
        
        Args:
            operations: Number of operations to benchmark
            
        Returns:
            Performance metrics in milliseconds
        """
        # Generate test vectors
        vector_a = self._generate_random_vector()
        vector_b = self._generate_random_vector()
        
        # Benchmark bind operations
        start_time = time.time()
        for _ in range(operations):
            self.hrr_ops.bind(vector_a, vector_b)
        bind_time = (time.time() - start_time) * 1000 / operations  # ms per operation
        
        # Benchmark unbind operations
        bound_vector = self.hrr_ops.bind(vector_a, vector_b)
        start_time = time.time()
        for _ in range(operations):
            self.hrr_ops.unbind(bound_vector, vector_a)
        unbind_time = (time.time() - start_time) * 1000 / operations  # ms per operation
        
        return {
            'bind_ms_per_op': bind_time,
            'unbind_ms_per_op': unbind_time,
            'total_ms_per_cycle': bind_time + unbind_time
        }
    
    def generate_comprehensive_metrics(self) -> HRRMetrics:
        """
        Generate comprehensive HRR metrics for validation report.
        
        Returns:
            Complete HRR metrics object
        """
        print("🧮 Generating comprehensive HRR validation metrics...")
        
        # Test bind/unbind accuracy
        print("  Testing bind/unbind correctness...")
        bind_unbind_accuracy = self.test_bind_unbind_correctness(k_pairs=50)
        
        # Test superposition capacity
        print("  Testing superposition capacity...")
        capacity = self.test_capacity_limits()
        
        # Test retrieval accuracy curve
        print("  Generating retrieval accuracy curve...")
        accuracy_curve = self.test_retrieval_accuracy_curve()
        
        # Test semantic drift
        print("  Testing semantic drift prevention...")
        drift_rate = self.test_semantic_drift_prevention(operations=1000)
        
        # Test performance
        print("  Benchmarking performance...")
        performance = self.test_performance_benchmarks(operations=1000)
        
        # Placeholder for temporal decay (to be implemented)
        temporal_decay_accuracy = 0.0  # TODO: Implement temporal testing
        
        metrics = HRRMetrics(
            bind_unbind_accuracy=bind_unbind_accuracy,
            superposition_capacity=capacity,
            retrieval_accuracy_at_k=accuracy_curve,
            semantic_drift_rate=drift_rate,
            temporal_decay_accuracy=temporal_decay_accuracy,
            performance_ms_per_operation=performance['total_ms_per_cycle']
        )
        
        return metrics
    
    def _generate_random_vector(self) -> np.ndarray:
        """Generate a random normalized vector"""
        vector = np.random.randn(self.dimensionality)
        return self._normalize_vector(vector)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Pytest test functions
class TestHRRValidation:
    """Pytest test class for HRR validation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = HRRValidationSuite(dimensionality=512)
    
    def test_bind_unbind_accuracy_target(self):
        """Test that bind/unbind accuracy meets >95% target"""
        accuracy = self.validator.test_bind_unbind_correctness(k_pairs=20)
        assert accuracy > 0.95, f"Bind/unbind accuracy {accuracy:.2%} below 95% target"
    
    def test_superposition_capacity_target(self):
        """Test that superposition capacity meets ~51 items target"""
        capacity = self.validator.test_capacity_limits()
        assert capacity >= 40, f"Superposition capacity {capacity} below expected ~51 items"
        assert capacity <= 70, f"Superposition capacity {capacity} unexpectedly high"
    
    def test_semantic_drift_target(self):
        """Test that semantic drift is <5% per 1000 operations"""
        drift_rate = self.validator.test_semantic_drift_prevention(operations=1000)
        assert drift_rate < 0.05, f"Semantic drift {drift_rate:.2%} exceeds 5% target"
    
    def test_performance_reasonable(self):
        """Test that performance is reasonable for production use"""
        performance = self.validator.test_performance_benchmarks(operations=100)
        total_time = performance['total_ms_per_cycle']
        assert total_time < 10.0, f"Performance {total_time:.2f}ms per cycle too slow"


if __name__ == "__main__":
    # Run comprehensive validation when executed directly
    validator = HRRValidationSuite(dimensionality=512)
    metrics = validator.generate_comprehensive_metrics()
    
    print("\n🎯 HRR VALIDATION RESULTS:")
    print(f"  Bind/Unbind Accuracy: {metrics.bind_unbind_accuracy:.2%}")
    print(f"  Superposition Capacity: {metrics.superposition_capacity} items")
    print(f"  Semantic Drift Rate: {metrics.semantic_drift_rate:.2%}")
    print(f"  Performance: {metrics.performance_ms_per_operation:.3f}ms per operation")
    
    print(f"\n📊 RETRIEVAL ACCURACY CURVE:")
    for items, accuracy in sorted(metrics.retrieval_accuracy_at_k.items()):
        print(f"  {items:2d} items: {accuracy:.2%}")
    
    # Determine if metrics meet targets
    targets_met = (
        metrics.bind_unbind_accuracy > 0.95 and
        metrics.superposition_capacity >= 40 and
        metrics.semantic_drift_rate < 0.05 and
        metrics.performance_ms_per_operation < 10.0
    )
    
    status = "✅ PASSED" if targets_met else "❌ FAILED"
    print(f"\n🎯 VALIDATION STATUS: {status}")