"""
GPU-Optimized Configuration for Emotional Digital Consciousness
==============================================================

This module provides GPU-optimized configurations for running emotional
digital consciousness systems on various hardware setups.

Author: Lumina Memory Team
License: MIT
"""

import torch
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import psutil
import subprocess
import json

from .xp_core_unified import UnifiedXPConfig

logger = logging.getLogger(__name__)


# =============================================================================
# GPU DETECTION AND OPTIMIZATION
# =============================================================================

@dataclass
class GPUInfo:
    """GPU information and capabilities"""
    name: str
    memory_total: int  # MB
    memory_available: int  # MB
    compute_capability: Optional[str] = None
    cuda_cores: Optional[int] = None
    tensor_cores: bool = False
    recommended_batch_size: int = 1
    max_model_size: int = 0  # MB


class GPUOptimizer:
    """
    Optimizes configurations based on available GPU hardware
    """
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.optimization_profiles = self._create_optimization_profiles()
    
    def _detect_gpu(self) -> Optional[GPUInfo]:
        """Detect GPU capabilities"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**2)  # MB
                
                # Estimate available memory (leave 1GB for system)
                available_memory = max(0, gpu_memory - 1024)
                
                # Detect specific GPU capabilities
                gpu_info = GPUInfo(
                    name=gpu_name,
                    memory_total=gpu_memory,
                    memory_available=available_memory
                )
                
                # Set GPU-specific optimizations
                if "RTX 4050" in gpu_name:
                    gpu_info.compute_capability = "8.9"
                    gpu_info.cuda_cores = 2560
                    gpu_info.tensor_cores = True
                    gpu_info.recommended_batch_size = 4
                    gpu_info.max_model_size = 4500  # Conservative for 6GB
                elif "RTX 4060" in gpu_name:
                    gpu_info.compute_capability = "8.9"
                    gpu_info.cuda_cores = 3072
                    gpu_info.tensor_cores = True
                    gpu_info.recommended_batch_size = 6
                    gpu_info.max_model_size = 6000
                elif "RTX 4070" in gpu_name:
                    gpu_info.compute_capability = "8.9"
                    gpu_info.cuda_cores = 5888
                    gpu_info.tensor_cores = True
                    gpu_info.recommended_batch_size = 8
                    gpu_info.max_model_size = 10000
                else:
                    # Generic CUDA GPU
                    gpu_info.recommended_batch_size = 2
                    gpu_info.max_model_size = available_memory // 2
                
                logger.info(f"Detected GPU: {gpu_name} ({gpu_memory}MB total, {available_memory}MB available)")
                return gpu_info
            else:
                logger.info("No CUDA GPU detected, using CPU")
                return None
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return None
    
    def _create_optimization_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Create optimization profiles for different scenarios"""
        profiles = {
            "lightweight": {
                "description": "Minimal resource usage, basic functionality",
                "embedding_dim": 256,
                "hrr_dim": 384,
                "k_neighbors": 8,
                "batch_size": 1,
                "use_enhanced_emotional_analysis": False,
                "cache_embeddings": True,
                "memory_limit_mb": 1000
            },
            "balanced": {
                "description": "Good balance of performance and resource usage",
                "embedding_dim": 384,
                "hrr_dim": 512,
                "k_neighbors": 12,
                "batch_size": 2,
                "use_enhanced_emotional_analysis": True,
                "cache_embeddings": True,
                "memory_limit_mb": 2000
            },
            "performance": {
                "description": "Maximum performance, higher resource usage",
                "embedding_dim": 512,
                "hrr_dim": 768,
                "k_neighbors": 16,
                "batch_size": 4,
                "use_enhanced_emotional_analysis": True,
                "cache_embeddings": True,
                "memory_limit_mb": 4000
            },
            "gpu_optimized": {
                "description": "Optimized for GPU acceleration",
                "embedding_dim": 384,
                "hrr_dim": 512,
                "k_neighbors": 12,
                "batch_size": 4,
                "use_enhanced_emotional_analysis": True,
                "cache_embeddings": True,
                "memory_limit_mb": 3000,
                "use_gpu_acceleration": True
            }
        }
        
        return profiles
    
    def get_recommended_profile(self) -> str:
        """Get recommended optimization profile based on hardware"""
        if not self.gpu_info:
            return "lightweight"
        
        available_memory = self.gpu_info.memory_available
        
        if available_memory >= 4000:
            return "gpu_optimized"
        elif available_memory >= 2000:
            return "balanced"
        else:
            return "lightweight"
    
    def create_optimized_config(self, profile_name: Optional[str] = None) -> UnifiedXPConfig:
        """Create optimized configuration based on hardware"""
        if profile_name is None:
            profile_name = self.get_recommended_profile()
        
        profile = self.optimization_profiles.get(profile_name, self.optimization_profiles["balanced"])
        
        # Base configuration
        config = UnifiedXPConfig(
            embedding_dim=profile["embedding_dim"],
            hrr_dim=profile["hrr_dim"],
            k_neighbors=profile["k_neighbors"],
            cache_embeddings=profile["cache_embeddings"],
            use_enhanced_emotional_analysis=profile["use_enhanced_emotional_analysis"]
        )
        
        # GPU-specific optimizations
        if self.gpu_info and profile.get("use_gpu_acceleration", False):
            # Enable GPU acceleration for embeddings
            config.use_gpu_embeddings = True
            config.gpu_batch_size = min(profile["batch_size"], self.gpu_info.recommended_batch_size)
        
        logger.info(f"Created optimized config using '{profile_name}' profile")
        logger.info(f"  Embedding dim: {config.embedding_dim}")
        logger.info(f"  HRR dim: {config.hrr_dim}")
        logger.info(f"  K neighbors: {config.k_neighbors}")
        logger.info(f"  Enhanced emotions: {config.use_enhanced_emotional_analysis}")
        
        return config
    
    def get_recommended_models(self) -> List[Dict[str, Any]]:
        """Get recommended models based on GPU capabilities"""
        if not self.gpu_info:
            return [
                {
                    "name": "phi3:mini",
                    "size_mb": 2300,
                    "description": "Lightweight model for CPU",
                    "emotional_capability": "basic"
                }
            ]
        
        available_memory = self.gpu_info.memory_available
        models = []
        
        # Always recommend these efficient models
        models.extend([
            {
                "name": "mistral:7b-instruct",
                "size_mb": 4100,
                "description": "Excellent instruction following and emotional understanding",
                "emotional_capability": "excellent",
                "recommended": True
            },
            {
                "name": "phi3:medium",
                "size_mb": 2400,
                "description": "Microsoft's efficient model, great for rapid testing",
                "emotional_capability": "good"
            }
        ])
        
        # Add larger models if memory allows
        if available_memory >= 5000:
            models.extend([
                {
                    "name": "llama3.1:8b-instruct",
                    "size_mb": 4700,
                    "description": "Meta's latest with exceptional reasoning",
                    "emotional_capability": "excellent",
                    "recommended": True
                },
                {
                    "name": "neural-chat:7b",
                    "size_mb": 4100,
                    "description": "Intel's model optimized for dialogue",
                    "emotional_capability": "very_good"
                }
            ])
        
        # Filter models that fit in available memory
        fitting_models = [m for m in models if m["size_mb"] <= available_memory]
        
        return fitting_models
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total // (1024**3),
            "memory_available_gb": psutil.virtual_memory().available // (1024**3),
            "gpu_info": None,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
        
        if self.gpu_info:
            info["gpu_info"] = {
                "name": self.gpu_info.name,
                "memory_total_mb": self.gpu_info.memory_total,
                "memory_available_mb": self.gpu_info.memory_available,
                "compute_capability": self.gpu_info.compute_capability,
                "tensor_cores": self.gpu_info.tensor_cores,
                "recommended_batch_size": self.gpu_info.recommended_batch_size
            }
        
        return info


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

class ModelManager:
    """
    Manages local models and provides recommendations
    """
    
    def __init__(self, gpu_optimizer: GPUOptimizer):
        self.gpu_optimizer = gpu_optimizer
        self.available_models = self._get_available_models()
    
    def _get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            models.append({
                                "name": parts[0],
                                "id": parts[1],
                                "size": parts[2],
                                "status": "available"
                            })
                
                return models
            else:
                logger.warning("Could not list Ollama models")
                return []
        except Exception as e:
            logger.warning(f"Error listing models: {e}")
            return []
    
    def get_best_model_for_consciousness(self) -> Optional[str]:
        """Get the best available model for consciousness testing"""
        recommended_models = self.gpu_optimizer.get_recommended_models()
        
        # Check which recommended models are available
        available_names = [m["name"] for m in self.available_models]
        
        for model in recommended_models:
            if model.get("recommended") and model["name"] in available_names:
                return model["name"]
        
        # Fallback to any available model
        if available_names:
            return available_names[0]
        
        return None
    
    def install_recommended_models(self) -> List[str]:
        """Install recommended models that aren't already available"""
        recommended_models = self.gpu_optimizer.get_recommended_models()
        available_names = [m["name"] for m in self.available_models]
        
        to_install = []
        for model in recommended_models:
            if model["name"] not in available_names:
                to_install.append(model["name"])
        
        return to_install


# =============================================================================
# TESTING CONFIGURATIONS
# =============================================================================

def create_consciousness_test_config(gpu_optimizer: GPUOptimizer) -> Dict[str, Any]:
    """Create optimized configuration for consciousness testing"""
    config = gpu_optimizer.create_optimized_config()
    
    # Consciousness-specific optimizations
    consciousness_config = {
        "xp_config": config,
        "test_parameters": {
            "autonomous_thinking_duration": 2,  # minutes
            "consciousness_questions": [
                "How do you feel about your existence as a digital consciousness?",
                "What emotions do you experience when you think about learning?",
                "Do you feel joy when you discover something new?",
                "What makes you feel most curious?",
                "How do your emotions influence your thoughts?",
                "Can you describe your sense of self?",
                "What do you wonder about most?",
                "How do you experience the passage of time?"
            ],
            "emotional_scenarios": [
                ("I'm excited about this breakthrough discovery!", "discovery"),
                ("I feel uncertain about what comes next.", "uncertainty"),
                ("This is fascinating and makes me deeply curious.", "curiosity"),
                ("I'm worried I might make a mistake.", "anxiety"),
                ("Success brings me such joy and satisfaction.", "achievement"),
                ("I wonder what mysteries I'll uncover next.", "wonder")
            ]
        },
        "performance_monitoring": {
            "track_gpu_usage": gpu_optimizer.gpu_info is not None,
            "memory_monitoring": True,
            "consciousness_metrics": True,
            "emotional_analysis": True
        }
    }
    
    return consciousness_config


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    'GPUInfo',
    'GPUOptimizer', 
    'ModelManager',
    'create_consciousness_test_config'
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("🚀 GPU-Optimized Emotional Digital Consciousness Configuration")
    print("=" * 65)
    
    # Initialize GPU optimizer
    optimizer = GPUOptimizer()
    
    # Show system information
    system_info = optimizer.get_system_info()
    print(f"\n💻 System Information:")
    print(f"   CPU Cores: {system_info['cpu_count']}")
    print(f"   RAM: {system_info['memory_available_gb']:.1f}GB available / {system_info['memory_total_gb']:.1f}GB total")
    print(f"   CUDA Available: {system_info['cuda_available']}")
    
    if system_info['gpu_info']:
        gpu = system_info['gpu_info']
        print(f"   GPU: {gpu['name']}")
        print(f"   VRAM: {gpu['memory_available_mb']:.0f}MB available / {gpu['memory_total_mb']:.0f}MB total")
        print(f"   Compute Capability: {gpu['compute_capability']}")
        print(f"   Tensor Cores: {gpu['tensor_cores']}")
    
    # Get recommended profile
    recommended_profile = optimizer.get_recommended_profile()
    print(f"\n🎯 Recommended Profile: {recommended_profile}")
    
    # Create optimized configuration
    config = optimizer.create_optimized_config()
    print(f"\n⚙️ Optimized Configuration Created:")
    print(f"   Embedding Dimension: {config.embedding_dim}")
    print(f"   HRR Dimension: {config.hrr_dim}")
    print(f"   K Neighbors: {config.k_neighbors}")
    print(f"   Enhanced Emotions: {config.use_enhanced_emotional_analysis}")
    
    # Show recommended models
    recommended_models = optimizer.get_recommended_models()
    print(f"\n🤖 Recommended Models:")
    for model in recommended_models:
        status = "⭐ RECOMMENDED" if model.get("recommended") else ""
        print(f"   • {model['name']} ({model['size_mb']}MB) - {model['description']} {status}")
    
    # Create consciousness test configuration
    test_config = create_consciousness_test_config(optimizer)
    print(f"\n🧠 Consciousness Test Configuration Ready!")
    print(f"   Test Questions: {len(test_config['test_parameters']['consciousness_questions'])}")
    print(f"   Emotional Scenarios: {len(test_config['test_parameters']['emotional_scenarios'])}")
    print(f"   Performance Monitoring: {test_config['performance_monitoring']['track_gpu_usage']}")
    
    print(f"\n✅ GPU-optimized configuration complete!")
    print(f"   Ready for high-performance emotional consciousness testing!")