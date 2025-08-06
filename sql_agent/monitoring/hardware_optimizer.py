"""
Hardware Optimizer for Local LLM Performance

Intelligent hardware detection and optimization for Ollama-based schema analysis.
Designed for NVIDIA GPUs (4090 24GB Desktop / 4090 16GB Laptop) with automatic
performance tuning for 200+ table database analysis.

Design Principles:
- Runtime hardware detection and profiling
- Dynamic optimization based on current system state
- Performance benchmarking for optimal settings
- Graceful degradation for unknown hardware

Author: ML Engineering Team
"""

import asyncio
import platform
import psutil
import subprocess
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import json
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU hardware information."""
    name: str
    memory_total_gb: float
    memory_free_gb: float
    memory_used_gb: float
    utilization_percent: float
    temperature_c: Optional[float]
    driver_version: str


@dataclass
class SystemInfo:
    """Complete system information."""
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    storage_type: str  # "ssd", "hdd", "nvme"
    platform: str
    gpu_info: Optional[GPUInfo]


@dataclass
class PerformanceBenchmark:
    """LLM performance benchmark results."""
    model_name: str
    tokens_per_second: float
    memory_usage_gb: float
    concurrent_capacity: int
    context_window_limit: int
    benchmark_timestamp: datetime


@dataclass
class OptimizationProfile:
    """Hardware-optimized configuration profile."""
    profile_name: str
    max_concurrent_requests: int
    batch_size: int
    recommended_model: str
    analysis_mode: str
    memory_buffer_mb: int
    enable_parallel: bool
    estimated_table_processing_time_seconds: float


class HardwareDetector:
    """Hardware detection and profiling system."""
    
    def __init__(self):
        self.gpu_available = False
        self.nvidia_smi_available = False
        self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check if GPU monitoring tools are available."""
        try:
            subprocess.run(["nvidia-smi", "--version"], 
                         capture_output=True, check=True, timeout=5)
            self.nvidia_smi_available = True
            self.gpu_available = True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.nvidia_smi_available = False
            self.gpu_available = False
    
    async def detect_system_info(self) -> SystemInfo:
        """Detect comprehensive system information."""
        try:
            # CPU information
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            
            # Memory information
            memory = psutil.virtual_memory()
            ram_total_gb = memory.total / (1024**3)
            ram_available_gb = memory.available / (1024**3)
            
            # Storage type detection
            storage_type = await self._detect_storage_type()
            
            # GPU information
            gpu_info = await self._detect_gpu_info() if self.gpu_available else None
            
            return SystemInfo(
                cpu_cores=cpu_cores,
                cpu_threads=cpu_threads,
                ram_total_gb=ram_total_gb,
                ram_available_gb=ram_available_gb,
                storage_type=storage_type,
                platform=platform.system(),
                gpu_info=gpu_info
            )
            
        except Exception as e:
            logger.error(f"System detection failed: {e}")
            # Return minimal fallback info
            return SystemInfo(
                cpu_cores=4, cpu_threads=8, ram_total_gb=16.0, ram_available_gb=8.0,
                storage_type="unknown", platform=platform.system(), gpu_info=None
            )
    
    async def _detect_gpu_info(self) -> Optional[GPUInfo]:
        """Detect GPU information using nvidia-smi."""
        if not self.nvidia_smi_available:
            return None
        
        try:
            # Query GPU information
            cmd = [
                "nvidia-smi", 
                "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,driver_version",
                "--format=csv,noheader,nounits"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            # Parse first GPU (assuming single GPU setup)
            lines = result.stdout.strip().split('\n')
            if not lines:
                return None
            
            gpu_data = lines[0].split(', ')
            if len(gpu_data) < 7:
                return None
            
            return GPUInfo(
                name=gpu_data[0].strip(),
                memory_total_gb=float(gpu_data[1]) / 1024,
                memory_free_gb=float(gpu_data[2]) / 1024,
                memory_used_gb=float(gpu_data[3]) / 1024,
                utilization_percent=float(gpu_data[4]),
                temperature_c=float(gpu_data[5]) if gpu_data[5] != "N/A" else None,
                driver_version=gpu_data[6].strip()
            )
            
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return None
    
    async def _detect_storage_type(self) -> str:
        """Detect primary storage type."""
        try:
            if platform.system() == "Linux":
                # Check for NVMe/SSD on Linux
                result = subprocess.run(["lsblk", "-d", "-o", "name,rota"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and "0" in result.stdout:
                    return "ssd"
            
            elif platform.system() == "Windows":
                # Basic Windows storage detection
                try:
                    result = subprocess.run(["wmic", "diskdrive", "get", "MediaType"], 
                                          capture_output=True, text=True, timeout=5)
                    if "SSD" in result.stdout:
                        return "ssd"
                except:
                    pass
            
            return "unknown"
            
        except Exception:
            return "unknown"


class OllamaModelDiscovery:
    """Dynamic Ollama model discovery and evaluation."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.model_cache: Dict[str, Dict[str, Any]] = {}
    
    async def discover_available_models(self) -> List[Dict[str, Any]]:
        """Discover all available models via Ollama API."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_info in data.get("models", []):
                        model_data = {
                            "name": model_info.get("name", ""),
                            "size_gb": model_info.get("size", 0) / (1024**3),
                            "family": self._extract_model_family(model_info.get("name", "")),
                            "parameters": self._estimate_parameters(model_info.get("size", 0)),
                            "modified_at": model_info.get("modified_at", ""),
                            "available": True
                        }
                        models.append(model_data)
                    
                    return models
                else:
                    logger.warning(f"Failed to fetch models: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            return []
    
    def _extract_model_family(self, model_name: str) -> str:
        """Extract model family from name."""
        name_lower = model_name.lower()
        
        families = {
            "llama": ["llama", "llama2", "llama3"],
            "mistral": ["mistral"],
            "codellama": ["codellama", "code-llama"],
            "phi": ["phi"],
            "qwen": ["qwen"],
            "gemma": ["gemma"],
            "deepseek": ["deepseek"],
            "tinyllama": ["tinyllama", "tiny-llama"]
        }
        
        for family, variants in families.items():
            if any(variant in name_lower for variant in variants):
                return family
        
        return "unknown"
    
    def _estimate_parameters(self, size_bytes: int) -> int:
        """Estimate parameter count from model size."""
        size_gb = size_bytes / (1024**3)
        
        # Rough estimation based on typical model sizes
        if size_gb < 2:
            return 1_000_000_000    # 1B
        elif size_gb < 5:
            return 3_000_000_000    # 3B
        elif size_gb < 8:
            return 7_000_000_000    # 7B
        elif size_gb < 15:
            return 13_000_000_000   # 13B
        elif size_gb < 25:
            return 20_000_000_000   # 20B
        else:
            return 30_000_000_000   # 30B+
    
    async def evaluate_model_for_schema_analysis(self, model_name: str) -> Dict[str, Any]:
        """Evaluate specific model for schema analysis tasks."""
        try:
            # Quick benchmark
            benchmark = await self._quick_benchmark_model(model_name)
            if not benchmark:
                return {"suitable": False, "reason": "benchmark_failed"}
            
            # Calculate suitability score
            score = self._calculate_suitability_score(model_name, benchmark)
            
            return {
                "model_name": model_name,
                "suitable": score >= 0.6,
                "suitability_score": score,
                "benchmark_results": benchmark,
                "recommendations": self._generate_model_recommendations(score, benchmark)
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed for {model_name}: {e}")
            return {"suitable": False, "reason": f"evaluation_error: {str(e)}"}
    
    async def _quick_benchmark_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Quick benchmark of model performance."""
        test_prompt = "Analyze this table structure: users (id, name, email, created_at). What business domain does this represent?"
        
        try:
            start_time = time.time()
            
            payload = {
                "model": model_name,
                "prompt": test_prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }
            
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    duration = time.time() - start_time
                    
                    response_text = data.get("response", "")
                    tokens = len(response_text.split()) * 1.3  # Rough estimation
                    
                    return {
                        "tokens_per_second": tokens / duration if duration > 0 else 0,
                        "response_time_seconds": duration,
                        "response_quality": self._assess_response_quality(response_text),
                        "memory_stable": True  # Assume stable if request succeeded
                    }
                else:
                    return None
                    
        except Exception as e:
            logger.warning(f"Quick benchmark failed for {model_name}: {e}")
            return None
    
    def _calculate_suitability_score(self, model_name: str, benchmark: Dict[str, Any]) -> float:
        """Calculate model suitability score for schema analysis."""
        score = 0.0
        
        # Speed component (0-0.3)
        tps = benchmark.get("tokens_per_second", 0)
        if tps >= 30:
            score += 0.3
        elif tps >= 15:
            score += 0.2
        elif tps >= 5:
            score += 0.1
        
        # Quality component (0-0.4)
        quality = benchmark.get("response_quality", 0)
        score += quality * 0.4
        
        # Model family bonus (0-0.15)
        family = self._extract_model_family(model_name)
        family_scores = {
            "mistral": 0.15,
            "llama": 0.12,
            "codellama": 0.15,
            "qwen": 0.10,
            "phi": 0.08,
            "gemma": 0.10
        }
        score += family_scores.get(family, 0.0)
        
        # Parameter count bonus (0-0.15)
        if "13b" in model_name.lower():
            score += 0.15
        elif "7b" in model_name.lower():
            score += 0.12
        elif "3b" in model_name.lower():
            score += 0.08
        
        return min(score, 1.0)
    
    def _assess_response_quality(self, response: str) -> float:
        """Assess response quality for schema analysis."""
        if not response or len(response) < 20:
            return 0.0
        
        quality_indicators = [
            "business" in response.lower(),
            "table" in response.lower() or "database" in response.lower(),
            "analysis" in response.lower() or "purpose" in response.lower(),
            len(response.split()) > 10,  # Sufficient detail
            len(response.split()) < 200,  # Not too verbose
        ]
        
        return sum(quality_indicators) / len(quality_indicators)
    
    async def get_best_models_for_hardware(self, gpu_memory_gb: float, 
                                         strategy: str = "balanced") -> List[Dict[str, Any]]:
        """Get best models for specific hardware configuration."""
        available_models = await self.discover_available_models()
        
        if not available_models:
            return []
        
        suitable_models = []
        
        for model in available_models:
            # Check memory compatibility
            estimated_memory = self._estimate_memory_usage(model["parameters"])
            if estimated_memory <= gpu_memory_gb * 0.8:  # 80% memory usage limit
                
                # Evaluate model
                evaluation = await self.evaluate_model_for_schema_analysis(model["name"])
                if evaluation.get("suitable", False):
                    model_info = {
                        **model,
                        **evaluation,
                        "estimated_memory_gb": estimated_memory
                    }
                    suitable_models.append(model_info)
        
        # Sort by suitability score
        suitable_models.sort(key=lambda x: x.get("suitability_score", 0), reverse=True)
        
        return suitable_models[:5]  # Return top 5
    
    def _estimate_memory_usage(self, parameters: int) -> float:
        """Estimate GPU memory usage based on parameter count."""
        # Rough estimation: 2 bytes per parameter + overhead
        base_memory_gb = (parameters * 2) / (1024**3)
        overhead_factor = 1.3  # 30% overhead for inference
        return base_memory_gb * overhead_factor


class OllamaBenchmarker:
    """Benchmark Ollama performance for optimization."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.model_discovery = OllamaModelDiscovery(base_url)
    
    async def discover_and_recommend_models(self, gpu_memory_gb: float) -> Dict[str, Any]:
        """Discover available models and recommend best ones for hardware."""
        try:
            # Discover available models
            available_models = await self.model_discovery.discover_available_models()
            
            if not available_models:
                return {
                    "available_models": [],
                    "recommended_models": [],
                    "fallback_recommendation": "install_models",
                    "suggested_downloads": ["mistral:7b", "llama3.1:7b"]
                }
            
            # Get best models for hardware
            best_models = await self.model_discovery.get_best_models_for_hardware(gpu_memory_gb)
            
            # Extract recommendations
            recommended = []
            for model in best_models[:3]:  # Top 3 recommendations
                recommended.append({
                    "name": model["name"],
                    "family": model["family"],
                    "suitability_score": model.get("suitability_score", 0),
                    "estimated_memory_gb": model.get("estimated_memory_gb", 0),
                    "reasoning": model.get("recommendations", [])
                })
            
            return {
                "available_models": [m["name"] for m in available_models],
                "recommended_models": recommended,
                "total_discovered": len(available_models),
                "evaluation_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            return {
                "available_models": [],
                "recommended_models": [],
                "error": str(e),
                "fallback_recommendation": "check_ollama_service"
            }
        """Benchmark specific Ollama model performance."""
        if not test_prompts:
            test_prompts = [
                "Analyze this table structure: users (id, name, email, created_at)",
                "What business domain does this represent: orders, customers, products, payments",
                "Explain the relationship between customer_id and customers table"
            ]
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
            
            # Test model availability
            if not await self._is_model_available(model_name):
                logger.warning(f"Model {model_name} not available")
                return None
            
            total_tokens = 0
            total_time = 0
            memory_samples = []
            
            for prompt in test_prompts:
                tokens, duration, memory_gb = await self._benchmark_single_request(model_name, prompt)
                if tokens > 0:
                    total_tokens += tokens
                    total_time += duration
                    memory_samples.append(memory_gb)
            
            if total_time == 0:
                return None
            
            tokens_per_second = total_tokens / total_time
            avg_memory_gb = sum(memory_samples) / len(memory_samples) if memory_samples else 0
            
            # Test concurrent capacity
            concurrent_capacity = await self._test_concurrent_capacity(model_name)
            
            return PerformanceBenchmark(
                model_name=model_name,
                tokens_per_second=tokens_per_second,
                memory_usage_gb=avg_memory_gb,
                concurrent_capacity=concurrent_capacity,
                context_window_limit=4096,  # Default assumption
                benchmark_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            return None
    
    async def _is_model_available(self, model_name: str) -> bool:
        """Check if model is available in Ollama."""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model.get("name", "") for model in data.get("models", [])]
                    return any(model_name in model for model in models)
                return False
        except Exception:
            return False
    
    async def _benchmark_single_request(self, model_name: str, prompt: str) -> Tuple[int, float, float]:
        """Benchmark single request and return (tokens, duration, memory_gb)."""
        try:
            # Get initial memory
            initial_memory = await self._get_gpu_memory_usage()
            
            start_time = time.time()
            
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    duration = time.time() - start_time
                    
                    # Estimate tokens (rough approximation)
                    response_text = data.get("response", "")
                    tokens = len(response_text.split()) * 1.3  # Rough token estimation
                    
                    # Get peak memory usage
                    peak_memory = await self._get_gpu_memory_usage()
                    memory_used = max(0, peak_memory - initial_memory)
                    
                    return int(tokens), duration, memory_used
                else:
                    return 0, 0, 0
                    
        except Exception as e:
            logger.warning(f"Benchmark request failed: {e}")
            return 0, 0, 0
    
    async def _test_concurrent_capacity(self, model_name: str) -> int:
        """Test maximum concurrent request capacity."""
        try:
            test_prompt = "Brief analysis of: id, name, email"
            
            # Test increasing concurrent requests
            for concurrent in [1, 2, 3, 4]:
                start_time = time.time()
                
                tasks = []
                for _ in range(concurrent):
                    tasks.append(self._benchmark_single_request(model_name, test_prompt))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start_time
                
                # Check if all requests succeeded
                successful = sum(1 for r in results if isinstance(r, tuple) and r[0] > 0)
                
                # If performance degrades significantly or failures occur
                if successful < concurrent or duration > concurrent * 8:  # 8 seconds per request threshold
                    return max(1, concurrent - 1)
            
            return 4  # Maximum tested
            
        except Exception as e:
            logger.warning(f"Concurrent capacity test failed: {e}")
            return 1
    
    async def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        try:
            if not self.nvidia_smi_available:
                return 0.0
            
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                memory_mb = float(result.stdout.strip().split('\n')[0])
                return memory_mb / 1024
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()


class HardwareOptimizer:
    """
    Main hardware optimization system.
    
    Provides intelligent configuration recommendations based on detected
    hardware capabilities and runtime performance benchmarks.
    """
    
    def __init__(self):
        self.detector = HardwareDetector()
        self.benchmarker = OllamaBenchmarker()
        self._system_info: Optional[SystemInfo] = None
        self._benchmark_cache: Dict[str, PerformanceBenchmark] = {}
    
    async def detect_and_optimize(self, ollama_base_url: str = "http://localhost:11434") -> Dict[str, Any]:
        """
        Detect hardware and provide optimization recommendations with dynamic model discovery.
        
        Returns:
            Complete optimization recommendations with discovered models
        """
        try:
            # Update URLs
            self.benchmarker.base_url = ollama_base_url.rstrip('/')
            self.benchmarker.model_discovery.base_url = ollama_base_url.rstrip('/')
            
            # Detect system capabilities
            self._system_info = await self.detector.detect_system_info()
            
            # Test Ollama connectivity
            ollama_available = await self._test_ollama_connectivity()
            
            if not ollama_available:
                return self._get_fallback_optimization()
            
            # Discover and evaluate models
            gpu_memory = self._system_info.gpu_info.memory_total_gb if self._system_info.gpu_info else 8.0
            model_recommendations = await self.benchmarker.discover_and_recommend_models(gpu_memory)
            
            # Get optimization profile based on discovered models
            optimization_profile = self._create_optimization_profile()
            
            return {
                "system_info": self._system_info.__dict__ if self._system_info else {},
                "optimization_profile": optimization_profile.__dict__,
                "model_recommendations": model_recommendations,
                "ollama_available": ollama_available,
                "recommendations": self._generate_recommendations(),
                "optimization_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Hardware optimization failed: {e}")
            return self._get_fallback_optimization()
    
    def _create_optimization_profile(self) -> OptimizationProfile:
        """Create optimization profile based on detected hardware."""
        if not self._system_info or not self._system_info.gpu_info:
            return self._get_generic_profile()
        
        gpu = self._system_info.gpu_info
        
        # Profile selection based on GPU memory
        if gpu.memory_total_gb >= 24:
            return OptimizationProfile(
                profile_name="desktop_24gb",
                max_concurrent_requests=4,
                batch_size=15,
                recommended_model="auto_select",  # Will be determined dynamically
                analysis_mode="deep",
                memory_buffer_mb=4096,
                enable_parallel=True,
                estimated_table_processing_time_seconds=4.0
            )
        elif gpu.memory_total_gb >= 16:
            return OptimizationProfile(
                profile_name="laptop_16gb",
                max_concurrent_requests=2,
                batch_size=10,
                recommended_model="auto_select",
                analysis_mode="standard", 
                memory_buffer_mb=2048,
                enable_parallel=True,
                estimated_table_processing_time_seconds=6.0
            )
        elif gpu.memory_total_gb >= 8:
            return OptimizationProfile(
                profile_name="gpu_8gb",
                max_concurrent_requests=1,
                batch_size=5,
                recommended_model="auto_select",
                analysis_mode="quick",
                memory_buffer_mb=1024,
                enable_parallel=False,
                estimated_table_processing_time_seconds=10.0
            )
        else:
            return self._get_generic_profile()
    
    def _get_generic_profile(self) -> OptimizationProfile:
        """Generic profile for unknown hardware."""
        return OptimizationProfile(
            profile_name="generic",
            max_concurrent_requests=1,
            batch_size=5,
            recommended_model="auto_select",
            analysis_mode="quick",
            memory_buffer_mb=1024,
            enable_parallel=False,
            estimated_table_processing_time_seconds=15.0
        )
    
    async def _test_ollama_connectivity(self) -> bool:
        """Test if Ollama service is accessible."""
        try:
            if not self.benchmarker.session:
                self.benchmarker.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                )
            
            async with self.benchmarker.session.get(f"{self.benchmarker.base_url}/api/tags") as response:
                return response.status == 200
                
        except Exception as e:
            logger.warning(f"Ollama connectivity test failed: {e}")
            return False
    
    async def _quick_benchmark(self) -> Optional[PerformanceBenchmark]:
        """Run quick performance benchmark if system is capable."""
        try:
            if not self._system_info or not self._system_info.gpu_info:
                return None
            
            # Only benchmark on capable hardware to save time
            if self._system_info.gpu_info.memory_total_gb < 8:
                return None
            
            # Use lightweight test for quick results
            return await self.benchmarker.benchmark_model(
                "mistral:7b",
                test_prompts=["Analyze: users table with id, name, email columns"]
            )
            
        except Exception as e:
            logger.warning(f"Quick benchmark failed: {e}")
            return None
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on detected hardware."""
        recommendations = []
        
        if not self._system_info:
            recommendations.append("Hardware detection failed - using conservative settings")
            return recommendations
        
        gpu = self._system_info.gpu_info
        
        if not gpu:
            recommendations.append("No GPU detected - consider GPU acceleration for better performance")
            recommendations.append("CPU-only processing will be significantly slower for large schemas")
            return recommendations
        
        # GPU-specific recommendations
        if "4090" in gpu.name:
            if gpu.memory_total_gb >= 24:
                recommendations.append("Excellent hardware detected - enabling aggressive optimization")
                recommendations.append("Consider deep analysis mode for comprehensive schema intelligence")
            else:
                recommendations.append("Good hardware detected - enabling standard optimization")
                recommendations.append("Monitor GPU memory usage during large schema analysis")
        
        # Temperature recommendations
        if gpu.temperature_c and gpu.temperature_c > 80:
            recommendations.append("GPU temperature high - consider reducing concurrent requests")
        
        # Memory recommendations
        memory_usage_percent = (gpu.memory_used_gb / gpu.memory_total_gb) * 100
        if memory_usage_percent > 90:
            recommendations.append("GPU memory nearly full - reduce batch sizes or concurrent requests")
        elif memory_usage_percent < 20:
            recommendations.append("GPU memory underutilized - consider increasing batch sizes")
        
        # RAM recommendations
        if self._system_info.ram_available_gb < 4:
            recommendations.append("Low available RAM - consider closing other applications")
        
        return recommendations
    
    def _get_fallback_optimization(self) -> Dict[str, Any]:
        """Fallback optimization when detection fails."""
        profile = self._get_generic_profile()
        
        return {
            "system_info": {"error": "Detection failed"},
            "optimization_profile": profile.__dict__,
            "benchmark_results": None,
            "ollama_available": False,
            "recommendations": [
                "Hardware detection failed - using conservative settings",
                "Verify Ollama installation and GPU drivers",
                "Manual configuration may be required"
            ],
            "optimization_timestamp": datetime.utcnow().isoformat()
        }
    
    def estimate_schema_analysis_time(self, table_count: int, profile: OptimizationProfile) -> Dict[str, float]:
        """Estimate analysis time for given table count and profile."""
        base_time_per_table = profile.estimated_table_processing_time_seconds
        
        if profile.enable_parallel:
            parallel_factor = min(profile.max_concurrent_requests, table_count) / table_count
            effective_time_per_table = base_time_per_table * (1 - parallel_factor * 0.6)
        else:
            effective_time_per_table = base_time_per_table
        
        total_seconds = table_count * effective_time_per_table
        
        return {
            "total_minutes": total_seconds / 60,
            "total_seconds": total_seconds,
            "per_table_seconds": effective_time_per_table,
            "parallel_efficiency": 0.6 if profile.enable_parallel else 0.0
        }
    
    async def get_runtime_optimization_adjustments(self) -> Dict[str, Any]:
        """Get real-time optimization adjustments based on current system state."""
        try:
            if not self._system_info:
                self._system_info = await self.detector.detect_system_info()
            
            adjustments = {
                "timestamp": datetime.utcnow().isoformat(),
                "adjustments": []
            }
            
            gpu = self._system_info.gpu_info
            if gpu:
                # Temperature-based adjustments
                if gpu.temperature_c and gpu.temperature_c > 85:
                    adjustments["adjustments"].append({
                        "type": "reduce_concurrent",
                        "reason": f"High GPU temperature: {gpu.temperature_c}°C",
                        "recommendation": "Reduce concurrent requests by 50%"
                    })
                
                # Memory-based adjustments
                memory_usage = (gpu.memory_used_gb / gpu.memory_total_gb) * 100
                if memory_usage > 95:
                    adjustments["adjustments"].append({
                        "type": "reduce_batch_size",
                        "reason": f"High GPU memory usage: {memory_usage:.1f}%",
                        "recommendation": "Reduce batch size by 30%"
                    })
                
                # Utilization-based adjustments
                if gpu.utilization_percent > 95:
                    adjustments["adjustments"].append({
                        "type": "queue_requests",
                        "reason": f"High GPU utilization: {gpu.utilization_percent}%",
                        "recommendation": "Queue additional requests"
                    })
            
            # RAM-based adjustments
            if self._system_info.ram_available_gb < 2:
                adjustments["adjustments"].append({
                    "type": "reduce_cache",
                    "reason": f"Low available RAM: {self._system_info.ram_available_gb:.1f}GB",
                    "recommendation": "Reduce cache sizes"
                })
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Runtime optimization failed: {e}")
            return {"error": str(e), "adjustments": []}
    
    async def close(self):
        """Cleanup resources."""
        await self.benchmarker.close()


# Global hardware optimizer instance
_hardware_optimizer: Optional[HardwareOptimizer] = None


async def get_hardware_optimizer() -> HardwareOptimizer:
    """Get global hardware optimizer instance."""
    global _hardware_optimizer
    if _hardware_optimizer is None:
        _hardware_optimizer = HardwareOptimizer()
    return _hardware_optimizer


async def optimize_for_hardware(ollama_base_url: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Main entry point for hardware optimization.
    
    Args:
        ollama_base_url: Ollama service URL
        
    Returns:
        Complete optimization results and recommendations
    """
    optimizer = await get_hardware_optimizer()
    return await optimizer.detect_and_optimize(ollama_base_url)