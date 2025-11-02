"""
Enhanced Sparse Instance Performance Benchmark for CapyMOA

This script comprehensively benchmarks sparse vs dense storage across all dimensions:
1. Instance creation time
2. Memory usage
3. Java conversion overhead
4. Feature access patterns (sequential, random, sparse iteration)
5. Training speed with real MOA classifiers
6. Scalability analysis
7. Real-world data comparison

Author: CapyMOA Team
Enhanced Version
"""

import time
import numpy as np
import psutil
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import gc
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.stream import Schema
from capymoa.instance import LabeledInstance
from capymoa._sparse_utils import SparseConfig, get_storage_info

# ============================================================================
# Enhanced Configuration
# ============================================================================

class BenchmarkConfig:
    """Configuration for benchmarks."""
    
    # Test dimensions - 增加更多维度范围
    DIMENSIONS = [100, 500, 1000, 5000, 10000, 50000, 100000]
    
    # Sparsity levels - 增加边界情况
    SPARSITY_LEVELS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    
    # Number of instances to create
    NUM_INSTANCES = 1000
    
    # Number of warmup iterations
    WARMUP_ITERATIONS = 10
    
    # Results directory
    RESULTS_DIR = "benchmark_results_enhanced"
    
    # Plot settings
    PLOT_DPI = 300
    PLOT_FIGSIZE = (14, 10)
    
    # ➕ 新增：特征访问测试
    ACCESS_PATTERNS = ['sequential', 'random', 'sparse_only']
    NUM_ACCESS_OPERATIONS = 10000


# ============================================================================
# Data Generation
# ============================================================================

def generate_sparse_data(
    num_samples: int,
    num_features: int,
    sparsity: float,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sparse data with controlled sparsity."""
    np.random.seed(random_seed)
    
    # Calculate number of non-zero elements per sample
    num_nonzero = max(1, int(num_features * (1 - sparsity)))
    
    # Generate sparse data
    X = np.zeros((num_samples, num_features))
    
    for i in range(num_samples):
        # Randomly select positions for non-zero values
        nonzero_indices = np.random.choice(
            num_features,
            size=num_nonzero,
            replace=False
        )
        # Assign random values
        X[i, nonzero_indices] = np.random.randn(num_nonzero)
    
    # Generate binary labels
    y = np.random.randint(0, 2, size=num_samples)
    
    return X, y


def generate_realistic_sparse_data(
    num_samples: int,
    num_features: int,
    sparsity: float,
    clustering: float = 0.8,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic sparse data with clustered non-zero values.
    
    Args:
        num_samples: Number of samples.
        num_features: Number of features.
        sparsity: Proportion of zero values.
        clustering: How clustered non-zero values are (0=random, 1=highly clustered).
        random_seed: Random seed.
    
    Returns:
        Tuple of (X, y).
    """
    np.random.seed(random_seed)
    
    X = np.zeros((num_samples, num_features))
    num_nonzero = max(1, int(num_features * (1 - sparsity)))
    
    # Create cluster centers
    num_clusters = max(1, int(num_features * 0.1))
    cluster_centers = np.random.choice(num_features, num_clusters, replace=False)
    
    for i in range(num_samples):
        if np.random.rand() < clustering:
            # Clustered: select features near cluster centers
            center = np.random.choice(cluster_centers)
            radius = max(1, int(num_features * 0.05))
            
            # Generate indices around center
            indices = []
            for _ in range(num_nonzero):
                offset = np.random.randint(-radius, radius)
                idx = (center + offset) % num_features
                if idx not in indices:
                    indices.append(idx)
            
            X[i, indices[:num_nonzero]] = np.random.randn(len(indices[:num_nonzero]))
        else:
            # Random
            nonzero_indices = np.random.choice(num_features, num_nonzero, replace=False)
            X[i, nonzero_indices] = np.random.randn(num_nonzero)
    
    y = np.random.randint(0, 2, size=num_samples)
    
    return X, y


# ============================================================================
# Memory Monitoring
# ============================================================================

def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self):
        self.baseline = get_memory_usage()
        self.peak = self.baseline
        self.samples = []
    
    def update(self):
        """Update peak memory usage."""
        current = get_memory_usage()
        self.peak = max(self.peak, current)
        self.samples.append(current)
    
    def get_delta(self) -> float:
        """Get memory increase from baseline in MB."""
        return self.peak - self.baseline
    
    def get_average(self) -> float:
        """Get average memory usage."""
        if not self.samples:
            return 0.0
        return np.mean(self.samples) - self.baseline
    
    def reset(self):
        """Reset baseline and peak."""
        gc.collect()  # Force garbage collection
        self.baseline = get_memory_usage()
        self.peak = self.baseline
        self.samples = []


# ============================================================================
# ➕ 新增：Feature Access Benchmarks
# ============================================================================

def benchmark_feature_access(
    instances: List[LabeledInstance],
    access_pattern: str,
    num_operations: int = 10000
) -> Dict[str, float]:
    """
    Benchmark feature access patterns.
    
    Args:
        instances: List of instances to access.
        access_pattern: 'sequential', 'random', or 'sparse_only'.
        num_operations: Number of access operations.
    
    Returns:
        Dictionary with timing results.
    """
    if not instances:
        return {'avg_time': 0.0, 'total_time': 0.0}
    
    num_features = len(instances[0].x)
    
    # Warmup
    for _ in range(100):
        _ = instances[0].x[0]
    
    start_time = time.perf_counter()
    
    if access_pattern == 'sequential':
        # Sequential access pattern
        for _ in range(num_operations):
            inst = instances[np.random.randint(len(instances))]
            for i in range(min(100, num_features)):  # Access first 100 features
                _ = inst.x[i]
    
    elif access_pattern == 'random':
        # Random access pattern
        for _ in range(num_operations):
            inst = instances[np.random.randint(len(instances))]
            idx = np.random.randint(num_features)
            _ = inst.x[idx]
    
    elif access_pattern == 'sparse_only':
        # Access only non-zero features
        for _ in range(num_operations):
            inst = instances[np.random.randint(len(instances))]
            nonzero_indices = np.nonzero(inst.x)[0]
            if len(nonzero_indices) > 0:
                idx = nonzero_indices[np.random.randint(len(nonzero_indices))]
                _ = inst.x[idx]
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / num_operations
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'ops_per_sec': num_operations / total_time
    }


# ============================================================================
# ➕ 新增：Training Speed Benchmark
# ============================================================================

def benchmark_training_speed(
    X: np.ndarray,
    y: np.ndarray,
    schema: Schema,
    use_sparse: bool,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark training speed with actual MOA classifier.
    
    Args:
        X: Feature matrix.
        y: Labels.
        schema: Schema object.
        use_sparse: Whether to use sparse storage.
        num_iterations: Number of training iterations.
    
    Returns:
        Dictionary with timing results.
    """
    try:
        from capymoa.classifier import NaiveBayes
        
        # Initialize classifier
        classifier = NaiveBayes(schema=schema)
        
        # Prepare instances
        instances = []
        for i in range(min(num_iterations, len(X))):
            inst = LabeledInstance(schema, (X[i], y[i]), force_sparse=use_sparse)
            instances.append(inst)
        
        # Warmup
        for inst in instances[:10]:
            classifier.train(inst)
        
        # Benchmark training
        start_time = time.perf_counter()
        
        for inst in instances:
            classifier.train(inst)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / len(instances)
        
        return {
            'total_time': total_time,
            'avg_time': avg_time,
            'throughput': len(instances) / total_time
        }
    
    except Exception as e:
        print(f"  Warning: Training benchmark failed: {e}")
        return {'total_time': 0.0, 'avg_time': 0.0, 'throughput': 0.0}


# ============================================================================
# ➕ 新增：Java Conversion Overhead Analysis
# ============================================================================

def benchmark_java_conversion_detailed(
    X: np.ndarray,
    y: np.ndarray,
    schema: Schema,
    use_sparse: bool
) -> Dict[str, float]:
    """
    Detailed benchmark of Java conversion overhead.
    
    Args:
        X: Feature matrix.
        y: Labels.
        schema: Schema object.
        use_sparse: Whether to use sparse storage.
    
    Returns:
        Dictionary with detailed timing breakdown.
    """
    num_samples = min(100, X.shape[0])
    
    # Time: Python instance creation only
    start_time = time.perf_counter()
    instances = []
    for i in range(num_samples):
        inst = LabeledInstance(schema, (X[i], y[i]), force_sparse=use_sparse)
        instances.append(inst)
    python_time = time.perf_counter() - start_time
    
    # Time: Java conversion only (first access)
    start_time = time.perf_counter()
    for inst in instances:
        _ = inst.java_instance  # Trigger lazy conversion
    java_time = time.perf_counter() - start_time
    
    # Time: Subsequent access (should be cached)
    start_time = time.perf_counter()
    for inst in instances:
        _ = inst.java_instance
    cache_time = time.perf_counter() - start_time
    
    return {
        'python_creation': python_time / num_samples,
        'java_conversion': java_time / num_samples,
        'cached_access': cache_time / num_samples,
        'total_overhead': (python_time + java_time) / num_samples
    }


# ============================================================================
# Enhanced Benchmarking Functions
# ============================================================================

def benchmark_instance_creation(
    X: np.ndarray,
    y: np.ndarray,
    schema: Schema,
    use_sparse: bool,
    warmup: int = 10
) -> Dict[str, float]:
    """Benchmark instance creation time and memory."""
    num_samples = X.shape[0]
    
    # Warmup
    for i in range(min(warmup, num_samples)):
        _ = LabeledInstance(schema, (X[i], y[i]), force_sparse=use_sparse)
    
    # Benchmark
    memory_monitor = MemoryMonitor()
    
    start_time = time.perf_counter()
    
    instances = []
    for i in range(num_samples):
        instance = LabeledInstance(schema, (X[i], y[i]), force_sparse=use_sparse)
        instances.append(instance)
        
        if i % 100 == 0:
            memory_monitor.update()
    
    end_time = time.perf_counter()
    memory_monitor.update()
    
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    memory_delta = memory_monitor.get_delta()
    memory_avg = memory_monitor.get_average()
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': num_samples / total_time,
        'memory_peak_mb': memory_delta,
        'memory_avg_mb': memory_avg,
        'instances': instances  # Return for further testing
    }


def run_single_benchmark(
    num_features: int,
    sparsity: float,
    num_samples: int = 1000,
    realistic: bool = False
) -> Dict:
    """
    Run a comprehensive single benchmark configuration.
    
    Args:
        num_features: Number of features.
        sparsity: Sparsity level (0.0 to 1.0).
        num_samples: Number of samples.
        realistic: Whether to use realistic clustered data.
    
    Returns:
        Dictionary with all benchmark results.
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {num_features:,} features, {sparsity:.1%} sparsity")
    print(f"{'='*80}")
    
    # Generate data
    if realistic:
        X, y = generate_realistic_sparse_data(num_samples, num_features, sparsity)
    else:
        X, y = generate_sparse_data(num_samples, num_features, sparsity)
    
    # Create schema
    schema = Schema.from_custom(
        [f"f{i}" for i in range(num_features)],
        dataset_name=f"Sparse{int(sparsity*100)}",
        values_for_class_label=["0", "1"]
    )
    
    # Get storage info
    storage_info = get_storage_info(X[0])
    print(f"  Non-zero elements: {storage_info['num_nonzero']}")
    print(f"  Actual sparsity: {storage_info['sparsity']:.2%}")
    print(f"  Recommended: {storage_info['recommended']}")
    
    # 1. Instance Creation Benchmark
    print("\n[1/7] Benchmarking Dense instance creation...")
    dense_creation = benchmark_instance_creation(X, y, schema, use_sparse=False)
    
    print("[2/7] Benchmarking Sparse instance creation...")
    sparse_creation = benchmark_instance_creation(X, y, schema, use_sparse=True)
    
    # 2. Java Conversion Detailed
    print("[3/7] Benchmarking Java conversion (Dense)...")
    dense_java_detailed = benchmark_java_conversion_detailed(X, y, schema, use_sparse=False)
    
    print("[4/7] Benchmarking Java conversion (Sparse)...")
    sparse_java_detailed = benchmark_java_conversion_detailed(X, y, schema, use_sparse=True)
    
    # 3. Feature Access Patterns
    print("[5/7] Benchmarking feature access patterns...")
    access_results = {}
    
    for pattern in BenchmarkConfig.ACCESS_PATTERNS:
        print(f"  Testing {pattern} access...")
        
        dense_access = benchmark_feature_access(
            dense_creation['instances'][:100],
            pattern,
            num_operations=1000
        )
        
        sparse_access = benchmark_feature_access(
            sparse_creation['instances'][:100],
            pattern,
            num_operations=1000
        )
        
        access_results[pattern] = {
            'dense': dense_access,
            'sparse': sparse_access,
            'speedup': dense_access['avg_time'] / (sparse_access['avg_time'] + 1e-10)
        }
    
    # 4. Training Speed
    print("[6/7] Benchmarking training speed...")
    dense_training = benchmark_training_speed(X, y, schema, use_sparse=False, num_iterations=100)
    sparse_training = benchmark_training_speed(X, y, schema, use_sparse=True, num_iterations=100)
    
    # Calculate speedups and savings
    creation_speedup = dense_creation['avg_time'] / sparse_creation['avg_time']
    java_speedup = dense_java_detailed['total_overhead'] / sparse_java_detailed['total_overhead']
    memory_saving = (dense_creation['memory_peak_mb'] - sparse_creation['memory_peak_mb']) / (dense_creation['memory_peak_mb'] + 1e-10)
    training_speedup = dense_training['avg_time'] / (sparse_training['avg_time'] + 1e-10)
    
    print("[7/7] Compiling results...")
    
    # Cleanup instances to free memory
    del dense_creation['instances']
    del sparse_creation['instances']
    gc.collect()
    
    result = {
        'num_features': num_features,
        'sparsity': sparsity,
        'num_samples': num_samples,
        'storage_info': storage_info,
        
        # Creation benchmarks
        'dense_creation': dense_creation,
        'sparse_creation': sparse_creation,
        'creation_speedup': creation_speedup,
        
        # Java conversion
        'dense_java': dense_java_detailed,
        'sparse_java': sparse_java_detailed,
        'java_speedup': java_speedup,
        
        # Access patterns
        'access_patterns': access_results,
        
        # Training
        'dense_training': dense_training,
        'sparse_training': sparse_training,
        'training_speedup': training_speedup,
        
        # Overall metrics
        'memory_saving': memory_saving,
        'overall_speedup': (creation_speedup + java_speedup + training_speedup) / 3
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"  Creation speedup: {creation_speedup:.2f}×")
    print(f"  Java conversion speedup: {java_speedup:.2f}×")
    print(f"  Training speedup: {training_speedup:.2f}×")
    print(f"  Memory saving: {memory_saving:.2%}")
    print(f"  Overall speedup: {result['overall_speedup']:.2f}×")
    print(f"{'='*80}")
    
    return result


# ============================================================================
# ➕ 新增：Enhanced Visualization
# ============================================================================

def plot_comprehensive_results(results: List[Dict], output_dir: str):
    """
    Create comprehensive visualization with multiple subplots.
    
    Args:
        results: List of benchmark results.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by dimension
    dimensions = sorted(set(r['num_features'] for r in results))
    
    for dim in dimensions:
        dim_results = [r for r in results if r['num_features'] == dim]
        dim_results = sorted(dim_results, key=lambda x: x['sparsity'])
        
        fig = plt.figure(figsize=(16, 12))
        
        sparsities = [r['sparsity'] for r in dim_results]
        
        # 1. Creation Time Comparison (subplot 1)
        ax1 = plt.subplot(3, 3, 1)
        dense_create = [r['dense_creation']['avg_time'] * 1000 for r in dim_results]
        sparse_create = [r['sparse_creation']['avg_time'] * 1000 for r in dim_results]
        
        ax1.plot(sparsities, dense_create, 'o-', label='Dense', linewidth=2)
        ax1.plot(sparsities, sparse_create, 's-', label='Sparse', linewidth=2)
        ax1.set_xlabel('Sparsity')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Instance Creation Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Creation Speedup (subplot 2)
        ax2 = plt.subplot(3, 3, 2)
        speedups = [r['creation_speedup'] for r in dim_results]
        ax2.plot(sparsities, speedups, 'o-', color='green', linewidth=2)
        ax2.axhline(y=1.0, color='red', linestyle='--', label='Break-even')
        ax2.set_xlabel('Sparsity')
        ax2.set_ylabel('Speedup (×)')
        ax2.set_title('Creation Speedup (Sparse/Dense)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory Usage (subplot 3)
        ax3 = plt.subplot(3, 3, 3)
        dense_mem = [r['dense_creation']['memory_peak_mb'] for r in dim_results]
        sparse_mem = [r['sparse_creation']['memory_peak_mb'] for r in dim_results]
        
        ax3.plot(sparsities, dense_mem, 'o-', label='Dense', linewidth=2)
        ax3.plot(sparsities, sparse_mem, 's-', label='Sparse', linewidth=2)
        ax3.set_xlabel('Sparsity')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('Peak Memory Usage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Java Conversion Breakdown (subplot 4)
        ax4 = plt.subplot(3, 3, 4)
        x_pos = np.arange(len(sparsities))
        width = 0.35
        
        dense_java = [r['dense_java']['total_overhead'] * 1000 for r in dim_results]
        sparse_java = [r['sparse_java']['total_overhead'] * 1000 for r in dim_results]
        
        ax4.bar(x_pos - width/2, dense_java, width, label='Dense', alpha=0.8)
        ax4.bar(x_pos + width/2, sparse_java, width, label='Sparse', alpha=0.8)
        ax4.set_xlabel('Sparsity')
        ax4.set_ylabel('Time (ms)')
        ax4.set_title('Java Conversion Overhead')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"{s:.0%}" for s in sparsities], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Access Pattern Comparison (subplot 5)
        ax5 = plt.subplot(3, 3, 5)
        
        # Average across all access patterns
        dense_access_avg = []
        sparse_access_avg = []
        
        for r in dim_results:
            dense_avg = np.mean([r['access_patterns'][p]['dense']['avg_time'] * 1e6 
                                for p in BenchmarkConfig.ACCESS_PATTERNS])
            sparse_avg = np.mean([r['access_patterns'][p]['sparse']['avg_time'] * 1e6 
                                 for p in BenchmarkConfig.ACCESS_PATTERNS])
            dense_access_avg.append(dense_avg)
            sparse_access_avg.append(sparse_avg)
        
        ax5.plot(sparsities, dense_access_avg, 'o-', label='Dense', linewidth=2)
        ax5.plot(sparsities, sparse_access_avg, 's-', label='Sparse', linewidth=2)
        ax5.set_xlabel('Sparsity')
        ax5.set_ylabel('Time (μs)')
        ax5.set_title('Average Feature Access Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Training Speed (subplot 6)
        ax6 = plt.subplot(3, 3, 6)
        training_speedup = [r['training_speedup'] for r in dim_results]
        ax6.plot(sparsities, training_speedup, 'o-', color='purple', linewidth=2)
        ax6.axhline(y=1.0, color='red', linestyle='--', label='Break-even')
        ax6.set_xlabel('Sparsity')
        ax6.set_ylabel('Speedup (×)')
        ax6.set_title('Training Speedup')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Memory Saving Percentage (subplot 7)
        ax7 = plt.subplot(3, 3, 7)
        memory_savings = [r['memory_saving'] * 100 for r in dim_results]
        ax7.plot(sparsities, memory_savings, 'o-', color='orange', linewidth=2)
        ax7.fill_between(sparsities, 0, memory_savings, alpha=0.3)
        ax7.set_xlabel('Sparsity')
        ax7.set_ylabel('Memory Saving (%)')
        ax7.set_title('Memory Savings')
        ax7.grid(True, alpha=0.3)
        
        # 8. Overall Speedup (subplot 8)
        ax8 = plt.subplot(3, 3, 8)
        overall_speedup = [r['overall_speedup'] for r in dim_results]
        ax8.plot(sparsities, overall_speedup, 'o-', color='red', linewidth=2, markersize=8)
        ax8.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax8.fill_between(sparsities, 1, overall_speedup, where=np.array(overall_speedup) > 1, 
                        alpha=0.3, color='green', label='Faster')
        ax8.fill_between(sparsities, overall_speedup, 1, where=np.array(overall_speedup) < 1, 
                        alpha=0.3, color='red', label='Slower')
        ax8.set_xlabel('Sparsity')
        ax8.set_ylabel('Overall Speedup (×)')
        ax8.set_title('Overall Performance (Avg of all metrics)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary Table (subplot 9)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        # Create summary table
        table_data = []
        for r in dim_results[-3:]:  # Last 3 sparsity levels
            table_data.append([
                f"{r['sparsity']:.1%}",
                f"{r['creation_speedup']:.2f}×",
                f"{r['training_speedup']:.2f}×",
                f"{r['memory_saving']:.1%}"
            ])
        
        table = ax9.table(
            cellText=table_data,
            colLabels=['Sparsity', 'Create', 'Train', 'Memory'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Overall title
        fig.suptitle(f'Comprehensive Performance Analysis - {dim:,} Features', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save
        filename = f"comprehensive_{dim}_features.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=BenchmarkConfig.PLOT_DPI, bbox_inches='tight')
        print(f"  Saved comprehensive plot: {filepath}")
        
        plt.close()


def save_enhanced_report(results: List[Dict], output_dir: str):
    """
    Save enhanced text summary report.
    
    Args:
        results: List of benchmark results.
        output_dir: Directory to save report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_report_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("CapyMOA Enhanced Sparse Instance Performance Benchmark Report\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total configurations: {len(results)}\n")
        f.write("="*100 + "\n\n")
        
        # Group by dimension
        dimensions = sorted(set(r['num_features'] for r in results))
        
        for dim in dimensions:
            dim_results = [r for r in results if r['num_features'] == dim]
            dim_results = sorted(dim_results, key=lambda x: x['sparsity'])
            
            f.write(f"\n{'='*100}\n")
            f.write(f"Dimension: {dim:,} features\n")
            f.write(f"{'='*100}\n\n")
            
            for r in dim_results:
                f.write(f"Sparsity: {r['sparsity']:.1%}\n")
                f.write(f"{'─'*100}\n")
                
                # Storage info
                f.write(f"  Storage Info:\n")
                f.write(f"    Non-zero elements: {r['storage_info']['num_nonzero']}\n")
                f.write(f"    Recommended: {r['storage_info']['recommended']}\n\n")
                
                # Creation metrics
                f.write(f"  Instance Creation:\n")
                f.write(f"    Dense time: {r['dense_creation']['avg_time']*1000:.3f}ms\n")
                f.write(f"    Sparse time: {r['sparse_creation']['avg_time']*1000:.3f}ms\n")
                f.write(f"    Speedup: {r['creation_speedup']:.2f}×\n\n")
                
                # Java conversion
                f.write(f"  Java Conversion:\n")
                f.write(f"    Dense overhead: {r['dense_java']['total_overhead']*1000:.3f}ms\n")
                f.write(f"    Sparse overhead: {r['sparse_java']['total_overhead']*1000:.3f}ms\n")
                f.write(f"    Speedup: {r['java_speedup']:.2f}×\n\n")
                
                # Access patterns
                f.write(f"  Feature Access:\n")
                for pattern in BenchmarkConfig.ACCESS_PATTERNS:
                    if pattern in r['access_patterns']:
                        pattern_result = r['access_patterns'][pattern]
                        f.write(f"    {pattern.capitalize()}:\n")
                        f.write(f"      Dense: {pattern_result['dense']['avg_time']*1e6:.2f}μs\n")
                        f.write(f"      Sparse: {pattern_result['sparse']['avg_time']*1e6:.2f}μs\n")
                        f.write(f"      Speedup: {pattern_result['speedup']:.2f}×\n")
                f.write("\n")
                
                # Training
                f.write(f"  Training:\n")
                f.write(f"    Dense time: {r['dense_training']['avg_time']*1000:.3f}ms\n")
                f.write(f"    Sparse time: {r['sparse_training']['avg_time']*1000:.3f}ms\n")
                f.write(f"    Speedup: {r['training_speedup']:.2f}×\n\n")
                
                # Memory
                f.write(f"  Memory:\n")
                f.write(f"    Dense peak: {r['dense_creation']['memory_peak_mb']:.2f}MB\n")
                f.write(f"    Sparse peak: {r['sparse_creation']['memory_peak_mb']:.2f}MB\n")
                f.write(f"    Saving: {r['memory_saving']:.2%}\n\n")
                
                # Overall
                f.write(f"  Overall Speedup: {r['overall_speedup']:.2f}×\n")
                f.write(f"{'─'*100}\n\n")
        
        # Overall statistics
        f.write(f"\n{'='*100}\n")
        f.write("Overall Statistics\n")
        f.write(f"{'='*100}\n\n")
        
        # Best cases
        best_creation = max(results, key=lambda x: x['creation_speedup'])
        best_training = max(results, key=lambda x: x['training_speedup'])
        best_memory = max(results, key=lambda x: x['memory_saving'])
        
        f.write(f"Best Creation Speedup:\n")
        f.write(f"  {best_creation['num_features']:,} features, {best_creation['sparsity']:.1%} sparsity\n")
        f.write(f"  Speedup: {best_creation['creation_speedup']:.2f}×\n\n")
        
        f.write(f"Best Training Speedup:\n")
        f.write(f"  {best_training['num_features']:,} features, {best_training['sparsity']:.1%} sparsity\n")
        f.write(f"  Speedup: {best_training['training_speedup']:.2f}×\n\n")
        
        f.write(f"Best Memory Saving:\n")
        f.write(f"  {best_memory['num_features']:,} features, {best_memory['sparsity']:.1%} sparsity\n")
        f.write(f"  Saving: {best_memory['memory_saving']:.2%}\n\n")
        
        # Averages
        avg_creation = np.mean([r['creation_speedup'] for r in results])
        avg_training = np.mean([r['training_speedup'] for r in results])
        avg_memory = np.mean([r['memory_saving'] for r in results])
        
        f.write(f"Average Metrics:\n")
        f.write(f"  Creation speedup: {avg_creation:.2f}×\n")
        f.write(f"  Training speedup: {avg_training:.2f}×\n")
        f.write(f"  Memory saving: {avg_memory:.2%}\n")
    
    print(f"  Saved enhanced report: {filepath}")


# ============================================================================
# Main Benchmark Suites
# ============================================================================

def run_quick_benchmark():
    """Run a quick benchmark with limited configurations."""
    print("\n" + "="*80)
    print("QUICK BENCHMARK - Enhanced Sparse Instance Performance")
    print("="*80)
    
    results = []
    
    configs = [
        (1000, 0.9, False),
        (10000, 0.95, False),
        (10000, 0.99, False),
    ]
    
    for num_features, sparsity, realistic in configs:
        result = run_single_benchmark(num_features, sparsity, num_samples=500, realistic=realistic)
        results.append(result)
    
    # Generate reports
    print(f"\n{'='*80}")
    print("Generating reports...")
    print(f"{'='*80}")
    
    output_dir = BenchmarkConfig.RESULTS_DIR
    plot_comprehensive_results(results, output_dir)
    save_enhanced_report(results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results saved to: {output_dir}/")
    print(f"{'='*80}")


def run_comprehensive_benchmark():
    """Run comprehensive benchmark with all configurations."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK - Enhanced Sparse Instance Performance")
    print("="*80)
    
    results = []
    
    dimensions = [1000, 10000, 50000]
    sparsity_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    total_configs = len(dimensions) * len(sparsity_levels)
    current = 0
    
    for dim in dimensions:
        for sparsity in sparsity_levels:
            current += 1
            print(f"\n[Progress: {current}/{total_configs}]")
            
            result = run_single_benchmark(dim, sparsity, num_samples=1000)
            results.append(result)
    
    # Generate reports
    print(f"\n{'='*80}")
    print("Generating reports...")
    print(f"{'='*80}")
    
    output_dir = BenchmarkConfig.RESULTS_DIR
    plot_comprehensive_results(results, output_dir)
    save_enhanced_report(results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results saved to: {output_dir}/")
    print(f"{'='*80}")


def run_realistic_benchmark():
    """Run benchmark with realistic clustered sparse data."""
    print("\n" + "="*80)
    print("REALISTIC DATA BENCHMARK - Clustered Sparse Features")
    print("="*80)
    
    results = []
    
    configs = [
        (10000, 0.95, True),
        (50000, 0.99, True),
    ]
    
    for num_features, sparsity, realistic in configs:
        result = run_single_benchmark(num_features, sparsity, num_samples=1000, realistic=realistic)
        results.append(result)
    
    # Generate reports
    print(f"\n{'='*80}")
    print("Generating reports...")
    print(f"{'='*80}")
    
    output_dir = BenchmarkConfig.RESULTS_DIR
    plot_comprehensive_results(results, output_dir)
    save_enhanced_report(results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results saved to: {output_dir}/")
    print(f"{'='*80}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced benchmark for sparse instance performance in CapyMOA"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='quick',
        choices=['quick', 'comprehensive', 'realistic'],
        help='Benchmark mode: quick (default), comprehensive, or realistic data'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CapyMOA Enhanced Sparse Instance Performance Benchmark")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print("="*80)
    
    if args.mode == 'quick':
        run_quick_benchmark()
    elif args.mode == 'comprehensive':
        run_comprehensive_benchmark()
    elif args.mode == 'realistic':
        run_realistic_benchmark()
    
    print("\nDone! 🎉")


if __name__ == '__main__':
    main()