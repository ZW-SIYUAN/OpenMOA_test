import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Literal, List

# ==========================================
# 1. 基础 Mock 环境 (无需 CapyMOA 库)
# ==========================================
class Schema:
    def __init__(self, d): self.d = d
    def get_num_attributes(self): return self.d
    def is_classification(self): return True

class Instance:
    def __init__(self, x, y):
        self.x = x
        self.y_index = y
        self.y_value = y

class LabeledInstance(Instance):
    @classmethod
    def from_array(cls, schema, x, y): return cls(x, y)

class RegressionInstance(Instance):
    @classmethod
    def from_array(cls, schema, x, y): return cls(x, y)

class Stream:
    def next_instance(self): pass
    def has_more_instances(self): pass
    def restart(self): pass
    def get_schema(self): pass

class MockBaseStream(Stream):
    """生成全 1 向量的流，用于测试"""
    def __init__(self, d=20, total=1000):
        self.d = d; self.total = total; self.count = 0; self.schema = Schema(d)
    def next_instance(self):
        if self.count >= self.total: return None
        self.count += 1
        return Instance(np.ones(self.d, dtype=float), 0)
    def has_more_instances(self): return self.count < self.total
    def get_schema(self): return self.schema
    def restart(self): self.count = 0

# ==========================================
# 2. 三大 OVFM 适配类定义 (已修复 d_max 问题)
# ==========================================

# --- 类 1: TrapezoidalStream ---
class TrapezoidalStream(Stream):
    def __init__(self, base_stream, d_min=2, d_max=None, evolution_mode="random", total_instances=1000, random_seed=42):
        self.base_stream = base_stream
        self.d_min = d_min
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        self.evolution_mode = evolution_mode
        self.total_instances = total_instances
        self._current_t = 0
        self._rng = np.random.RandomState(random_seed)
        self._schema = base_stream.get_schema()
        
        self._feature_ranking = self._gen_ranking()
        self._schedule = self._gen_schedule()

    def _gen_ranking(self):
        idx = np.arange(self.d_max)
        if self.evolution_mode == "random": self._rng.shuffle(idx)
        return idx 

    def _gen_schedule(self):
        if self.evolution_mode == "pyramid":
            half = self.total_instances // 2
            return np.concatenate([
                np.linspace(self.d_min, self.d_max, half),
                np.linspace(self.d_max, self.d_min, self.total_instances - half)
            ]).astype(int)
        return np.linspace(self.d_min, self.d_max, self.total_instances).astype(int)

    def next_instance(self):
        if not self.has_more_instances(): return None
        base = self.base_stream.next_instance()
        if not base: return None
        
        t_idx = min(self._current_t, self.total_instances-1)
        k = self._schedule[t_idx]
        active = self._feature_ranking[:k]
        
        x_full = np.full(self.d_max, np.nan)
        x_base = np.array(base.x)[:self.d_max]
        x_full[active] = x_base[active]
        
        self._current_t += 1
        return LabeledInstance(x_full, base.y_index)

    def has_more_instances(self): return self._current_t < self.total_instances and self.base_stream.has_more_instances()
    def get_schema(self): return self._schema


# --- 类 2: CapriciousStream (已修复 d_max 报错) ---
class CapriciousStream(Stream):
    def __init__(
        self, 
        base_stream: Stream, 
        d_max: Optional[int] = None,   # <--- 关键修复点：这里必须接收 d_max
        missing_ratio: float = 0.5, 
        total_instances: int = 1000, 
        min_features: int = 1, 
        random_seed: int = 42
    ):
        self.base_stream = base_stream
        
        # 即使 base_stream 有维度，也允许 d_max 覆盖
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        self.missing_ratio = missing_ratio
        self.total_instances = total_instances
        self.min_features = min_features
        self.random_seed = random_seed
        self._current_t = 0
        self._schema = base_stream.get_schema()

    def next_instance(self):
        if not self.has_more_instances(): return None
        base = self.base_stream.next_instance()
        if not base: return None

        rng = np.random.RandomState(self.random_seed + self._current_t)
        mask = rng.rand(self.d_max) > self.missing_ratio
        if mask.sum() < self.min_features:
            mask[rng.choice(self.d_max, self.min_features, replace=False)] = True
            
        x_full = np.array(base.x, dtype=float).copy()
        if len(x_full) > self.d_max:
            x_full = x_full[:self.d_max]
            
        x_full[~mask] = np.nan 
        
        self._current_t += 1
        return LabeledInstance(x_full, base.y_index)

    def has_more_instances(self): return self._current_t < self.total_instances and self.base_stream.has_more_instances()
    def get_schema(self): return self._schema


# --- 类 3: EvolvableTrapezoidalStream ---
class EvolvableTrapezoidalStream(Stream):
    def __init__(self, base_stream, n_segments=2, overlap_ratio=1.0, feature_split="sequential", d_max=None, total_instances=1000, random_seed=42):
        self.base_stream = base_stream
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        self.n_segments = n_segments
        self.overlap_ratio = overlap_ratio
        self.total_instances = total_instances
        self._current_t = 0
        self._schema = base_stream.get_schema()
        
        # Partitions
        all_idx = np.arange(self.d_max)
        if feature_split == "random":
            np.random.RandomState(random_seed).shuffle(all_idx)
        self.partitions = [np.sort(p) for p in np.array_split(all_idx, n_segments)]
        
        # Boundaries
        L = total_instances / (n_segments + overlap_ratio * (n_segments - 1))
        self.boundaries = []
        curr = 0.0
        for i in range(2 * n_segments - 1):
            curr += L if i % 2 == 0 else L * overlap_ratio
            self.boundaries.append(int(curr))
        self.boundaries[-1] = total_instances

    def next_instance(self):
        if not self.has_more_instances(): return None
        base = self.base_stream.next_instance()
        
        stage = 0
        for i, b in enumerate(self.boundaries):
            if self._current_t < b:
                stage = i; break
        else: stage = len(self.boundaries) - 1
            
        active_idx = []
        if stage % 2 == 0: active_idx = self.partitions[stage//2]
        else: 
            prev = (stage-1)//2
            active_idx = np.concatenate([self.partitions[prev], self.partitions[prev+1]])
            
        x_full = np.full(self.d_max, np.nan)
        x_base = np.array(base.x)[:self.d_max]
        x_full[active_idx] = x_base[active_idx]
        
        self._current_t += 1
        return LabeledInstance(x_full, base.y_index)

    def has_more_instances(self): return self._current_t < self.total_instances and self.base_stream.has_more_instances()
    def get_schema(self): return self._schema


# ==========================================
# 3. 绘图逻辑
# ==========================================
def run_and_collect(stream_class, params, d_max=20, total=100):
    base = MockBaseStream(d=d_max, total=total)
    # 这里 d_max=d_max 会传递给所有类，包括 CapriciousStream
    stream = stream_class(base_stream=base, d_max=d_max, total_instances=total, **params)
    
    matrix = np.zeros((d_max, total))
    t = 0
    while stream.has_more_instances():
        inst = stream.next_instance()
        if inst is None: break
        # 1=Value, 0=NaN
        matrix[:, t] = ~np.isnan(inst.x)
        t += 1
    return matrix

def plot_ovfm_demo():
    D_MAX = 20
    N = 100
    
    scenarios = [
        # Row 1: TrapezoidalStream
        (TrapezoidalStream, {"evolution_mode": "ordered", "d_min": 2}, "Trapezoidal: Ordered"),
        (TrapezoidalStream, {"evolution_mode": "random", "d_min": 2}, "Trapezoidal: Random"),
        (TrapezoidalStream, {"evolution_mode": "pyramid", "d_min": 2}, "Trapezoidal: Pyramid"),
        
        # Row 2: CapriciousStream
        (CapriciousStream, {"missing_ratio": 0.2}, "Capricious: Low Missing (20%)"),
        (CapriciousStream, {"missing_ratio": 0.5}, "Capricious: Med Missing (50%)"),
        (CapriciousStream, {"missing_ratio": 0.8}, "Capricious: High Missing (80%)"),
        
        # Row 3: EvolvableTrapezoidalStream
        (EvolvableTrapezoidalStream, {"n_segments": 2, "overlap_ratio": 0.0}, "Evolvable: 2 Segs (No Overlap)"),
        (EvolvableTrapezoidalStream, {"n_segments": 2, "overlap_ratio": 1.0}, "Evolvable: 2 Segs (Full Overlap)"),
        (EvolvableTrapezoidalStream, {"n_segments": 3, "overlap_ratio": 0.2, "feature_split": "random"}, "Evolvable: 3 Segs (Random Split)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    cmap = ListedColormap(['#f0f0f0', '#084594']) 

    for i, (cls_obj, params, title) in enumerate(scenarios):
        print(f"Simulating: {title}...")
        mask = run_and_collect(cls_obj, params, d_max=D_MAX, total=N)
        ax = axes[i]
        ax.imshow(mask, aspect='auto', cmap=cmap, interpolation='nearest', origin='lower', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.3)
        if i % 3 == 0: ax.set_ylabel("Feature Index")
        if i >= 6: ax.set_xlabel("Time Step")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#f0f0f0', edgecolor='gray', label='NaN (Missing)'),
        Patch(facecolor='#084594', edgecolor='gray', label='Value (Active)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.suptitle("OVFM-Compatible Streams (NaN Padding) Demo", y=0.98, fontsize=16)
    plt.show()

if __name__ == "__main__":
    plot_ovfm_demo()