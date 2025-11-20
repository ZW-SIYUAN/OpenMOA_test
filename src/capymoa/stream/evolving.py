"""capymoa/stream/evolving.py - 特征演化流包装器"""
import numpy as np
from typing import Literal, Optional
from capymoa.stream import Stream
from capymoa.stream._stream import Schema
from capymoa.instance import LabeledInstance, RegressionInstance


class EvolvingFeatureStream(Stream):
    """将固定特征的数据流包装为特征演化流。
    
    支持多种演化模式：
    - pyramid: 特征数先增后减（OASF 论文）
    - incremental: 特征数单调增长
    - decremental: 特征数单调减少
    - tds: 梯形数据流，特征有"出生时间"（ORF3V 论文）
    - vfs: 完全随机缺失（ORF3V 论文）
    """

    def __init__(
        self,
        base_stream: Stream,
        d_min: int = 2,
        d_max: Optional[int] = None,
        evolution_pattern: Literal["pyramid", "incremental", "decremental", "tds", "vfs"] = "pyramid",
        total_instances: int = 10000,
        feature_selection: Literal["prefix", "suffix", "random"] = "prefix",
        missing_ratio: float = 0.0,
        random_seed: int = 42
    ):
        """初始化特征演化流
        
        :param base_stream: 原始数据流（特征固定）
        :param d_min: 最小特征维度
        :param d_max: 最大特征维度（None 则使用原始特征数）
        :param evolution_pattern: 演化模式
            - pyramid: 前半程增长，后半程减少
            - incremental: 单调增长
            - decremental: 单调减少
            - tds: 梯形数据流（特征有出生时间）
            - vfs: 完全随机缺失（每个特征独立缺失）
        :param total_instances: 总样本数
        :param feature_selection: 特征选择方式（仅用于 pyramid/incremental/decremental）
        :param missing_ratio: VFS 模式下的特征缺失率（0.0-1.0）
        :param random_seed: 随机种子
        """
        self.base_stream = base_stream
        self.d_min = d_min
        
        # 获取原始特征数
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        if self.d_max > original_d:
            raise ValueError(
                f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})"
            )
        
        self.evolution_pattern = evolution_pattern
        self.total_instances = total_instances
        self.feature_selection = feature_selection
        self.missing_ratio = missing_ratio
        self.random_seed = random_seed
        
        # 设置随机种子
        self._rng = np.random.RandomState(random_seed)
        
        # 当前时间步
        self._current_t = 0
        
        # 预计算演化序列
        if evolution_pattern in ["pyramid", "incremental", "decremental"]:
            self._dimension_schedule = self._generate_dimension_schedule()
            self._feature_indices_cache = self._generate_feature_indices()
        elif evolution_pattern == "tds":
            self._feature_offsets = self._generate_tds_offsets()
        elif evolution_pattern == "vfs":
            # VFS 模式不需要预计算
            pass
        
        # Schema
        self._schema = base_stream.get_schema()

    def _generate_dimension_schedule(self) -> np.ndarray:
        """生成特征维度演化序列"""
        dims = np.zeros(self.total_instances, dtype=int)
        
        if self.evolution_pattern == "pyramid":
            half = self.total_instances // 2
            # 前半程：从 d_min 增长到 d_max
            dims[:half] = np.linspace(self.d_min, self.d_max, half).astype(int)
            # 后半程：从 d_max 减少到 d_min
            dims[half:] = np.linspace(
                self.d_max, self.d_min, self.total_instances - half
            ).astype(int)
        
        elif self.evolution_pattern == "incremental":
            # 单调增长
            dims = np.linspace(self.d_min, self.d_max, self.total_instances).astype(int)
        
        elif self.evolution_pattern == "decremental":
            # 单调减少
            dims = np.linspace(self.d_max, self.d_min, self.total_instances).astype(int)
        
        return dims

    def _generate_feature_indices(self) -> list:
        """生成每个时间步的特征索引（用于 pyramid/incremental/decremental）"""
        indices_list = []
        
        for t in range(self.total_instances):
            d_current = self._dimension_schedule[t]
            
            if self.feature_selection == "prefix":
                indices = np.arange(d_current)
            elif self.feature_selection == "suffix":
                indices = np.arange(self.d_max - d_current, self.d_max)
            elif self.feature_selection == "random":
                rng_t = np.random.RandomState(self.random_seed + t)
                indices = rng_t.choice(self.d_max, d_current, replace=False)
                indices.sort()
            else:
                raise ValueError(f"Unknown feature_selection: {self.feature_selection}")
            
            indices_list.append(indices)
        
        return indices_list

    def _generate_tds_offsets(self) -> np.ndarray:
        """生成 TDS 模式下的特征 offset（出生时间）"""
        # 将特征随机分配到 10 个时间段
        offsets = np.zeros(self.d_max, dtype=int)
        indices = self._rng.permutation(self.d_max)
        
        for i in range(self.d_max):
            feature_idx = indices[i]
            time_slot = i % 10
            offsets[feature_idx] = time_slot * (self.total_instances // 10)
        
        return offsets

    def _get_tds_indices(self, t: int) -> np.ndarray:
        """获取 TDS 模式下当前时刻的可用特征"""
        # 只返回已经"出生"的特征
        available = np.where(self._feature_offsets <= t)[0]
        return available

    def _get_vfs_indices(self, t: int) -> np.ndarray:
        """获取 VFS 模式下当前时刻的可用特征（随机缺失）"""
        # 每个特征独立地以 (1 - missing_ratio) 概率存在
        rng_t = np.random.RandomState(self.random_seed + t)
        mask = rng_t.rand(self.d_max) > self.missing_ratio
        available = np.where(mask)[0]
        return available

    def next_instance(self):
        """获取下一个实例（特征已演化）"""
        if not self.has_more_instances():
            return None
        
        # 从基础流获取原始实例
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # 根据演化模式获取特征索引
        if self.evolution_pattern in ["pyramid", "incremental", "decremental"]:
            active_indices = self._feature_indices_cache[self._current_t]
        elif self.evolution_pattern == "tds":
            active_indices = self._get_tds_indices(self._current_t)
        elif self.evolution_pattern == "vfs":
            active_indices = self._get_vfs_indices(self._current_t)
        else:
            raise ValueError(f"Unknown evolution_pattern: {self.evolution_pattern}")
        
        # 如果没有可用特征，至少保留一个（避免空特征向量）
        if len(active_indices) == 0:
            active_indices = np.array([0])
        
        # 提取子集特征
        x_full = np.array(base_instance.x)
        x_subset = x_full[active_indices]
        
        # 创建新实例
        if self._schema.is_classification():
            modified_instance = LabeledInstance.from_array(
                self._schema,
                x_subset,
                base_instance.y_index
            )
        else:
            modified_instance = RegressionInstance.from_array(
                self._schema,
                x_subset,
                base_instance.y_value
            )
        
        self._current_t += 1
        return modified_instance

    def has_more_instances(self) -> bool:
        """检查是否还有更多实例"""
        return (
            self._current_t < self.total_instances 
            and self.base_stream.has_more_instances()
        )

    def restart(self):
        """重启流"""
        self.base_stream.restart()
        self._current_t = 0

    def get_schema(self) -> Schema:
        """返回 schema"""
        return self._schema

    def get_moa_stream(self):
        """自定义流不支持 MOA 加速"""
        return None

    def get_current_dimension(self) -> int:
        """获取当前特征维度（用于调试）"""
        if self.evolution_pattern in ["pyramid", "incremental", "decremental"]:
            if self._current_t < self.total_instances:
                return self._dimension_schedule[self._current_t]
            return self.d_min
        elif self.evolution_pattern == "tds":
            return len(self._get_tds_indices(self._current_t))
        elif self.evolution_pattern == "vfs":
            return len(self._get_vfs_indices(self._current_t))

    def get_dimension_schedule(self) -> Optional[np.ndarray]:
        """获取完整的维度演化序列（用于可视化）
        
        注意：仅适用于 pyramid/incremental/decremental 模式
        TDS 和 VFS 模式返回 None
        """
        if self.evolution_pattern in ["pyramid", "incremental", "decremental"]:
            return self._dimension_schedule.copy()
        return None
    
    def __len__(self) -> int:
        """返回流的长度"""
        return min(self.total_instances, len(self.base_stream))

class TrapezoidalStream(Stream):
    """OVFM 专用梯形流（支持顺序/随机模式）
    
    特点：
    - 返回固定维度实例，未出生特征用 NaN 填充
    - 支持前缀顺序或随机顺序出生
    
    Example:
        >>> stream = TrapezoidalStream(
        ...     base_stream=Electricity(),
        ...     d_min=1, d_max=6,
        ...     feature_order="sequential"
        ... )
        >>> 
        >>> stream = TrapezoidalStream(
        ...     base_stream=Electricity(),
        ...     d_min=1, d_max=6,
        ...     feature_order="random"
        ... )
    """
    
    def __init__(
        self,
        base_stream: Stream,
        d_min: int = 2,
        d_max: Optional[int] = None,
        total_instances: int = 10000,
        num_phases: int = 10,
        feature_order: Literal["sequential", "random"] = "sequential",
        random_seed: int = 42
    ):
        """初始化 OVFM 梯形流
        
        :param base_stream: 原始数据流
        :param d_min: 起始特征数
        :param d_max: 最终特征数（None 则使用全部）
        :param total_instances: 总样本数
        :param num_phases: 特征出现的阶段数（默认10，即每10%实例新增一批特征）
        :param feature_order: 特征出现顺序
            - "sequential": 按索引顺序（0,1,2...）
            - "random": 随机打乱顺序
        :param random_seed: 随机种子
        """
        self.base_stream = base_stream
        self.d_min = d_min
        
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        if self.d_max > original_d:
            raise ValueError(
                f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})"
            )
        
        self.total_instances = total_instances
        self.num_phases = num_phases
        self.feature_order = feature_order
        self.random_seed = random_seed
        
        self._current_t = 0
        self._schema = base_stream.get_schema()
        
        # 初始化随机数生成器
        self._rng = np.random.RandomState(random_seed)
        
        # 预计算每个特征的出生时间
        self._feature_birth_times = self._compute_birth_times()
    
    def _compute_birth_times(self) -> np.ndarray:
        """计算每个特征的出生时间"""
        birth_times = np.zeros(self.d_max, dtype=int)
        
        if self.feature_order == "sequential":
            # 顺序模式：特征按索引顺序出生
            for i in range(self.d_max):
                phase = int((i * self.num_phases) / self.d_max)
                birth_times[i] = phase * (self.total_instances // self.num_phases)
        
        elif self.feature_order == "random":
            # 随机模式：特征随机打乱后分配出生时间
            indices = self._rng.permutation(self.d_max)  # 随机排列
            
            for i in range(self.d_max):
                feature_idx = indices[i]
                phase = int((i * self.num_phases) / self.d_max)
                birth_times[feature_idx] = phase * (self.total_instances // self.num_phases)
        
        else:
            raise ValueError(f"Unknown feature_order: {self.feature_order}")
        
        return birth_times
    
    def _get_active_features_mask(self, t: int) -> np.ndarray:
        """获取时刻 t 的活跃特征掩码（True=已出生，False=未出生）"""
        return self._feature_birth_times <= t
    
    def next_instance(self):
        """获取下一个实例（固定维度，缺失用 NaN）"""
        if not self.has_more_instances():
            return None
        
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # 获取当前已出生的特征掩码
        active_mask = self._get_active_features_mask(self._current_t)
        num_active = np.sum(active_mask)
        
        # 确保至少有 d_min 个特征（如果不够，随机激活）
        if num_active < self.d_min:
            inactive_indices = np.where(~active_mask)[0]
            if len(inactive_indices) > 0:
                # 随机选择一些未出生的特征提前激活
                num_needed = min(self.d_min - num_active, len(inactive_indices))
                early_birth = self._rng.choice(inactive_indices, num_needed, replace=False)
                active_mask[early_birth] = True
        
        # 创建固定维度的特征向量（全部用 NaN 初始化）
        x_full = np.full(self.d_max, np.nan)
        
        # 填充已出生的特征
        x_base = np.array(base_instance.x)
        x_full[active_mask] = x_base[active_mask]
        
        # 创建新实例（固定维度）
        if self._schema.is_classification():
            modified_instance = LabeledInstance.from_array(
                self._schema,
                x_full,
                base_instance.y_index
            )
        else:
            modified_instance = RegressionInstance.from_array(
                self._schema,
                x_full,
                base_instance.y_value
            )
        
        self._current_t += 1
        return modified_instance
    
    def has_more_instances(self) -> bool:
        return (
            self._current_t < self.total_instances 
            and self.base_stream.has_more_instances()
        )
    
    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
    
    def get_schema(self) -> Schema:
        return self._schema
    
    def get_moa_stream(self):
        return None
    
    def __len__(self) -> int:
        return min(self.total_instances, len(self.base_stream))


class CapriciousStream(Stream):
    """专为 OVFM 设计的任意变化流（VFS）
    
    特点：
    - 特征随机缺失（每个样本独立）
    - 返回固定维度实例，缺失特征用 NaN 填充
    - 保证与 OVFM 的 OnlineExpectationMaximization 兼容
    
    Example:
        >>> from capymoa.datasets import Electricity
        >>> from capymoa.stream import OVFMCapriciousStream
        >>> 
        >>> stream = OVFMCapriciousStream(
        ...     base_stream=Electricity(),
        ...     missing_ratio=0.5,
        ...     total_instances=3000
        ... )
        >>> 
        >>> # 每个实例都是固定8维，但约50%的特征是NaN
        >>> inst = stream.next_instance()
        >>> print(len(inst.x))  # 8
        >>> print(np.sum(np.isnan(inst.x)))  # 约 4
    """
    
    def __init__(
        self,
        base_stream: Stream,
        missing_ratio: float = 0.5,
        total_instances: int = 10000,
        min_features: int = 1,
        random_seed: int = 42
    ):
        """初始化 OVFM 任意变化流
        
        :param base_stream: 原始数据流
        :param missing_ratio: 特征缺失率（0.0-1.0）
        :param total_instances: 总样本数
        :param min_features: 每个实例至少保留的特征数
        :param random_seed: 随机种子
        """
        self.base_stream = base_stream
        self.missing_ratio = missing_ratio
        self.total_instances = total_instances
        self.min_features = min_features
        self.random_seed = random_seed
        
        self._current_t = 0
        self._schema = base_stream.get_schema()
        self._num_features = base_stream.get_schema().get_num_attributes()
        self._rng = np.random.RandomState(random_seed)
    
    def _get_feature_mask(self, t: int) -> np.ndarray:
        # 使用时间步作为种子，保证可重复性
        rng_t = np.random.RandomState(self.random_seed + t)
        
        # 生成掩码：每个特征以 (1 - missing_ratio) 概率保留
        mask = rng_t.rand(self._num_features) > self.missing_ratio
        
        # 确保至少有 min_features 个特征
        if np.sum(mask) < self.min_features:
            # 随机选择 min_features 个特征保留
            indices = rng_t.choice(
                self._num_features, 
                self.min_features, 
                replace=False
            )
            mask = np.zeros(self._num_features, dtype=bool)
            mask[indices] = True
        
        return mask
    
    def next_instance(self):
        if not self.has_more_instances():
            return None
        
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # 获取特征掩码
        mask = self._get_feature_mask(self._current_t)
        
        # 创建固定维度的特征向量
        x_base = np.array(base_instance.x)
        x_masked = x_base.copy()
        
        # 将缺失的特征设为 NaN
        x_masked[~mask] = np.nan
        
        # 创建新实例
        if self._schema.is_classification():
            modified_instance = LabeledInstance.from_array(
                self._schema,
                x_masked,
                base_instance.y_index
            )
        else:
            modified_instance = RegressionInstance.from_array(
                self._schema,
                x_masked,
                base_instance.y_value
            )
        
        self._current_t += 1
        return modified_instance
    
    def has_more_instances(self) -> bool:
        return (
            self._current_t < self.total_instances 
            and self.base_stream.has_more_instances()
        )
    
    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
    
    def get_schema(self) -> Schema:
        return self._schema
    
    def get_moa_stream(self):
        return None
    
    def get_current_missing_ratio(self) -> float:
        """获取当前实例的实际缺失率（用于调试）"""
        if self._current_t == 0:
            return 0.0
        mask = self._get_feature_mask(self._current_t - 1)
        return 1.0 - np.sum(mask) / len(mask)
    
    def __len__(self) -> int:
        return min(self.total_instances, len(self.base_stream))
    

class EvolvableStream(Stream):
    """Evolvable Stream: 模拟特征空间从 S1 → 全部特征 → S2 的三阶段演化。
    
    三阶段结构：
    - Phase 1: 只有前 d1 个特征（prefix）
    - Phase 2 (Overlap): 全部 d_max 个特征
    - Phase 3: 只有后 d2 个特征（suffix）
    
    这种模式常用于模拟特征空间切换场景（如 FESL、OLD³S 论文），
    其中有一个重叠期用于学习特征空间之间的映射关系。
    
    Example:
        >>> from capymoa.datasets import Electricity
        >>> from capymoa.stream import EvolvableStream
        >>> 
        >>> # 配置三阶段演化
        >>> stream = EvolvableStream(
        ...     base_stream=Electricity(),
        ...     d1=4,                    # Phase 1: 前 4 个特征
        ...     d2=4,                    # Phase 3: 后 4 个特征
        ...     phase1_ratio=0.4,        # Phase 1 占 40% 实例
        ...     overlap_ratio=0.2,       # Overlap 占 20% 实例
        ...     phase3_ratio=0.4,        # Phase 3 占 40% 实例
        ...     total_instances=10000
        ... )
        >>> 
        >>> # Phase 1 (t=0-3999):   x = [f0, f1, f2, f3]
        >>> # Overlap (t=4000-5999): x = [f0, f1, ..., f7]
        >>> # Phase 3 (t=6000-9999): x = [f4, f5, f6, f7]
        >>> 
        >>> # 配合 FESL 算法使用
        >>> learner = FESLClassifier(
        ...     schema=stream.get_schema(),
        ...     s1_feature_indices=[0, 1, 2, 3],
        ...     s2_feature_indices=[4, 5, 6, 7],
        ...     overlap_size=2000,
        ...     switch_point=6000
        ... )
    """

    def __init__(
        self,
        base_stream: Stream,
        d1: int,
        d2: int,
        d_max: Optional[int] = None,
        phase1_ratio: float = 0.4,
        overlap_ratio: float = 0.2,
        phase3_ratio: float = 0.4,
        total_instances: int = 10000,
        random_seed: int = 42
    ):
        """初始化 Evolvable Stream
        
        :param base_stream: 原始数据流
        :param d1: Phase 1 的特征数量（前缀特征）
        :param d2: Phase 3 的特征数量（后缀特征）
        :param d_max: 总特征数（None 则使用原始特征数）
        :param phase1_ratio: Phase 1 占总实例数的比例
        :param overlap_ratio: Overlap 占总实例数的比例
        :param phase3_ratio: Phase 3 占总实例数的比例
        :param total_instances: 总样本数
        :param random_seed: 随机种子
        
        注意：
        - phase1_ratio + overlap_ratio + phase3_ratio 必须等于 1.0
        - d1 和 d2 不应该重叠（d1 + d2 <= d_max）
        """
        # 验证比例参数
        ratio_sum = phase1_ratio + overlap_ratio + phase3_ratio
        if not np.isclose(ratio_sum, 1.0):
            raise ValueError(
                f"Phase ratios must sum to 1.0, got {ratio_sum:.3f} "
                f"(phase1={phase1_ratio}, overlap={overlap_ratio}, phase3={phase3_ratio})"
            )
        
        self.base_stream = base_stream
        self.d1 = d1
        self.d2 = d2
        
        # 获取原始特征数
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        if self.d_max > original_d:
            raise ValueError(
                f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})"
            )
        
        if d1 > self.d_max or d2 > self.d_max:
            raise ValueError(
                f"d1 ({d1}) and d2 ({d2}) must not exceed d_max ({self.d_max})"
            )
        
        if d1 + d2 > self.d_max:
            raise ValueError(
                f"d1 ({d1}) + d2 ({d2}) = {d1 + d2} exceeds d_max ({self.d_max}). "
                "Phase 1 and Phase 3 features should not overlap to ensure distinct feature spaces."
            )
        
        self.phase1_ratio = phase1_ratio
        self.overlap_ratio = overlap_ratio
        self.phase3_ratio = phase3_ratio
        self.total_instances = total_instances
        self.random_seed = random_seed
        
        # 设置随机种子
        self._rng = np.random.RandomState(random_seed)
        
        # 计算三个阶段的时间边界
        self.phase1_end = int(total_instances * phase1_ratio)
        self.overlap_end = self.phase1_end + int(total_instances * overlap_ratio)
        
        # 预计算特征索引
        self._s1_indices = np.arange(d1)  # Phase 1: [0, 1, ..., d1-1]
        self._s2_indices = np.arange(self.d_max - d2, self.d_max)  # Phase 3: 后 d2 个
        self._full_indices = np.arange(self.d_max)  # Overlap: 全部特征
        
        # 当前时间步
        self._current_t = 0
        
        # Schema
        self._schema = base_stream.get_schema()

    def __str__(self):
        return (
            f"EvolvableStream(d1={self.d1}, d2={self.d2}, d_max={self.d_max}, "
            f"phases=[{self.phase1_ratio:.1%}, {self.overlap_ratio:.1%}, {self.phase3_ratio:.1%}])"
        )

    def _get_active_indices(self, t: int) -> np.ndarray:
        """根据时间步获取活跃特征索引
        
        :param t: 当前时间步
        :return: 活跃特征的索引数组
        """
        if t < self.phase1_end:
            # Phase 1: 前 d1 个特征
            return self._s1_indices
        elif t < self.overlap_end:
            # Overlap: 全部特征
            return self._full_indices
        else:
            # Phase 3: 后 d2 个特征
            return self._s2_indices

    def next_instance(self):
        """获取下一个实例（特征已演化）
        
        根据当前时间步，返回相应阶段的特征子集。
        
        :return: 演化后的 Instance 对象，或 None（流结束时）
        """
        if not self.has_more_instances():
            return None
        
        # 从基础流获取原始实例
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # 根据当前时间步获取活跃特征
        active_indices = self._get_active_indices(self._current_t)
        
        # 提取子集特征
        x_full = np.array(base_instance.x)
        x_subset = x_full[active_indices]
        
        # 创建新实例
        if self._schema.is_classification():
            modified_instance = LabeledInstance.from_array(
                self._schema,
                x_subset,
                base_instance.y_index
            )
        else:
            modified_instance = RegressionInstance.from_array(
                self._schema,
                x_subset,
                base_instance.y_value
            )
        
        self._current_t += 1
        return modified_instance

    def has_more_instances(self) -> bool:
        """检查是否还有更多实例"""
        return (
            self._current_t < self.total_instances 
            and self.base_stream.has_more_instances()
        )

    def restart(self):
        """重启流"""
        self.base_stream.restart()
        self._current_t = 0

    def get_schema(self) -> Schema:
        """返回 schema"""
        return self._schema

    def get_moa_stream(self):
        """自定义流不支持 MOA 加速"""
        return None
    
    def get_current_phase(self) -> str:
        """获取当前所处阶段（用于调试和监控）
        
        :return: 'phase1', 'overlap', 或 'phase3'
        """
        t = self._current_t
        if t < self.phase1_end:
            return "phase1"
        elif t < self.overlap_end:
            return "overlap"
        else:
            return "phase3"
    
    def get_current_dimension(self) -> int:
        """获取当前特征维度（用于调试）
        
        :return: 当前时刻的特征数量
        """
        return len(self._get_active_indices(self._current_t))
    
    def get_phase_boundaries(self) -> dict:
        """获取三个阶段的时间边界（用于可视化和分析）
        
        :return: 字典，包含每个阶段的 (start, end) 时间步
        """
        return {
            "phase1": (0, self.phase1_end),
            "overlap": (self.phase1_end, self.overlap_end),
            "phase3": (self.overlap_end, self.total_instances)
        }
    
    def get_feature_spaces(self) -> dict:
        """获取各阶段的特征索引（用于算法配置）
        
        :return: 字典，包含各阶段的特征索引数组
        """
        return {
            "s1_indices": self._s1_indices.copy(),
            "s2_indices": self._s2_indices.copy(),
            "full_indices": self._full_indices.copy()
        }
    
    def __len__(self) -> int:
        """返回流的长度"""
        return min(self.total_instances, len(self.base_stream))