"""OSLMF Classifier - Online Semi-supervised Learning with Mix-typed Features"""

import numpy as np
from capymoa.base import Classifier
from capymoa.instance import LabeledInstance
from capymoa._utils import build_cli_str_from_mapping_and_locals
from ._oslmf_copula import GaussianCopula
from ._oslmf_density_peaks import DensityPeakClustering


class OSLMFClassifier(Classifier):
    """Online Semi-supervised Learning with Mix-typed streaming Features
    
    论文：Wu et al., "Online Semi-supervised Learning with Mix-Typed Streaming 
    Features", AAAI 2023
    
    核心思想：
    1. Gaussian Copula 处理混合数据类型（连续+离散）
    2. Density-peak clustering 在半监督场景下传播标签
    3. Ensemble 两个 learner：observed space + latent space
    
    Parameters
    ----------
    schema : Schema
        数据 schema
    window_size : int, default=200
        Copula 的滑动窗口大小
    buffer_size : int, default=200
        Density-peak 的缓冲区大小
    learning_rate : float, default=0.01
        SGD 学习率
    decay_coef : float, default=0.5
        Copula 协方差更新的衰减系数
    max_ord_levels : int, default=14
        判断 ordinal 的最大类别数
    random_seed : int, default=42
        随机种子
    """
    
    def __init__(
        self,
        schema=None,
        window_size: int = 200,
        buffer_size: int = 200,
        learning_rate: float = 0.01,
        decay_coef: float = 0.5,
        max_ord_levels: int = 14,
        random_seed: int = 42
    ):
        super().__init__(schema=schema, random_seed=random_seed)
        
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.decay_coef = decay_coef
        self.max_ord_levels = max_ord_levels
        
        # 在 schema 可用后初始化
        self._initialized = False
        self._copula = None
        self._density_peaks = None
        self._w_observed = None
        self._w_latent = None
        self.ensemble_weight = 0.5  # α
        
        # 训练统计
        self._loss_observed = 0.0
        self._loss_latent = 0.0
        self._num_updates = 0
        
        np.random.seed(random_seed)
    
    def _lazy_init(self, instance):
        """延迟初始化（等 schema 可用）"""
        if self._initialized:
            return
        
        num_features = len(instance.x)
        
        # 检测特征类型
        self._cont_indices, self._ord_indices = self._detect_feature_types(instance.x)
        
        # 初始化 Copula
        self._copula = GaussianCopula(
            self._cont_indices,
            self._ord_indices,
            window_size=self.window_size
        )
        
        # 初始化 Density-peak
        self._density_peaks = DensityPeakClustering(
            buffer_size=self.buffer_size
        )
        
        # 初始化权重（+1 for bias）
        self._w_observed = np.zeros(num_features + 1)
        self._w_latent = np.zeros(num_features + 1)
        
        self._initialized = True
    
    def _detect_feature_types(self, x):
        """检测特征类型：连续 vs 离散
        
        简化版：假设所有特征都是连续的
        实际应该基于历史数据的唯一值数量判断
        """
        num_features = len(x)
        # 简化：这里应该用滑动窗口统计唯一值
        # 暂时假设都是连续的
        cont_indices = np.ones(num_features, dtype=bool)
        ord_indices = np.zeros(num_features, dtype=bool)
        return cont_indices, ord_indices
    
    def train(self, instance: LabeledInstance):
        """训练（可能有或没有标签）"""
        if not self._initialized:
            self._lazy_init(instance)
        
        x = np.array(instance.x)
        y = 1 if instance.y_index == 1 else -1
        has_true_label = hasattr(instance, '_has_true_label') and instance._has_true_label
        
        # 1-6 步：Copula + Density-peak（不变）
        X_batch = x.reshape(1, -1)
        self._copula.partial_fit(X_batch)
        Z_latent = self._copula.transform_to_latent(X_batch)
        self._density_peaks.add_instance(x, y, is_labeled=has_true_label)
        pseudo_labels, confidence = self._density_peaks.propagate_labels()
        
        if has_true_label:
            self._copula.update_covariance_em(X_batch, Z_latent, self.decay_coef)
        
        X_reconstructed = self._copula.reconstruct_features(X_batch, Z_latent)
        
        # 7. 训练两个 learner
        x_obs = self._pad_with_bias(X_reconstructed[0])
        z_lat = self._pad_with_bias(Z_latent[0])
        
        effective_y = y
        if not has_true_label and pseudo_labels[-1] is not None:
            effective_y = 1 if pseudo_labels[-1] == 1 else -1
        
        # Observed learner
        pred_obs = np.dot(self._w_observed, x_obs)
        loss_obs = max(0, 1 - effective_y * pred_obs)
        if loss_obs > 0:
            grad_obs = -effective_y * x_obs
            self._w_observed -= self.learning_rate * grad_obs
        
        # Latent learner
        pred_lat = np.dot(self._w_latent, z_lat)
        loss_lat = max(0, 1 - effective_y * pred_lat)
        if loss_lat > 0:
            grad_lat = -effective_y * z_lat
            self._w_latent -= self.learning_rate * grad_lat
        
        # 8. 更新 ensemble 权重（修复：数值稳定版）
        self._loss_observed += loss_obs
        self._loss_latent += loss_lat
        self._num_updates += 1
        
        if self._num_updates > 10:  # 等积累一些样本再更新
            T = self._num_updates
            mu = 2 * np.sqrt(2 * np.log(2) / T)
            
            # Log-space 计算避免下溢
            log_w_obs = -mu * self._loss_observed
            log_w_lat = -mu * self._loss_latent
            
            # Log-sum-exp trick
            max_log = max(log_w_obs, log_w_lat)
            log_sum = max_log + np.log(
                np.exp(log_w_obs - max_log) + np.exp(log_w_lat - max_log)
            )
            
            # 防止 NaN
            if np.isfinite(log_sum):
                self.ensemble_weight = np.exp(log_w_obs - log_sum)
            else:
                self.ensemble_weight = 0.5
        # 在 train 方法的最后加上
        if self._num_updates % 500 == 0:
            print(f"\n[DEBUG at {self._num_updates}]")
            print(f"  Cumulative loss: {self._loss_observed:.1f} / {self._loss_latent:.1f}")
            print(f"  mu: {mu:.6f}")
            print(f"  log_w: {log_w_obs:.2f} / {log_w_lat:.2f}")
            print(f"  alpha: {self.ensemble_weight:.4f}")
    
    def predict(self, instance):
        """预测（只用 x，不用 y）
        
        Parameters
        ----------
        instance : Instance
            预测实例
        
        Returns
        -------
        prediction : int
            预测的类别索引
        """
        if not self._initialized:
            # 第一次预测前没有训练，随机猜测
            return 0
        
        x = np.array(instance.x)
        
        # 1. 变换到潜在空间（用历史学到的 Copula）
        X_batch = x.reshape(1, -1)
        Z_latent = self._copula.transform_to_latent(X_batch)
        
        # 2. 重构特征
        X_reconstructed = self._copula.reconstruct_features(X_batch, Z_latent)
        
        # 3. Ensemble 预测
        x_obs = self._pad_with_bias(X_reconstructed[0])
        z_lat = self._pad_with_bias(Z_latent[0])
        
        pred_obs = np.dot(self._w_observed, x_obs)
        pred_lat = np.dot(self._w_latent, z_lat)
        
        pred_ensemble = self.ensemble_weight * pred_obs + (1 - self.ensemble_weight) * pred_lat
        
        # 转换为类别索引
        return 1 if pred_ensemble > 0 else 0
    
    def predict_proba(self, instance):
        """预测概率（简化版，用 sigmoid）"""
        if not self._initialized:
            return np.array([0.5, 0.5])
        
        x = np.array(instance.x)
        X_batch = x.reshape(1, -1)
        Z_latent = self._copula.transform_to_latent(X_batch)
        X_reconstructed = self._copula.reconstruct_features(X_batch, Z_latent)
        
        x_obs = self._pad_with_bias(X_reconstructed[0])
        z_lat = self._pad_with_bias(Z_latent[0])
        
        pred_obs = np.dot(self._w_observed, x_obs)
        pred_lat = np.dot(self._w_latent, z_lat)
        
        pred_ensemble = self.ensemble_weight * pred_obs + (1 - self.ensemble_weight) * pred_lat
        
        # Sigmoid
        p_class1 = 1 / (1 + np.exp(-pred_ensemble))
        return np.array([1 - p_class1, p_class1])
    
    def _pad_with_bias(self, x):
        """添加 bias 项"""
        x_clean = np.nan_to_num(x, nan=0.0)
        return np.append(x_clean, 1.0)
    
    def __str__(self):
        return f"OSLMFClassifier(window={self.window_size}, buffer={self.buffer_size})"