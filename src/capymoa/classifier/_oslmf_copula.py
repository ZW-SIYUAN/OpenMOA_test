"""Gaussian Copula for OSLMF - 处理混合数据类型"""

import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF


class GaussianCopula:
    """在线 Gaussian Copula 模型，用于混合数据类型的特征对齐
    
    核心功能：
    1. 边缘分布估计（ECDF）
    2. 协方差矩阵 Σ 的在线 EM 更新
    3. 特征重构（observed → latent → reconstructed）
    """
    
    def __init__(self, cont_indices, ord_indices, window_size=200):
        """
        Parameters
        ----------
        cont_indices : np.ndarray (bool)
            连续特征的索引掩码
        ord_indices : np.ndarray (bool)
            离散（ordinal）特征的索引掩码
        window_size : int
            滑动窗口大小（用于边缘分布估计）
        """
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        self.window_size = window_size
        
        p = len(cont_indices)
        self.p = p
        
        # 滑动窗口（用于 ECDF）
        self.window = np.full((window_size, p), np.nan)
        self.update_pos = np.zeros(p, dtype=int)
        
        # 协方差矩阵（ordinal first, continuous second）
        self.Sigma = np.eye(p)
        self.iteration = 1
        
    def partial_fit(self, X_batch):
        """更新边缘分布的滑动窗口
        
        Parameters
        ----------
        X_batch : np.ndarray, shape (batch_size, p)
            批次数据
        """
        for row in X_batch:
            for col_idx in range(self.p):
                value = row[col_idx]
                if not np.isnan(value):
                    self.window[self.update_pos[col_idx], col_idx] = value
                    self.update_pos[col_idx] = (self.update_pos[col_idx] + 1) % self.window_size
    
    def transform_to_latent(self, X_batch):
        """X (observed) → Z (latent normals)
        
        Parameters
        ----------
        X_batch : np.ndarray, shape (batch_size, p)
        
        Returns
        -------
        Z : np.ndarray, shape (batch_size, p)
            潜在正态空间的表示
        """
        Z = np.empty_like(X_batch)
        Z[:] = np.nan
        
        # 连续特征变换
        for i in np.where(self.cont_indices)[0]:
            missing = np.isnan(X_batch[:, i])
            if np.sum(~missing) > 0:
                Z[~missing, i] = self._continuous_to_latent(
                    X_batch[~missing, i], 
                    self.window[:, i]
                )
        
        # 离散特征变换（取区间中点）
        for i in np.where(self.ord_indices)[0]:
            missing = np.isnan(X_batch[:, i])
            if np.sum(~missing) > 0:
                z_lower, z_upper = self._ordinal_to_latent(
                    X_batch[~missing, i],
                    self.window[:, i]
                )
                Z[~missing, i] = (z_lower + z_upper) / 2
                Z[~missing, i] = np.clip(Z[~missing, i], -5, 5)  # 防止无穷值
        
        return Z
    
    def _continuous_to_latent(self, x_obs, window):
        """连续特征 → 标准正态"""
        window_clean = window[~np.isnan(window)]
        if len(window_clean) == 0:
            return np.zeros_like(x_obs)
        
        ecdf = ECDF(window_clean)
        H = len(window_clean) / (len(window_clean) + 1)
        u = H * ecdf(x_obs)
        u = np.clip(u, 1e-10, 1 - 1e-10)  # 避免 0 和 1
        return norm.ppf(u)
    
    def _ordinal_to_latent(self, x_obs, window):
        """离散特征 → 标准正态区间"""
        window_clean = window[~np.isnan(window)]
        if len(window_clean) == 0:
            return np.zeros_like(x_obs), np.zeros_like(x_obs)
        
        ecdf = ECDF(window_clean)
        unique = np.unique(window_clean)
        
        if len(unique) > 1:
            threshold = np.min(np.abs(unique[1:] - unique[:-1])) / 2
            z_lower = norm.ppf(np.clip(ecdf(x_obs - threshold), 1e-10, 1 - 1e-10))
            z_upper = norm.ppf(np.clip(ecdf(x_obs + threshold), 1e-10, 1 - 1e-10))
        else:
            z_lower = np.full_like(x_obs, -5.0)
            z_upper = np.full_like(x_obs, 5.0)
        
        return z_lower, z_upper
    
    def update_covariance_em(self, X_batch, Z_latent, decay_coef=0.5):
        """EM 更新协方差矩阵 Σ
        
        Parameters
        ----------
        X_batch : np.ndarray
            观测数据
        Z_latent : np.ndarray
            潜在表示（可能有 NaN）
        decay_coef : float
            衰减系数（新 vs 旧）
        """
        batch_size = len(X_batch)
        
        # 至少需要 2 个样本才能计算协方差
        if batch_size < 2:
            return
        
        # E-step: 用当前 Σ 推断缺失值
        Z_imputed = Z_latent.copy()
        for i in range(len(Z_imputed)):
            missing_idx = np.where(np.isnan(Z_imputed[i]))[0]
            if len(missing_idx) > 0:
                # 简化：用均值填充
                Z_imputed[i, missing_idx] = 0
        
        # M-step: 更新 Σ（添加安全检查）
        try:
            # 检查是否有足够的有效数据
            valid_rows = ~np.any(np.isnan(Z_imputed), axis=1)
            if np.sum(valid_rows) < 2:
                return
            
            Sigma_new = np.cov(Z_imputed, rowvar=False)
            
            # 检查有效性
            if np.any(np.isnan(Sigma_new)) or np.any(np.isinf(Sigma_new)):
                return
            
            # 确保维度正确
            if Sigma_new.shape != (self.p, self.p):
                return
            
            # 正则化：确保正定
            Sigma_new = Sigma_new + np.eye(self.p) * 1e-6
            
            # 标准化为相关矩阵
            D = np.sqrt(np.diag(Sigma_new))
            D[D < 1e-10] = 1  # 避免除零
            Sigma_new = Sigma_new / D[:, None] / D[None, :]
            
            # 再次检查（防止除零产生 NaN）
            if np.any(np.isnan(Sigma_new)) or np.any(np.isinf(Sigma_new)):
                return
            
            # 指数衰减更新
            self.Sigma = decay_coef * Sigma_new + (1 - decay_coef) * self.Sigma
            self.iteration += 1
            
        except (ValueError, np.linalg.LinAlgError):
            # 如果协方差计算失败，跳过这次更新
            pass
    
    def reconstruct_features(self, X_batch, Z_latent):
        """重构完整特征（包括缺失的）
        
        Parameters
        ----------
        X_batch : np.ndarray
            原始观测（有 NaN）
        Z_latent : np.ndarray
            潜在表示
        
        Returns
        -------
        X_reconstructed : np.ndarray
            重构后的完整特征
        """
        X_rec = X_batch.copy()
        
        # 对于缺失的特征，用潜在值反变换
        for i in range(self.p):
            missing = np.isnan(X_rec[:, i])
            if np.sum(missing) == 0:
                continue
            
            z_missing = Z_latent[missing, i]
            if np.any(np.isnan(z_missing)):
                z_missing = np.nan_to_num(z_missing, nan=0.0)
            
            if self.cont_indices[i]:
                X_rec[missing, i] = self._latent_to_continuous(
                    z_missing, self.window[:, i]
                )
            else:
                X_rec[missing, i] = self._latent_to_ordinal(
                    z_missing, self.window[:, i]
                )
        
        return X_rec
    
    def _latent_to_continuous(self, z, window):
        """标准正态 → 连续特征"""
        window_clean = window[~np.isnan(window)]
        if len(window_clean) == 0:
            return np.zeros_like(z)
        
        u = norm.cdf(z)
        return np.quantile(window_clean, u)
    
    def _latent_to_ordinal(self, z, window):
        """标准正态 → 离散特征"""
        window_clean = window[~np.isnan(window)]
        if len(window_clean) == 0:
            return np.zeros_like(z)
        
        u = norm.cdf(z)
        n = len(window_clean)
        indices = np.ceil(np.round((n + 1) * u - 1, 3))
        indices = np.clip(indices, 0, n - 1).astype(int)
        return np.sort(window_clean)[indices]