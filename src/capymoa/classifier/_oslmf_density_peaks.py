"""Density-peak clustering for semi-supervised label propagation"""

import numpy as np
from typing import Tuple, List


class DensityPeakClustering:
    """在线 Density-peak 聚类，用于半监督标签传播
    
    核心思想：
    1. 计算每个样本的局部密度 ρ 和距离 δ
    2. 识别聚类中心（高 ρ + 高 δ）
    3. 构建指向图（每个点指向密度更高的最近邻）
    4. 沿指向图传播标签
    """
    
    def __init__(self, buffer_size=200, p_arr=0.02):
        """
        Parameters
        ----------
        buffer_size : int
            缓冲区大小
        p_arr : float
            用于计算 cutoff distance 的百分比
        """
        self.buffer_size = buffer_size
        self.p_arr = p_arr
        
        self.buffer_X = []
        self.buffer_y = []
        self.buffer_labeled = []  # 是否有标签
        
    def add_instance(self, x, y=None, is_labeled=False):
        """添加实例到缓冲区
        
        Parameters
        ----------
        x : np.ndarray
            特征向量
        y : int or None
            标签（如果有）
        is_labeled : bool
            是否有标签
        """
        if len(self.buffer_X) >= self.buffer_size:
            # 移除最老的实例
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)
            self.buffer_labeled.pop(0)
        
        self.buffer_X.append(x)
        self.buffer_y.append(y if is_labeled else None)
        self.buffer_labeled.append(is_labeled)
    
    def propagate_labels(self) -> Tuple[List, List]:
        """通过 density-peak 结构传播标签
        
        Returns
        -------
        pseudo_labels : List
            传播后的伪标签
        confidence : List
            置信度分数
        """
        if len(self.buffer_X) < 2:
            return self.buffer_y.copy(), [1.0] * len(self.buffer_y)
        
        X = np.array(self.buffer_X)
        n = len(X)
        
        # 计算距离矩阵
        dist_matrix = self._compute_distances(X)
        
        # 计算局部密度 ρ 和距离 δ
        rho, delta, nearest_higher = self._compute_density_peaks(dist_matrix)
        
        # 传播标签
        pseudo_labels = self.buffer_y.copy()
        confidence = [1.0 if labeled else 0.0 for labeled in self.buffer_labeled]
        
        # 按密度从高到低排序
        sorted_indices = np.argsort(-rho)
        
        for idx in sorted_indices:
            if pseudo_labels[idx] is not None:
                continue  # 已经有标签
            
            # 沿着 nearest_higher 链找到有标签的点
            current = idx
            path = []
            max_depth = 10  # 防止无限循环
            
            for _ in range(max_depth):
                if current == -1:
                    break
                path.append(current)
                
                if pseudo_labels[current] is not None:
                    # 找到标签，传播给路径上的所有点
                    label = pseudo_labels[current]
                    conf = confidence[current] * 0.9  # 衰减置信度
                    
                    for p in path[:-1]:
                        if pseudo_labels[p] is None:
                            pseudo_labels[p] = label
                            confidence[p] = conf
                    break
                
                current = nearest_higher[current]
        
        return pseudo_labels, confidence
    
    def _compute_distances(self, X):
        """计算欧氏距离矩阵"""
        n = len(X)
        dist = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(X[i] - X[j])
                dist[i, j] = d
                dist[j, i] = d
        
        return dist
    
    def _compute_density_peaks(self, dist_matrix):
        """计算 ρ, δ, 和 nearest_higher
        
        Returns
        -------
        rho : np.ndarray
            局部密度
        delta : np.ndarray
            到更高密度点的距离
        nearest_higher : np.ndarray
            指向的更高密度点的索引
        """
        n = len(dist_matrix)
        
        # 计算 cutoff distance
        upper_tri_indices = np.triu_indices(n, k=1)
        all_distances = dist_matrix[upper_tri_indices]
        position = int(len(all_distances) * self.p_arr)
        d_cut = np.sort(all_distances)[min(position, len(all_distances) - 1)]
        
        # 计算局部密度 ρ (Gaussian kernel)
        rho = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    rho[i] += np.exp(-(dist_matrix[i, j] / d_cut) ** 2)
        
        # 计算 δ 和 nearest_higher
        delta = np.zeros(n)
        nearest_higher = np.full(n, -1, dtype=int)
        
        sorted_indices = np.argsort(-rho)  # 密度从高到低
        
        for i, idx in enumerate(sorted_indices):
            if i == 0:
                # 密度最高的点
                delta[idx] = np.max(dist_matrix[idx])
            else:
                # 找到密度更高的点中最近的
                higher_density_indices = sorted_indices[:i]
                distances_to_higher = dist_matrix[idx, higher_density_indices]
                nearest_idx = np.argmin(distances_to_higher)
                
                delta[idx] = distances_to_higher[nearest_idx]
                nearest_higher[idx] = higher_density_indices[nearest_idx]
        
        return rho, delta, nearest_higher