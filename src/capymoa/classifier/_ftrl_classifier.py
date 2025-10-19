"""
FTRL Classifier - Final Production Version
Implements Follow-The-Regularized-Leader algorithm from McMahan (2011)
Optimized for both sparse and dense features
"""

from __future__ import annotations
import numpy as np
from typing import Union, Dict, Optional
from collections import defaultdict

from capymoa.base import Classifier
from capymoa.stream import Schema
from capymoa.instance import Instance


class FTRLClassifier(Classifier):
    """Follow-The-Regularized-Leader (FTRL-Proximal) Classifier.
    
    Reference:
    McMahan, H. B. (2011). Follow-the-Regularized-Leader and Mirror Descent:
    Equivalence Theorems and L1 Regularization. AISTATS 2011.
    
    Supports both sparse high-dimensional data (rcv1, news20) and dense data (Electricity).
    
    Parameters:
        alpha: Learning rate parameter (default 0.5)
        beta: Smoothing parameter for adaptive learning rate (default 1.0)
        l1: L1 regularization strength - induces sparsity (default 0.1)
        l2: L2 regularization strength - smoothing (default 1.0)
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 0.5,
        beta: float = 1.0,
        l1: float = 0.1,
        l2: float = 1.0,
        random_seed: int = 1,
    ):
        """Initialize FTRL Classifier.
        
        :param schema: Stream schema
        :param alpha: Learning rate parameter
        :param beta: Smoothing parameter
        :param l1: L1 regularization (sparsity)
        :param l2: L2 regularization (smoothness)
        :param random_seed: Random seed
        """
        super().__init__(schema=schema, random_seed=random_seed)
        
        if schema.get_num_classes() != 2:
            raise ValueError("FTRL only supports binary classification")
        
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        
        # FTRL state - sparse representation for efficiency
        self.z = defaultdict(float)  # Coefficients
        self.n = defaultdict(float)  # Sum of squared gradients
        self.w = {}  # Weights (only non-zero stored)
        
        self.t = 0  # Timestep counter
        self._is_sparse = None  # Auto-detect input type

    def __str__(self):
        return (f"FTRLClassifier(alpha={self.alpha}, beta={self.beta}, "
                f"l1={self.l1}, l2={self.l2})")

    def _parse_features(self, instance: Instance) -> Union[Dict[int, float], Dict[str, float]]:
        """Parse features from instance - supports both sparse dict and dense array."""
        
        # ✅ 优先检查自定义的稀疏特征属性
        if hasattr(instance, '_sparse_x'):
            self._is_sparse = True
            return instance._sparse_x
        
        # 原有的逻辑（处理标准numpy数组）
        if isinstance(instance.x, dict):
            self._is_sparse = True
            return instance.x
        else:
            x = np.array(instance.x)
            if self._is_sparse is None:
                sparsity = 1.0 - (np.count_nonzero(x) / len(x))
                self._is_sparse = sparsity > 0.5
            
            if self._is_sparse:
                sparse = {}
                for i, val in enumerate(x):
                    if val != 0.0:
                        sparse[i] = float(val)
                return sparse
            else:
                return x

    def _get_weight(self, feature_id: int) -> float:
        """Get weight for feature, computing on-the-fly if sparse."""
        if feature_id not in self.w:
            self._compute_weight(feature_id)
        return self.w.get(feature_id, 0.0)

    def _compute_weight(self, feature_id: int):
        """Compute weight using FTRL-Proximal formula.
        
        w[i] = 0 if |z[i]| <= l1
        w[i] = -(z[i] - sign(z[i])*l1) / ((beta + sqrt(n[i]))/alpha + l2) otherwise
        """
        z_val = self.z[feature_id]
        n_val = self.n[feature_id]
        
        if np.abs(z_val) <= self.l1:
            # Soft-thresholding
            if feature_id in self.w:
                del self.w[feature_id]
        else:
            # Compute weight
            numerator = -(z_val - np.sign(z_val) * self.l1)
            denominator = (self.beta + np.sqrt(n_val)) / self.alpha + self.l2
            self.w[feature_id] = numerator / denominator

    def train(self, instance: Instance):
        """FTRL-Proximal update step.
        
        Process:
        1. Parse features (sparse or dense)
        2. Compute prediction
        3. Calculate gradient
        4. Update z, n, w for each feature
        """
        features = self._parse_features(instance)
        y = float(instance.y_index)
        self.t += 1
        
        # Prediction
        y_hat = self._predict_proba(features)
        
        # FTRL update
        if isinstance(features, dict):
            # Sparse update - O(k) where k is number of non-zero features
            for feature_id, x_val in features.items():
                self._ftrl_update_feature(feature_id, x_val, y_hat, y)
        else:
            # Dense update - O(n)
            for i, x_val in enumerate(features):
                if x_val != 0.0:  # Skip zero contributions
                    self._ftrl_update_feature(i, float(x_val), y_hat, y)

    def _ftrl_update_feature(self, feature_id: int, x_val: float, y_hat: float, y: float):
        """Update FTRL state for a single feature."""
        gradient = (y_hat - y) * x_val
        old_n = self.n[feature_id]
        new_n = old_n + gradient ** 2
        
        # Adaptive learning rate
        sigma = (np.sqrt(new_n) - np.sqrt(old_n)) / self.alpha
        
        # Update z
        old_weight = self._get_weight(feature_id)
        self.z[feature_id] = self.z[feature_id] + gradient - sigma * old_weight
        self.n[feature_id] = new_n
        
        # Recompute weight
        self._compute_weight(feature_id)

    def predict(self, instance: Instance) -> int:
        """Predict class label (0 or 1)."""
        prob = self._predict_proba(self._parse_features(instance))
        return 1 if prob > 0.5 else 0

    def predict_proba(self, instance: Instance) -> np.ndarray:
        """Predict class probabilities."""
        prob_class1 = self._predict_proba(self._parse_features(instance))
        return np.array([1 - prob_class1, prob_class1])

    def _predict_proba(self, features: Union[Dict[int, float], np.ndarray]) -> float:
        """Compute prediction probability using logistic function."""
        logit = 0.0
        
        if isinstance(features, dict):
            # Sparse prediction - O(k)
            for feature_id, x_val in features.items():
                w_val = self._get_weight(feature_id)
                logit += w_val * x_val
        else:
            # Dense prediction - O(n)
            for i, x_val in enumerate(features):
                if x_val != 0.0:
                    w_val = self._get_weight(i)
                    logit += w_val * x_val
        
        return self._sigmoid(logit)

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Sigmoid with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # ============ Analysis methods ============

    def get_sparsity(self) -> float:
        """Get proportion of zero weights among all encountered features.
        
        For sparse models, we calculate sparsity based on all features
        that have been encountered during training (stored in self.z).
        """
        if not hasattr(self, 'z') or not self.z:
            return 1.0
        
        # Total features = all features encountered (stored in z)
        total_features = len(self.z)
        
        # Non-zero weights = features in self.w with non-zero values
        # Note: zero weights are removed from self.w by _compute_weight
        num_nonzero = len(self.w)
        
        # Zero weights = total - non-zero
        num_zero = total_features - num_nonzero
        
        return num_zero / total_features if total_features > 0 else 1.0

    def get_density(self) -> float:
        """Get proportion of non-zero weights."""
        return 1.0 - self.get_sparsity()

    def get_num_active_features(self) -> int:
        """Get number of non-zero weights."""
        return len(self.w)

    def get_num_zero_weights(self) -> int:
        """Get number of zero weights."""
        if not hasattr(self, 'z'):
            return 0
        return len(self.z) - len(self.w)

    def get_num_total_features(self) -> int:
        """Get total number of features encountered."""
        return len(self.z) if hasattr(self, 'z') else 0

    def get_num_active_features(self) -> int:
        """Get number of non-zero weights."""
        return len([v for v in self.w.values() if np.abs(v) > 1e-8])

    def get_weights_sparse(self) -> Dict[int, float]:
        """Get weights as sparse dictionary (memory efficient)."""
        return dict(self.w)

    def get_top_weights(self, k: int = 10) -> list:
        """Get top-k features by absolute weight value."""
        if not self.w:
            return []
        sorted_weights = sorted(self.w.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_weights[:k]

    def get_weight(self, feature_id: int) -> float:
        """Get weight for specific feature."""
        return self._get_weight(feature_id)