"""capymoa/classifier/_rsol_classifier.py"""
from __future__ import annotations
from typing import Optional
import numpy as np

from capymoa.base import Classifier
from capymoa.stream import Schema
from capymoa.instance import Instance


class RSOLClassifier(Classifier):
    """Robust Sparse Online Learning (RSOL) Classifier.
    
    Handles online binary classification with dynamically evolving feature spaces
    using ℓ1,2-norm regularization for sparsity. Designed for scenarios where
    features can appear (incremental) or disappear (decremental) over time.
    
    Reference:
    
    Chen, Z., He, Y., Wu, D., Zhan, H., Sheng, V., & Zhang, K. (2024).
    Robust Sparse Online Learning for Data Streams with Streaming Features.
    SIAM International Conference on Data Mining (SDM).
    
    Example:
    
    >>> from capymoa.datasets import Electricity
    >>> from capymoa.classifier import RSOLClassifier
    >>> from capymoa.stream import EvolvingFeatureStream
    >>> from capymoa.evaluation import prequential_evaluation
    >>> 
    >>> base_stream = Electricity()
    >>> evolving_stream = EvolvingFeatureStream(base_stream, d_min=2, d_max=6)
    >>> learner = RSOLClassifier(
    ...     schema=evolving_stream.get_schema(),
    ...     lambda_param=50,
    ...     mu=1,
    ...     L=1000
    ... )
    >>> results = prequential_evaluation(evolving_stream, learner, max_instances=10000)
    """

    def __init__(
        self,
        schema: Schema,
        lambda_param: float = 50.0,
        mu: float = 1.0,
        L: int = 1000,
        d_max: int = 1000,
        random_seed: int = 1,
    ):
        """Initialize RSOL Classifier.
        
        :param schema: The schema of the stream
        :param lambda_param: Regularization parameter for ℓ1,2-norm sparsity (larger = more sparse)
        :param mu: Penalty parameter for PA update (smaller = more aggressive)
        :param L: Sliding window size for storing historical weights
        :param d_max: Maximum expected feature dimension
        :param random_seed: Random seed for reproducibility
        """
        super().__init__(schema=schema, random_seed=random_seed)
        
        np.random.seed(random_seed)
        
        # Algorithm parameters
        self.lambda_param = lambda_param
        self.mu = mu
        self.L = L
        self.d_max = d_max
        
        # Weight matrix W: d_max × L (features × window)
        self.W = np.zeros((d_max, L))
        
        # Current feature dimension
        self.current_dim = 0
        
        # Time step counter
        self.t = 0

    def __str__(self):
        return (f"RSOLClassifier(lambda={self.lambda_param}, mu={self.mu}, "
                f"L={self.L}, d_max={self.d_max})")

    def train(self, instance: Instance):
        """Train on a single instance with potentially evolved features."""
        self.t += 1
        
        # Extract features and label
        x_full = np.array(instance.x)
        d_current = len(x_full)
        y = 1 if instance.y_index == 1 else -1  # Convert to {-1, +1}
        
        # Determine if decremental or incremental
        if self.current_dim >= d_current:
            # Decremental case: features disappeared
            w_new = self._update_decremental(x_full, y, d_current)
        else:
            # Incremental case: new features appeared
            w_new = self._update_incremental(x_full, y, d_current)
        
        # Shift sliding window and add new weight vector
        self.W = np.roll(self.W, -1, axis=1)
        self.W[:, -1] = 0
        self.W[:d_current, -1] = w_new
        
        # Apply ℓ1,2-norm sparsification (Theorem 3.3)
        self._apply_l12_sparsity()
        
        # Update current dimension
        self.current_dim = d_current

    def predict(self, instance: Instance) -> int:
        """Predict class label."""
        x_full = np.array(instance.x)
        d_current = len(x_full)
        
        # Use the most recent weight vector
        if self.current_dim >= d_current:
            # Decremental: use survival features
            w_pred = self.W[:d_current, -1]
            margin = np.dot(w_pred, x_full)
        else:
            # Incremental: pad with zeros
            w_padded = np.concatenate([
                self.W[:self.current_dim, -1],
                np.zeros(d_current - self.current_dim)
            ])
            margin = np.dot(w_padded, x_full)
        
        return 1 if margin > 0 else 0

    def predict_proba(self, instance: Instance) -> np.ndarray:
        """Predict class probabilities."""
        x_full = np.array(instance.x)
        d_current = len(x_full)
        
        if self.current_dim >= d_current:
            w_pred = self.W[:d_current, -1]
            margin = np.dot(w_pred, x_full)
        else:
            w_padded = np.concatenate([
                self.W[:self.current_dim, -1],
                np.zeros(d_current - self.current_dim)
            ])
            margin = np.dot(w_padded, x_full)
        
        # Use sigmoid for probability
        prob_class_1 = 1.0 / (1.0 + np.exp(-margin))
        prob_class_0 = 1.0 - prob_class_1
        
        return np.array([prob_class_0, prob_class_1])

    def _update_decremental(self, xt: np.ndarray, yt: float, d_current: int) -> np.ndarray:
        """Update for decremental case (Theorem 3.1)."""
        # Survival features weights
        w_s = self.W[:d_current, -1]
        
        # Calculate hinge loss
        loss = max(0, 1 - yt * np.dot(w_s, xt))
        
        # Closed-form PA update
        gamma = loss / (np.linalg.norm(xt)**2 + 1/(2*self.mu))
        w_new = w_s + gamma * yt * xt
        
        return w_new

    def _update_incremental(self, xt: np.ndarray, yt: float, d_current: int) -> np.ndarray:
        """Update for incremental case (Theorem 3.2)."""
        # Survival and new features
        x_s = xt[:self.current_dim]
        x_n = xt[self.current_dim:]
        
        # Pad weight vector with zeros for new features
        w_padded = np.concatenate([
            self.W[:self.current_dim, -1],
            np.zeros(d_current - self.current_dim)
        ])
        
        # Calculate hinge loss
        loss = max(0, 1 - yt * np.dot(w_padded, xt))
        
        # Closed-form PA update
        gamma = loss / (np.linalg.norm(xt)**2 + 1/(2*self.mu))
        w_s_new = self.W[:self.current_dim, -1] + gamma * yt * x_s
        w_n_new = gamma * yt * x_n
        w_new = np.concatenate([w_s_new, w_n_new])
        
        return w_new

    def _apply_l12_sparsity(self):
        """Apply ℓ1,2-norm regularization for sparsity (Theorem 3.3)."""
        for i in range(self.d_max):
            row = self.W[i, :]
            row_norm = np.linalg.norm(row)
            
            if row_norm <= self.lambda_param:
                # Set entire row to zero (aggressive sparsification)
                self.W[i, :] = 0
            else:
                # Soft-thresholding
                self.W[i, :] = (1 - self.lambda_param / row_norm) * row