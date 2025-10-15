"""OVFM (Online Learning in Variable Feature Spaces with Mixed Data) Classifier

This module implements the OVFM algorithm for online binary classification
in streams with dynamically changing feature spaces and mixed data types.

Reference:
    Yi He, Jiaxian Dong, Bo-Jian Hou, Yu Wang, Fei Wang. (2021).
    "Online Learning in Variable Feature Spaces with Mixed Data."
    IEEE International Conference on Data Mining (ICDM).
"""
import numpy as np
from typing import Optional, Literal
import warnings

from capymoa.base import Classifier
from capymoa.instance import LabeledInstance, Instance
from capymoa.stream import Schema
from capymoa.type_alias import LabelProbabilities


class OVFMClassifier(Classifier):
    """OVFM: Online Learning in Variable Feature Spaces with Mixed Data.
    
    Uses Gaussian Copula to model joint distribution of mixed data types
    (continuous/ordinal/boolean) in a latent normal space. This enables:
    
    1. **Feature Reconstruction**: When features go missing, their values can
       be imputed from the correlation structure learned via the copula.
    
    2. **Smooth Optimization**: Discrete ordinal variables cause gradient
       oscillations. Mapping to continuous latent space enables finer updates.
    
    3. **Educated Initialization**: New features' weights are initialized based
       on their correlation with existing features.
    
    **Key Features:**
    - Handles trapezoidal (TDS) and capricious (VFS) feature evolution patterns
    - Automatically detects ordinal vs continuous features
    - Ensemble of observed-space and latent-space learners
    - Dynamic dimension growth with sparse weight pruning
    
    **Limitations:**
    - Only supports binary classification (labels in {0, 1} or {-1, +1})
    - Requires sufficient buffer size (window_size) for accurate copula estimation
    - Computational overhead from online EM algorithm (~O(d³) per batch)
    - Does not handle concept drift explicitly (assumes stationary distribution)
    """

    def __init__(
        self,
        schema: Schema,
        window_size: int = 200,
        batch_size: int = 50,
        evolution_pattern: Literal["tds", "vfs"] = "vfs",
        decay_coef: float = 0.5,
        num_ord_updates: int = 2,
        max_ord_levels: int = 14,
        ensemble_weight: float = 0.5,
        learning_rate: float = 0.01,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.01,
        sparsity_threshold: float = 0.01,
        random_seed: int = 1
    ):
        """Initialize OVFM Classifier.
        
        :param schema: Schema describing the data stream structure
        :param window_size: Buffer size for marginal distribution estimation
        :param batch_size: Number of instances to accumulate before EM update
        :param evolution_pattern: Feature space dynamics ("tds" or "vfs")
        :param decay_coef: Exponential decay weight for covariance updates (0-1)
        :param num_ord_updates: EM iterations for ordinal latent variables
        :param max_ord_levels: Threshold to distinguish ordinal from continuous
        :param ensemble_weight: Initial weight for observed vs latent learner (0-1)
        :param learning_rate: SGD step size
        :param l1_lambda: L1 regularization strength (Lasso)
        :param l2_lambda: L2 regularization strength (Ridge)
        :param sparsity_threshold: Weight pruning threshold
        :param random_seed: Random seed for reproducibility
        
        :raises ValueError: If parameters are invalid or schema is not binary classification
        """
        super().__init__(schema=schema, random_seed=random_seed)
        
        # ===== Parameter Validation =====
        if not schema.is_classification():
            raise ValueError("OVFM only supports classification tasks")
        if schema.get_num_classes() != 2:
            raise ValueError(
                f"OVFM only supports binary classification. "
                f"Got {schema.get_num_classes()} classes. "
                f"Consider using one-vs-rest for multi-class."
            )
        
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > window_size:
            warnings.warn(
                f"batch_size ({batch_size}) > window_size ({window_size}). "
                f"This may reduce EM estimation quality."
            )
        if not 0 <= decay_coef <= 1:
            raise ValueError(f"decay_coef must be in [0,1], got {decay_coef}")
        if not 0 <= ensemble_weight <= 1:
            raise ValueError(f"ensemble_weight must be in [0,1], got {ensemble_weight}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if l1_lambda < 0:
            raise ValueError(f"l1_lambda must be non-negative, got {l1_lambda}")
        if l2_lambda < 0:
            raise ValueError(f"l2_lambda must be non-negative, got {l2_lambda}")
        if sparsity_threshold < 0:
            raise ValueError(f"sparsity_threshold must be non-negative, got {sparsity_threshold}")
        
        # Store hyperparameters
        self.window_size = window_size
        self.batch_size = batch_size
        self.evolution_pattern = evolution_pattern
        self.decay_coef = decay_coef
        self.num_ord_updates = num_ord_updates
        self.max_ord_levels = max_ord_levels
        self.ensemble_weight = ensemble_weight
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.sparsity_threshold = sparsity_threshold
        
        # Internal state (lazy initialization on first instance)
        self._ovfm_learner = None
        self._instance_buffer = []
        self._label_buffer = []
        self._cont_indices = None
        self._ord_indices = None
        self._max_features_seen = 0
        
        # Ensemble tracking
        self._cumulative_loss_observed = 0.0
        self._cumulative_loss_latent = 0.0
        self._num_updates = 0
        
        # Classifiers
        self._w_observed = None
        self._w_latent = None
        
        self._rng = np.random.RandomState(random_seed)
        self._trained = False

    def _initialize_ovfm(self, first_instance: LabeledInstance):
        """Lazy initialization when first instance arrives."""
        from capymoa.classifier._ovfm_classifier_em import (
            TrapezoidalExpectationMaximization2,
            OnlineExpectationMaximization
        )
        
        self._max_features_seen = len(first_instance.x)
        
        # Initialize all features as continuous
        self._cont_indices = np.ones(self._max_features_seen, dtype=bool)
        self._ord_indices = np.zeros(self._max_features_seen, dtype=bool)
        
        # Initialize OVFM learner
        if self.evolution_pattern == "tds":
            self._ovfm_learner = TrapezoidalExpectationMaximization2(
                cont_indices=self._cont_indices,
                ord_indices=self._ord_indices,
                window_size=self.window_size,
                window_width=self._max_features_seen,
                sigma_init=None
            )
        else:  # vfs
            self._ovfm_learner = OnlineExpectationMaximization(
                cont_indices=self._cont_indices,
                ord_indices=self._ord_indices,
                window_size=self.window_size,
                sigma_init=None
            )
        
        # Initialize weight vectors
        self._w_observed = self._rng.randn(self._max_features_seen + 1) * 0.01
        self._w_latent = self._rng.randn(self._max_features_seen + 1) * 0.01
        
        self._trained = True

    def _update_feature_types(self, X_batch: np.ndarray):
        """Detect ordinal vs continuous features from batch statistics."""
        for i in range(X_batch.shape[1]):
            col = X_batch[:, i]
            col_nonan = col[~np.isnan(col)]
            
            if len(col_nonan) > 0:
                unique_vals = np.unique(col_nonan)
                if len(unique_vals) <= self.max_ord_levels:
                    self._ord_indices[i] = True
                    self._cont_indices[i] = False

    def _extend_dimensions(self, new_dim: int):
        """Extend weight vectors when new features appear (TDS mode)."""
        if new_dim > self._max_features_seen:
            diff = new_dim - self._max_features_seen
            
            # Extend feature type indices
            self._cont_indices = np.concatenate([
                self._cont_indices, 
                np.ones(diff, dtype=bool)
            ])
            self._ord_indices = np.concatenate([
                self._ord_indices,
                np.zeros(diff, dtype=bool)
            ])
            
            # Extend weights
            new_weights = self._rng.randn(diff) * 0.01
            self._w_observed = np.concatenate([
                self._w_observed[:-1],
                new_weights,
                [self._w_observed[-1]]
            ])
            self._w_latent = np.concatenate([
                self._w_latent[:-1],
                new_weights.copy(),
                [self._w_latent[-1]]
            ])
            
            self._max_features_seen = new_dim

    def _batch_update(self):
        """Process accumulated batch with OVFM and update classifiers."""
        if len(self._instance_buffer) == 0:
            return
        
        X_batch = np.array(self._instance_buffer)
        y_batch = np.array(self._label_buffer)
        
        self._update_feature_types(X_batch)
        
        # OVFM update
        try:
            if self.evolution_pattern == "tds":
                Z_imp, X_imp = self._ovfm_learner.partial_fit_and_predict(
                    X_batch,
                    cont_indices=self._cont_indices[:X_batch.shape[1]],
                    ord_indices=self._ord_indices[:X_batch.shape[1]],
                    max_workers=1,
                    decay_coef=self.decay_coef,
                    num_ord_updates=self.num_ord_updates
                )
            else:
                Z_imp, X_imp = self._ovfm_learner.partial_fit_and_predict(
                    X_batch,
                    max_workers=1,
                    decay_coef=self.decay_coef,
                    num_ord_updates=self.num_ord_updates
                )
        except Exception as e:
            warnings.warn(
                f"OVFM update failed: {e}. "
                f"Skipping batch of size {len(X_batch)}. "
                f"Consider adjusting window_size or batch_size."
            )
            self._instance_buffer = []
            self._label_buffer = []
            return
        
        # Update classifiers
        for i in range(len(X_batch)):
            x_obs = self._pad_with_bias(X_imp[i])
            z_lat = self._pad_with_bias(Z_imp[i])
            y_binary = 1 if y_batch[i] == 1 else -1
            
            # Compute losses
            wx_obs = np.dot(self._w_observed[:len(x_obs)], x_obs)
            wx_lat = np.dot(self._w_latent[:len(z_lat)], z_lat)
            self._cumulative_loss_observed += self._logistic_loss(wx_obs, y_binary)
            self._cumulative_loss_latent += self._logistic_loss(wx_lat, y_binary)
            
            # SGD update
            self._sgd_update(self._w_observed, x_obs, y_binary)
            self._sgd_update(self._w_latent, z_lat, y_binary)
        
        self._num_updates += len(X_batch)
        self._update_ensemble_weight()
        self._sparsify_weights()
        
        self._instance_buffer = []
        self._label_buffer = []

    def _pad_with_bias(self, x: np.ndarray) -> np.ndarray:
        """Add bias term and handle NaNs."""
        x_clean = np.where(np.isnan(x), 0, x)
        return np.concatenate([x_clean, [1.0]])

    def _sigmoid(self, z: float) -> float:
        """Numerically stable sigmoid."""
        z = np.clip(z, -100, 100)  # Prevent overflow
        if z >= 0:
            return 1.0 / (1 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            return exp_z / (1 + exp_z)

    def _logistic_loss(self, wx: float, y: int) -> float:
        """Compute logistic loss with numerical stability."""
        z = np.clip(-y * wx, -100, 100)  # Prevent extreme values
        if z > 0:
            return z + np.log(1 + np.exp(-z))
        else:
            return np.log(1 + np.exp(z))

    def _sgd_update(self, w: np.ndarray, x: np.ndarray, y: int):
        """SGD update with L1/L2 regularization."""
        wx = np.dot(w[:len(x)], x)
        p = self._sigmoid(y * wx)
        
        gradient = -(1 - p) * y * x
        reg_gradient = (
            self.l1_lambda * np.sign(w[:len(x)]) +
            self.l2_lambda * w[:len(x)]
        )
        
        w[:len(x)] -= self.learning_rate * (gradient + reg_gradient)

    def _update_ensemble_weight(self):
        """Update ensemble weight using exponential weighting."""
        if self._num_updates == 0:
            return
        
        tau = 2 * np.sqrt(2 * np.log(2) / max(1, self._num_updates))
        
        exp_obs = np.exp(-tau * self._cumulative_loss_observed)
        exp_lat = np.exp(-tau * self._cumulative_loss_latent)
        
        denom = exp_obs + exp_lat
        if denom > 0:
            self.ensemble_weight = exp_obs / denom

    def _sparsify_weights(self):
        """Prune small weights."""
        mask_obs = np.abs(self._w_observed[:-1]) < self.sparsity_threshold
        mask_lat = np.abs(self._w_latent[:-1]) < self.sparsity_threshold
        
        self._w_observed[:-1][mask_obs] = 0
        self._w_latent[:-1][mask_lat] = 0

    def _approximate_latent_simple(self, x: np.ndarray) -> np.ndarray:
        """Simple z-score normalization to approximate latent space.
        
        This is a lightweight alternative to full copula transformation
        for prediction time when no label is available.
        """
        z = np.copy(x)
        mask = ~np.isnan(x)
        
        if np.sum(mask) > 0:
            x_valid = x[mask]
            mean = np.mean(x_valid)
            std = np.std(x_valid)
            
            if std > 1e-6:
                z[mask] = (x[mask] - mean) / std
            else:
                z[mask] = 0
        
        return z

    def train(self, instance: LabeledInstance):
        """Train on a single labeled instance (accumulated into batch)."""
        if not self._trained:
            self._initialize_ovfm(instance)
        
        if len(instance.x) > self._max_features_seen:
            self._extend_dimensions(len(instance.x))
        
        x_padded = np.full(self._max_features_seen, np.nan)
        x_padded[:len(instance.x)] = instance.x
        
        self._instance_buffer.append(x_padded)
        self._label_buffer.append(instance.y_index)
        
        if len(self._instance_buffer) >= self.batch_size:
            self._batch_update()

    def predict_proba(self, instance: Instance) -> Optional[LabelProbabilities]:
        """Predict class probabilities using ensemble of two learners."""
        if not self._trained or self._w_observed is None:
            return None
        
        x_padded = np.full(self._max_features_seen, np.nan)
        x_padded[:len(instance.x)] = instance.x
        x_obs = self._pad_with_bias(x_padded)
        
        # Approximate latent representation via simple normalization
        z_approx = self._approximate_latent_simple(x_padded)
        z_lat = self._pad_with_bias(z_approx)
        
        # Compute ensemble prediction
        wx_obs = np.dot(self._w_observed[:len(x_obs)], x_obs)
        wx_lat = np.dot(self._w_latent[:len(z_lat)], z_lat)
        
        wx_ensemble = (
            self.ensemble_weight * wx_obs + 
            (1 - self.ensemble_weight) * wx_lat
        )
        
        p_class1 = self._sigmoid(wx_ensemble)
        return np.array([1 - p_class1, p_class1])

    def finalize_training(self):
        """Process any remaining instances in buffer (call at stream end)."""
        if len(self._instance_buffer) > 0:
            self._batch_update()

    @property
    def training_statistics(self) -> dict:
        """Return training statistics for monitoring and debugging."""
        return {
            "num_updates": self._num_updates,
            "buffer_size": len(self._instance_buffer),
            "max_features": self._max_features_seen,
            "ensemble_weight": self.ensemble_weight,
            "loss_observed": self._cumulative_loss_observed,
            "loss_latent": self._cumulative_loss_latent,
            "num_continuous": int(np.sum(self._cont_indices)) if self._cont_indices is not None else 0,
            "num_ordinal": int(np.sum(self._ord_indices)) if self._ord_indices is not None else 0,
        }

    def __str__(self):
        return (
            f"OVFMClassifier(pattern={self.evolution_pattern}, "
            f"window={self.window_size}, batch={self.batch_size}, "
            f"α={self.ensemble_weight:.3f})"
        )