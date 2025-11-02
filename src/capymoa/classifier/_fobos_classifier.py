"""capymoa/classifier/_fobos_classifier.py"""
from __future__ import annotations
import numpy as np
from typing import Literal

from capymoa.base import Classifier
from capymoa.stream import Schema
from capymoa.instance import Instance


class FOBOSClassifier(Classifier):
    """Forward-Backward Splitting (FOBOS) for Online Binary Classification.
    
    Implements the FOBOS algorithm with multiple regularization options for
    sparse online learning. Supports logistic loss and hinge loss.
    
    Reference:
        Duchi, J., & Singer, Y. (2009). Efficient Online and Batch Learning Using
        Forward Backward Splitting. Journal of Machine Learning Research, 10, 2899-2934.
    
    Parameters:
        schema: Stream schema
        alpha: Base learning rate (initial η in paper)
        lambda_: Regularization strength parameter
        regularization: Type of regularization
            - "l1": Section 5.1, Eq. 19 (soft thresholding, promotes sparsity)
            - "l2": Section 5.3 (spherical shrinkage)
            - "l2_squared": Section 5.2, Eq. 20 (ridge regression style)
            - "elastic_net": L1 + L2² combination
        elastic_net_ratio: Ratio of L1 to L2² for elastic net (0=pure L2², 1=pure L1)
        step_schedule: Learning rate schedule
            - "sqrt": η_t = α/√t (Theorem 6, general convex)
            - "linear": η_t = α/t (Theorem 8, strongly convex)
            - "constant": η_t = α/√T (batch mode)
        loss: Loss function
            - "logistic": Logistic loss for probabilistic predictions
            - "hinge": Hinge loss for SVM-style classification
        random_seed: Random seed for weight initialization
    
    Example:
        >>> from capymoa.datasets import Electricity
        >>> from capymoa.classifier import FOBOSClassifier
        >>> stream = Electricity()
        >>> # L1 regularization (sparse)
        >>> learner = FOBOSClassifier(schema=stream.get_schema(), 
        ...                           regularization="l1", lambda_=0.01)
        >>> # Elastic Net (sparse + stable)
        >>> learner = FOBOSClassifier(schema=stream.get_schema(),
        ...                           regularization="elastic_net",
        ...                           elastic_net_ratio=0.7)
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 1.0,
        lambda_: float = 0.01,
        regularization: Literal["l1", "l2", "l2_squared", "elastic_net"] = "l1",
        elastic_net_ratio: float = 0.5,
        step_schedule: Literal["sqrt", "linear", "constant"] = "sqrt",
        loss: Literal["logistic", "hinge"] = "logistic",
        random_seed: int = 1,
    ):
        """Initialize FOBOS Binary Classifier."""
        super().__init__(schema=schema, random_seed=random_seed)
        
        if schema.get_num_classes() != 2:
            raise ValueError("FOBOSClassifier only supports binary classification. "
                           "Use FOBOSMulticlassClassifier for multi-class problems.")
        
        # Validate parameters
        if not 0 <= elastic_net_ratio <= 1:
            raise ValueError("elastic_net_ratio must be in [0, 1]")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization
        self.elastic_net_ratio = elastic_net_ratio
        self.step_schedule = step_schedule
        self.loss = loss
        
        # Initialize weights
        n_features = schema.get_num_attributes()
        np.random.seed(random_seed)
        self.w = np.random.randn(n_features) * 0.01
        
        # Step counter (starts at 1 for η_t = α/√t)
        self.step = 1
        
        # For constant step schedule, we need to know T
        self._total_steps = None

    def __str__(self):
        return (f"FOBOSClassifier(alpha={self.alpha}, lambda={self.lambda_}, "
                f"regularization={self.regularization}, loss={self.loss})")

    def train(self, instance: Instance):
        """Train on a single instance using forward-backward splitting.
        
        Implements Algorithm from Section 2:
        1. Forward step: w_{t+1/2} = w_t - η_t * ∇f_t(w_t)
        2. Backward step: w_{t+1} = argmin_w { 1/2||w - w_{t+1/2}||² + η_{t+1/2} r(w) }
        """
        x = np.array(instance.x)
        y = float(instance.y_index)
        
        # Convert label to {-1, +1} for hinge loss
        if self.loss == "hinge":
            y_signed = 2.0 * y - 1.0  # 0 -> -1, 1 -> +1
        
        # Compute gradient based on loss function
        if self.loss == "logistic":
            gradient = self._compute_logistic_gradient(x, y)
        elif self.loss == "hinge":
            gradient = self._compute_hinge_gradient(x, y_signed)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
        
        # Learning rate schedule
        eta_t = self._compute_step_size()
        
        # Forward step: gradient descent
        w_half = self.w - eta_t * gradient
        
        # Backward step: apply regularization
        self.w = self._backward_step(w_half, eta_t)
        
        self.step += 1

    def predict(self, instance: Instance) -> int:
        """Predict class label (0 or 1)."""
        x = np.array(instance.x)
        
        if self.loss == "logistic":
            prob = self._sigmoid(np.dot(self.w, x))
            return 1 if prob > 0.5 else 0
        elif self.loss == "hinge":
            score = np.dot(self.w, x)
            return 1 if score > 0 else 0
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def predict_proba(self, instance: Instance) -> np.ndarray:
        """Predict class probabilities.
        
        Note: For hinge loss, returns deterministic 0/1 probabilities.
        """
        x = np.array(instance.x)
        
        if self.loss == "logistic":
            prob_class1 = self._sigmoid(np.dot(self.w, x))
            return np.array([1 - prob_class1, prob_class1])
        elif self.loss == "hinge":
            # Hinge loss doesn't provide probabilistic predictions
            # Return deterministic prediction
            pred = self.predict(instance)
            return np.array([1 - pred, pred], dtype=float)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def _compute_logistic_gradient(self, x: np.ndarray, y: float) -> np.ndarray:
        """Compute gradient of logistic loss."""
        y_hat = self._sigmoid(np.dot(self.w, x))
        return (y_hat - y) * x

    def _compute_hinge_gradient(self, x: np.ndarray, y_signed: float) -> np.ndarray:
        """Compute subgradient of hinge loss.
        
        Hinge loss: max(0, 1 - y * (w·x))
        Subgradient: -y*x if y*(w·x) < 1, else 0
        """
        margin = y_signed * np.dot(self.w, x)
        if margin < 1.0:
            return -y_signed * x
        else:
            return np.zeros_like(x)

    def _compute_step_size(self) -> float:
        """Compute learning rate based on schedule.
        
        Returns:
            Learning rate η_t for current step
        """
        if self.step_schedule == "sqrt":
            # η_t = α/√t (Theorem 6, Corollary 7)
            return self.alpha / np.sqrt(self.step)
        elif self.step_schedule == "linear":
            # η_t = α/t (Theorem 8, for strongly convex)
            return self.alpha / self.step
        elif self.step_schedule == "constant":
            # η_t = α/√T (batch mode, Corollary 3)
            # For online learning, use sqrt schedule as fallback
            if self._total_steps is None:
                return self.alpha / np.sqrt(self.step)
            return self.alpha / np.sqrt(self._total_steps)
        else:
            raise ValueError(f"Unknown step_schedule: {self.step_schedule}")

    def _backward_step(self, w_half: np.ndarray, eta_t: float) -> np.ndarray:
        """Apply regularization in backward step.
        
        Args:
            w_half: Result of forward step (w - η_t * gradient)
            eta_t: Current learning rate
            
        Returns:
            Regularized weight vector w_{t+1}
        """
        if self.regularization == "l1":
            return self._l1_backward(w_half, eta_t)
        elif self.regularization == "l2":
            return self._l2_backward(w_half, eta_t)
        elif self.regularization == "l2_squared":
            return self._l2_squared_backward(w_half, eta_t)
        elif self.regularization == "elastic_net":
            return self._elastic_net_backward(w_half, eta_t)
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")

    def _l1_backward(self, w_half: np.ndarray, eta_t: float) -> np.ndarray:
        """L1 regularization: soft thresholding (Section 5.1, Eq. 19).
        
        Solution: w_j = sign(w_{1/2,j}) * [|w_{1/2,j}| - λη_t]_+
        """
        threshold = eta_t * self.lambda_
        return np.sign(w_half) * np.maximum(0.0, np.abs(w_half) - threshold)

    def _l2_backward(self, w_half: np.ndarray, eta_t: float) -> np.ndarray:
        """L2 regularization: spherical shrinkage (Section 5.3).
        
        Solution: w = [1 - λη_t / ||w_{1/2}||]_+ * w_{1/2}
        """
        norm = np.linalg.norm(w_half)
        threshold = eta_t * self.lambda_
        
        if norm > threshold:
            return (1.0 - threshold / norm) * w_half
        else:
            return np.zeros_like(w_half)

    def _l2_squared_backward(self, w_half: np.ndarray, eta_t: float) -> np.ndarray:
        """L2² regularization: simple scaling (Section 5.2, Eq. 20).
        
        Solution: w = w_{1/2} / (1 + λη_t)
        """
        return w_half / (1.0 + eta_t * self.lambda_)

    def _elastic_net_backward(self, w_half: np.ndarray, eta_t: float) -> np.ndarray:
        """Elastic Net: L1 + L2² regularization.
        
        Solution: Apply L2² shrinkage then L1 soft thresholding
        """
        # Split λ into L1 and L2² parts
        lambda_l1 = self.elastic_net_ratio * self.lambda_
        lambda_l2_sq = (1.0 - self.elastic_net_ratio) * self.lambda_
        
        # First apply L2² shrinkage
        w_l2 = w_half / (1.0 + eta_t * lambda_l2_sq)
        
        # Then apply L1 soft thresholding
        threshold = eta_t * lambda_l1
        return np.sign(w_l2) * np.maximum(0.0, np.abs(w_l2) - threshold)

    @staticmethod
    def _sigmoid(z: float | np.ndarray) -> float | np.ndarray:
        """Sigmoid activation with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def get_sparsity(self) -> float:
        """Get proportion of zero weights (sparsity level).
        
        Returns:
            Fraction of weights that are effectively zero (< 1e-8)
        """
        return np.sum(np.abs(self.w) < 1e-8) / len(self.w)

    def get_weights(self) -> np.ndarray:
        """Get copy of current weight vector.
        
        Returns:
            Copy of weight vector to prevent external modification
        """
        return self.w.copy()
    
    def get_model_description(self) -> str:
        """Get detailed model description."""
        sparsity = self.get_sparsity()
        n_zeros = int(sparsity * len(self.w))
        
        desc = f"FOBOS Binary Classifier\n"
        desc += f"  Regularization: {self.regularization}"
        if self.regularization == "elastic_net":
            desc += f" (L1 ratio={self.elastic_net_ratio:.2f})"
        desc += f"\n"
        desc += f"  Loss function: {self.loss}\n"
        desc += f"  Learning rate: α={self.alpha} (schedule={self.step_schedule})\n"
        desc += f"  Regularization: λ={self.lambda_}\n"
        desc += f"  Steps trained: {self.step - 1}\n"
        desc += f"  Sparsity: {sparsity:.2%} ({n_zeros}/{len(self.w)} zeros)\n"
        desc += f"  Weight norm: ||w||₂={np.linalg.norm(self.w):.4f}"
        
        return desc