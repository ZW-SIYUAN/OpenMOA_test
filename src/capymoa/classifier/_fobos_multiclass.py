"""capymoa/classifier/_fobos_multiclass.py"""
from __future__ import annotations
import numpy as np
from typing import Literal

from capymoa.base import Classifier
from capymoa.stream import Schema
from capymoa.instance import Instance


class FOBOSMulticlassClassifier(Classifier):
    """Forward-Backward Splitting (FOBOS) for Online Multi-class Classification.
    
    Implements FOBOS with mixed-norm regularization for group sparsity in
    multi-class learning. Uses softmax loss and supports L1/L2 and L1/L∞
    regularization.
    
    Reference:
        Duchi, J., & Singer, Y. (2009). Efficient Online and Batch Learning Using
        Forward Backward Splitting. Journal of Machine Learning Research, 10, 2899-2934.
        See Section 5.6 (Mixed-norm regularization) and Section 7.6 (Experiments).
    
    Parameters:
        schema: Stream schema
        alpha: Base learning rate
        lambda_: Regularization strength parameter
        regularization: Type of mixed-norm regularization
            - "l1_l2": Section 5.6, Eq. 27 (promotes row sparsity)
            - "l1_linf": Section 5.6 (L1/L∞ mixed norm)
        step_schedule: Learning rate schedule
            - "sqrt": η_t = α/√t (default for online learning)
            - "linear": η_t = α/t (for strongly convex)
        random_seed: Random seed for weight initialization
    
    Note:
        Weight matrix W has shape (n_features, n_classes), where each column
        corresponds to one class. Mixed-norm regularization encourages entire
        rows to become zero, implementing feature selection across all classes.
    
    Example:
        >>> from capymoa.datasets import CoverType
        >>> from capymoa.classifier import FOBOSMulticlassClassifier
        >>> stream = CoverType()
        >>> # Convert to binary for testing (or use with multi-class)
        >>> learner = FOBOSMulticlassClassifier(schema=stream.get_schema(),
        ...                                      regularization="l1_l2")
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 1.0,
        lambda_: float = 0.01,
        regularization: Literal["l1_l2", "l1_linf"] = "l1_l2",
        step_schedule: Literal["sqrt", "linear"] = "sqrt",
        random_seed: int = 1,
    ):
        """Initialize FOBOS Multi-class Classifier."""
        super().__init__(schema=schema, random_seed=random_seed)
        
        self.n_classes = schema.get_num_classes()
        if self.n_classes < 2:
            raise ValueError("Number of classes must be at least 2")
        
        # Validate parameters
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization
        self.step_schedule = step_schedule
        
        # Initialize weight matrix: W ∈ R^{d×k}
        # Each column is the weight vector for one class
        n_features = schema.get_num_attributes()
        np.random.seed(random_seed)
        self.W = np.random.randn(n_features, self.n_classes) * 0.01
        
        # Step counter
        self.step = 1

    def __str__(self):
        return (f"FOBOSMulticlassClassifier(alpha={self.alpha}, lambda={self.lambda_}, "
                f"regularization={self.regularization}, n_classes={self.n_classes})")

    def train(self, instance: Instance):
        """Train on a single instance using forward-backward splitting.
        
        Implements multi-class FOBOS with softmax loss and mixed-norm
        regularization (Section 5.6).
        """
        x = np.array(instance.x)
        y = int(instance.y_index)
        
        # Compute softmax probabilities
        scores = self.W.T @ x  # shape: (n_classes,)
        probs = self._softmax(scores)
        
        # Compute gradient for each class
        # ∇L = (p_j - 1_{j=y}) * x for class j
        gradient = np.outer(x, probs)  # shape: (n_features, n_classes)
        gradient[:, y] -= x.reshape(-1, 1)[:, 0]
        
        # Learning rate
        eta_t = self._compute_step_size()
        
        # Forward step
        W_half = self.W - eta_t * gradient
        
        # Backward step: apply mixed-norm regularization
        self.W = self._backward_step(W_half, eta_t)
        
        self.step += 1

    def predict(self, instance: Instance) -> int:
        """Predict class label (argmax over class scores)."""
        x = np.array(instance.x)
        scores = self.W.T @ x
        return int(np.argmax(scores))

    def predict_proba(self, instance: Instance) -> np.ndarray:
        """Predict class probabilities using softmax."""
        x = np.array(instance.x)
        scores = self.W.T @ x
        return self._softmax(scores)

    def _compute_step_size(self) -> float:
        """Compute learning rate based on schedule."""
        if self.step_schedule == "sqrt":
            return self.alpha / np.sqrt(self.step)
        elif self.step_schedule == "linear":
            return self.alpha / self.step
        else:
            raise ValueError(f"Unknown step_schedule: {self.step_schedule}")

    def _backward_step(self, W_half: np.ndarray, eta_t: float) -> np.ndarray:
        """Apply mixed-norm regularization to weight matrix.
        
        Args:
            W_half: Result of forward step (W - η_t * gradient)
            eta_t: Current learning rate
            
        Returns:
            Regularized weight matrix W_{t+1}
        """
        if self.regularization == "l1_l2":
            return self._l1_l2_backward(W_half, eta_t)
        elif self.regularization == "l1_linf":
            return self._l1_linf_backward(W_half, eta_t)
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")

    def _l1_l2_backward(self, W_half: np.ndarray, eta_t: float) -> np.ndarray:
        """L1/L2 mixed-norm regularization (Section 5.6, Eq. 27).
        
        For each row i:
            - Compute row norm: ||w̄_i||_2
            - If ||w̄_i||_2 > λη_t: shrink by (1 - λη_t/||w̄_i||_2)
            - Otherwise: set entire row to zero
        
        This promotes row sparsity (entire features become zero across all classes).
        """
        W_new = np.zeros_like(W_half)
        threshold = eta_t * self.lambda_
        
        for i in range(W_half.shape[0]):
            row = W_half[i, :]
            row_norm = np.linalg.norm(row)
            
            if row_norm > threshold:
                # Shrink the row
                W_new[i, :] = (1.0 - threshold / row_norm) * row
            else:
                # Zero out the entire row
                W_new[i, :] = 0.0
        
        return W_new

    def _l1_linf_backward(self, W_half: np.ndarray, eta_t: float) -> np.ndarray:
        """L1/L∞ mixed-norm regularization (Section 5.6).
        
        For each row i:
            - Compute row L∞ norm: ||w̄_i||_∞ = max_j |w_{i,j}|
            - Apply projection to L1 ball (Section 5.4)
        
        This is more complex and uses the projection algorithm from
        Section 5.4 (Equation 25).
        """
        W_new = np.zeros_like(W_half)
        threshold = eta_t * self.lambda_
        
        for i in range(W_half.shape[0]):
            row = W_half[i, :]
            
            # Compute L1 norm of row
            row_l1_norm = np.sum(np.abs(row))
            
            if row_l1_norm <= threshold:
                # Entire row becomes zero
                W_new[i, :] = 0.0
            else:
                # Apply element-wise capping (simplified L∞ projection)
                # This is a simplified version; full version needs
                # the algorithm from Duchi et al. (2008)
                W_new[i, :] = self._project_linf_row(row, threshold)
        
        return W_new

    def _project_linf_row(self, row: np.ndarray, threshold: float) -> np.ndarray:
        """Project a row to L∞ constraint (simplified version).
        
        This is a simplified implementation. Full implementation would use
        the algorithm from Section 5.4 and Duchi et al. (2008).
        """
        # Find the threshold θ such that sum of capped values = threshold
        # For simplicity, we use a heuristic: cap all values at θ
        
        abs_row = np.abs(row)
        if np.sum(abs_row) <= threshold:
            return row
        
        # Binary search for θ
        sorted_abs = np.sort(abs_row)[::-1]  # descending
        
        cumsum = 0.0
        theta = 0.0
        for k, val in enumerate(sorted_abs):
            cumsum += val
            if cumsum > threshold:
                # Found the threshold
                excess = cumsum - threshold
                theta = sorted_abs[k-1] if k > 0 else val
                break
        
        # Cap all values at theta
        return np.sign(row) * np.minimum(abs_row, theta)

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities with numerical stability.
        
        Args:
            scores: Class scores, shape (n_classes,)
            
        Returns:
            Probabilities, shape (n_classes,), sum to 1
        """
        # Subtract max for numerical stability
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / np.sum(exp_scores)

    def get_row_sparsity(self) -> float:
        """Get proportion of zero rows (row sparsity level).
        
        Returns:
            Fraction of rows that are effectively zero
        """
        row_norms = np.linalg.norm(self.W, axis=1)
        return np.sum(row_norms < 1e-8) / self.W.shape[0]

    def get_element_sparsity(self) -> float:
        """Get proportion of zero elements (element-wise sparsity).
        
        Returns:
            Fraction of individual weights that are effectively zero
        """
        return np.sum(np.abs(self.W) < 1e-8) / self.W.size

    def get_weights(self) -> np.ndarray:
        """Get copy of current weight matrix.
        
        Returns:
            Copy of weight matrix W (shape: n_features × n_classes)
        """
        return self.W.copy()
    
    def get_model_description(self) -> str:
        """Get detailed model description."""
        row_sparsity = self.get_row_sparsity()
        elem_sparsity = self.get_element_sparsity()
        n_zero_rows = int(row_sparsity * self.W.shape[0])
        
        desc = f"FOBOS Multi-class Classifier\n"
        desc += f"  Classes: {self.n_classes}\n"
        desc += f"  Features: {self.W.shape[0]}\n"
        desc += f"  Regularization: {self.regularization}\n"
        desc += f"  Learning rate: α={self.alpha} (schedule={self.step_schedule})\n"
        desc += f"  Regularization: λ={self.lambda_}\n"
        desc += f"  Steps trained: {self.step - 1}\n"
        desc += f"  Row sparsity: {row_sparsity:.2%} ({n_zero_rows}/{self.W.shape[0]} zero rows)\n"
        desc += f"  Element sparsity: {elem_sparsity:.2%}\n"
        desc += f"  Weight Frobenius norm: ||W||_F={np.linalg.norm(self.W, 'fro'):.4f}"
        
        return desc