"""capymoa/classifier/_fobos_classifier.py"""
from __future__ import annotations
import numpy as np

from capymoa.base import Classifier
from capymoa.stream import Schema
from capymoa.instance import Instance


class FOBOSClassifier(Classifier):
    """Forward-Backward Splitting (FOBOS) for Online Learning.
    
    Reference:
    Duchi, J., & Singer, Y. (2009). Efficient Online and Batch Learning Using
    Forward Backward Splitting. Journal of Machine Learning Research.
    
    Example:
    >>> from capymoa.datasets import Electricity
    >>> from capymoa.classifier import FOBOSClassifier
    >>> stream = Electricity()
    >>> learner = FOBOSClassifier(schema=stream.get_schema())
    >>> # Use prequential_evaluation for testing
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 1.0,
        lambda_: float = 0.01,
        random_seed: int = 1,
    ):
        """Initialize FOBOS Classifier.
        
        :param schema: Stream schema
        :param alpha: Base learning rate
        :param lambda_: L1 regularization parameter
        :param random_seed: Random seed
        """
        super().__init__(schema=schema, random_seed=random_seed)
        
        if schema.get_num_classes() != 2:
            raise ValueError("FOBOS only supports binary classification")
        
        self.alpha = alpha
        self.lambda_ = lambda_
        
        # Initialize weights to small random values (not zeros)
        n_features = schema.get_num_attributes()
        np.random.seed(random_seed)
        self.w = np.random.randn(n_features) * 0.01
        
        # Step counter (start from 1 to avoid division issues)
        self.step = 1

    def __str__(self):
        return f"FOBOSClassifier(alpha={self.alpha}, lambda={self.lambda_})"

    def train(self, instance: Instance):
        """Train on a single instance using forward-backward splitting."""
        x = np.array(instance.x)
        y = float(instance.y_index)
        
        # Forward step: compute prediction and gradient
        y_hat = self._sigmoid(self.w.dot(x))
        gradient = (y_hat - y) * x
        
        # Damped learning rates (following original implementation)
        eta_t = self.alpha / np.sqrt(self.step)
        eta_t_half = self.alpha / np.sqrt(self.step + 1)
        
        # Forward-Backward splitting
        for i in range(len(self.w)):
            # Forward step: gradient descent
            w_half = self.w[i] - eta_t * gradient[i]
            
            # Backward step: soft thresholding
            threshold = eta_t_half * self.lambda_
            self.w[i] = np.sign(w_half) * max(0.0, np.abs(w_half) - threshold)
        
        self.step += 1

    def predict(self, instance: Instance) -> int:
        """Predict class label (0 or 1)."""
        x = np.array(instance.x)
        prob = self._sigmoid(self.w.dot(x))
        return 1 if prob > 0.5 else 0

    def predict_proba(self, instance: Instance) -> np.ndarray:
        """Predict class probabilities."""
        x = np.array(instance.x)
        prob_class1 = self._sigmoid(self.w.dot(x))
        return np.array([1 - prob_class1, prob_class1])

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Sigmoid activation function with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def get_sparsity(self) -> float:
        """Get the proportion of zero weights."""
        return np.sum(np.abs(self.w) < 1e-8) / len(self.w)

    def get_weights(self) -> np.ndarray:
        """Get current weight vector."""
        return self.w.copy()