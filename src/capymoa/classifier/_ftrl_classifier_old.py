"""capymoa/classifier/_ftrl_classifier.py"""
from __future__ import annotations
import numpy as np

from capymoa.base import Classifier
from capymoa.stream import Schema
from capymoa.instance import Instance


class FTRLClassifier(Classifier):
    """Follow-The-Regularized-Leader (FTRL) for Online Learning.
    
    Reference:
    McMahan, H. B., et al. (2013). Ad Click Prediction: A View from the Trenches.
    
    Example:
    >>> from capymoa.datasets import Electricity
    >>> from capymoa.classifier import FTRLClassifier
    >>> stream = Electricity()
    >>> learner = FTRLClassifier(schema=stream.get_schema())
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
        
        # Initialize FTRL state
        n_features = schema.get_num_attributes()
        np.random.seed(random_seed)
        
        self.z = np.zeros(n_features)
        self.n = np.zeros(n_features)  # Sum of squared gradients
        self.w = np.zeros(n_features)

    def __str__(self):
        return f"FTRLClassifier(alpha={self.alpha}, beta={self.beta}, l1={self.l1}, l2={self.l2})"

    def train(self, instance: Instance):
        """Train using FTRL-Proximal update."""
        x = np.array(instance.x)
        y = float(instance.y_index)
        
        # Update weights
        self._update_weights()
        
        # Compute prediction and gradient
        y_hat = self._sigmoid(np.dot(self.w, x))
        gradient = (y_hat - y) * x
        
        # FTRL update
        for i in range(len(self.w)):
            sigma = (np.sqrt(self.n[i] + gradient[i]**2) - np.sqrt(self.n[i])) / self.alpha
            self.z[i] += gradient[i] - sigma * self.w[i]
            self.n[i] += gradient[i]**2

    def _update_weights(self):
        """Update weights using FTRL formula."""
        for i in range(len(self.w)):
            if np.abs(self.z[i]) <= self.l1:
                self.w[i] = 0.0
            else:
                numerator = -(self.z[i] - np.sign(self.z[i]) * self.l1)
                denominator = (self.beta + np.sqrt(self.n[i])) / self.alpha + self.l2
                self.w[i] = numerator / denominator

    def predict(self, instance: Instance) -> int:
        """Predict class label."""
        self._update_weights()
        x = np.array(instance.x)
        prob = self._sigmoid(np.dot(self.w, x))
        return 1 if prob > 0.5 else 0

    def predict_proba(self, instance: Instance) -> np.ndarray:
        """Predict class probabilities."""
        self._update_weights()
        x = np.array(instance.x)
        prob_class1 = self._sigmoid(np.dot(self.w, x))
        return np.array([1 - prob_class1, prob_class1])

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Sigmoid with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def get_sparsity(self) -> float:
        """Get proportion of zero weights."""
        self._update_weights()
        return np.sum(np.abs(self.w) < 1e-8) / len(self.w)

    def get_weights(self) -> np.ndarray:
        """Get current weights."""
        self._update_weights()
        return self.w.copy()