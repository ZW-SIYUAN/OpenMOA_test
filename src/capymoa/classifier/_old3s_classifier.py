"""_old3s_classifier.py - OLD³S Classifier for CapyMOA"""
from __future__ import annotations
from typing import Optional, Literal, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from capymoa.base import Classifier
from capymoa.stream import Schema
from capymoa.instance import Instance


# ======================== VAE Components ========================

class VAE_Shallow(nn.Module):
    """Variational Autoencoder for shallow feature spaces."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super(VAE_Shallow, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))  # ✅ 保留 sigmoid，配合 BCE 损失
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return z, x_recon, mu, logvar


# ======================== HBP MLP Components ========================

class HBPLayer(nn.Module):
    """Single layer block for HBP."""
    
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(HBPLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class HBPMLP(nn.Module):
    """Multi-layer perceptron with Hedge Backpropagation (HBP).
    
    Each layer outputs a prediction, and HBP adaptively weights them.
    """
    
    def __init__(self, input_dim: int, num_classes: int, num_layers: int = 5) -> None:
        super(HBPMLP, self).__init__()
        
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        
        current_dim = input_dim
        for _ in range(num_layers):
            self.hidden_layers.append(HBPLayer(current_dim, current_dim))
            self.output_layers.append(nn.Linear(current_dim, num_classes))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning predictions from all layers."""
        predictions = []
        hidden = x
        
        for i in range(self.num_layers):
            hidden = self.hidden_layers[i](hidden)
            pred = self.output_layers[i](hidden)
            predictions.append(pred)
        
        return predictions


# ======================== OLD³S Classifier ========================

class OLD3SClassifier(Classifier):
    """Online Learning Deep models from Data of Double Streams (OLD³S).
    
    This classifier handles evolving feature spaces using:
    - Variational Autoencoders (VAE) to learn shared latent representations
    - Hedge Backpropagation (HBP) for adaptive model depth
    - Ensemble learning to combine old and new feature space classifiers
    
    Reference:
    
    Lian, H., Wu, D., Hou, B.-J., Wu, J., & He, Y. (2024).
    Online Learning From Evolving Feature Spaces With Deep Variational Models.
    IEEE Transactions on Knowledge and Data Engineering, 36(8), 4144-4162.
    
    Example:
    
    >>> from capymoa.datasets import Electricity
    >>> from capymoa.classifier import OLD3SClassifier
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = Electricity()
    >>> schema = stream.get_schema()
    >>> learner = OLD3SClassifier(
    ...     schema=schema,
    ...     s1_feature_indices=[0, 1, 2, 3],
    ...     s2_feature_indices=[2, 3, 4, 5],
    ...     overlap_size=500,
    ...     switch_point=5000,
    ...     latent_dim=20,
    ...     hidden_dim=128
    ... )
    >>> results = prequential_evaluation(stream, learner, max_instances=10000)
    """
    
    def __init__(
        self,
        schema: Schema,
        s1_feature_indices: List[int],
        s2_feature_indices: List[int],
        overlap_size: int = 500,
        switch_point: int = 5000,
        latent_dim: int = 20,
        hidden_dim: int = 128,
        num_hbp_layers: int = 5,
        learning_rate: float = 0.001,
        beta: float = 0.9,
        eta: float = -0.05,
        random_seed: int = 1,
    ) -> None:
        """Initialize OLD³S Classifier.
        
        :param schema: Stream schema
        :param s1_feature_indices: Feature indices for first space S1
        :param s2_feature_indices: Feature indices for second space S2
        :param overlap_size: Number of instances in overlapping period B
        :param switch_point: Instance number where S1 -> S2 transition occurs
        :param latent_dim: Dimensionality of VAE latent space
        :param hidden_dim: Hidden layer size for VAE
        :param num_hbp_layers: Number of layers in HBP classifier
        :param learning_rate: Learning rate for model updates
        :param beta: Decay rate for HBP hedge weights
        :param eta: Exponential weight parameter for ensemble
        :param random_seed: Random seed
        """
        super().__init__(schema=schema, random_seed=random_seed)
        
        # ============ Input Validation ============
        self._validate_inputs(
            s1_feature_indices, s2_feature_indices, overlap_size, 
            switch_point, latent_dim, hidden_dim, num_hbp_layers,
            learning_rate, beta, eta
        )
        
        # Set seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature space configuration
        self.s1_indices = np.array(s1_feature_indices)
        self.s2_indices = np.array(s2_feature_indices)
        self.d1 = len(s1_feature_indices)
        self.d2 = len(s2_feature_indices)
        
        # Temporal configuration
        self.B = overlap_size
        self.T1 = switch_point
        self.overlap_start = self.T1 - self.B
        
        # Model hyperparameters
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_hbp_layers = num_hbp_layers
        self.lr = learning_rate
        self.beta = beta
        self.eta = eta
        
        # Number of classes
        self.num_classes = schema.get_num_classes()
        
        # Initialize VAE models
        self.vae_s1 = VAE_Shallow(self.d1, hidden_dim, latent_dim).to(self.device)
        self.vae_s2 = VAE_Shallow(self.d2, hidden_dim, latent_dim).to(self.device)
        
        # Initialize HBP classifiers
        self.classifier_s1 = HBPMLP(latent_dim, self.num_classes, num_hbp_layers).to(self.device)
        self.classifier_s2 = HBPMLP(latent_dim, self.num_classes, num_hbp_layers).to(self.device)
        
        # Optimizers
        self.optimizer_vae_s1 = torch.optim.Adam(self.vae_s1.parameters(), lr=learning_rate)
        self.optimizer_vae_s2 = torch.optim.Adam(self.vae_s2.parameters(), lr=learning_rate)
        self.optimizer_clf_s1 = torch.optim.Adam(self.classifier_s1.parameters(), lr=learning_rate)
        self.optimizer_clf_s2 = torch.optim.Adam(self.classifier_s2.parameters(), lr=learning_rate)
        
        # HBP hedge weights (one per layer)
        self.alpha_s1 = torch.ones(num_hbp_layers, device=self.device) / num_hbp_layers
        self.alpha_s2 = torch.ones(num_hbp_layers, device=self.device) / num_hbp_layers
        
        # Ensemble weights
        self.ensemble_weight_s1 = 0.5
        self.ensemble_weight_s2 = 0.5
        
        # Cumulative loss tracking
        self.cumulative_loss_s1: List[float] = []
        self.cumulative_loss_s2: List[float] = []
        
        # Instance counter
        self.instance_count = 0
        
        # Flag for first instance validation
        self._validated = False
        
        # ✅ 统计量用于归一化（滑动窗口统计）
        self.s1_min = None
        self.s1_max = None
        self.s2_min = None
        self.s2_max = None

    def _validate_inputs(
        self,
        s1_indices: List[int],
        s2_indices: List[int],
        overlap_size: int,
        switch_point: int,
        latent_dim: int,
        hidden_dim: int,
        num_hbp_layers: int,
        learning_rate: float,
        beta: float,
        eta: float
    ) -> None:
        """Validate constructor inputs."""
        # Check feature indices
        if not s1_indices or not s2_indices:
            raise ValueError("Feature indices cannot be empty")
        
        if len(s1_indices) != len(set(s1_indices)):
            raise ValueError("S1 feature indices contain duplicates")
        
        if len(s2_indices) != len(set(s2_indices)):
            raise ValueError("S2 feature indices contain duplicates")
        
        if any(idx < 0 for idx in s1_indices):
            raise ValueError("S1 feature indices must be non-negative")
        
        if any(idx < 0 for idx in s2_indices):
            raise ValueError("S2 feature indices must be non-negative")
        
        # Check temporal parameters
        if overlap_size <= 0:
            raise ValueError(f"overlap_size must be positive, got {overlap_size}")
        
        if switch_point <= overlap_size:
            raise ValueError(
                f"switch_point ({switch_point}) must be greater than "
                f"overlap_size ({overlap_size})"
            )
        
        # Check model dimensions
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        
        if num_hbp_layers <= 0:
            raise ValueError(f"num_hbp_layers must be positive, got {num_hbp_layers}")
        
        # Check hyperparameters
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        
        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")

    def _validate_instance_features(self, instance: Instance) -> None:
        """Validate instance features against configured indices."""
        if self._validated:
            return
        
        x_full = np.array(instance.x)
        num_features = len(x_full)
        
        max_s1_idx = self.s1_indices.max()
        max_s2_idx = self.s2_indices.max()
        
        if max_s1_idx >= num_features:
            raise ValueError(
                f"S1 feature index {max_s1_idx} exceeds available features "
                f"(0-{num_features - 1})"
            )
        
        if max_s2_idx >= num_features:
            raise ValueError(
                f"S2 feature index {max_s2_idx} exceeds available features "
                f"(0-{num_features - 1})"
            )
        
        self._validated = True

    def _normalize_to_01(self, x: np.ndarray, indices: np.ndarray, is_s1: bool) -> np.ndarray:
        """归一化特征到 [0, 1] 范围（Min-Max normalization）
        
        :param x: 完整特征向量
        :param indices: 要提取的特征索引
        :param is_s1: 是否为 S1 特征空间
        :return: 归一化后的特征
        """
        x_subset = x[indices]
        
        # 更新统计量（简单的滑动最小最大值）
        if is_s1:
            if self.s1_min is None:
                self.s1_min = x_subset.copy()
                self.s1_max = x_subset.copy()
            else:
                self.s1_min = np.minimum(self.s1_min, x_subset)
                self.s1_max = np.maximum(self.s1_max, x_subset)
            
            min_val = self.s1_min
            max_val = self.s1_max
        else:
            if self.s2_min is None:
                self.s2_min = x_subset.copy()
                self.s2_max = x_subset.copy()
            else:
                self.s2_min = np.minimum(self.s2_min, x_subset)
                self.s2_max = np.maximum(self.s2_max, x_subset)
            
            min_val = self.s2_min
            max_val = self.s2_max
        
        # Min-Max 归一化到 [0, 1]
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0  # 避免除零
        x_normalized = (x_subset - min_val) / range_val
        
        return x_normalized

    def __str__(self) -> str:
        return (f"OLD3SClassifier(s1_dim={self.d1}, s2_dim={self.d2}, "
                f"latent_dim={self.latent_dim}, overlap={self.B}, switch={self.T1})")

    def train(self, instance: Instance) -> None:
        """Train the classifier on a single instance."""
        # Validate features on first instance
        self._validate_instance_features(instance)
        
        self.instance_count += 1
        t = self.instance_count
        
        # Extract features
        x_full = np.array(instance.x, dtype=np.float32)
        
        # Check for NaN or Inf
        if not np.isfinite(x_full).all():
            raise ValueError(f"Instance {t} contains NaN or Inf values")
        
        # ✅ 归一化到 [0, 1]
        x_s1_normalized = self._normalize_to_01(x_full, self.s1_indices, is_s1=True)
        x_s2_normalized = self._normalize_to_01(x_full, self.s2_indices, is_s1=False)
        
        x_s1 = torch.tensor(x_s1_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_s2 = torch.tensor(x_s2_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get true label
        y = torch.tensor([instance.y_index], dtype=torch.long).to(self.device)
        
        # ============ Stage 1: Only S1 available ============
        if t <= self.overlap_start:
            self._train_vae_and_classifier(
                self.vae_s1, self.classifier_s1, x_s1, y,
                self.optimizer_vae_s1, self.optimizer_clf_s1,
                self.alpha_s1
            )
        
        # ============ Overlap period: Both S1 and S2 ============
        elif t <= self.T1:
            # Train S1 VAE and classifier
            self._train_vae_and_classifier(
                self.vae_s1, self.classifier_s1, x_s1, y,
                self.optimizer_vae_s1, self.optimizer_clf_s1,
                self.alpha_s1
            )
            
            # Train S2 VAE with alignment to S1
            with torch.no_grad():
                z_s1, _, mu_s1, logvar_s1 = self.vae_s1(x_s1)
            
            z_s2, x_s2_recon, mu_s2, logvar_s2 = self.vae_s2(x_s2)
            
            # VAE loss: reconstruction + KL divergence + alignment
            vae_loss_s2 = self._vae_loss(x_s2_recon, x_s2, mu_s2, logvar_s2)
            alignment_loss = F.smooth_l1_loss(z_s2, z_s1.detach())
            total_loss_s2 = vae_loss_s2 + alignment_loss
            
            self.optimizer_vae_s2.zero_grad()
            total_loss_s2.backward()
            self.optimizer_vae_s2.step()
        
        # ============ Stage 2: Only S2 available ============
        else:
            # 1. Train VAE first (independent forward pass)
            z_s2_vae, x_s2_recon, mu_s2, logvar_s2 = self.vae_s2(x_s2)
            vae_loss_s2 = self._vae_loss(x_s2_recon, x_s2, mu_s2, logvar_s2)
            self.optimizer_vae_s2.zero_grad()
            vae_loss_s2.backward()
            self.optimizer_vae_s2.step()
            
            # 2. Get latent representation for classifiers (fresh forward pass)
            with torch.no_grad():
                z_s2, _, _, _ = self.vae_s2(x_s2)
            
            # 3. Train both classifiers
            loss_s1 = self._train_classifier_hbp(
                self.classifier_s1, z_s2.detach(), y,
                self.optimizer_clf_s1, self.alpha_s1
            )
            
            loss_s2 = self._train_classifier_hbp(
                self.classifier_s2, z_s2, y,
                self.optimizer_clf_s2, self.alpha_s2
            )
            
            # 4. Update ensemble weights
            self.cumulative_loss_s1.append(loss_s1.item())
            self.cumulative_loss_s2.append(loss_s2.item())
            
            if len(self.cumulative_loss_s1) > 100:
                self.cumulative_loss_s1.pop(0)
                self.cumulative_loss_s2.pop(0)
            
            self._update_ensemble_weights()

    def predict(self, instance: Instance) -> int:
        """Make a prediction on an instance."""
        x_full = np.array(instance.x, dtype=np.float32)
        
        # Check for NaN or Inf
        if not np.isfinite(x_full).all():
            raise ValueError("Instance contains NaN or Inf values")
        
        # Check feature dimensions
        if max(self.s1_indices.max(), self.s2_indices.max()) >= len(x_full):
            raise ValueError(
                f"Instance has {len(x_full)} features, but indices require "
                f"at least {max(self.s1_indices.max(), self.s2_indices.max()) + 1}"
            )
        
        t = self.instance_count + 1
        
        with torch.no_grad():
            # Stage 1: Use S1 only
            if t <= self.T1:
                x_s1_normalized = self._normalize_to_01(x_full, self.s1_indices, is_s1=True)
                x_s1 = torch.tensor(x_s1_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
                z_s1, _, _, _ = self.vae_s1(x_s1)
                pred = self._hbp_predict(self.classifier_s1, z_s1, self.alpha_s1)
            
            # Stage 2: Ensemble S1 and S2
            else:
                x_s2_normalized = self._normalize_to_01(x_full, self.s2_indices, is_s1=False)
                x_s2 = torch.tensor(x_s2_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
                z_s2, _, _, _ = self.vae_s2(x_s2)
                
                pred_s1 = self._hbp_predict(self.classifier_s1, z_s2, self.alpha_s1)
                pred_s2 = self._hbp_predict(self.classifier_s2, z_s2, self.alpha_s2)
                
                # Weighted ensemble
                pred = (self.ensemble_weight_s1 * pred_s1 + 
                       self.ensemble_weight_s2 * pred_s2)
        
        return int(torch.argmax(pred, dim=1).item())

    def predict_proba(self, instance: Instance) -> np.ndarray:
        """Predict class probabilities."""
        x_full = np.array(instance.x, dtype=np.float32)
        
        # Check for NaN or Inf
        if not np.isfinite(x_full).all():
            raise ValueError("Instance contains NaN or Inf values")
        
        # Check feature dimensions
        if max(self.s1_indices.max(), self.s2_indices.max()) >= len(x_full):
            raise ValueError(
                f"Instance has {len(x_full)} features, but indices require "
                f"at least {max(self.s1_indices.max(), self.s2_indices.max()) + 1}"
            )
        
        t = self.instance_count + 1
        
        with torch.no_grad():
            if t <= self.T1:
                x_s1_normalized = self._normalize_to_01(x_full, self.s1_indices, is_s1=True)
                x_s1 = torch.tensor(x_s1_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
                z_s1, _, _, _ = self.vae_s1(x_s1)
                pred = self._hbp_predict(self.classifier_s1, z_s1, self.alpha_s1)
            else:
                x_s2_normalized = self._normalize_to_01(x_full, self.s2_indices, is_s1=False)
                x_s2 = torch.tensor(x_s2_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
                z_s2, _, _, _ = self.vae_s2(x_s2)
                
                pred_s1 = self._hbp_predict(self.classifier_s1, z_s2, self.alpha_s1)
                pred_s2 = self._hbp_predict(self.classifier_s2, z_s2, self.alpha_s2)
                
                pred = (self.ensemble_weight_s1 * pred_s1 + 
                       self.ensemble_weight_s2 * pred_s2)
        
        probs = F.softmax(pred, dim=1).squeeze(0).cpu().numpy()
        return probs

    # ============ Helper Methods ============
    
    def _vae_loss(
        self, 
        x_recon: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Compute VAE loss: BCE + KL divergence (符合原论文)."""
        bce_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce_loss + kl_loss
    
    def _train_vae_and_classifier(
        self, 
        vae: VAE_Shallow, 
        classifier: HBPMLP, 
        x: torch.Tensor, 
        y: torch.Tensor,
        optimizer_vae: torch.optim.Optimizer, 
        optimizer_clf: torch.optim.Optimizer, 
        alpha: torch.Tensor
    ) -> None:
        """Train VAE and classifier together."""
        # Forward VAE
        z, x_recon, mu, logvar = vae(x)
        
        # VAE loss
        vae_loss = self._vae_loss(x_recon, x, mu, logvar)
        optimizer_vae.zero_grad()
        vae_loss.backward()
        optimizer_vae.step()
        
        # Classifier loss with HBP
        with torch.no_grad():
            z, _, _, _ = vae(x)
        
        self._train_classifier_hbp(classifier, z, y, optimizer_clf, alpha)
    
    def _train_classifier_hbp(
        self, 
        classifier: HBPMLP, 
        z: torch.Tensor, 
        y: torch.Tensor, 
        optimizer: torch.optim.Optimizer, 
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """Train classifier using Hedge Backpropagation."""
        predictions = classifier(z)
        
        # Compute loss for each layer
        losses = []
        for pred in predictions:
            loss = F.cross_entropy(pred, y)
            losses.append(loss)
        
        # Weighted loss based on hedge weights
        weighted_loss = sum(alpha[i] * losses[i] for i in range(len(losses)))
        
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        
        # Update hedge weights
        with torch.no_grad():
            for i in range(len(losses)):
                alpha[i] *= torch.pow(torch.tensor(self.beta, device=self.device), losses[i])
            alpha[:] = alpha / alpha.sum()
        
        return weighted_loss
    
    def _hbp_predict(
        self, 
        classifier: HBPMLP, 
        z: torch.Tensor, 
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """Make prediction using HBP ensemble."""
        predictions = classifier(z)
        weighted_pred = sum(alpha[i] * predictions[i] for i in range(len(predictions)))
        return weighted_pred
    
    def _update_ensemble_weights(self) -> None:
            """Update ensemble weights using exponential weights."""
            if len(self.cumulative_loss_s1) > 0:
                sum_loss_s1 = sum(self.cumulative_loss_s1)
                sum_loss_s2 = sum(self.cumulative_loss_s2)
                
                try:
                    exp_s1 = np.exp(self.eta * sum_loss_s1)
                    exp_s2 = np.exp(self.eta * sum_loss_s2)
                    
                    total = exp_s1 + exp_s2
                    self.ensemble_weight_s1 = exp_s1 / total
                    self.ensemble_weight_s2 = exp_s2 / total
                except OverflowError:
                    # Keep current weights if overflow
                    pass