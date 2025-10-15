"""Simple Demo for OVFM Classifier"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

import numpy as np
from capymoa.datasets import Electricity
from capymoa.stream import CapriciousStream  # 使用新流类
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import OVFMClassifier

print("=" * 70)
print("OVFM Classifier Demo - Capricious Stream (VFS)")
print("=" * 70)

# Setup
base_stream = Electricity()
schema = base_stream.schema

print(f"\n[Stream Info]")
print(f"  Base features: {schema.get_num_attributes()}")
print(f"  Classes: {schema.get_num_classes()} {schema.get_label_values()}")

stream = CapriciousStream(
    base_stream=base_stream,
    missing_ratio=0.1,
    total_instances=10000,
    min_features=2,
    random_seed=42
)

learner = OVFMClassifier(
    schema=schema,
    window_size=300,
    batch_size=100,
    evolution_pattern="vfs",
    decay_coef=0.6,
    learning_rate=0.005,
    l2_lambda=0.001,
    random_seed=42
)

print(f"\n[Classifier Config]")
print(f"  Window: {learner.window_size}, Batch: {learner.batch_size}")
print(f"  Learning rate: {learner.learning_rate}")

# Evaluation
print("\n[Running Evaluation...]")
print("-" * 70)

try:
    results = prequential_evaluation(
        stream=stream,
        learner=learner,
        max_instances=10000,
        window_size=500,
        progress_bar=True
    )
    
    # Process remaining buffer
    learner.finalize_training()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    if learner._trained:
        print(f"Debug: Processed {learner._num_updates} instances")
    raise

# Results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

cumulative = results['cumulative']
print(f"\n[Performance]")
print(f"  Accuracy: {cumulative.accuracy():.4f}")
print(f"  Kappa:    {cumulative.kappa():.4f}")

print(f"\n[Confusion Matrix]")
cm = cumulative.confusion_matrix  # 移除括号
if cm is not None and len(cm) == 2:
    print(f"  [[{cm[0][0]:4d}, {cm[0][1]:4d}]")
    print(f"   [{cm[1][0]:4d}, {cm[1][1]:4d}]]")
    
    # 手动计算每类指标
    print(f"\n[Per-Class Metrics]")
    # Class 0
    prec_0 = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    rec_0 = cm[0][0] / (cm[0][0] + cm[1][0]) if (cm[0][0] + cm[1][0]) > 0 else 0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0
    print(f"  Class 0: P={prec_0:.4f}, R={rec_0:.4f}, F1={f1_0:.4f}")
    
    # Class 1
    prec_1 = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    rec_1 = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0
    print(f"  Class 1: P={prec_1:.4f}, R={rec_1:.4f}, F1={f1_1:.4f}")
else:
    print("  Confusion matrix not available")

print(f"\n[OVFM Info]")
stats = learner.training_statistics
print(f"  Ensemble α:      {learner.ensemble_weight:.4f}")
print(f"  Total updates:   {stats['num_updates']}")
print(f"  Loss (obs/lat):  {stats['loss_observed']:.1f} / {stats['loss_latent']:.1f}")
print(f"  Features:        {stats['num_continuous']} cont, {stats['num_ordinal']} ord")

print("\n" + "=" * 70)
print("✓ Demo completed")
print("=" * 70)