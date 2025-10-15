"""demo_ftrl_classifier.py"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import FTRLClassifier
import numpy as np

stream = Electricity()

# 更保守的参数
learner = FTRLClassifier(
    schema=stream.get_schema(),
    alpha=0.5,    # 降低学习率
    beta=1.0,
    l1=0.1,       # 增大 L1（更多稀疏性）
    l2=1.0,       # 增大 L2（更多平滑）
    random_seed=42
)

print(f"Evaluating {learner} on Electricity")
print("-" * 60)

results = prequential_evaluation(
    stream=stream,
    learner=learner,
    max_instances=10000,
    window_size=100
)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Accuracy:              {results['cumulative'].accuracy():.2f}%")
print(f"Kappa:                 {results['cumulative'].kappa():.4f}")
print(f"F1 Score:              {results['cumulative'].f1_score():.2f}%")
print(f"Precision:             {results['cumulative'].precision():.2f}%")
print(f"Recall:                {results['cumulative'].recall():.2f}%")

print("\n" + "=" * 60)
print("SPARSITY STATISTICS")
print("=" * 60)
print(f"Model sparsity:        {learner.get_sparsity():.2%}")
print(f"Non-zero weights:      {np.sum(np.abs(learner.get_weights()) > 1e-8)}")

weights = learner.get_weights()
print(f"\nWeight statistics:")
print(f"  Mean (abs):          {np.mean(np.abs(weights)):.4f}")
print(f"  Max (abs):           {np.max(np.abs(weights)):.4f}")