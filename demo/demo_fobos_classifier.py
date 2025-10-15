"""demo_fobos_classifier.py - FOBOS Classifier Demo"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Hyper100k
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import FOBOSClassifier
import numpy as np

stream = Hyper100k()

# 使用更合理的参数
learner = FOBOSClassifier(
    schema=stream.get_schema(),
    alpha=1.0,       # 更大的基础学习率
    lambda_=0.01,    # 适中的正则化
    random_seed=42
)

print(f"Evaluating {learner} on Hyper100k")
print(f"Task: Binary classification with L1 regularization")
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
print(f"Kappa Temporal:        {results['cumulative'].kappa_t():.4f}")
print(f"Kappa M:               {results['cumulative'].kappa_m():.4f}")
print(f"F1 Score:              {results['cumulative'].f1_score():.2f}%")
print(f"Precision:             {results['cumulative'].precision():.2f}%")
print(f"Recall:                {results['cumulative'].recall():.2f}%")
print(f"Instances processed:   {results['cumulative'].instances_seen}")
print(f"Correctly classified:  {results['cumulative'].correctly_classified}")
print(f"Incorrectly classified: {results['cumulative'].incorrectly_classified}")

print("\n" + "=" * 60)
print("SPARSITY STATISTICS")
print("=" * 60)
print(f"Model sparsity:        {learner.get_sparsity():.2%}")
print(f"Non-zero weights:      {np.sum(np.abs(learner.get_weights()) > 1e-8)}")
print(f"Total weights:         {len(learner.get_weights())}")

weights = learner.get_weights()
print(f"\nWeight statistics:")
print(f"  Mean (abs):          {np.mean(np.abs(weights)):.4f}")
print(f"  Max (abs):           {np.max(np.abs(weights)):.4f}")
if np.any(np.abs(weights) > 1e-8):
    print(f"  Min (non-zero):      {np.min(np.abs(weights[np.abs(weights) > 1e-8])):.4f}")