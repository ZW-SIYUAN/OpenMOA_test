"""demo_ftrl_vs_fobos.py - Compare FTRL and FOBOS"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import FTRLClassifier, FOBOSClassifier
import numpy as np

print("Comparing FTRL vs FOBOS on Electricity")
print("=" * 60)

# FOBOS
print("\n1. FOBOS")
print("-" * 60)
stream = Electricity()
fobos = FOBOSClassifier(
    schema=stream.get_schema(),
    alpha=1.0,
    lambda_=0.01,
    random_seed=42
)
results_fobos = prequential_evaluation(
    stream=stream,
    learner=fobos,
    max_instances=10000,
    window_size=100,
    progress_bar=False
)
print(f"Accuracy:  {results_fobos['cumulative'].accuracy():.2f}%")
print(f"Kappa:     {results_fobos['cumulative'].kappa():.4f}")
print(f"Sparsity:  {fobos.get_sparsity():.2%}")

# FTRL
print("\n2. FTRL")
print("-" * 60)
stream = Electricity()
ftrl = FTRLClassifier(
    schema=stream.get_schema(),
    alpha=0.5,
    beta=1.0,
    l1=0.1,
    l2=1.0,
    random_seed=42
)
results_ftrl = prequential_evaluation(
    stream=stream,
    learner=ftrl,
    max_instances=10000,
    window_size=100,
    progress_bar=False
)
print(f"Accuracy:  {results_ftrl['cumulative'].accuracy():.2f}%")
print(f"Kappa:     {results_ftrl['cumulative'].kappa():.4f}")
print(f"Sparsity:  {ftrl.get_sparsity():.2%}")

# 总结
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Algorithm':<15} {'Accuracy':<12} {'Kappa':<12} {'Sparsity'}")
print("-" * 60)
print(f"{'FOBOS':<15} {results_fobos['cumulative'].accuracy():<12.2f} "
      f"{results_fobos['cumulative'].kappa():<12.4f} {fobos.get_sparsity():.2%}")
print(f"{'FTRL':<15} {results_ftrl['cumulative'].accuracy():<12.2f} "
      f"{results_ftrl['cumulative'].kappa():<12.4f} {ftrl.get_sparsity():.2%}")

diff_acc = results_ftrl['cumulative'].accuracy() - results_fobos['cumulative'].accuracy()
print(f"\nFTRL improvement: {diff_acc:+.2f}%")