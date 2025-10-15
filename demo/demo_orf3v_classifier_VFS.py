"""demo_orf3v_vfs.py - ORF3V with VFS mode"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import ORF3VClassifier
from capymoa.stream import EvolvingFeatureStream

# VFS 模式（10% 缺失）
stream = EvolvingFeatureStream(
    base_stream=Electricity(),
    evolution_pattern="vfs",
    missing_ratio=0.10,
    total_instances=10000,
    random_seed=42
)

learner = ORF3VClassifier(
    schema=stream.get_schema(),
    n_stumps=20,
    alpha=0.3,
    grace_period=200,
    replacement_interval=50,
    random_seed=42
)

print(f"Evaluating {learner} on EvolvingFeatureStream (VFS mode, 10% missing)")
print(f"Dataset: Electricity")
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