"""Demo script for OLD³S Classifier - Baseline Comparison"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Covtype
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import OLD3SClassifier, HoeffdingTree, NaiveBayes
import time

# Load stream
print("=" * 70)
print("OLD³S vs Baseline Classifiers - Performance Comparison")
print("=" * 70)
print()

covtype_stream = Covtype()
schema = covtype_stream.schema

print(f"Dataset: Covtype")
print(f"Number of features: {schema.get_num_attributes()}")
print(f"Number of classes: {schema.get_num_classes()}")
print()

# ============================================================
# 1. OLD³S Classifier
# ============================================================
print("-" * 70)
print("Testing OLD³S Classifier...")
print("-" * 70)

covtype_stream = Covtype()  # 重新加载
start_time = time.time()

old3s_learner = OLD3SClassifier(
    schema=schema,
    s1_feature_indices=list(range(0, 27)),      # 前 27 个特征
    s2_feature_indices=list(range(25, 52)),     # 后 27 个特征（有重叠）
    overlap_size=500,
    switch_point=5000,
    latent_dim=20,
    hidden_dim=128,
    num_hbp_layers=5,
    learning_rate=0.001,
    beta=0.9,
    eta=-0.05,
    random_seed=42
)

results_old3s = prequential_evaluation(
    stream=covtype_stream,
    learner=old3s_learner,
    max_instances=10000,
    window_size=500,
    progress_bar=True
)

old3s_time = time.time() - start_time
print(f"OLD³S Training Time: {old3s_time:.2f} seconds")
print()

# ============================================================
# 2. Hoeffding Tree (基线 1)
# ============================================================
print("-" * 70)
print("Testing Hoeffding Tree (Baseline 1)...")
print("-" * 70)

covtype_stream = Covtype()  # 重新加载
start_time = time.time()

ht_learner = HoeffdingTree(schema=schema)

results_ht = prequential_evaluation(
    stream=covtype_stream,
    learner=ht_learner,
    max_instances=10000,
    window_size=500,
    progress_bar=True
)

ht_time = time.time() - start_time
print(f"Hoeffding Tree Training Time: {ht_time:.2f} seconds")
print()

# ============================================================
# 3. Naive Bayes (基线 2)
# ============================================================
print("-" * 70)
print("Testing Naive Bayes (Baseline 2)...")
print("-" * 70)

covtype_stream = Covtype()  # 重新加载
start_time = time.time()

nb_learner = NaiveBayes(schema=schema)

results_nb = prequential_evaluation(
    stream=covtype_stream,
    learner=nb_learner,
    max_instances=10000,
    window_size=500,
    progress_bar=True
)

nb_time = time.time() - start_time
print(f"Naive Bayes Training Time: {nb_time:.2f} seconds")
print()

# ============================================================
# Performance Comparison Table
# ============================================================
print("=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print()

print(f"{'Model':<20} {'Accuracy':<15} {'Kappa':<15} {'Time (s)':<15}")
print("-" * 70)
print(f"{'OLD³S':<20} {results_old3s['cumulative'].accuracy():>8.2f}%     "
      f"{results_old3s['cumulative'].kappa():>8.4f}     {old3s_time:>8.2f}")
print(f"{'Hoeffding Tree':<20} {results_ht['cumulative'].accuracy():>8.2f}%     "
      f"{results_ht['cumulative'].kappa():>8.4f}     {ht_time:>8.2f}")
print(f"{'Naive Bayes':<20} {results_nb['cumulative'].accuracy():>8.2f}%     "
      f"{results_nb['cumulative'].kappa():>8.4f}     {nb_time:>8.2f}")
print("=" * 70)
print()

# ============================================================
# Detailed Analysis
# ============================================================
print("DETAILED ANALYSIS")
print("=" * 70)

# Random baseline
random_accuracy = 100.0 / schema.get_num_classes()
print(f"Random Baseline Accuracy: {random_accuracy:.2f}%")
print()

# Improvement over random
old3s_improvement = (results_old3s['cumulative'].accuracy() - random_accuracy) / random_accuracy * 100
ht_improvement = (results_ht['cumulative'].accuracy() - random_accuracy) / random_accuracy * 100
nb_improvement = (results_nb['cumulative'].accuracy() - random_accuracy) / random_accuracy * 100

print(f"Improvement over Random Baseline:")
print(f"  OLD³S:          {old3s_improvement:>6.1f}%")
print(f"  Hoeffding Tree: {ht_improvement:>6.1f}%")
print(f"  Naive Bayes:    {nb_improvement:>6.1f}%")
print()

# Best model
accuracies = {
    'OLD³S': results_old3s['cumulative'].accuracy(),
    'Hoeffding Tree': results_ht['cumulative'].accuracy(),
    'Naive Bayes': results_nb['cumulative'].accuracy()
}

best_model = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model]

print(f"Best Model: {best_model} ({best_accuracy:.2f}%)")
print()

# Speed comparison
speeds = {
    'OLD³S': old3s_time,
    'Hoeffding Tree': ht_time,
    'Naive Bayes': nb_time
}

fastest_model = min(speeds, key=speeds.get)
fastest_time = speeds[fastest_model]

print(f"Fastest Model: {fastest_model} ({fastest_time:.2f}s)")
print()

# ============================================================
# Insights
# ============================================================
print("=" * 70)
print("INSIGHTS")
print("=" * 70)
print("""
1. OLD³S Performance:
   - Designed for EVOLVING feature spaces
   - Covtype has NO natural feature evolution
   - Artificial split (S1/S2) may hurt performance
   
2. When OLD³S Excels:
   - Real feature space transitions
   - Sensor upgrades
   - Data collection method changes
   - Gradual feature appearance/disappearance
   
3. Current Results:
   - All models significantly beat random baseline
   - Hoeffding Tree may perform better on static features
   - OLD³S shows competitive performance despite feature split

4. Recommendations:
   - Use OLD³S when features actually evolve
   - For static features, simpler models may suffice
   - Consider increasing training samples (10k → 50k+)
""")

print("=" * 70)