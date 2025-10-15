"""Demo script for FESL Classifier - Simplified Version"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import FESLClassifier

# Load stream
elec_stream = Electricity()
schema = elec_stream.schema

print(f"Number of features: {schema.get_num_attributes()}")

# Create FESL learner
fesl_learner = FESLClassifier(
    schema=schema,
    s1_feature_indices=[0, 1, 2, 3],
    s2_feature_indices=[2, 3, 4, 5],
    overlap_size=100,
    switch_point=5000,
    ensemble_method="selection",
    learning_rate_scale=0.1,
    random_seed=42
)

# Run evaluation with progress bar
results = prequential_evaluation(
    stream=elec_stream,
    learner=fesl_learner,
    max_instances=10000,
    window_size=100,
    progress_bar=True  # 开启进度条
)

# Print final results
print(f"Ensemble Method: {fesl_learner.ensemble_method}")
print(f"\nFinal Accuracy: {results['cumulative'].accuracy():.2f}%")
print(f"Kappa: {results['cumulative'].kappa():.4f}")